
// FIXME inconsistent usage of DIMENSION and state.dimension
const DIMENSION: Trip<Float> = (500., 500., 100.);
const NPARTICLE: usize = 1500;

const CORE_RADIUS: Float = 5.0;
const INTRA_CHAIN_SEP: Cart = Cart(1.);
const PARTICLE_RADIUS: Cart = Cart(1.);
const MOVE_RADIUS: Cart = Cart(1.);

const THETA_STRENGTH: Float = 1.0;
const RADIUS_STRENGTH: Float = 1.0;
const TARGET_RADIUS: Float = 1.0;

// could use an enum here but bleh
const LABEL_SILICON: &'static str = "Si";
const LABEL_CARBON:  &'static str = "C";

extern crate time;
extern crate rand;
#[macro_use(zip_with)]
extern crate homogenous;
#[macro_use(iproduct)]
extern crate itertools;
#[macro_use]
extern crate newtype_ops;
extern crate dla_sys;
extern crate libc;

mod sp2;

use rand::Rng;
use rand::distributions::{IndependentSample,Normal};
use itertools::Itertools;
use homogenous::prelude::*;
use homogenous::numeric::prelude::*;
use time::precise_time_ns;

use std::ops::Range;
use std::io::Write;
use std::io::stderr;

type Float = f64;
type Pair<T> = (T,T);
type Trip<T> = (T,T,T);

//--------------------
// For statically proving that fractional/cartesian conversions are handled properly.
#[derive(Debug,PartialEq,PartialOrd,Copy,Clone)]
pub struct Frac(Float);
#[derive(Debug,PartialEq,PartialOrd,Copy,Clone)]
pub struct Cart(Float);
impl Frac { pub fn cart(self, dimension: Float) -> Cart { Cart(self.0*dimension) } }
impl Cart { pub fn frac(self, dimension: Float) -> Frac { Frac(self.0/dimension) } }

// add common binops to eliminate the majority of reasons I might need to
// convert back into floats (which would render the type system useless)
newtype_ops!{ {[Frac][Cart]} arithmetic {:=} {^&}Self {^&}{Self Float} }

// cart() and frac() methods for triples
trait ToCart { fn cart(self, dimension: Trip<Float>) -> Trip<Cart>; }
trait ToFrac { fn frac(self, dimension: Trip<Float>) -> Trip<Frac>; }
impl ToCart for Trip<Frac> { fn cart(self, dimension: Trip<Float>) -> Trip<Cart> { zip_with!((self,dimension), |x:Frac,d| x.cart(d)) } }
impl ToFrac for Trip<Cart> { fn frac(self, dimension: Trip<Float>) -> Trip<Frac> { zip_with!((self,dimension), |x:Cart,d| x.frac(d)) } }

fn reduce_pbc(this: Trip<Frac>) -> Trip<Frac> { this.map(|Frac(x)| Frac((x + 1.0).fract())) }

//--------------------
// fulfills two needs which BTreeMap fails to satisfy:
//  * support for PartialOrd
//  * multiple values may have same key
// and also has a few oddly-placed bits of logic specific to our application.
type Key = Frac;
type Value = usize;
struct SortedIndices {keys: Vec<Key>, values: Vec<Value>}
impl SortedIndices {
	fn new() -> Self {
		// specific to our application:
		//    Keys at the far reaches of outer space help simplify edge cases.
		//    The corresponding values should never be used.
		SortedIndices {
			keys:   vec![Frac(::std::f64::NEG_INFINITY), Frac(::std::f64::INFINITY)],
			values: vec![0, ::std::usize::MAX],
		}
	}

	// initialize with indices into an enumerable sequence
	fn rebuild<I:IntoIterator<Item=Key>>(iter: I) -> Self {
		let mut this = SortedIndices::new();
		for (v,x) in iter.into_iter().enumerate() {
			this.insert_images(x, v);
		}
		this
	}

	// specific to our application;
	//    Images one period above and below are stored to simplify edge cases.
	fn insert_images(&mut self, k: Key, v: Value) {
		assert!(Frac(0.) <= k && k <= Frac(1.));
		self.insert(k, v);
		self.insert(k - Frac(1.), v);
		self.insert(k + Frac(1.), v);
	}

	fn insert(&mut self, k: Key, v: Value) {
		let i = self.lower_bound(k);
		self.keys.insert(i, k); self.values.insert(i, v);
	}

	fn lower_bound(&self, k: Key) -> usize {
		match self.keys.binary_search_by(|b| b.partial_cmp(&k).unwrap()) {
			Ok(x) => x, Err(x) => x,
		}
	}
}

fn update_lower_hint(mut hint: usize, sorted: &[Frac], needle: Frac) -> usize {
	while needle <= sorted[hint] { hint -= 1; }
	while needle >  sorted[hint] { hint += 1; }
	hint
}
fn update_upper_hint(mut hint: usize, sorted: &[Frac], needle: Frac) -> usize {
	while needle <  sorted[hint] { hint -= 1; }
	while needle >= sorted[hint] { hint += 1; }
	hint
}

fn intersection(a: Vec<usize>, b: Vec<usize>) -> Vec<usize> {
	// hm, can't find anything on cargo. Itertools only has unions (merge).
	// we'll do O(m*n) because the sets are almost always expected to be size 0.
	a.into_iter().filter(|x| b.iter().any(|y| x == y)).collect()
}

//--------------------------
struct State {
	labels: Vec<&'static str>,
	positions: Vec<Trip<Cart>>,
	dimension: Trip<Float>,
	// tracks x +/- move_radius index on each axis
	hints: Trip<Range<usize>>,
	// Contains images in the fractional range [-1, 2] along each axis
	sorted: Trip<SortedIndices>,
}

impl State {
	fn new(dimension: Trip<Float>) -> State {
		State {
			labels: vec![],
			positions: vec![],
			hints: ((),(),()).map(|_| 0..0),
			dimension: dimension,
			sorted: ((),(),()).map(|_| SortedIndices::new()),
		}
	}

	fn update_cursors(&mut self, frac: Trip<Frac>, radius: Cart) {
		let radii = self.dimension.map(|d| radius.frac(d));

		zip_with!((frac, radii, self.sorted.as_ref(), self.hints.as_mut()),
		|x,r,set: &SortedIndices,hints: &mut Range<usize>| {
			hints.start = update_lower_hint(hints.start, &set.keys, x - r);
			hints.end   = update_upper_hint(hints.end,   &set.keys, x + r);
		});
	}

	fn insert(&mut self, label: &'static str, point: Trip<Frac>) {
		let i = self.positions.len();
		self.positions.push(point.cart(self.dimension));
		self.labels.push(label);

		zip_with!((point, self.sorted.as_mut()), |x, set: &mut SortedIndices| {
			set.insert_images(x, i);
		});
	}

	// returns neighborhood size for debug info
	fn relax_neighborhood(&mut self, center: Trip<Frac>, radius: Cart) -> usize {
		let mut fixed = vec![true; self.positions.len()];
		for i in self.neighborhood(center, radius) {
			if self.labels[i] != LABEL_SILICON {
				fixed[i] = false;
			}
		}

		let n_free = fixed.iter().filter(|&x| !x).count();

		let State { ref mut positions, ref mut sorted, dimension, .. } = *self;

		*positions = sp2::relax(positions.clone(), fixed, dimension);

		// Relaxation is infrequent; we shall just rebuild the index lists from scratch.
		sorted.as_mut().enumerate().map(|(axis, set)| {
			let projected: Vec<_> = positions.iter()
				.map(|&x| x.frac(dimension))
				.map(|x| x.into_nth(axis))
				.collect();
			*set = SortedIndices::rebuild(projected);
		});

		n_free
	}


	fn cursor_neighborhood(&self) -> Vec<usize> {
		// should we even really bother?
		if self.hints.clone().any(|x| x.len() == 0) { return vec![]; }

		zip_with!((self.sorted.as_ref(), self.hints.clone()),
		|set: &SortedIndices, range: Range<_>| {
			set.values[range].to_vec()
		}).fold1(intersection)
	}

	fn neighborhood_from_candidates<I: IntoIterator<Item=usize>>(&self, frac: Trip<Frac>, radius: Cart, indices: I) -> Vec<usize> {
		let cart = frac.cart(self.dimension);

		indices.into_iter().filter(|&i| {
			let displacement = cart.sub_v(self.positions[i]);
			displacement.sqnorm() <= radius*radius
		}).collect()
	}

	fn neighborhood(&mut self, point: Trip<Frac>, radius: Cart) -> Vec<usize> {
		self.update_cursors(point, radius);
		let candidates = self.cursor_neighborhood();
		self.neighborhood_from_candidates(point, radius, candidates)
	}

	fn bruteforce_neighborhood(&self, point: Trip<Frac>, radius: Cart) -> Vec<usize> {
		self.neighborhood_from_candidates(point, radius, 0..self.positions.len())
	}
}

fn output<W: Write>(state: &State, file: &mut W) {
	writeln!(file, "[").unwrap();
	let mut first = true;
	for (&(Cart(x),Cart(y),Cart(z)), label) in state.positions.iter().zip(&state.labels) {
		write!(file, "{}", if first { "" } else { ",\n " }).unwrap();
		write!(file, "[{:?},[{},{},{}]]", label, x, y, z).unwrap();
		first = false;
	}
	writeln!(file, "]").unwrap();
}

//---------- DLA

fn random_direction<R:Rng>(rng: &mut R) -> Trip<Cart> {
	let normal = Normal::new(0.0, 1.0);
	let x = normal.ind_sample(rng) as Float;
	let y = normal.ind_sample(rng) as Float;
	let z = normal.ind_sample(rng) as Float;

	let vec = (x,y,z);
	let length = vec.sqnorm().sqrt();
	vec.map(|x| Cart(x / length))
}

fn random_border_position<R:Rng>(rng: &mut R) -> Trip<Frac> {
	// this makes no attempt to be isotropic,
	// as evidenced by the fact that it works entirely in terms of fractional coords

	// place onto either the i=0 or j=0 face of the cuboid
	let o = Frac(0.);
	let x = Frac(rng.next_f64() as Float);
	let z = Frac(rng.next_f64() as Float);
	match rng.gen_range(0, 2) {
		0 => (x, o, z),
		1 => (o, x, z),
		_ => unreachable!(),
	}
}

fn add_nucleation_site(mut state: State) -> State {
	let n = (Cart(state.dimension.2) / INTRA_CHAIN_SEP).0.round() as i32;
	for i in 0i32..n {
		state.insert(LABEL_SILICON, (Frac(0.5), Frac(0.5), Frac(i as Float / n as Float)));
	}
	state
}

use std::collections::vec_deque::VecDeque;
struct Timer { deque: VecDeque<u64> }
impl Timer {
	pub fn new(n: usize) -> Timer {
		let mut this = Timer { deque: VecDeque::new() };
		// Fill solely for ease of implementation (the first few outputs may be inaccurate)
		while this.deque.len() < n { this.deque.push_back(precise_time_ns()) }
		this
	}
	pub fn push(&mut self) {
		self.deque.pop_front();
		self.deque.push_back(precise_time_ns());
	}
	pub fn last_ms(&self) -> u64 {
		(self.deque[self.deque.len()-1] - self.deque[self.deque.len()-2]) / 1000
	}
	pub fn average_ms(&self) -> u64 {
		(self.deque[self.deque.len()-1] - self.deque[0]) / ((self.deque.len() as u64 - 1) * 1000)
	}
}

fn dla_run() -> State {
	let state = State::new(DIMENSION);
	let state = add_nucleation_site(state);
	let mut state = state;

	let mut rng = rand::weak_rng();

	let nbr_radius = Cart(1.)*MOVE_RADIUS + PARTICLE_RADIUS;

	let mut timer = Timer::new(30);

	for n in 0..NPARTICLE {
		write!(stderr(), "Particle {:8} of {:8}: ", n, NPARTICLE).unwrap();

		// loop to skip relaxation failures
		'restart: loop {

			let mut pos = random_border_position(&mut rng);

			// move until ready to place
			while state.neighborhood(pos, nbr_radius).is_empty() {
				//writeln!(stderr(), "({:4},{:4},{:4})  ({:8?} ms)",
				//	(pos.0).0, (pos.1).0, (pos.2).0, (precise_time_ns() - start_time)/1000).unwrap();

				let disp = random_direction(&mut rng)
					.mul_s(MOVE_RADIUS).frac(state.dimension);

				pos = reduce_pbc(pos.add_v(disp));
			}

			// Place the particle.
			state.insert(LABEL_CARBON, pos);
			let n_free = state.relax_neighborhood(pos, MOVE_RADIUS*3.);

			// debugging info
			timer.push();
			writeln!(stderr(), "({:22},{:22},{:22})  ({:8?} ms)  (avg: {:8?} ms)  (relaxed {:8})",
				(pos.0).0, (pos.1).0, (pos.2).0, timer.last_ms(), timer.average_ms(), n_free
			).unwrap();

			break 'restart;
		}
	}
	state
}

fn main() {
	let state = dla_run();
	output(&state, &mut std::io::stdout());
}
