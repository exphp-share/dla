const DIMENSION: Trip<Float> = (240., 240., 40.);
const NPARTICLE: usize = 10000;

const CORE_RADIUS: Float = 5.0;
const INTRA_CHAIN_SEP: Cart = Cart(2.);
const PARTICLE_RADIUS: Cart = Cart(1.);
const MOVE_RADIUS: Cart = Cart(1.);

extern crate time;
extern crate rand;
#[macro_use(zip_with)]
extern crate homogenous;
#[macro_use(iproduct)]
extern crate itertools;

use rand::Rng;
use rand::distributions::{IndependentSample,Normal};
use homogenous::prelude::*;
use homogenous::numeric::prelude::*;
use time::precise_time_ns;

use std::io::Write;

type Float = f32;
type Trip<T> = (T,T,T);

// For statically proving that fractional/cartesian conversions are handled properly.
#[derive(Debug,PartialEq,PartialOrd,Copy,Clone)]
struct Frac(Float);
#[derive(Debug,PartialEq,PartialOrd,Copy,Clone)]
struct Cart(Float);
impl Frac { pub fn cart(self, dimension: Float) -> Cart { Cart(self.0*dimension) } }
impl Cart { pub fn frac(self, dimension: Float) -> Frac { Frac(self.0/dimension) } }

// add common binops to eliminate the majority of reasons I might need to
// convert back into floats (which would render the type system useless)
macro_rules! impl_binop { ($T:ident, $trt:ident, $func:ident, $op:tt) => {
	impl $trt<$T> for $T {
		type Output = $T;
		fn $func(self, other: $T) -> $T { $T(self.0 $op other.0) }
	}
	impl $trt<Float> for $T {
		type Output = $T;
		fn $func(self, other: Float) -> $T { $T(self.0 $op other) }
	}
};}
use ::std::ops::{Add,Sub,Mul,Div,Rem};
impl_binop!(Cart, Mul, mul, *);
impl_binop!(Cart, Add, add, +);
impl_binop!(Cart, Sub, sub, -);
impl_binop!(Cart, Div, div, /);
impl_binop!(Cart, Rem, rem, %);
impl_binop!(Frac, Mul, mul, *);
impl_binop!(Frac, Add, add, +);
impl_binop!(Frac, Sub, sub, -);
impl_binop!(Frac, Div, div, /);
impl_binop!(Frac, Rem, rem, %);

// fulfills two needs which BTreeMap fails to satisfy:
//  * support for PartialOrd
//  * multiple values may have same key
type Key = Frac;
type Value = usize;
struct SortedIndices {keys: Vec<Key>, values: Vec<Value>}
impl SortedIndices {
	fn new() -> Self { SortedIndices { keys: vec![], values: vec![] } }

	fn insert(&mut self, k: Key, v: Value) {
		let i = self.lower_bound(k);
		self.keys.insert(i, k); self.values.insert(i, v);
	}

	fn lower_bound(&self, k: Key) -> usize {
		match self.keys.binary_search_by(|b| b.partial_cmp(&k).unwrap()) {
			Ok(x) => x, Err(x) => x,
		}
	}

	fn upper_bound(&self, k: Key) -> usize {
		let i = self.lower_bound(k);
		for i in i+1..self.keys.len() {
			if self.keys[i] > k { return i; }
		}
		return self.keys.len();
	}
}

#[derive(Copy,Clone,Debug,Hash,Eq,PartialEq,Ord,PartialOrd)]
struct Cursor { index: usize }
impl Cursor {
	// since we have a good hint, linear search should outperform binary search
	pub fn update_as_lower(&mut self, set: &SortedIndices, key: Frac) {
		while key <= set.keys[self.index] { self.index -= 1; }
		while key >  set.keys[self.index] { self.index += 1; }
	}
	pub fn update_as_upper(&mut self, set: &SortedIndices, key: Frac) {
		while key >= set.keys[self.index] { self.index += 1; }
		while key <  set.keys[self.index] { self.index -= 1; }
	}
}

fn intersection(a: Vec<usize>, b: Vec<usize>) -> Vec<usize> {
	// hm, can't find anything on cargo. Itertools only has unions (merge).
	// we'll do O(m*n) because the sets are almost always expected to be size 0.
	a.into_iter().filter(|x| b.iter().any(|y| x == y)).collect()
}


struct State {
	labels: Vec<&'static str>,
	positions: Vec<Trip<Cart>>,
	dimension: Trip<Float>,
	// tracks x - move_radius index on each axis
	lowers: Trip<Cursor>,
	// tracks x + move_radius index on each axis
	uppers: Trip<Cursor>,
	// Contains images in the fractional range [-1, 2] along each axis
	sorted: Trip<SortedIndices>,
}

impl State {
	fn new(dimension: Trip<Float>) -> State {
		State {
			labels: vec![],
			positions: vec![],
			lowers: ((),(),()).map(|_| Cursor { index: 0 }),
			uppers: ((),(),()).map(|_| Cursor { index: 0 }),
			dimension: dimension,
			sorted: ((),(),()).map(|_| SortedIndices::new()),
		}
	}

	// init with binary search
	fn init_cursors(&mut self, frac: Trip<Frac>, radius: Cart) {
		let radii = self.dimension.map(|d| radius.frac(d));

		zip_with!((frac, radii, self.sorted.as_ref(), self.lowers.as_mut(), self.uppers.as_mut()),
		|x, r, set: &SortedIndices, lower: &mut Cursor, upper: &mut Cursor| { // type inference y u be hatin
			*lower = Cursor { index: set.lower_bound(x - r) };
			*upper = Cursor { index: set.upper_bound(x + r) };
		});
	}

	// update with linear search
	fn update_cursors(&mut self, frac: Trip<Frac>, radius: Cart) {
		let radii = self.dimension.map(|d| radius.frac(d));

		zip_with!((frac, radii, self.sorted.as_ref(), self.lowers.as_mut(), self.uppers.as_mut()),
		|x,r,set,lower: &mut Cursor,upper: &mut Cursor| {
			lower.update_as_lower(set, x - r);
			upper.update_as_upper(set, x + r);
		});
	}

	fn insert(&mut self, label: &'static str, point: Trip<Frac>) {
		let i = self.positions.len();
		self.positions.push(point.cart(self.dimension));
		self.labels.push(label);

		zip_with!((point, self.sorted.as_mut()), |Frac(x), set: &mut SortedIndices| {
			assert!(0.0 <= x && x <= 1.0);
			set.insert(Frac(x), i);
			set.insert(Frac(x - 1.0), i);
			set.insert(Frac(x + 1.0), i);
		});
	}

	fn cursor_neighborhood(&self) -> Vec<usize> {
		let ranges = zip_with!((self.lowers, self.uppers), |a:Cursor,b:Cursor| a.index..b.index);

		// should we even really bother?
		if ranges.clone().any(|x| x.len() == 0) { return vec![]; }
		
		zip_with!((self.sorted.as_ref(), ranges), |set: &SortedIndices, range: ::std::ops::Range<_>| {
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

	fn neighborhood(&self, point: Trip<Frac>, radius: Cart) -> Vec<usize> {
		let candidates = self.cursor_neighborhood();
		self.neighborhood_from_candidates(point, radius, candidates)
	}

	fn bruteforce_neighborhood(&self, point: Trip<Frac>, radius: Cart) -> Vec<usize> {
		self.neighborhood_from_candidates(point, radius, 0..self.positions.len())
	}
}

trait ToCart { fn cart(self, dimension: Trip<Float>) -> Trip<Cart>; }
trait ToFrac { fn frac(self, dimension: Trip<Float>) -> Trip<Frac>; }
impl ToCart for Trip<Frac> { fn cart(self, dimension: Trip<Float>) -> Trip<Cart> { zip_with!((self,dimension), |x:Frac,d| x.cart(d)) } }
impl ToFrac for Trip<Cart> { fn frac(self, dimension: Trip<Float>) -> Trip<Frac> { zip_with!((self,dimension), |x:Cart,d| x.frac(d)) } }

fn output<W: Write>(state: &State, file: &mut W) {
	writeln!(file, "[").unwrap();
	let mut first = false;
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
		state.insert("Si", (Frac(0.5), Frac(0.5), Frac(i as Float / n as Float)));
	}
	state
}

use std::collections::vec_deque::VecDeque;
#[derive(Default)]
struct Timer { deque: VecDeque<u64> }
impl Timer {
	pub fn new(n: usize) -> Timer {
		let mut this = Timer { deque: VecDeque::new() };
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

	let nbr_radius = Cart(2.)*MOVE_RADIUS + PARTICLE_RADIUS;

	let mut timer = Timer::new(30);

	for n in 0..NPARTICLE {
		write!(std::io::stderr(), "Particle {:8} of {:8}: ", n, NPARTICLE).unwrap();

		let mut pos = random_border_position(&mut rng);
		state.init_cursors(pos, nbr_radius);

		// move until ready to place
		loop {
			//writeln!(std::io::stderr(), "({:4},{:4},{:4})  ({:8?} ms)",
			//	(pos.0).0, (pos.1).0, (pos.2).0, (precise_time_ns() - start_time)/1000).unwrap();
			let neighbors = state.neighborhood(pos, nbr_radius);
			if !neighbors.is_empty() { break }

			let c_dir = random_direction(&mut rng);
			let c_disp = c_dir.mul_s(MOVE_RADIUS);
			let f_disp = c_disp.frac(state.dimension);

			pos = pos.add_v(f_disp);
			pos = pos.map(|Frac(x)| Frac((x + 1.0).fract()));
			state.update_cursors(pos, nbr_radius);
		}

		// place the particle
		state.insert("C", pos);

		timer.push();
		writeln!(std::io::stderr(), "({:14},{:14},{:14})  ({:8?} ms)  (avg: {:8?} ms)",
			(pos.0).0, (pos.1).0, (pos.2).0, timer.last_ms(), timer.average_ms()
		).unwrap();
	}
	state
}

fn main() {
	let state = dla_run();
	output(&state, &mut std::io::stdout());
}
