const DIMENSION: Trip<f64> = (240., 240., 40.);
const NPARTICLE: usize = 10000;

const CORE_RADIUS: f64 = 5f64;
const INTRA_CHAIN_SEP: Cart = Cart(2f64);
const PARTICLE_RADIUS: Cart = Cart(1f64);
const MOVE_RADIUS: Cart = Cart(1f64);

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

type Trip<T> = (T,T,T);

// For statically proving that fractional/cartesian conversions are handled properly.
#[derive(PartialEq,PartialOrd,Copy,Clone)]
struct Frac(f64);
#[derive(PartialEq,PartialOrd,Copy,Clone)]
struct Cart(f64);
impl Frac { pub fn cart(self, dimension: f64) -> Cart { Cart(self.0*dimension) } }
impl Cart { pub fn frac(self, dimension: f64) -> Frac { Frac((self.0/dimension).fract()) } }

// add common binops to eliminate the majority of reasons I might need to
// convert back into floats (which would render the type system useless)
macro_rules! impl_binop { ($T:ident, $trt:ident, $func:ident, $op:tt) => {
	impl $trt<$T> for $T {
		type Output = $T;
		fn $func(self, other: $T) -> $T { $T(self.0 $op other.0) }
	}
	impl $trt<f64> for $T {
		type Output = $T;
		fn $func(self, other: f64) -> $T { $T(self.0 $op other) }
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
		match self.keys.binary_search_by(|b| k.partial_cmp(b).unwrap()) {
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

	fn range(&self, from: Key, to: Key) -> &[Value] {
		&self.values[self.lower_bound(from)..self.upper_bound(to)]
	}
}

fn intersection(a: Vec<usize>, b: Vec<usize>) -> Vec<usize> {
	// hm, can't find anything on cargo. Itertools only has unions (merge).
	// we'll do O(m*n) because the sets are almost always expected to be size 0.
	a.into_iter().filter(|x| b.iter().any(|y| x == y)).collect()
}

// We track two sorted lists of atoms along each axis; one in the range [0,1],
// and one in the range [0.5,1.5].
#[derive(Copy,Clone,Hash,Debug,Ord,PartialOrd,Eq,PartialEq)]
enum Region { Center = 0, Boundary = 1 }
use Region::*;
impl Region {
	fn categorize(Frac(x): Frac) -> Region {
		if 0.25 <= x && x <= 0.75 { Center } else { Boundary }
	}
	fn image_of(self, Frac(x): Frac) -> Frac { match self {
		Center => Frac(x),
		Boundary => Frac(0.5 + (x + 0.5).fract()),
	}}
}

struct State {
	labels: Vec<&'static str>,
	positions: Vec<Trip<Cart>>,
	dimension: Trip<f64>,
	sorted: Trip<[SortedIndices; 2]>,
}

impl State {
	fn new(dimension: Trip<f64>) -> State {
		State {
			labels: vec![],
			positions: vec![],
			dimension: dimension,
			sorted: (0,0,0).map(|_| [SortedIndices::new(), SortedIndices::new()]),
		}
	}

	fn insert(&mut self, label: &'static str, point: Trip<Frac>) {
		let i = self.positions.len();
		self.positions.push(point.cart(self.dimension));
		self.labels.push(label);
		for (x,sets) in point.zip(self.sorted.as_mut()).into_iter() {
			sets[Center   as usize].insert(Center.image_of(x),   i);
			sets[Boundary as usize].insert(Boundary.image_of(x), i);
		}
	}

	fn cubic_neighborhood(&self, point: Trip<Frac>, radius: Cart) -> Vec<usize> {
		zip_with!((point, self.sorted.as_ref(), self.dimension), |x, sets: &[SortedIndices; 2], dim| {
			let region = Region::categorize(x);
			let radius = radius.frac(dim);
			let x = region.image_of(x);
			sets[region as usize].range(x - radius, x + radius).to_vec()
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
		let candidates = self.cubic_neighborhood(point, radius);
		self.neighborhood_from_candidates(point, radius, candidates)
	}

	fn bruteforce_neighborhood(&self, point: Trip<Frac>, radius: Cart) -> Vec<usize> {
		self.neighborhood_from_candidates(point, radius, 0..self.positions.len())
	}
}

trait ToCart { fn cart(self, dimension: Trip<f64>) -> Trip<Cart>; }
trait ToFrac { fn frac(self, dimension: Trip<f64>) -> Trip<Frac>; }
impl ToCart for Trip<Frac> { fn cart(self, dimension: Trip<f64>) -> Trip<Cart> { zip_with!((self,dimension), |x:Frac,d| x.cart(d)) } }
impl ToFrac for Trip<Cart> { fn frac(self, dimension: Trip<f64>) -> Trip<Frac> { zip_with!((self,dimension), |x:Cart,d| x.frac(d)) } }

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
	let x = normal.ind_sample(rng);
	let y = normal.ind_sample(rng);
	let z = normal.ind_sample(rng);

	let vec = (x,y,z);
	let length = vec.sqnorm().sqrt();
	vec.map(|x| Cart(x / length))
}

fn random_border_position<R:Rng>(rng: &mut R) -> Trip<Frac> {
	// this makes no attempt to be isotropic,
	// as evidenced by the fact that it works entirely in terms of fractional coords

	// place onto either the i=0 or j=0 face of the cuboid
	let o = Frac(0.);
	let x = Frac(rng.next_f64());
	let z = Frac(rng.next_f64());
	match rng.gen_range(0, 2) {
		0 => (x, o, z),
		1 => (o, x, z),
		_ => unreachable!(),
	}
}

fn add_nucleation_site(mut state: State) -> State {
	let n = (Cart(state.dimension.2) / INTRA_CHAIN_SEP).0.round() as i32;
	for i in 0i32..n {
		state.insert("Si", (Frac(0.), Frac(0.), Frac(i as f64 / n as f64)));
	}
	state
}

fn dla_run() -> State {
	let state = State::new(DIMENSION);
	let state = add_nucleation_site(state);
	let mut state = state;

	let mut rng = rand::weak_rng();

	for n in 0..NPARTICLE {
		write!(std::io::stderr(), "Particle {:8} of {:8}: ", n, NPARTICLE).unwrap();
		let start_time = precise_time_ns();

		let mut pos = random_border_position(&mut rng);

		// move until ready to place
		loop {
			writeln!(std::io::stderr(), "({:4},{:4},{:4})  ({:8?} ms)",
				(pos.0).0, (pos.1).0, (pos.2).0, (precise_time_ns() - start_time)/1000).unwrap();
			let neighbors = state.neighborhood(pos, Cart(2.)*MOVE_RADIUS + PARTICLE_RADIUS);
			if !neighbors.is_empty() { break }

			let c_dir = random_direction(&mut rng);
			let c_disp = c_dir.mul_s(MOVE_RADIUS);
			let f_disp = c_disp.frac(state.dimension);
			pos = pos.add_v(f_disp);
		}

		// place the particle
		state.insert("C", pos);
		writeln!(std::io::stderr(), "({:4},{:4},{:4})  ({:8?} ms)",
			(pos.0).0, (pos.1).0, (pos.2).0, (precise_time_ns() - start_time)/1000).unwrap();
	}
	state
}

fn main() {
	let state = dla_run();
	output(&state, &mut std::io::stdout());
}
