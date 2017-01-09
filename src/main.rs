
// FIXME inconsistent usage of DIMENSION and state.dimension
const DIMENSION: Trip<Float> = (100., 100., 6.);
const NPARTICLE: usize = 100;

// VM: * Dimer sep should be 1.4 (Angstrom)
//     * Interaction radius (to begin relaxation) should be 2

// const CORE_RADIUS: Float = 5.0;
// const INTRA_CHAIN_SEP: Cart = Cart(1.);
const PARTICLE_RADIUS: Cart = Cart(1.);//Cart(0.4);
const MOVE_RADIUS: Cart = Cart(1.);
const DIMER_INITIAL_SEP: Cart = Cart(1.4);
const HEX_INITIAL_RADIUS: Cart = Cart(0.5);
// const RELAX_FREE_RADIUS: Cart = Cart(10.);
// const RELAX_NEIGHBORHOOD_FACTOR: f64 = 10.;

const CART_ORIGIN: Trip<Cart> = (Cart(0.), Cart(0.), Cart(0.));
// const FRAC_ORIGIN: Trip<Frac> = (Frac(0.), Frac(0.), Frac(0.));
// const ORIGIN: Trip<Float> = (0., 0., 0.);

// const THETA_STRENGTH: Float = 1.0;
// const RADIUS_STRENGTH: Float = 1.0;
// const TARGET_RADIUS: Float = 1.0;

// what a mess I've made; how did we accumulate so many dependencies? O_o
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
extern crate nalgebra;


mod sp2;

use rand::Rng;
use rand::distributions::{IndependentSample,Normal};
use itertools::Itertools;
use homogenous::prelude::*;
use homogenous::numeric::prelude::*;
use time::precise_time_ns;
use nalgebra as na;
use nalgebra::{Rotation as NaRotation, Translation as NaTranslation};

use std::ops::Range;
use std::io::Write;
use std::io::{stderr,stdout};

use std::f64::consts::PI;

type Float = f64;
type Pair<T> = (T,T);
type Trip<T> = (T,T,T);

type NaIso = na::Mat3<Float>;
type NaVector3 = na::Vector3<Float>;
type NaPoint3 = na::Point3<Float>;

// could use an enum but meh
type Label = &'static str;
const LABEL_CARBON: Label = "C";
const LABEL_SILICON: Label = "Si";

struct Tree {
	// FIXME 'label' is misleading;  These are metadata, not identifiers.
	labels: Vec<Label>,
	// transformation from a plain cartesian basis into one
	// where this atom is at the origin, and x points away from parent.
	isos: Vec<NaIso>,
	parents: Vec<Option<usize>>,
}

fn identity_iso() -> NaIso { NaIso::new(na::zero(), na::zero()) }

impl Tree {
	fn from_two(Cart(length): Cart, labels: (Label,Label)) -> Self {
		// Begins with two atoms; one at the origin and one at (length, 0., 0.).

		// Beginning with two atoms allows us to define a root node easily;
		// * the atom at the origin will be the child of the second
		// * the second atom will be a child of a "ghost" at the origin.
		//   (hence both represent the same bond, but there's no parent cycle)

		// at origin, let x point away from atom 2
		let iso_1 = NaIso::new(na::zero(), na::Vector3 { y: 180f64.to_radians(), ..na::zero() });
		// at atom 2, let x point away from origin
		let iso_2 = NaIso::new(na::Vector3 { x: length, ..na::zero() }, na::zero());
		Tree {
//			classes: vec![Class::Ghost, classes.1, classes.0],
//			isos:    vec![identity_iso(), iso_2, iso_1],
//			parents: vec![None, Some(0), Some(1)],
			labels: vec![labels.0, labels.1],
			isos:    vec![iso_1, iso_2],
			parents: vec![None, Some(0)],
		}
	}

	fn len(&self) -> usize { self.labels.len() }

	// mutators simply to contain all the junk relating to nalgebra
	fn translate_mut(&mut self, t: Trip<Cart>) {
		for iso in &mut self.isos { iso.append_translation_mut(&to_na_vector(t)); }
	}
	fn rotate_z_mut(&mut self, r: f64) {
		for iso in &mut self.isos { iso.append_rotation_mut(&na::Vector3 { z: r, ..na::zero() }); }
	}

	fn attach_new(&mut self, parent: usize, label: Label, Cart(length): Cart, beta: f64) -> usize {
		assert!(parent < self.len());

		// NOTE: nalgebra's "append/prepend" are defined in chronological terms;
		// The appended operation is performed last. (it is the first matrix when read ltr)
		let iso = self.isos[parent]
			.append_translation(&na::Vector3 { x: length, ..na::zero() })
			.append_rotation(&na::Vector3 { y: 60f64.to_radians(), ..na::zero() })
			.append_rotation(&na::Vector3 { x: beta, ..na::zero() })
			;

		self.parents.push(Some(parent));
		self.isos.push(iso);
		self.labels.push(label);
		self.isos.len()-1
	}

	fn positions(&self) -> Vec<Trip<Cart>> {
		self.isos.iter().map(|iso| from_na_vector(iso.translation)).collect()
	}
}


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
trait ToVec3 { fn na_vector(self) -> na::Vector3<Float>; }
impl ToCart for Trip<Frac> { fn cart(self, dimension: Trip<Float>) -> Trip<Cart> { zip_with!((self,dimension) |x,d| x.cart(d)) } }
impl ToFrac for Trip<Cart> { fn frac(self, dimension: Trip<Float>) -> Trip<Frac> { zip_with!((self,dimension) |x,d| x.frac(d)) } }
impl ToCart for Trip<Cart> { fn cart(self, dimension: Trip<Float>) -> Trip<Cart> { self } }
impl ToFrac for Trip<Frac> { fn frac(self, dimension: Trip<Float>) -> Trip<Frac> { self } }

// nalgebra interop, but strictly for cartesian
fn to_na_vector((x,y,z): Trip<Cart>) -> na::Vector3<Float> { na::Vector3 { x: x.0, y: y.0, z: z.0 } }
fn to_na_point((x,y,z): Trip<Cart>)  -> na::Point3<Float>  { na::Point3  { x: x.0, y: y.0, z: z.0 } }

fn from_na_vector(na::Vector3 {x,y,z}: na::Vector3<Float>) -> Trip<Cart> { (x,y,z).map(|x| Cart(x)) }
fn from_na_point(na::Point3 {x,y,z}: na::Point3<Float>) -> Trip<Cart> { (x,y,z).map(|x| Cart(x)) }

fn reduce_pbc(this: Trip<Frac>) -> Trip<Frac> { this.map(|Frac(x)| Frac((x.fract() + 1.0).fract())) }
fn nearest_image_dist_sq<P:ToFrac,Q:ToFrac>(this: P, that: Q, dimension: Trip<Float>) -> Cart {
	// assumes a diagonal cell
	let this = this.frac(dimension);
	let that = that.frac(dimension);
	let fdiff = reduce_pbc(this.sub_v(that)); // range [0, 1.]
	let fdiff = fdiff.map(|Frac(x)| Frac(x.min(1. - x))); // range [0, 0.5]

	fdiff.map(|Frac(x)| assert!(0. <= x && x <= 0.5, "{} {:?} {:?}", x, this.map(|x| x.0), that.map(|x| x.0)));
	fdiff.cart(dimension).sqnorm()
}

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
		let k = Frac((k.0 + 1.).fract()); // FIXME HACK
		assert!(Frac(0.) <= k && k <= Frac(1.), "{:?}", k);
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

//--------------------------
struct State {
	labels: Vec<Label>,
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
			dimension: dimension,
			hints: ((),(),()).map(|_| 0..0),
			sorted: ((),(),()).map(|_| SortedIndices::new()),
		}
	}

	fn from_positions<P:ToFrac,I:IntoIterator<Item=(P,Label)>>(dimension: Trip<f64>, pos: I) -> State {
		let mut this = State::new(dimension);
		for (x, lbl) in pos {
			this.insert(lbl, reduce_pbc(x.frac(dimension)));
		}
		this
	}

	fn update_cursors(&mut self, frac: Trip<Frac>, radius: Cart) {
		let radii = self.dimension.map(|d| radius.frac(d));

		zip_with!((frac, radii, self.sorted.as_ref(), self.hints.as_mut())
		|x,r,set,hints| {
			hints.start = update_lower_hint(hints.start, &set.keys, x - r);
			hints.end   = update_upper_hint(hints.end,   &set.keys, x + r);
		});
	}

	fn insert(&mut self, label: Label, point: Trip<Frac>) {
		let point = reduce_pbc(point);
		let i = self.positions.len();
		self.positions.push(point.cart(self.dimension));
		self.labels.push(label);

		zip_with!((point, self.sorted.as_mut()) |x, set| set.insert_images(x, i));
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

		*positions = sp2::relax(positions.clone(), fixed, (0.,0.,0.));// dimension); // FIXME

		// Relaxation is infrequent; we shall just rebuild the index lists from scratch.
		sorted.as_mut().enumerate().map(|(axis, set)| {
			let projected: Vec<_> = positions.iter()
				.map(|&x| reduce_pbc(x.frac(dimension)))
				.map(|x| x.into_nth(axis))
				.collect();
			*set = SortedIndices::rebuild(projected);
		});

		n_free
	}

	fn extend_and_relax<P:ToFrac,I:IntoIterator<Item=P>>(&mut self, iter: I) {
		let mut fixed = vec![true; self.positions.len()];
		for p in iter {
			let dim = self.dimension;
			self.insert(LABEL_CARBON, reduce_pbc(p.frac(dim)))
		}
		fixed.resize(self.positions.len(), false);

		let tmp = self.positions.clone();
		self.positions = sp2::relax(tmp, fixed, (0.,0.,self.dimension.2));
	}

	fn cursor_neighborhood(&self) -> Vec<usize> {
		// should we even really bother?
		if self.hints.clone().any(|x| x.len() == 0) { return vec![]; }

		zip_with!((self.sorted.as_ref(), self.hints.clone())
			|set, range| { set.values[range].to_vec() }
		).fold1(|a, b| a.into_iter().filter(|x| b.iter().any(|y| x == y)).collect())
	}

	fn neighborhood_from_candidates<I: IntoIterator<Item=usize>>(&self, frac: Trip<Frac>, radius: Cart, indices: I) -> Vec<usize> {
		let cart = frac.cart(self.dimension);

		indices.into_iter().filter(|&i| {
			nearest_image_dist_sq(cart, self.positions[i], self.dimension) <= radius*radius
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

fn write_xyz<W: Write>(tree: &Tree, file: &mut W, final_length: usize) {
	let mut lab = tree.labels.clone();
	let mut pos = tree.positions();
	let first = *pos.first().unwrap();

	lab.resize(final_length, LABEL_CARBON);
	pos.resize(final_length, first);
	writeln!(file, "{}", final_length);
	writeln!(file, "blah blah blah");
	for (&label, &(Cart(x),Cart(y),Cart(z))) in lab.iter().zip(&pos) {
		writeln!(file, "{} {} {} {}", label, x, y, z);
	}
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

fn rotate((Cart(x),Cart(y)): (Cart,Cart), angle: Float) -> (Cart,Cart) {
	let (sin,cos) = (angle.sin(), angle.cos());
	(Cart(cos * x - sin * y), Cart(cos * y + sin * x))
}

fn hexagon_nucleus(dimension: Trip<f64>) -> Tree {
	let mut tree = Tree::from_two(DIMER_INITIAL_SEP, (LABEL_SILICON, LABEL_SILICON));
	let i = tree.attach_new(0, LABEL_SILICON, DIMER_INITIAL_SEP, PI/2.);
	let i = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, PI/2.);
	let i = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, PI/2.);
	let _ = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, PI/2.);
//	let _ = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, PI/2.);
//	let i = tree.attach_new(1, LABEL_SILICON, DIMER_INITIAL_SEP, -PI/2.);
//	let _ = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, -PI/2.);

	// optimize bond length
	//let pos = sp2::relax_all(pos, (0.,0.,0.));
	tree
}

fn center(pos: &Vec<Trip<Cart>>) -> Trip<Cart> {
	let n = Cart(pos.len() as Float);
	pos.iter().fold(CART_ORIGIN, |u,&b| u.add_v(b)).div_s(n)
}

// TODO
fn do_mst() { unimplemented!() }

/*
fn seven_hexagon_nucleus(dimension: Trip<f64>) -> State {
	let mut pos = hexagon_sites(dimension);
	pos.sort_by(|&(x,y,_), &(x2,y2,_)| (y.0.atan2(x.0)).partial_cmp(&y2.0.atan2(x2.0)).unwrap());

	let hex_disps = pos.iter().cloned()
		.cycle().tuple_windows::<(_,_)>().take(6)
		.map(|(v1,v2)| v1.add_v(v2)).collect_vec();

	for p in ::std::mem::replace(&mut pos, vec![])  {
		for &d in &hex_disps {
			pos.push(p.add_v(d));
		}
	}

	let pos = remove_overlapping(pos, Cart(1e-4));
	let pos = recenter_midpoint(pos, dimension);
	let pos = sp2::relax_all(pos, (0.,0.,0.));
	let pos = remove_overlapping(pos, Cart(1e-4));
	let pos = sp2::relax_all(pos, (0.,0.,0.));
	assert_eq!(pos.len(), 24);
	State::from_sites(dimension, pos)
}
*/

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

fn dla_run() -> Tree {
	let mut tree = hexagon_nucleus(DIMENSION);

	let mut rng = rand::weak_rng();

	let nbr_radius = Cart(1.)*MOVE_RADIUS + PARTICLE_RADIUS;

	let mut timer = Timer::new(30);

	let final_particles = 2*NPARTICLE + tree.len();
	for n in 0..NPARTICLE {
/*
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

			// Egads! The particle was actually a dimer^H^H^H^H^Htrimer all along!
			let sites = {
//				let disp = random_direction(&mut rng).mul_s(DIMER_INITIAL_SEP * 0.5);
//				let disp = disp.frac(state.dimension);
//				(reduce_pbc(pos.add_v(disp)), reduce_pbc(pos.sub_v(disp)))
				((),(),).map(|_|
					reduce_pbc(pos.add_v(random_direction(&mut rng).mul_s(DIMER_INITIAL_SEP * 0.5).frac(state.dimension)))
				)
			};

			// Place the particle.
			//state.extend_and_relax(sites.into_iter());
			//state.extend_and_relax(sites.into_iter());
			for p in sites.into_iter() { state.insert(LABEL_CARBON, p) }
			let n_free = state.relax_neighborhood(pos, Cart(5.));

			// debugging info
			timer.push();
			writeln!(stderr(), "({:22},{:22},{:22})  ({:8?} ms)  (avg: {:8?} ms)  (relaxed {:8})",
				(pos.0).0, (pos.1).0, (pos.2).0, timer.last_ms(), timer.average_ms(), n_free
			).unwrap();

			break 'restart;
		}

		write_xyz(&state, &mut stdout(), final_particles);
*/
	}
	write_xyz(&tree, &mut stdout(), final_particles);
	assert_eq!(final_particles, tree.len());
	tree
}

fn main() {
	dla_run();
}
