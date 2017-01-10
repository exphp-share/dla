
#![feature(iter_min_by)]

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
use nalgebra::{Rotation as NaRotation, Translation as NaTranslation, Transformation as NaTransformation};

use std::ops::Range;
use std::io::Write;
use std::io::{stderr,stdout};

use std::f64::consts::PI;

type Float = f64;
type Pair<T> = (T,T);
type Trip<T> = (T,T,T);

type NaIso = na::Isometry3<Float>;
type NaVector3 = na::Vector3<Float>;
type NaPoint3 = na::Point3<Float>;

fn identity() -> NaIso { NaIso::new(na::zero(), na::zero()) }
fn na_x(r: Float) -> NaVector3 { na::Vector3{x:r, ..na::zero()} }
fn na_y(r: Float) -> NaVector3 { na::Vector3{y:r, ..na::zero()} }
fn na_z(r: Float) -> NaVector3 { na::Vector3{z:r, ..na::zero()} }

// could use an enum but meh
type Label = &'static str;
const LABEL_CARBON: Label = "C";
const LABEL_SILICON: Label = "Si";

//---------------------------------

struct Tree {
	// FIXME 'label' is misleading;  These are metadata, not identifiers.
	labels: Vec<Label>,
	// transformation from a plain cartesian basis into one
	// where this atom is at the origin, and x points away from parent.
	isos: Vec<NaIso>,
	parents: Vec<Option<usize>>,
}

/*
fn closest_pair_indices_bipartite<I,J>(iter: I, jtre: J, dimension: Trip<Float>) -> Option<(usize,usize)>
where I: IntoIterator<Item=Trip<Cart>>, J: IntoIterator<Item=Trip<Cart>>,
{
	iproduct!(iter.into_iter().enumerate(), jtre.into_iter().enumerate())
		.map(|((i,p),(j,q))| (Some((i,j)), nearest_image_dist_sq(p,q,dimension)))
		.fold((None, Cart(::std::f64::MAX)), |(i,p),(j,q)| if p < q { (i,p) } else { (j,q) })
		.0
}

fn closest_pair_indices_within<I,J>(iter: I, jtre: J, dimension: Trip<Float>) -> Option<(usize,usize)>
where I: IntoIterator<Item=Trip<Cart>>, J: IntoIterator<Item=Trip<Cart>>,
{
	iproduct!(iter.into_iter().enumerate(), jtre.into_iter().enumerate())
		.map(|((i,p),(j,q))| (Some((i,j)), nearest_image_dist_sq(p,q,dimension)))
		.fold((None, Cart(::std::f64::MAX)), |(i,p),(j,q)| if p < q { (i,p) } else { (j,q) })
		.0
}
*/

impl Tree {
	fn from_two(Cart(cart): Cart, labels: (Label,Label)) -> Self {
		// Begins with two atoms; one at the origin and one at (length, 0., 0.).

		// Beginning with two atoms allows us to define a root node easily;
		// * the atom at the origin will be the child of the second
		// * the second atom will be a child of a "ghost" at the origin.
		//   (hence both represent the same bond, but there's no parent cycle)

		// at origin, let x point away from atom 2
		let iso_1 = identity().append_rotation(&na_z(180f64.to_radians()));
		// at atom 2, let x point away from origin
		let iso_2 = identity().append_translation(&na_x(cart));
		Tree {
//			classes: vec![Class::Ghost, classes.1, classes.0],
//			isos:    vec![identity_iso(), iso_2, iso_1],
//			parents: vec![None, Some(0), Some(1)],
			labels:  vec![labels.0, labels.1],
			isos:    vec![iso_1, iso_2],
			parents: vec![None, Some(0)],
		}
	}

	fn from_two_pos((p,q): (Trip<Cart>, Trip<Cart>), labels: (Label,Label)) -> Self {
		unimplemented!() // TODO ffs
	}

	fn len(&self) -> usize { self.labels.len() }

	fn transform_mut(&mut self, iso: NaIso) {
		for x in &mut self.isos { x.append_transformation_mut(&iso) }
	}

	fn attach_new(&mut self, parent: usize, label: Label, Cart(cart): Cart, beta: f64) -> usize {
		assert!(parent < self.len());

		// NOTE: a.append_something(b) is defined to apply b after a, *chronologically* speaking.
		// (i.e. reading the matrices right-to-left, assuming they operate on column vectors)
		let iso = identity()
			.append_translation(&na_x(cart))
			.append_rotation(&na_y(60f64.to_radians()))
			.append_rotation(&na_x(beta))
			.append_transformation(&self.isos[parent])
			;

//		// NOTE: This should be equivalent to the next four lines. (check with squiggle_core)
//		self.attach_at(parent, label, from_na_vector(iso.translation))

		self.parents.push(Some(parent));
		self.isos.push(iso);
		self.labels.push(label);
		self.isos.len()-1
	}

	fn attach_at(&mut self, parent: usize, label: Label, point: Trip<Cart>, dimension: Trip<Float>) -> usize {
		assert!(parent < self.len());

		// this method is disgusting
		// the goal is to produce take an arbitrary point and break it down into the
		// parameterization that attach_new uses.

		// the implementation... is the result of pure trial and error.
		let (x,y,z) = point.map(|x| x.0);
		let na::Point3 {x,y,z} = self.isos[parent].inverse_transformation() * na::Point3 { x:x, y:y, z:z };

		// nearest image
		// TODO: Test that this works/is needed
		let (x,y,z) = zip_with!(((x,y,z), dimension) |x,d| x - (x/d).round()*d);

		let beta = y.atan2(-z);
		let na::Point3 {x,y,z} = identity().append_rotation(&na_x(-beta)) * na::Point3 { x:x, y:y, z:z };
		assert!(y.abs() < 1e-5, "{}", z);

		let alpha = (-z).atan2(x);
		let na::Point3 {x,y,z} = identity().append_rotation(&na_y(-alpha)) * na::Point3 { x:x, y:y, z:z };
		assert!(z.abs() < 1e-5, "{}", x);

		let t = x;

		writeln!(stderr(), " ALPHA {} BETA {} ", alpha.to_degrees(), beta.to_degrees());
		let iso = identity()
			.append_translation(&na_x(t))
			.append_rotation(&na_y(alpha))
			.append_rotation(&na_x(beta))
			.append_transformation(&self.isos[parent])
			;

		self.parents.push(Some(parent));
		self.isos.push(iso);
		self.labels.push(label);
		self.isos.len()-1
	}

	fn extend<I:IntoIterator<Item=Label>, J: IntoIterator<Item=Trip<Cart>>>(&mut self, dimension: Trip<Float>, new_pos: J, new_labels: I) {
		let mut new_labels = new_labels.into_iter().collect_vec();
		let mut new_pos = new_pos.into_iter().collect_vec();
		assert_eq!(new_labels.len(), new_pos.len());

		// extend the tree greedily a la prim's algorithm
		// NOTE: brute force implementation
		while !new_pos.is_empty() {

			let tree_positions = self.positions();

			let (i_child, i_parent) = {
				let mut best_dist = Cart(::std::f64::MAX);
				let mut best_idxs = None;
				for (i_child,&p) in new_pos.iter().enumerate() {
					for (i_parent,&q) in tree_positions.iter().enumerate() {
						let sqdist = nearest_image_dist_sq(p,q,dimension);
						if sqdist < best_dist {
							best_dist = sqdist;
							best_idxs = Some((i_child,i_parent));
						}
					}
				}
				best_idxs
			}.unwrap(); // tree and new_pos each have at least one node

			let label = new_labels.remove(i_child);
			let point = new_pos.remove(i_child);
			self.attach_at(i_parent, label, point, dimension);
		}
	}

	fn from_iter<I:IntoIterator<Item=Label>, J: IntoIterator<Item=Trip<Cart>>>(dimension: Trip<Float>, pos: J, labels: I) -> Self {
		let mut labels = labels.into_iter().collect_vec();
		let mut pos = pos.into_iter().collect_vec();
		assert_eq!(labels.len(), pos.len());
		assert!(pos.len() >= 2);

		// find a minimum weight edge
		// NOTE: this subtly differs from the search in extend() in that we are now
		//  seeking two different indices from the SAME iterable.
		let (i, j) = {
			let mut best_dist = Cart(::std::f64::MAX);
			let mut best_idxs = None;
			for i in 0..pos.len() {
				for j in 0..i {
					let sqdist = nearest_image_dist_sq(pos[i],pos[j],dimension);
					if sqdist < best_dist {
						best_dist = sqdist;
						best_idxs = Some((i,j));
					}
				}
			}
			best_idxs
		}.unwrap(); // there are at least two nodes

		let label_1 = labels.swap_remove(i);
		let label_2 = labels.swap_remove(j);
		let pos_1 = pos.swap_remove(i);
		let pos_2 = pos.swap_remove(j);

		let mut tree = Tree::from_two_pos((pos_1, pos_2), (label_1, label_2));
		tree.extend(dimension, pos, labels);
		tree
	}

	// FIXME hack
	fn pop(&mut self) -> Option<(Label, Option<usize>, Trip<Cart>)> {
		self.labels.pop().map(|label| {
			let parent = self.parents.pop().unwrap();
			let pos = from_na_vector(self.isos.pop().unwrap().translation);
			(label, parent, pos)
		})
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
fn nearest_image_sub<P:ToFrac,Q:ToFrac>(this: P, that: Q, dimension: Trip<Float>) -> Trip<Cart> {
	// assumes a diagonal cell
	let this = this.frac(dimension);
	let that = that.frac(dimension);
	let diff = this.sub_v(that)
		.map(|Frac(x)| Frac(x - x.round())); // range [0.5, -0.5]

	diff.map(|Frac(x)| assert!(-0.5-1e-5 <= x && x <= 0.5+1e-5));
	diff.cart(dimension)
}

fn nearest_image_dist_sq<P:ToFrac,Q:ToFrac>(this: P, that: Q, dimension: Trip<Float>) -> Cart {
	nearest_image_sub(this, that, dimension).sqnorm()
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

	fn closest_neighbor(&mut self, point: Trip<Frac>, radius: Cart) -> Option<usize> {
		let indices = self.neighborhood(point, radius);
		indices.into_iter()
			.map(|i| (Some(i), nearest_image_dist_sq(point, self.positions[i], self.dimension)))
			// locate tuple with minimum distance; unfortunately, Iterator::min_by is unstable.
			.fold((None,Cart(std::f64::MAX)), |(i,p),(k,q)| if p < q { (i,p) } else { (k,q) })
			.0
	}

	fn neighborhood_from_candidates<I: IntoIterator<Item=usize>>(&self, point: Trip<Frac>, radius: Cart, indices: I) -> Vec<usize> {
		indices.into_iter().filter(|&i| {
			nearest_image_dist_sq(point, self.positions[i], self.dimension) <= radius*radius
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

// for debugging the isometries; this should be a barbell shape, with
// 4 atoms protruding from each end in 4 diagonal directions
fn barbell_nucleus(dimension: Trip<f64>) -> Tree {
	let mut tree = Tree::from_two(DIMER_INITIAL_SEP, ("Si", "C"));
	tree.attach_new(0, "Si", DIMER_INITIAL_SEP, PI*0.0);
	tree.attach_new(0, "Si", DIMER_INITIAL_SEP, PI*0.5);
	tree.attach_new(0, "Si", DIMER_INITIAL_SEP, PI*1.0);
	tree.attach_new(0, "Si", DIMER_INITIAL_SEP, PI*1.5);
	tree.attach_new(1, "C", DIMER_INITIAL_SEP, PI*0.0);
	tree.attach_new(1, "C", DIMER_INITIAL_SEP, PI*0.5);
	tree.attach_new(1, "C", DIMER_INITIAL_SEP, PI*1.0);
	tree.attach_new(1, "C", DIMER_INITIAL_SEP, PI*1.5);
	tree
}

// for debugging reattachment;
fn squiggle_nucleus(dimension: Trip<f64>) -> Tree {
	let mut tree = Tree::from_two(DIMER_INITIAL_SEP, ("Si", "Si"));
	let mut i = 0;
	for k in 0..16 {
		i = tree.attach_new(i, "Si", DIMER_INITIAL_SEP, PI*(k as f64)/(16 as f64));
	}
	// test on the "ghost" nucleus
	let mut i = 1;
	for k in 0..16 {
		i = tree.attach_new(i, "Si", DIMER_INITIAL_SEP, PI*(k as f64)/(16 as f64));
	}
	tree
}

fn hexagon_nucleus(dimension: Trip<f64>) -> Tree {
	let mut tree = Tree::from_two(DIMER_INITIAL_SEP, (LABEL_SILICON, LABEL_SILICON));
	let i = tree.attach_new(1, LABEL_SILICON, DIMER_INITIAL_SEP, 0.);
	let i = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, 0.);
	let i = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, 0.);
	let _ = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, 0.);

	// FIXME uncomment
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
	let mut tree = squiggle_nucleus(DIMENSION);

	let mut rng = rand::weak_rng();

	let nbr_radius = Cart(1.)*MOVE_RADIUS + PARTICLE_RADIUS;

	let mut timer = Timer::new(30);

	let final_particles = 2*NPARTICLE + tree.len();

/*
	for n in 0..NPARTICLE {
		write!(stderr(), "Particle {:8} of {:8}: ", n, NPARTICLE).unwrap();

		let mut state = State::from_positions(DIMENSION, tree.positions().into_iter().zip(tree.labels.clone()));

		let mut pos = random_border_position(&mut rng);

		// move until ready to place
		while state.neighborhood(pos, nbr_radius).is_empty() {
			//writeln!(stderr(), "({:4},{:4},{:4})  ({:8?} ms)",
			//	(pos.0).0, (pos.1).0, (pos.2).0, (precise_time_ns() - start_time)/1000).unwrap();

			let disp = random_direction(&mut rng)
				.mul_s(MOVE_RADIUS).frac(state.dimension);

			pos = reduce_pbc(pos.add_v(disp));
		}

		{ // make the two angles random
			let i = state.closest_neighbor(pos, nbr_radius).unwrap();
			let i = tree.attach_new(i, LABEL_CARBON, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
			let i = tree.attach_new(i, LABEL_CARBON, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
		}

		// HACK to get relaxed positions, ignoring the code in state
		// (tbh neighbor finding and relaxation should be separated out of state)
		{
			let pos = tree.positions();
			let mut fixed = vec![true; pos.len()-2];
			fixed.resize(pos.len(), false);

			let mut pos = sp2::relax(pos, fixed, state.dimension);

			// Replace positions in tree through even more terrible hax
			// (NOTE: using Prim's algorithm would allow this to be done more efficiently)
			let pos_2 = pos.pop().unwrap();
			let pos_1 = pos.pop().unwrap();
			let (label_2, parent_2, _old_pos) = tree.pop().unwrap();
			let (label_1, parent_1, _old_pos) = tree.pop().unwrap();
			tree.attach_at(parent_1.unwrap(), label_1, pos_1);
			tree.attach_at(parent_2.unwrap(), label_2, pos_2);
		}
		let n_free = 2;

		// debugging info
		timer.push();
		writeln!(stderr(), "({:22},{:22},{:22})  ({:8?} ms)  (avg: {:8?} ms)  (relaxed {:3})",
			(pos.0).0, (pos.1).0, (pos.2).0, timer.last_ms(), timer.average_ms(), n_free
		).unwrap();

		write_xyz(&tree, &mut stdout(), final_particles);
	}
*/

	write_xyz(&tree, &mut stdout(), final_particles);
	assert_eq!(final_particles, tree.len());
	tree
}

fn main() {
	dla_run();
}
