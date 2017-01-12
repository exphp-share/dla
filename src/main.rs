
// FIXME inconsistent usage of DIMENSION and state.dimension
const DIMENSION: Trip<Float> = (100., 6., 100.);
const NPARTICLE: usize = 100;

const THETA_STRENGTH:  Float = 1000.;
const RADIUS_STRENGTH: Float = 1000.;
const TARGET_RADIUS: Float = 1.4;

// VM: * Dimer sep should be 1.4 (Angstrom)
//     * Interaction radius (to begin relaxation) should be 2

// const CORE_RADIUS: Float = 5.0;
// const INTRA_CHAIN_SEP: Cart = Cart(1.);
const PARTICLE_RADIUS: Cart = 1.;//Cart(0.4);
const MOVE_RADIUS: Cart = 1.;
const DIMER_INITIAL_SEP: Cart = 1.4;
const HEX_INITIAL_RADIUS: Cart = 0.5;
// const RELAX_FREE_RADIUS: Cart = 10.;
// const RELAX_NEIGHBORHOOD_FACTOR: f64 = 10.;

const CART_ORIGIN: Trip<Cart> = (0., 0., 0.);
// const FRAC_ORIGIN: Trip<Frac> = (Frac(0.), Frac(0.), Frac(0.));
// const ORIGIN: Trip<Float> = (0., 0., 0.);

// const TARGET_RADIUS: Float = 1.0;

fn main() {
//	test_outputs();
//	dla_run_test(Dee::Two);
	dla_run(Dee::Three);
}

fn dla_run(dee: Dee) -> Tree {
	let mut tree = hexagon_nucleus(DIMENSION);

	let mut rng = rand::weak_rng();

	let nbr_radius = MOVE_RADIUS + PARTICLE_RADIUS;

	let mut timer = Timer::new(30);

	let final_particles = 2*NPARTICLE + tree.len();

	for n in 0..NPARTICLE {
		write!(stderr(), "Particle {:8} of {:8}: ", n, NPARTICLE).unwrap();

		let mut state = State::from_positions(DIMENSION, tree.pos.clone().into_iter().zip(tree.labels.clone()));

		let mut pos = random_border_position(&mut rng);

		// move until ready to place
		while state.neighborhood(pos, nbr_radius).is_empty() {
			//writeln!(stderr(), "({:4},{:4},{:4})  ({:8?} ms)",
			//	(pos.0).0, (pos.1).0, (pos.2).0, (precise_time_ns() - start_time)/1000).unwrap();

			let disp = random_direction(&mut rng)
				.mul_s(MOVE_RADIUS).frac(state.dimension);

			pos = reduce_pbc(pos.add_v(disp));

			if dee == Dee::Two { pos.1 = Frac(0.5); }
		}

		{ // make the two angles random
			let i = state.closest_neighbor(pos, nbr_radius).unwrap();
			let i = tree.attach_new(i, LABEL_CARBON, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
			let i = tree.attach_new(i, LABEL_CARBON, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
		}

		// HACK to get relaxed positions, ignoring the code in state
		// (tbh neighbor finding and relaxation should be separated out of state)
		let mut n_free = 2;
		{
			let unrelaxed = tree.pos.clone();
			let n_total = unrelaxed.len();

			let n_fixed = n_total - n_free;

			tree = relax_suffix_using_fire(tree, n_fixed);
		}

		// debugging info
		timer.push();
		writeln!(stderr(), "({:22},{:22},{:22})  ({:8?} ms)  (avg: {:8?} ms)  (relaxed {:3})",
			(pos.0).0, (pos.1).0, (pos.2).0, timer.last_ms(), timer.average_ms(), n_free
		).unwrap();

		write_xyz(&mut stdout(), &tree, final_particles);
	}

	{
		let dimension = tree.dimension;
		let pos = tree.pos.clone().into_iter().map(|x| reduce_pbc(x.frac(dimension)).cart(dimension));
		write_xyz_(&mut stdout(), pos, tree.labels.clone(), final_particles);
	}
	assert_eq!(final_particles, tree.len());
	tree
}

fn dla_run_test(dee: Dee) -> Tree {
	let mut tree = one_dimer(DIMENSION);

	let mut rng = rand::weak_rng();

	let nbr_radius = MOVE_RADIUS + PARTICLE_RADIUS;

	let mut timer = Timer::new(30);

	let final_particles = 4;

	for n in 0..1 {
		let mut state = State::from_positions(DIMENSION, tree.pos.clone().into_iter().zip(tree.labels.clone()));

		let mut pos = random_border_position(&mut rng);

		// move until ready to place
		while state.neighborhood(pos, nbr_radius).is_empty() {
			//writeln!(stderr(), "({:4},{:4},{:4})  ({:8?} ms)",
			//	(pos.0).0, (pos.1).0, (pos.2).0, (precise_time_ns() - start_time)/1000).unwrap();

			let disp = random_direction(&mut rng)
				.mul_s(MOVE_RADIUS).frac(state.dimension);

			pos = reduce_pbc(pos.add_v(disp));

			if dee == Dee::Two { pos.1 = Frac(0.5); }
		}

		{ // make the two angles random
			let i = state.closest_neighbor(pos, nbr_radius).unwrap();
			let i = tree.attach_new(i, LABEL_CARBON, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
			let i = tree.attach_new(i, LABEL_CARBON, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
		}

		// HACK to get relaxed positions, ignoring the code in state
		// (tbh neighbor finding and relaxation should be separated out of state)
		let n_free = 2;
		{
			let unrelaxed = tree.pos.clone();
			let n_total = unrelaxed.len();
			let n_fixed = n_total - n_free;

			writeln!(stderr(), "{:?}", distances_from_tree(&tree));

			let oldf = ::sp2::calc_potential(tree.pos.clone(), tree.dimension).1;
			let oldx = tree.pos.clone();
			tree = relax_suffix_using_fire(tree, 2);
			let newf = ::sp2::calc_potential(tree.pos.clone(), tree.dimension).1;
			let newx = tree.pos.clone();

			for (u,v) in izip!(oldf,newf) {
				let u = u.map(|x| (x*1000.).round()/1000.);
				let v = v.map(|x| (x*1000.).round()/1000.);
				writeln!(stderr(), "FORCE {:?} -> {:?}", u, v);
			}
			for (u,v) in izip!(oldx,newx) {
				let u = u.map(|x| (x*1000.).round()/1000.);
				let v = v.map(|x| (x*1000.).round()/1000.);
				writeln!(stderr(), "POS   {:?} -> {:?}", u, v);
			}
			writeln!(stderr(), "{:?}", distances_from_tree(&tree));
		}

		write_xyz(&mut stdout(), &tree, final_particles);
	}

	{
		let dimension = tree.dimension;
		let pos = tree.pos.clone().into_iter().map(|x| reduce_pbc(x.frac(dimension)).cart(dimension));
		write_xyz_(&mut stdout(), pos, tree.labels.clone(), final_particles);
	}
	assert_eq!(final_particles, tree.len());
	tree
}

// Relaxes the last few atoms on the tree (which can safely be worked with without having
// to unroot other atoms)
fn relax_suffix_using_fire(mut tree: Tree, n_fixed: usize) -> Tree {
	match n_fixed {
		0 => { return relax_all_using_fire(tree); },
		1 => { panic!("n_fixed == 1"); }, // a PITA edge case that's pointless anyways
		n if n == tree.len() => { return tree; },
		_ => {}
	};

	let free_indices = (n_fixed..tree.len()).collect_vec();

	// force computation works with [(f64,f64,f64)] and includes all positions;
	// relaxation works with flattened [f64], and... also now includes all positions.
	let labels: Vec<_> = tree.labels[n_fixed..].to_vec();

	// FIXME Params should not be in ::sp2
	let params = ::sp2::Params {
		timestep_start: 1e-3,
		timestep_max:   0.05,
		force_tolerance: Some(1e-5),
		step_limit: Some(4000),
	//	timestep_max:   1e-6,
	//	timestep_start: 1e-6,
	//	force_tolerance: Some(1e-7),
	//	step_limit: Some(40000),
		..Default::default()
	};

	let pos = {
		let mut relax = sp2::Relax::init(params, flatten(&tree.pos.clone()));
		//let force_fn = |md| just_write_forces(md, &free_indices[..], tree.dimension);
		//let force_fn = |md| force_canceling_force_writer(md, &free_indices[..], tree.dimension, &tree.parents[..]);
		//let force_fn = |md| centripetal_force_writer(md, &free_indices[..], tree.dimension, &tree.parents[..]);
		//let force_fn = |md| velocity_canceling_force_writer(md, &free_indices[..], tree.dimension, &tree.parents[..]);
		//let force_fn = |md| radius_resetting_force_writer(md, &free_indices[..], tree.dimension, &tree.parents[..], 1.41);
		//let force_fn = |md| cone_adjusting_force_writer(md, &free_indices[..], tree.dimension, &tree.parents[..], 1.41);
		//let force_fn = |md| zero_forces(md);
		let force_fn = |md| new_writer(md, &free_indices[..], tree.dimension, &tree.parents[..], RADIUS_STRENGTH, THETA_STRENGTH, TARGET_RADIUS);
		relax.relax(force_fn)
	};

	tree.dangerously_reassign_positions(unflatten(&pos));

	for p in &mut tree.pos {
		*p = reduce_pbc((*p).frac(tree.dimension)).cart(tree.dimension)
	}

	tree
}



// what a mess I've made; how did we accumulate so many dependencies? O_o
extern crate time;
extern crate rand;
#[macro_use(zip_with)]
extern crate homogenous;
#[macro_use(iproduct,izip)]
extern crate itertools;
#[macro_use]
extern crate newtype_ops;
extern crate dla_sys;
extern crate libc;
extern crate num;
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
fn translate((x,y,z): Trip<Cart>) -> NaIso { NaIso::new(NaVector3{x:x,y:y,z:z}, na::zero()) }
fn rotate((x,y,z): Trip<Float>) -> NaIso { NaIso::new(na::zero(), NaVector3{x:x,y:y,z:z}) }
fn rotate_x(x: Float) -> NaIso { NaIso::new(na::zero(), na_x(x)) }
fn rotate_y(x: Float) -> NaIso { NaIso::new(na::zero(), na_y(x)) }
fn rotate_z(x: Float) -> NaIso { NaIso::new(na::zero(), na_z(x)) }
fn na_x(r: Float) -> NaVector3 { na::Vector3{x:r, ..na::zero()} }
fn na_y(r: Float) -> NaVector3 { na::Vector3{y:r, ..na::zero()} }
fn na_z(r: Float) -> NaVector3 { na::Vector3{z:r, ..na::zero()} }

// could use an enum but meh
type Label = &'static str;
const LABEL_CARBON: Label = "C";
const LABEL_SILICON: Label = "Si";

//---------------------------------

// FIXME misnomer; not actually a tree; the first two atoms point to each other
#[derive(Clone)]
struct Tree {
	// FIXME 'label' is misleading;  These are metadata, not identifiers.
	labels: Vec<Label>,
	pos: Vec<Trip<Cart>>,
	parents: Vec<usize>,
	dimension: Trip<Float>,
}

// get an isometry that maps the origin to a given atom,
// and maps the x unit vector away from its parent.
fn bond_iso(pos: Trip<Cart>, parent: Trip<Cart>, dimension: Trip<Float>) -> NaIso {
	// this implementation... is the result of pure trial and error.
	// FIXME it is also wrong, as evidenced by the need for various hax in
	//       the angle-based potential (add_theta_terms)
	let (x,y,z) = nearest_image_sub(pos, parent, dimension);

	let beta = y.atan2(-z);
	let na::Point3 {x,y,z} = identity().append_rotation(&na_x(-beta)) * na::Point3 { x:x, y:y, z:z };
	assert!(y.abs() < 1e-5, "{}", y);

	let alpha = (-z).atan2(x);
	let na::Point3 {x,y,z} = identity().append_rotation(&na_y(-alpha)) * na::Point3 { x:x, y:y, z:z };
	assert!(z.abs() < 1e-5, "{}", z);

	let t = x;

	identity()
		.append_translation(&na_x(t))
		.append_rotation(&na_y(alpha))
		.append_rotation(&na_x(beta))
		.append_translation(&to_na_vector(parent))
}

impl Tree {
	fn from_two(dimension: Trip<Float>, length: Cart, labels: (Label,Label)) -> Self {
		Tree::from_two_pos(dimension, (CART_ORIGIN, (length, 0., 0.)), labels)
	}
	fn from_two_pos(dimension: Trip<Float>, pos: (Trip<Cart>, Trip<Cart>), labels: (Label,Label)) -> Self {
		Tree {
			labels:  vec![labels.0, labels.1],
			pos:     vec![pos.0, pos.1],
			parents: vec![1, 0],
			dimension: dimension,
		}
	}

	fn transform_mut(&mut self, iso: NaIso) {
		for p in &mut self.pos {
			*p = from_na_point(iso * to_na_point(*p));
		}
	}

	fn len(&self) -> usize { self.labels.len() }

	fn bond_iso(&self, index: usize) -> NaIso {
		bond_iso(self.pos[index], self.pos[self.parents[index]], self.dimension)
	}

	fn attach_new(&mut self, parent: usize, label: Label, length: Cart, beta: f64) -> usize {
		assert!(parent < self.len());

		// NOTE: a.append_something(b) is defined to apply b after a, *chronologically* speaking.
		// (i.e. reading the matrices right-to-left, assuming they operate on column vectors)
		let iso = identity()
			.append_translation(&na_x(length))
			.append_rotation(&na_y(60f64.to_radians()))
			.append_rotation(&na_x(beta))
			.append_transformation(&self.bond_iso(parent))
			;
		let na::Vector3 { x, y, z } = iso.translation;

		self.attach_at(parent, label, (x,y,z))
	}

	fn attach_at(&mut self, parent: usize, label: Label, point: Trip<Cart>) -> usize {
		assert!(parent < self.len());
		self.parents.push(parent);
		self.pos.push(point);
		self.labels.push(label);
		self.pos.len()-1
	}

	// Reassigns positions without regenerating tree structure.
	// The old bonds must remain valid.
	fn dangerously_reassign_positions(&mut self, pos: Vec<Trip<Cart>>) {
		// (this method only exists for emphasis; self.pos is actually visible at the callsite)
		assert_eq!(pos.len(), self.pos.len());
		self.pos = pos;
	}
}


//--------------------
// Make frac coords a newtype which is incompatible with other floats, to help prove that
// fractional/cartesian conversions are handled properly.
#[derive(Debug,PartialEq,PartialOrd,Copy,Clone)]
pub struct Frac(Float);
pub type Cart = Float;
trait CartExt { fn frac(self, dimension: Float) -> Frac; }
impl CartExt for Cart { fn frac(self, dimension: Float) -> Frac { Frac(self/dimension) } }

impl Frac { pub fn cart(self, dimension: Float) -> Cart { self.0*dimension } }

// add common binops to eliminate the majority of reasons I might need to
// convert back into floats (which would render the type system useless)
newtype_ops!{ [Frac] arithmetic {:=} {^&}Self {^&}{Self} }

// cart() and frac() methods for triples
trait ToCart { fn cart(self, dimension: Trip<Float>) -> Trip<Cart>; }
trait ToFrac { fn frac(self, dimension: Trip<Float>) -> Trip<Frac>; }
impl ToCart for Trip<Frac> { fn cart(self, dimension: Trip<Float>) -> Trip<Cart> { zip_with!((self,dimension) |x,d| x.cart(d)) } }
impl ToFrac for Trip<Cart> { fn frac(self, dimension: Trip<Float>) -> Trip<Frac> { zip_with!((self,dimension) |x,d| x.frac(d)) } }
impl ToCart for Trip<Cart> { fn cart(self, dimension: Trip<Float>) -> Trip<Cart> { self } }
impl ToFrac for Trip<Frac> { fn frac(self, dimension: Trip<Float>) -> Trip<Frac> { self } }

// nalgebra interop, but strictly for cartesian
fn to_na_vector((x,y,z): Trip<Cart>) -> na::Vector3<Float> { na::Vector3 { x: x, y: y, z: z } }
fn to_na_point((x,y,z): Trip<Cart>)  -> na::Point3<Float>  { na::Point3  { x: x, y: y, z: z } }

fn from_na_vector(na::Vector3 {x,y,z}: na::Vector3<Float>) -> Trip<Cart> { (x,y,z) }
fn from_na_point(na::Point3 {x,y,z}: na::Point3<Float>) -> Trip<Cart> { (x,y,z) }

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
			.fold((None,std::f64::MAX), |(i,p),(k,q)| if p < q { (i,p) } else { (k,q) })
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

fn write_xyz<W: Write>(mut file: W, tree: &Tree, final_length: usize) {
	write_xyz_(file, tree.pos.clone(), tree.labels.clone(), final_length);
}

fn write_xyz_<W: Write, I,J>(mut file: W, pos: I, labels: J, final_length: usize)
where I: IntoIterator<Item=Trip<Cart>>, J: IntoIterator<Item=Label> {
	let mut pos = pos.into_iter().collect_vec();
	let mut labels = labels.into_iter().collect_vec();
	let first = pos[0];

	labels.resize(final_length, LABEL_CARBON);
	pos.resize(final_length, first);
	writeln!(file, "{}", final_length);
	writeln!(file, "blah blah blah");
	for (label, (x,y,z)) in labels.into_iter().zip(pos) {
		writeln!(file, "{} {} {} {}", label, x, y, z);
	}
}

//---------------------------

fn relax_all_using_fire(mut tree: Tree) -> Tree {
	unimplemented!()
}

// part of a that is parallel to b
fn par(a: Trip<f64>, b: Trip<f64>) -> Trip<f64> {
	let b_norm = b.dot(b).sqrt();
	let b_unit = b.div_s(b_norm);
	b_unit.mul_s(a.dot(b_unit))
}
fn perp(a: Trip<f64>, b: Trip<f64>) -> Trip<f64> {
	let c = a.sub_v(par(a,b));
	assert!(c.dot(b).abs() <= 1e-4 * c.dot(c).abs(), "{:?} {:?}", c.dot(c), c.dot(b));
	c
}

fn zero_forces(mut md: ::sp2::Relax) -> ::sp2::Relax {
	md.force.resize(0, 0.);
	md.force.resize(md.position.len(), 0.);
	md
}

fn just_write_forces(mut md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>) -> ::sp2::Relax {
	let (potential,force) = sp2::calc_potential_flat(md.position.clone(), dim);

	let mut md = zero_forces(md);
	for i in free_indices {
		md.force[3*i..3*(i+1)].copy_from_slice(&force[3*i..3*(i+1)]);
	}
	md
}

fn cancel_something(something: &mut [f64], positions: &[f64], free_indices: &[usize], dim: Trip<f64>, parents: &[usize]) {
	assert_eq!(parents.len() * 3, positions.len());
	assert_eq!(parents.len() * 3, something.len());

	let tup3 = |xs: &[f64], i:usize| { (0,1,2).map(|k| xs[3*i+k]) };
	let tup3set = |xs: &mut [f64], i:usize, v:(f64,f64,f64)| { v.enumerate().map(|(k,x)| xs[3*i+k] = x); };

	// verify dependencies are in order (parents before children)
	for (ii,&i) in free_indices.iter().enumerate() {
		if let Some(jj) = free_indices.iter().position(|&x| x == parents[i]) {
			assert!(ii > jj);
		}
	}

	for &i in free_indices {
		let j = parents[i];

		let fi = tup3(something, i);
		let fj = tup3(something, j);
		let ri = tup3(positions, i);
		let rj = tup3(positions, j);
		let dr = nearest_image_sub(ri, rj, dim);
		let df = fi.sub_v(fj);

		// old (is this right?)
		let fpar = par(df,dr);
		let fi = fi.sub_v(fpar);

		// new
//		let fi = perp(fi,dr).add_v(par(fj,dr));

		tup3set(something, i, fi);
		tup3set(something, j, fj);

		// dubba check
		let fi = tup3(something, i);
		let fj = tup3(something, j);
		let dr = nearest_image_sub(ri, rj, dim);
		let df = fi.sub_v(fj);
		assert!(dr.dot(df).abs() < 1e-8);
	}
}

fn add_radius_term(mut md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize], strength: f64, target_radius: f64) -> ::sp2::Relax {
	let tup3 = |xs: &[f64], i:usize| { (0,1,2).map(|k| xs[3*i+k]) };
	let tup3set = |xs: &mut [f64], i:usize, v:(f64,f64,f64)| { v.enumerate().map(|(k,x)| xs[3*i+k] = x); };

	// verify dependencies are in order (parents before children)
	for (ii,&i) in free_indices.iter().enumerate() {
		if let Some(jj) = free_indices.iter().position(|&x| x == parents[i]) {
			assert!(ii > jj);
		}
	}

	for &i in free_indices {
		let j = parents[i];

		let ri = tup3(&md.position, i);
		let rj = tup3(&md.position, j);
		let dr = nearest_image_sub(ri, rj, dim);
		let r = dr.sqnorm().sqrt();
		let f_add = dr.mul_s(2. * strength * (target_radius/r - 1.));

		let f = tup3(&md.force, i);

		tup3set(&mut md.force, i, f.add_v(f_add));
	}
	md
}

fn add_theta_term(mut md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize], strength: f64, target_angle: f64) -> ::sp2::Relax {
	let tup3 = |xs: &[f64], i:usize| { (0,1,2).map(|k| xs[3*i+k]) };
	let tup3set = |xs: &mut [f64], i:usize, v:(f64,f64,f64)| { v.enumerate().map(|(k,x)| xs[3*i+k] = x); };
	let apply = |iso: NaIso, v: (f64,f64,f64)| { from_na_point(iso * to_na_point(v)) };
	let applyV = |iso: NaIso, v: (f64,f64,f64)| { from_na_vector(iso * to_na_vector(v)) };

	// verify dependencies are in order (parents before children)
	for (ii,&i) in free_indices.iter().enumerate() {
		if let Some(jj) = free_indices.iter().position(|&x| x == parents[i]) {
			assert!(ii > jj);
		}
	}

	for &i in free_indices {
		let ri = tup3(&md.position, i);
		let rj = tup3(&md.position, parents[i]);
		let rk = tup3(&md.position, parents[parents[i]]);

		let rj = ri.add_v(nearest_image_sub(rj, ri, dim));
		let rk = rj.add_v(nearest_image_sub(rk, rj, dim));

		// move into the basis of i's parent bond  (rj at center, x axis away from rk)

		// FIXME why won't this work?  seems something is up...
		//let inv = bond_iso(rj, rk, dim).inverse_transformation();
		//let ri = apply(inv, ri);
		//let rj = apply(inv, rj);
		//let rk = apply(inv, rk);
		let (beta,alpha,ri,rj,rk) = {
			let ri = ri.sub_v(rj);
			let rj = rj.sub_v(rj);
			let rk = rk.sub_v(rj);

			let (x,y,z) = rk;

			let beta = z.atan2(y);
			let iso = rotate_x(-beta);
			let ri      = apply(iso, ri);
			let (x,y,z) = apply(iso, (x,y,z));
			assert!(z.abs() < 1e-5, "{}", z);

			let alpha = y.atan2(x);
			let iso = rotate_z(-alpha + PI);
			let ri      = apply(iso, ri);
			let (x,y,z) = apply(iso, (x,y,z));
			assert!(y.abs() < 1e-5, "{}", y);

			(beta,alpha,ri, rj, (x,y,z))
		};

		{
			// ensure rj is on the origin
			assert!(rj.sqnorm() <= 1e-7, "{:?}", rj);

			// ensure rk looks like it is along -x
			let mut rk = rk;
			assert!(rk.0 < -0.3);
			rk.0 = 0.;
			assert!(rk.sqnorm() <= 1e-7, "{:?}", rk);
		}

		// Define  gamma := cos theta (== x/r),  w/ theta being the exterior angle (angle to +x axis).
		// Add a potential term of  K * (gamma - cos target)^2

		let (x,y,z) = ri;
		let r = ri.sqnorm().sqrt();
		let gam = x / r;

		let prefac  = 2. * strength * (gam - target_angle.cos());
		let dgam_dr = (y*y + z*z, -x*y, -x*z).div_s(r*r*r);

		let f_add = dgam_dr.mul_s(-1.0 * prefac);

		// bring f_add back into cartesian.
		// (note: transformation from  grad' V  to  grad V  is transpose matrix of the one
		//   that maps x to x')
		let f_add = applyV(rotate_z(alpha - PI), f_add);
		let f_add = applyV(rotate_x(beta), f_add);

		let f = tup3(&md.force, i);
		tup3set(&mut md.force, i, f.add_v(f_add));
	}
	md
}

fn add_centripetal_terms(mut md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize]) -> ::sp2::Relax {
	let tup3 = |xs: &[f64], i:usize| { (0,1,2).map(|k| xs[3*i+k]) };
	let tup3set = |xs: &mut [f64], i:usize, v:(f64,f64,f64)| { v.enumerate().map(|(k,x)| xs[3*i+k] = x); };

	// verify dependencies are in order (parents before children)
	for (ii,&i) in free_indices.iter().enumerate() {
		if let Some(jj) = free_indices.iter().position(|&x| x == parents[i]) {
			assert!(ii > jj);
		}
	}

	for &i in free_indices {
		let j = parents[i];

		let ri = tup3(&md.position, i);
		let rj = tup3(&md.position, j);
		let vi = tup3(&md.velocity, i);
		let vj = tup3(&md.velocity, j);
		let fi = tup3(&md.force, i);
		let fj = tup3(&md.force, j);
		
		let dr = nearest_image_sub(ri, rj, dim);
		let dv = vi.sub_v(vj);

		let vsq_tangential = perp(dv, dr).sqnorm();
		let f_centrip = dr.mul_s(-1.0 * perp(dv, dr).sqnorm() / dr.sqnorm());

		tup3set(&mut md.force, i, fi.add_v(f_centrip));
	}
	md
}


fn reset_radii(mut md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize], target_radius: f64) -> ::sp2::Relax {
	let tup3 = |xs: &[f64], i:usize| { (0,1,2).map(|k| xs[3*i+k]) };
	let tup3set = |xs: &mut [f64], i:usize, v:(f64,f64,f64)| { v.enumerate().map(|(k,x)| xs[3*i+k] = x); };

	for &i in free_indices {
		let j = parents[i];
		let ri = tup3(&md.position, i);
		let rj = tup3(&md.position, j);
		let dr = nearest_image_sub(ri, rj, dim);

		let dr_norm = dr.sqnorm().sqrt();
		let dr_unit = dr.div_s(dr_norm);

		let dr = dr_unit.mul_s(0.95 * target_radius + 0.05 * dr_norm);
		tup3set(&mut md.position, i, rj.add_v(dr));
	}
	md
}

fn adjust_onto_cones(mut md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize], target_radius: f64) -> ::sp2::Relax {
	let tup3 = |xs: &[f64], i:usize| { (0,1,2).map(|k| xs[3*i+k]) };
	let tup3set = |xs: &mut [f64], i:usize, v:(f64,f64,f64)| { v.enumerate().map(|(k,x)| xs[3*i+k] = x); };

	for &i in free_indices {
		let ri = tup3(&md.position, i);
		let rj = tup3(&md.position, parents[i]);
		let rk = tup3(&md.position, parents[parents[i]]);

		let rcos = (60f64).to_radians().cos()*target_radius;
		let rsin = (60f64).to_radians().sin()*target_radius;

		// move the valid circle of positions onto the YZ plane with an inverse transform
		let iso = bond_iso(rj, rk, dim).append_transformation(&translate((rcos,0.,0.)));
		let inv = iso.inverse_transformation();
		let ri = from_na_point(inv * to_na_point(ri));

		// move the position onto the nearest point on that circle
		let ryz = (ri.1,ri.2);
		let ryz = ryz.div_s(ryz.sqnorm().sqrt()).mul_s(rsin);
		let ri = (0., ryz.0, ryz.1);

		// transform back
		let ri = from_na_point(iso * to_na_point(ri));
		tup3set(&mut md.position, i, ri);
	}
	md
}


fn cone_adjusting_force_writer(md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize], target_radius: f64) -> ::sp2::Relax {
	let md = adjust_onto_cones(md, free_indices, dim, parents, target_radius);
	let md = just_write_forces(md, free_indices, dim);
	md
}

fn radius_resetting_force_writer(md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize], target_radius: f64) -> ::sp2::Relax {
	let md = reset_radii(md, free_indices, dim, parents, target_radius);
	let md = just_write_forces(md, free_indices, dim);
	md
}

fn force_canceling_force_writer(md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize]) -> ::sp2::Relax {
	let mut md = just_write_forces(md, free_indices, dim);
	cancel_something(&mut md.force, &md.position, free_indices, dim, parents);
	md
}

fn centripetal_force_writer(md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize]) -> ::sp2::Relax {
	let mut md = just_write_forces(md, free_indices, dim);
	cancel_something(&mut md.force, &md.position, free_indices, dim, parents);
//	cancel_something(&mut md.velocity, &md.position, free_indices, dim, parents);
	let md = add_centripetal_terms(md, free_indices, dim, parents);
	md
}

fn new_writer(md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize], strength1: f64, strength2: f64, target_radius: f64) -> ::sp2::Relax {
	let mut md = just_write_forces(md, free_indices, dim);
	let md = add_radius_term(md, free_indices, dim, parents, strength1, target_radius);
	let md = add_theta_term(md, free_indices, dim, parents, strength2, 60f64.to_radians());
	md
}

fn velocity_canceling_force_writer(md: ::sp2::Relax, free_indices: &[usize], dim: Trip<f64>, parents: &[usize]) -> ::sp2::Relax {
	let mut md = just_write_forces(md, free_indices, dim);
	cancel_something(&mut md.velocity, &md.position, free_indices, dim, parents);
	md
}

fn unflatten<T:Copy>(slice: &[T]) -> Vec<(T,T,T)> {
	let mut iter = slice.iter().cloned();
	let mut out = vec![];
	loop {
		if let Some(x) = iter.next() {
			let y = iter.next().unwrap();
			let z = iter.next().unwrap();
			out.push((x, y, z));
		} else { break }
	}
	out
}

fn flatten<T:Copy>(slice: &[(T,T,T)]) -> Vec<T> {
	slice.iter().cloned().flat_map(|x| x.into_iter()).collect()
}

//---------- DLA

fn random_direction<R:Rng>(rng: &mut R) -> Trip<Cart> {
	let normal = Normal::new(0.0, 1.0);
	let x = normal.ind_sample(rng) as Float;
	let y = normal.ind_sample(rng) as Float;
	let z = normal.ind_sample(rng) as Float;

	let vec = (x,y,z);
	let length = vec.sqnorm().sqrt();
	vec.map(|x| x / length)
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

// for debugging the isometries; this should be a barbell shape, with
// 4 atoms protruding from each end in 4 diagonal directions
fn barbell_nucleus(dimension: Trip<f64>) -> Tree {
	let mut tree = Tree::from_two(dimension, DIMER_INITIAL_SEP, ("Si", "C"));
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

fn random_barbell_nucleus(dimension: Trip<f64>) -> Tree {
	let dir = random_direction(&mut rand::weak_rng());
	let pos1 = ((),(),()).map(|_| Frac(rand::weak_rng().next_f64())).cart(dimension);
	let pos2 = pos1.add_v(dir.mul_s(DIMER_INITIAL_SEP));
	let mut tree = Tree::from_two_pos(dimension, (pos1,pos2), ("Si", "C"));
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
	let mut tree = Tree::from_two(dimension, DIMER_INITIAL_SEP, ("Si", "Si"));
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

fn one_dimer(dimension: Trip<f64>) -> Tree {
	let mut tree = Tree::from_two(dimension, DIMER_INITIAL_SEP, (LABEL_SILICON, LABEL_SILICON));
	tree.transform_mut(translate(dimension.mul_s(0.5)));
	tree
}

fn hexagon_nucleus(dimension: Trip<f64>) -> Tree {
	let mut tree = Tree::from_two(dimension, DIMER_INITIAL_SEP, (LABEL_SILICON, LABEL_SILICON));
	let i = tree.attach_new(1, LABEL_SILICON, DIMER_INITIAL_SEP, 0.);
	let i = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, 0.);
	let i = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, 0.);
	let _ = tree.attach_new(i, LABEL_SILICON, DIMER_INITIAL_SEP, 0.);
	tree.transform_mut(translate(dimension.mul_s(0.5)));

	let pos = sp2::relax_all(tree.pos.clone(), dimension);
	tree.dangerously_reassign_positions(pos);
	tree
}

fn center(pos: &Vec<Trip<Cart>>) -> Trip<Cart> {
	let n = pos.len() as Cart;
	pos.iter().fold(CART_ORIGIN, |u,&b| u.add_v(b)).div_s(n)
}

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

fn test_outputs() {
	let doit = |tree, path| {
		write_xyz(::std::fs::File::create(path).unwrap(), &tree, tree.len());
	};
	doit(hexagon_nucleus(DIMENSION), "hexagon.xyz");

	doit(barbell_nucleus(DIMENSION), "barbell.xyz");
	doit(random_barbell_nucleus(DIMENSION), "barbell-random.xyz");

	let tree = squiggle_nucleus(DIMENSION);
	doit(tree.clone(), "squiggle.xyz");
}

#[derive(Eq,PartialEq,Copy,Clone)]
enum Dee { Two, Three }

fn distances_from_tree(tree: &Tree) -> Vec<f64> {
	let mut out = vec![];
	for i in 0..tree.len() {
		let j = tree.parents[i];
		out.push(
			tree.pos[i].sub_v(tree.pos[j]).sqnorm().sqrt()
		);
	}
	out
}


