
#![feature(non_ascii_idents)]

// FIXME inconsistent usage of DIMENSION and state.dimension
const DIMENSION: Trip<Float> = (75., 75., 75.);
const IS_VACUUM: Trip<bool> = (true, false, true);
const NPARTICLE: usize = 100;

const DEBUG: bool = false;
const XYZ_DEBUG: bool = true;
const FORCE_DEBUG: bool = true;

#[derive(PartialEq,Copy,Clone,Debug)]
enum Force {
	Morse { center: f64, D: f64, k: f64, },
	Quadratic { center: f64, k: f64, },
	Zero,
}

//const RADIUS_FORCE: Force = Force::Morse { center: 1.41, D: 100., k: 0. };
//const RADIUS_FORCE: Force = Force::Quadratic { center: 1.41, k: 100. };

const RADIUS_FORCE: Force = Force::Morse { center: 1.41, D: 100., k: 100. };
const THETA_FORCE: Force = Force::Quadratic { center: (120.*PI/180.), k: 100. };
const USE_REBO: bool = true;

//const RADIUS_FORCE: Force = Force::Zero;
//const THETA_FORCE: Force = Force::Zero;

const RELAX_PARAMS: ::sp2::Params =
	::sp2::Params {
		timestep_start: 1e-3,
		timestep_max:   0.05,
//		timestep_start: 1e-2,
//		timestep_max:   0.15,
		force_tolerance: Some(1e-5),
		step_limit: Some(4000),
		flail_step_limit: Some(10),
		turn_condition: ::sp2::TurnCondition::Potential,
		..::sp2::DEFAULT_PARAMS
	};

// VM: * Dimer sep should be 1.4 (Angstrom)
//     * Interaction radius (to begin relaxation) should be 2

const RELAX_NEIGHBORHOOD_RADIUS: Cart = 5.;
const RELAX_MAX_PARTICLE_COUNT: usize = 12;
const PARTICLE_RADIUS: Cart = 1.;//Cart(0.4);
const MOVE_RADIUS: Cart = 1.;
const DIMER_INITIAL_SEP: Cart = 1.4;
const HEX_INITIAL_RADIUS: Cart = 0.5;

const CART_ORIGIN: Trip<Cart> = (0., 0., 0.);

fn main() {
//	test_outputs();
//	dla_run_test();
//	let tree = dla_run();
	run_relax_on(&::std::env::args().nth(1).unwrap_or("xyz-debug/tree.json".to_string()));
}

fn dla_run() {
	let tree = dla_run_();
	serde_json::to_writer(&mut File::create("xyz-debug/tree.json").unwrap(), &tree);
}

fn run_relax_on(path: &str) {
	let tree = serde_json::from_reader(&mut File::open("xyz-debug/tree.json").unwrap()).unwrap();
	println!("{:?}", tree);
	run_relax_on_(tree)
}

fn dla_run_() -> Tree {
	let PER_STEP = 2;

	let mut tree = hexagon_nucleus(DIMENSION);

	let mut rng = rand::weak_rng();

	let nbr_radius = MOVE_RADIUS + PARTICLE_RADIUS;

	let mut timer = Timer::new(30);

	let final_particles = PER_STEP*NPARTICLE + tree.len();

	let mut debug_file = File::create("debug").unwrap();

	for dla_step in 0..NPARTICLE {
		write!(stderr(), "Particle {:8} of {:8}: ", dla_step, NPARTICLE).unwrap();

		let mut state = State::from_positions(DIMENSION, tree.pos.clone().into_iter().zip(tree.labels.clone()));

		let mut pos = random_border_position(&mut rng);

		// move until ready to place
		while state.neighborhood(pos, nbr_radius).is_empty() {
			//writeln!(stderr(), "({:4},{:4},{:4})  ({:8?} ms)",
			//	(pos.0).0, (pos.1).0, (pos.2).0, (precise_time_ns() - start_time)/1000).unwrap();

			let disp = random_direction(&mut rng)
				.mul_s(MOVE_RADIUS).frac(state.dimension);

			pos = reduce_pbc(pos.add_v(disp));
		}

		// introduce at random angles
		match PER_STEP {
			// dimer
			2 => {
				let i = state.closest_neighbor(pos, nbr_radius).unwrap();
				let i = tree.attach_new(i, Label::C, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
				let i = tree.attach_new(i, Label::C, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
			},

			// trimer
			3 => {
				let i = state.closest_neighbor(pos, nbr_radius).unwrap();
				let i = tree.attach_new(i, Label::C, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
				let r = rng.next_f64()*2.*PI;
				tree.attach_new(i, Label::C, DIMER_INITIAL_SEP, r);
				tree.attach_new(i, Label::C, DIMER_INITIAL_SEP, r+PI);
			},

			_ => panic!(),
		}

		// HACK find who to relax
		let free_indices = {
			let first_new_index = tree.len() - PER_STEP;
			let nbrhood_center = tree.pos[first_new_index];

			state.neighborhood(nbrhood_center.frac(tree.dimension), RELAX_NEIGHBORHOOD_RADIUS)
			.into_iter()
			.chain(first_new_index..tree.len()) // the new indices are not yet in state
			.filter(|&i| tree.labels[i] != Label::Si)
			// ascending by distance
			.map(|i| (i, nearest_image_dist_sq(nbrhood_center, tree.pos[i], tree.dimension)))
			.sorted_by(|&(ia,a), &(ib,b)| a.partial_cmp(&b).unwrap())
			.into_iter().take(RELAX_MAX_PARTICLE_COUNT)
			.map(|(i,_)| i)
			.collect_vec()
		};

		// HACK to get relaxed positions, ignoring the code in state
		// (tbh neighbor finding and relaxation should be separated out of state)
		let n_free = free_indices.len();
//		let mut n_free = PER_STEP;
		let (n_relax_steps, stop_reason) = {
//			let n_total = tree.len();

//			let n_fixed = n_total - n_free;

			let mut xyz_debug_file = if XYZ_DEBUG {
				let path = format!("xyz-debug/event-{:06}.xyz", dla_step);
				Some(File::create(&path).unwrap())
			} else { None };

			let mut force_debug_file = if FORCE_DEBUG {
				let path = format!("xyz-debug/force-{:06}", dla_step);
				Some(File::create(&path).unwrap())
			} else { None };

			let mut n_relax_steps = 0; // prepare for more abominable hax...

			let labels = tree.labels.clone();
			//relax_suffix_using_fire(&mut tree, n_fixed, |md| { // relax new
			//relax_suffix_using_fire(&mut tree, 6, force_debug_file, |md| { // relax all XXX
			//relax_suffix_using_fire(&mut tree, 0, force_debug_file, |md| { // relax all XXX
			let reason = relax_using_fire(&mut tree, &free_indices, force_debug_file, |md| { // relax all XXX

				for file in &mut xyz_debug_file {
					write_xyz_(file, unflatten(&md.position), labels.clone(), labels.len());
				}

				if DEBUG {
					writeln!(debug_file, "F {} {} {} {} {}", dla_step, md.nstep, md.alpha, md.timestep, md.cooldown);
					for i in 0..md.position.len()/3 {
						let v = (md.velocity[3*i+0], md.velocity[3*i+1], md.velocity[3*i+2]).sqnorm().sqrt();
						let f = (md.force[3*i+0], md.force[3*i+1], md.force[3*i+2]).sqnorm().sqrt();
						writeln!(debug_file, "A {} {} {} {:.6} {:.6}", dla_step, md.nstep, i, v, f);
					}
				}

				n_relax_steps = md.nstep; // only the final assigned value will remain
			});
			(n_relax_steps, reason)
		};

		// debugging info
		timer.push();
		writeln!(stderr(), "({:8.6}, {:8.6}, {:8.6})  ({:5?} ms, avg: {:5?})  (relaxed {:3} in {:6} after {:?})",
			(pos.0).0, (pos.1).0, (pos.2).0, timer.last_ms(), timer.average_ms(), n_free, n_relax_steps, stop_reason
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

fn run_relax_on_(mut tree: Tree) {
}

#[derive(Copy,Clone,Debug,PartialEq)]
struct ForceOut { potential: f64, force: f64 }

impl Force {
	fn data(self: Force, x: f64) -> ForceOut {
		let square = |x| x*x;
		match self {
			Force::Quadratic { center, k, } => {
				ForceOut {
					potential: k * square(center - x),
					force:     2. * k * (center - x),
				}
			},
			Force::Morse { center, k, D } => {
				let a = (k/(2. * D)).sqrt();
				let f = (a * (center - x)).exp();
				ForceOut {
					potential: a * D * square(f - 1.),
					force:     2. * a * D * f * (f - 1.),
				}
			},
			Force::Zero => { ForceOut { potential: 0., force: 0. } },
		}
	}

	// signed value of force along +x
	fn force(self: Force, x: f64) -> f64 { self.data(x).force }
}

// Relaxes the last few atoms on the tree (which can safely be worked with without having
// to unroot other atoms)
fn relax_suffix_using_fire<C: FnMut(&Relax), W:Write>(tree: &mut Tree, n_fixed: usize, ffile: Option<W>, cb: C) -> ::sp2::StopReason {
	let free_indices = (n_fixed..tree.len()).collect_vec();
	relax_using_fire(tree, &free_indices, ffile, cb)
}

fn relax_using_fire<C: FnMut(&Relax), W:Write>(tree: &mut Tree, free_indices: &[usize], mut ffile: Option<W>, mut cb: C) -> ::sp2::StopReason {
	let info = compute_force_info(free_indices, &tree.parents);
	let dim = tree.dimension;

	let ffile = ::std::cell::RefCell::new(ffile);
	let (pos,reason) = {
		sp2::Relax::init(RELAX_PARAMS, flatten(&tree.pos.clone()))
		// cb to write forces before fire
		.relax(|md| {
			let mut ffile = ffile.borrow_mut();
			if let Some(file) = ffile.as_mut() {
				writeln!(file, "STEP {}", md.nstep);
			}

			let mut md = zero_forces(md);
			let mut md = add_rebo(md, info.free_indices.iter().cloned(), dim, ffile.as_mut());
			let mut md = add_corrections(md, &info, dim, ffile.as_mut());
			cb(&md);

			md

		// cb that is invoked after fire (so that f_dot_v is known)
		}, |md| {
			if let Some(file) = ffile.borrow_mut().as_mut() {
				writeln!(file, "TOTAL_E {:23.18} F_DOT_V {:23.18} DT {:23.18}", md.potential, md.f_dot_v, md.timestep);
			}
		})
	};

	tree.dangerously_reassign_positions(unflatten(&pos));
	reason
}


// what a mess I've made; how did we accumulate so many dependencies? O_o
extern crate time;
extern crate rand;
#[macro_use] extern crate homogenous;
#[macro_use] extern crate itertools;
#[macro_use] extern crate newtype_ops;
extern crate dla_sys;
extern crate libc;
extern crate num;
extern crate nalgebra;
#[macro_use] extern crate serde_derive;
extern crate serde_json;
extern crate serde;


mod sp2;
use sp2::Relax;

use rand::Rng;
use rand::distributions::{IndependentSample,Normal};
use itertools::Itertools;
use homogenous::prelude::*;
use homogenous::numeric::prelude::*;
use time::precise_time_ns;
use nalgebra as na;
use nalgebra::{Rotation, Translation, Transformation, Transform};
use serde::{Serialize,Deserialize};

use std::collections::HashSet as Set;

use std::ops::Range;
use std::io::Write;
use std::io::{stderr,stdout};
use std::fs::File;

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

#[derive(Copy,Clone,Eq,PartialEq,Ord,PartialOrd,Hash,Serialize,Deserialize,Debug)]
enum Label { C, Si }
impl Label {
	pub fn as_str(&self) -> &'static str {
		match *self {
			Label::C => "C",
			Label::Si => "Si",
		}
	}
}

//---------------------------------

// NOTE misnomer; not actually a tree; the first two atoms point to each other
#[derive(Debug,Clone,Serialize,Deserialize)]
struct Tree {
	labels: Vec<Label>,
	pos: Vec<Trip<Cart>>,
	parents: Vec<usize>,
	dimension: Trip<Float>,
}

// Get a "look at" isometry; it maps the origin to the eye, and +z towards the target.
// (The up direction is arbitrarily chosen, without risk of it being invalid)
fn look_at_pbc(eye: Trip<Cart>, target: Trip<Cart>, dimension: Trip<Float>) -> NaIso {
	let (_,θ,φ) = spherical_from_cart(nearest_image_sub(target, eye, dimension));
	translate(eye) * rotate_z(φ) * rotate_y(θ)
}

impl Tree {
	fn from_two(dimension: Trip<Float>, length: Cart, labels: (Label,Label)) -> Self {
		Tree::from_two_pos(dimension, (CART_ORIGIN, (0., length, 0.)), labels)
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

	fn attach_new(&mut self, parent: usize, label: Label, length: Cart, beta: f64) -> usize {
		assert!(parent < self.len());

		// put the parent at the origin and its parent along +Z
		let iso = look_at_pbc(self.pos[parent], self.pos[self.parents[parent]], self.dimension);
		// add an atom in this basis
		let pos = cart_from_spherical((length, 120f64.to_radians(), beta));
		// back to cartesian
		let pos = from_na_point(iso * to_na_point(pos));
		self.attach_at(parent, label, pos)
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
// FIXME why does state still have labels
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

	labels.resize(final_length, Label::C);
	pos.resize(final_length, first);
	writeln!(file, "{}", final_length);
	writeln!(file, "blah blah blah");
	for (label, (x,y,z)) in labels.into_iter().zip(pos) {
		writeln!(file, "{} {} {} {}", label.as_str(), x, y, z);
	}
}

//---------------------------

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

fn zero_forces(mut md: Relax) -> Relax {
	md.potential = 0.;
	md.force.resize(0, 0.);
	md.force.resize(md.position.len(), 0.);
	md
}

fn add_rebo<I:IntoIterator<Item=usize>,W:Write>(mut md: Relax, free_indices: I, dim: Trip<f64>, mut ffile: Option<W>) -> Relax {
	if !USE_REBO { return md; }

	let (potential,force) = sp2::calc_potential_flat(md.position.clone(), dim);

	md.potential += potential;

	for i in free_indices {
		if let Some(file) = ffile.as_mut() {
			writeln!(file, "REB:{} F= {} {} {}", i, force[3*i], force[3*i+1], force[3*i+2]);
		}

		for k in 0..3 {
			md.force[3*i + k] += force[3*i + k];
		}
	}
	md
}

// precomputed info about what r terms and θ terms exist.
// contains index pairs/triples which are associated with at least one nonzero
//  term in the correction forces. (taking into account that fixed atoms have zero force)
struct ForceTermInfo {
	free_indices: Set<usize>,
	radial_pairs: Set<Pair<usize>>,
	angle_triples: Set<Trip<usize>>,
}

fn compute_force_info(free_indices: &[usize], parents: &[usize]) -> ForceTermInfo {
	let free_indices: Set<_> = free_indices.iter().cloned().collect();

	let radial_pairs = {
		let mut set = Set::new();
		for i in 0..parents.len() {
			let j = parents[i];
			if free_indices.contains(&i) { set.insert((i, j)); }
			if free_indices.contains(&j) { set.insert((j, i)); }
		}
		set
	};

	let angle_triples = {
		let mut set = Set::new();
		for i in 0..parents.len() {
			let j = parents[i];
			let k = parents[j];
			if i != k {
				if free_indices.contains(&i) || free_indices.contains(&j) { set.insert((i,j,k)); }
				if free_indices.contains(&k) || free_indices.contains(&j) { set.insert((k,j,i)); }
			}
		}
		set
	};

	ForceTermInfo {
		free_indices: free_indices,
		radial_pairs: radial_pairs,
		angle_triples: angle_triples,
	}
}

fn add_corrections<W:Write>(mut md: Relax, info: &ForceTermInfo, dim: Trip<f64>, mut ffile: Option<W>) -> Relax {
	let tup3 = |xs: &[f64], i:usize| { (0,1,2).map(|k| xs[3*i+k]) };
	let tup3set = |xs: &mut [f64], i:usize, v:(f64,f64,f64)| { v.enumerate().map(|(k,x)| xs[3*i+k] = x); };
	let tup3add = |xs: &mut [f64], i:usize, v:(f64,f64,f64)| {
		if info.free_indices.contains(&i) {
			let tmp = tup3(&xs, i);
			tup3set(xs, i, tmp.add_v(v));
		}
	};
	let apply  = |iso: NaIso, v: (f64,f64,f64)| { from_na_point(iso * to_na_point(v)) };
	let applyV = |iso: NaIso, v: (f64,f64,f64)| { from_na_vector(iso * to_na_vector(v)) };

	for &(i,j) in &info.radial_pairs {
		let pi = tup3(&md.position, i);
		let pj = tup3(&md.position, j);
		let dvec = nearest_image_sub(pi, pj, dim);

		let ForceOut { force: signed_force, potential } = RADIUS_FORCE.data(dvec.sqnorm().sqrt());
		let f = normalize(dvec).mul_s(signed_force);

		if let Some(file) = ffile.as_mut() {
			writeln!(file, "RAD:{}:{} V= {} F= {} {} {}", i, j, potential, f.0, f.1, f.2);
		}

		md.potential += potential;
		tup3add(&mut md.force, i, f);
	}

	for &(i,j,k) in &info.angle_triples {
		let pi = tup3(&md.position, i);
		let pj = tup3(&md.position, j);
		let pk = tup3(&md.position, k);

		let pj = pi.add_v(nearest_image_sub(pj, pi, dim));
		let pk = pj.add_v(nearest_image_sub(pk, pj, dim));

		// move parent to origin
		let (pi,pj,pk) = (pi,pj,pk).map(|x| x.sub_v(pj));

		// rotate parent's parent to +z
		let (_,θk,φk) = spherical_from_cart(pk);
		let iso = rotate_y(-θk) * rotate_z(-φk);
		let inv = iso.inverse_transformation();
		let (pi,pj,pk) = (pi,pj,pk).map(|x| apply(iso, x));

		// are things placed about where we expect?
		assert!(pj.0.abs() < 1e-5, "{}", pj.0);
		assert!(pj.1.abs() < 1e-5, "{}", pj.1);
		assert!(pj.2.abs() < 1e-5, "{}", pj.2);
		assert!(pk.0.abs() < 1e-5, "{}", pk.0);
		assert!(pk.1.abs() < 1e-5, "{}", pk.1);
		assert!(pk.2 > 0.,         "{}", pk.2);
		assert_eq!(pi, pi);

		// get dat angle
		let (_,θi,_) = spherical_from_cart(pi);
		let θ_hat = unit_θ_from_cart(pi);

		// force
		let ForceOut { force: signed_force, potential } = THETA_FORCE.data(θi);
		let f = θ_hat.mul_s(signed_force);

		// bring force back into cartesian.
		// (note: transformation from  grad' V  to  grad V  is more generally the
		//   transpose matrix of the one that maps x to x'. But for a rotation,
		//   this is also the inverse.)
		let f = applyV(inv, f);

		if let Some(file) = ffile.as_mut() {
			writeln!(file, "ANG:{}:{}:{} V= {} F= {} {} {}", i, j, k, potential, f.0, f.1, f.2);
		}

		// Note to self:
		// Yes, it is correct for the potential to always be added once,
		// regardless of how many of the atoms are fixed.
		md.potential += potential;

		// ultimately, the two outer atoms (i, k) get pulled in similar directions,
		// and the middle one (j) receives the opposing forces
		tup3add(&mut md.force, i, f);
		tup3add(&mut md.force, j, f.mul_s(-1.));
	}
	md
}

fn spherical_from_cart((x,y,z): Trip<f64>) -> Trip<f64> {
	let ρ = (x*x + y*y).sqrt();
	let r = (ρ*ρ + z*z).sqrt();
	(r, ρ.atan2(z), y.atan2(x))
}

fn normalize(p: Trip<f64>) -> Trip<f64> {
	let r = p.sqnorm().sqrt();
	p.map(|x| x/r)
}

fn unit_θ_from_cart((x,y,z): Trip<f64>) -> Trip<f64> {
	// rats, would be safer to compute these from spherical
	if x == 0. && y == 0. { (z.signum(),0.,0.) }
	else {
		let ρ = (x*x + y*y).sqrt();
		let r = (ρ*ρ + z*z).sqrt();
		(x*z/ρ/r, y*z/ρ/r, -ρ/r)
	}
}

fn cart_from_spherical((r,θ,φ): Trip<f64>) -> Trip<f64> {
	let (sinθ,cosθ) = (θ.sin(), θ.cos());
	let (sinφ,cosφ) = (φ.sin(), φ.cos());
	(r*sinθ*cosφ, r*sinθ*sinφ, r*cosθ)
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
	normalize(((),(),()).map(|_| normal.ind_sample(rng) as Float))
}

fn random_border_position<R:Rng>(rng: &mut R) -> Trip<Frac> {
	// random point
	let mut point = ((),(),()).map(|_| Frac(rng.next_f64() as Float));

	// project onto a vacuum face
	loop {
		let k = rng.gen_range(0, 3);
		if IS_VACUUM.into_nth(k) {
			*point.mut_nth(k) = Frac(0.);
			break;
		}
	}
	point
}

// for debugging the isometries; this should be a barbell shape, with
// 4 atoms protruding from each end in 4 diagonal directions
fn barbell_nucleus(dimension: Trip<f64>) -> Tree {
	let mut tree = Tree::from_two(dimension, DIMER_INITIAL_SEP, (Label::Si, Label::C));
	tree.attach_new(0, Label::Si, DIMER_INITIAL_SEP, PI*0.0);
	tree.attach_new(0, Label::Si, DIMER_INITIAL_SEP, PI*0.5);
	tree.attach_new(0, Label::Si, DIMER_INITIAL_SEP, PI*1.0);
	tree.attach_new(0, Label::Si, DIMER_INITIAL_SEP, PI*1.5);
	tree.attach_new(1, Label::C, DIMER_INITIAL_SEP, PI*0.0);
	tree.attach_new(1, Label::C, DIMER_INITIAL_SEP, PI*0.5);
	tree.attach_new(1, Label::C, DIMER_INITIAL_SEP, PI*1.0);
	tree.attach_new(1, Label::C, DIMER_INITIAL_SEP, PI*1.5);
	tree
}

fn random_barbell_nucleus(dimension: Trip<f64>) -> Tree {
	let dir = random_direction(&mut rand::weak_rng());
	let pos1 = ((),(),()).map(|_| Frac(rand::weak_rng().next_f64())).cart(dimension);
	let pos2 = pos1.add_v(dir.mul_s(DIMER_INITIAL_SEP));
	let mut tree = Tree::from_two_pos(dimension, (pos1,pos2), (Label::Si, Label::C));
	tree.attach_new(0, Label::Si, DIMER_INITIAL_SEP, PI*0.0);
	tree.attach_new(0, Label::Si, DIMER_INITIAL_SEP, PI*0.5);
	tree.attach_new(0, Label::Si, DIMER_INITIAL_SEP, PI*1.0);
	tree.attach_new(0, Label::Si, DIMER_INITIAL_SEP, PI*1.5);
	tree.attach_new(1, Label::C, DIMER_INITIAL_SEP, PI*0.0);
	tree.attach_new(1, Label::C, DIMER_INITIAL_SEP, PI*0.5);
	tree.attach_new(1, Label::C, DIMER_INITIAL_SEP, PI*1.0);
	tree.attach_new(1, Label::C, DIMER_INITIAL_SEP, PI*1.5);
	tree
}

fn one_dimer(dimension: Trip<f64>) -> Tree {
	let mut tree = Tree::from_two(dimension, DIMER_INITIAL_SEP, (Label::Si, Label::Si));
	tree.transform_mut(translate(dimension.mul_s(0.5)));
	tree
}

fn hexagon_nucleus(dimension: Trip<f64>) -> Tree {
	// HACK meaning of attachment angle is arbitrary (beyond being fixed on a per-atom basis)
	//  so hardcoded angles are trouble
	let mut tree = Tree::from_two(dimension, DIMER_INITIAL_SEP, (Label::Si, Label::Si));
	let i = tree.attach_new(1, Label::Si, DIMER_INITIAL_SEP, PI*0.5);
	let i = tree.attach_new(i, Label::Si, DIMER_INITIAL_SEP, PI*0.5);
	let i = tree.attach_new(i, Label::Si, DIMER_INITIAL_SEP, PI*0.5);
	let _ = tree.attach_new(i, Label::Si, DIMER_INITIAL_SEP, PI*0.5);
	tree.transform_mut(translate(dimension.mul_s(0.5)));

	let labels = tree.labels.clone();
	let mut force_file = if FORCE_DEBUG {
		let path = format!("xyz-debug/force-start");
		Some(File::create(&path).unwrap())
	} else { None };
	let mut xyz_debug_file = if XYZ_DEBUG {
		let path = format!("xyz-debug/event-start.xyz");
		Some(File::create(&path).unwrap())
	} else { None };

	let reason = relax_suffix_using_fire(&mut tree, 0, force_file, |md| {
		if let Some(file) = xyz_debug_file.as_mut() {
			let pos = unflatten(&md.position);
			let lab = labels.clone();
			write_xyz_(file, pos, lab.clone(), lab.len());
		}
	});

	match reason {
		Convergence => tree,
		x => panic!("could not relax hexagon: {:?}", x),
	}
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
		(self.deque[self.deque.len()-1] - self.deque[self.deque.len()-2]) / 1_000_000
	}
	pub fn average_ms(&self) -> u64 {
		(self.deque[self.deque.len()-1] - self.deque[0]) / ((self.deque.len() as u64 - 1) * 1_000_000)
	}
}

fn test_outputs() {
	let doit = |tree, path| {
		write_xyz(File::create(path).unwrap(), &tree, tree.len());
	};

	doit(barbell_nucleus(DIMENSION), "barbell.xyz");
	doit(random_barbell_nucleus(DIMENSION), "barbell-random.xyz");

	doit(hexagon_nucleus(DIMENSION), "hexagon.xyz");
}

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


