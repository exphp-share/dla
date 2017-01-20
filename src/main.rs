
// BEWARE OF DOG

#![feature(non_ascii_idents)]
#![feature(test)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

const DIMENSION: Trip<Float> = (75., 75., 75.);
const IS_VACUUM: Trip<bool> = (true, true, false);
const NPARTICLE: usize = 100;

const DEBUG: bool = false; // generates a general debug file
const XYZ_DEBUG: bool = true; // creates "xyz-debug/event-*.xyz"  files
const FORCE_DEBUG: bool = true; // creates "xyz-debug/force-*"

// For easily switching forces on/off
const FORCE_PARAMS: ::force::Params = FORCE_PARAMS_SRC;
//const FORCE_PARAMS: ::force::Params = JUST_REBO;
//const FORCE_PARAMS: ::force::Params = WITHOUT_REBO;

const FORCE_PARAMS_SRC: ::force::Params = ::force::Params {
	radial: Model::Morse { center: 1.41, D: 100., k: 100. },
	angular: Model::Quadratic { center: (120.*PI/180.), k: 100. },
	rebo: true,
};

const ZERO_FORCE: ::force::Params = ::force::Params { rebo: false, radial: Model::Zero, angular: Model::Zero };
const WITHOUT_REBO: ::force::Params = ::force::Params { rebo: false, ..FORCE_PARAMS_SRC };
const JUST_REBO: ::force::Params = ::force::Params { rebo: true, ..ZERO_FORCE };

// Simulates recent bugs... (these options exist to help identify the bug's impact)
const DOUBLE_COUNTED_RADIAL_POTENTIAL: bool = false;
const DOUBLE_COUNTED_ANGULAR_POTENTIAL: bool = false;
const ERRONEOUS_MORSE_PREFACTOR: bool = false;

const RELAX_PARAMS: ::fire::Params =
	::fire::Params {
		timestep_start: 1e-3,
		timestep_max:   0.05,
		force_tolerance: Some(1e-5),
//		step_limit: Some(4000),
		flail_step_limit: Some(50),
		turn_condition: ::fire::TurnCondition::Potential,
		//turn_condition: ::fire::TurnCondition::FDotV,
		..::fire::DEFAULT_PARAMS
	};

// VM: * Dimer sep should be 1.4 (Angstrom)
//     * Interaction radius (to begin relaxation) should be 2

const BROWNIAN_STOP_RADIUS: Cart = 2.;
const RELAX_NEIGHBORHOOD_RADIUS: Cart = 5.;
const RELAX_MAX_PARTICLE_COUNT: usize = 12;
const MOVE_RADIUS: Cart = 1.;
const DIMER_INITIAL_SEP: Cart = 1.4;

macro_rules! cond_file {
	($cond:expr, $($fmt_args:tt)+) => {
		if $cond {
			Some(File::create(&format!($($fmt_args)+)).unwrap())
		} else { None }
	};
}

fn main() {
//	test_outputs();
//	dla_run_test();
//	hexagon_nucleus(DIMENSION);
//	let tree = dla_run();
	run_relax_on(&::std::env::args().nth(1).unwrap_or("xyz-debug/tree.json".to_string()));
}

fn dla_run() {
	let tree = dla_run_();
	serde_json::to_writer(&mut File::create("xyz-debug/tree.json").unwrap(), &tree).unwrap();
}

fn run_relax_on(path: &str) {
	let tree = serde_json::from_reader(&mut File::open(path).unwrap()).unwrap();
	run_relax_on_(tree)
}

fn dla_run_() -> Tree {
	let PER_STEP = 2;

	let mut tree = hexagon_nucleus(DIMENSION);

	let mut rng = rand::weak_rng();
	let mut timer = Timer::new(30);
	let mut debug_file = cond_file!(DEBUG, "debug");

	let nbr_radius = BROWNIAN_STOP_RADIUS;
	let final_particles = PER_STEP*NPARTICLE + tree.len();

	for dla_step in 0..NPARTICLE {
		write!(stderr(), "Particle {:8} of {:8}: ", dla_step, NPARTICLE).unwrap();

		let mut finder = NeighborFinder::from_positions(tree.dimension, tree.pos.clone());

		let mut pos = random_border_position(&mut rng);

		// move until ready to place
		while finder.neighborhood(pos, nbr_radius).is_empty() {
			//writeln!(stderr(), "({:4},{:4},{:4})  ({:8?} ms)",
			//	(pos.0).0, (pos.1).0, (pos.2).0, (precise_time_ns() - start_time)/1000).unwrap();

			let disp = random_direction(&mut rng)
				.mul_s(MOVE_RADIUS).frac(tree.dimension);

			pos = reduce_pbc(pos.add_v(disp));
		}

		// introduce at random angles
		match PER_STEP {
			// dimer
			2 => {
				let i = finder.closest_neighbor(pos, nbr_radius).unwrap();
				let i = tree.attach_new(i, Label::C, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
				let _ = tree.attach_new(i, Label::C, DIMER_INITIAL_SEP, rng.next_f64()*2.*PI);
			},

			// trimer
			3 => {
				let i = finder.closest_neighbor(pos, nbr_radius).unwrap();
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

			finder.neighborhood(nbrhood_center.frac(tree.dimension), RELAX_NEIGHBORHOOD_RADIUS)
			.into_iter()
			.chain(first_new_index..tree.len()) // the new indices are not yet in state
			.filter(|&i| tree.labels[i] != Label::Si)
			// ascending by distance
			.map(|i| (i, nearest_image_dist_sq(nbrhood_center, tree.pos[i], tree.dimension)))
			.sorted_by(|&(_,a), &(_,b)| a.partial_cmp(&b).unwrap())
			.into_iter().take(RELAX_MAX_PARTICLE_COUNT)
			.map(|(i,_)| i)
			.collect_vec()
		};

		let n_free = free_indices.len();
//		let mut n_free = PER_STEP;
		let (n_relax_steps, stop_reason) = {
//			let n_total = tree.len();

//			let n_fixed = n_total - n_free;

			let mut n_relax_steps = 0; // prepare for more abominable hax...

			//relax_suffix_using_fire(&mut tree, n_fixed, |md| { // relax new
			//relax_suffix_using_fire(&mut tree, 6, force_debug_file, |md| { // relax all XXX
			//relax_suffix_using_fire(&mut tree, 0, force_debug_file, |md| { // relax all XXX
			let reason = relax_with_files(FORCE_PARAMS, &mut tree, free_indices, &format!("{:06}", dla_step), |md| { // relax all XXX

				for file in &mut debug_file {
					writeln!(file, "F {} {} {} {} {}", dla_step, md.nstep, md.alpha, md.timestep, md.cooldown).unwrap();
					for i in 0..md.position.len()/3 {
						let v = (md.velocity[3*i+0], md.velocity[3*i+1], md.velocity[3*i+2]).sqnorm().sqrt();
						let f = (md.force[3*i+0], md.force[3*i+1], md.force[3*i+2]).sqnorm().sqrt();
						writeln!(file, "A {} {} {} {:.6} {:.6}", dla_step, md.nstep, i, v, f).unwrap();
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

		write_xyz(&mut stdout(), &tree, Some(final_particles));
	}

	{
		let dimension = tree.dimension;
		let pos = tree.pos.clone().into_iter().map(|x| reduce_pbc(x.frac(dimension)).cart(dimension));
		write_xyz_(&mut stdout(), pos, tree.labels.clone(), None);
	}
	assert_eq!(final_particles, tree.len());
	tree
}

fn run_relax_on_(mut tree: Tree) {
	let free_indices = (0..tree.len()).filter(|&i| tree.labels[i] != Label::Si).collect_vec();

	let mut params = FORCE_PARAMS;

	let mut file = File::create(&format!("reruns-loop.xyz")).unwrap();

	let ks_asc = (  0..100).map(|i| (100 - i) as f64 / 100f64);
	let ks_dsc = (100..201).map(|i| (i - 100) as f64 / 100f64);
	let ks = ks_asc.chain(ks_dsc);

	for (step, k) in ks.enumerate() {
		params.radial.set_spring_constant(k).expect("kek");
		params.angular.set_spring_constant(k).expect("kek");

		let mut n_relax_steps = 0;
		let reason = relax_with_files(params, &mut tree, free_indices.clone(), &format!("reruns-{:03}", step),
			|md| {n_relax_steps = md.nstep});

		writeln!(stderr(), "Step {:03} ended in {} steps after {:?}", step, n_relax_steps, reason).unwrap();
		write_xyz(&mut file, &tree, None);
	}
}

fn relax_with_files<C,I>(params: ::force::Params, tree: &mut Tree, free_indices: I, file_id: &str, mut cb: C) -> ::fire::StopReason
where C:FnMut(&Fire), I: IntoIterator<Item=usize>,
{
	let labels = tree.labels.clone();
	let force_file = cond_file!(FORCE_DEBUG, "xyz-debug/force-{}", file_id);
	let mut xyz_file = cond_file!(XYZ_DEBUG, "xyz-debug/event-{}.xyz", file_id);

	relax_using_fire(params, tree, free_indices, force_file, |md| {
		for file in &mut xyz_file {
			write_xyz_(file, unflatten(&md.position), labels.clone(), None);
		}
		cb(&md);
	})
}

fn relax_using_fire<C, I, W>(params: ::force::Params, tree: &mut Tree, free_indices: I, ffile: Option<W>, mut cb: C) -> ::fire::StopReason
where C:FnMut(&Fire), I: IntoIterator<Item=usize>, W: Write,
{
	let free_indices = free_indices.into_iter().collect();

	let force = ::force::Composite::prepare(params, &free_indices, &tree.parents);

	let ffile = ::std::cell::RefCell::new(ffile);
	let (pos,reason) = {
		Fire::init(RELAX_PARAMS, flatten(&tree.pos.clone()))
		// cb to write forces before fire
		.relax(|md| {
			let mut ffile = ffile.borrow_mut();
			for file in &mut *ffile {
				writeln!(file, "STEP {}", md.nstep).unwrap();
			}

			let mut md = zero_forces(md);
			force.tally(&mut md, ffile.as_mut(), &free_indices, tree.dimension);
			cb(&md);

			md

		// cb that is invoked after fire (so that f_dot_v is known)
		}, |md| {
			for file in &mut *ffile.borrow_mut() {
				writeln!(file, "TOTAL_E {:23.18} F_DOT_V {:23.18} DT {:23.18}", md.potential, md.f_dot_v, md.timestep).unwrap();
			}
		})
	};

	tree.dangerously_reassign_positions(unflatten(&pos));
	reason
}


// what a mess I've made; how did we accumulate so many dependencies? O_o
extern crate time;
extern crate test;
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


pub mod common;
use common::*;

pub mod force;
pub mod fire;
pub mod ffi;

pub mod brownian;
use brownian::NeighborFinder;

pub mod timer;
use timer::Timer;

use rand::Rng;
use rand::distributions::{IndependentSample,Normal};
use itertools::Itertools;
use homogenous::prelude::*;
use homogenous::numeric::prelude::*;

use std::ops::Range;
use std::io::Write;
use std::io::{stderr,stdout};
use std::fs::File;

use std::f64::consts::PI;

use force::Model;
use fire::Fire;

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
			*p = as_na_point(*p, |p| iso * p);
		}
	}

	fn perturb_mut(&mut self, r: Cart) {
		let mut rng = rand::weak_rng();
		for p in &mut self.pos {
			*p = (*p).add_v(random_direction(&mut rng).mul_s(r));
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
		let pos = as_na_point(pos, |p| iso * p);
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

fn write_xyz<W: Write>(file: W, tree: &Tree, final_length: Option<usize>) {
	write_xyz_(file, tree.pos.clone(), tree.labels.clone(), final_length);
}

fn write_xyz_<W: Write, I,J>(mut file: W, pos: I, labels: J, final_length: Option<usize>)
where I: IntoIterator<Item=Trip<Cart>>, J: IntoIterator<Item=Label> {
	let mut pos = pos.into_iter().collect_vec();
	let mut labels = labels.into_iter().collect_vec();
	let first = pos[0];

	let final_length = final_length.unwrap_or(labels.len());

	labels.resize(final_length, Label::C);
	pos.resize(final_length, first);
	writeln!(file, "{}", final_length).unwrap();
	writeln!(file, "blah blah blah").unwrap();
	for (label, (x,y,z)) in labels.into_iter().zip(pos) {
		writeln!(file, "{} {} {} {}", label.as_str(), x, y, z).unwrap();
	}
}

//---------------------------

// odd placement...
fn zero_forces(mut md: Fire) -> Fire {
	md.potential = 0.;
	md.force.resize(0, 0.);
	md.force.resize(md.position.len(), 0.);
	md
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
	tree.perturb_mut(0.1);

	let free_indices = 0..tree.len();
	let reason = relax_with_files(FORCE_PARAMS, &mut tree, free_indices, "start", |_| {});
	match reason {
		::fire::StopReason::Convergence => tree,
		::fire::StopReason::Flailing => {
			writeln!(stderr(), "Warning: Core relaxation ended in flailing!").unwrap();
			tree
		},
		::fire::StopReason::Timeout => panic!("could not relax core"),
	}
}

fn test_outputs() {
	let doit = |tree, path| {
		write_xyz(File::create(path).unwrap(), &tree, None);
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


