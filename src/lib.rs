
// BEWARE OF DOG

#![feature(non_ascii_idents)]
#![feature(test)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

#[macro_use]
pub mod common;
use common::*;

const PBC: Pbc = Pbc {
	dim: (100., 100., 100.),
	vacuum: (true, true, true),
};
const NPARTICLE: usize = 200;

const DEBUG: bool = false; // generates a general debug file
const XYZ_DEBUG: Option<usize> = Some(30); // creates "xyz-debug/event-*.xyz"  files
const FORCE_DEBUG: ForceDebug = ForceDebug::Summary; // creates "xyz-debug/force-*"
const VALIDATE_REBO: bool = false;

#[derive(Copy,Clone,Debug,PartialOrd,Ord,Eq,PartialEq,Hash)]
enum ForceDebug { None, Summary, Full }

// For easily switching forces on/off
const FORCE_PARAMS: ::force::Params = FORCE_PARAMS_SRC;
//const FORCE_PARAMS: ::force::Params = JUST_REBO;
//const FORCE_PARAMS: ::force::Params = WITHOUT_REBO;

const FORCE_PARAMS_SRC: ::force::Params = ::force::Params {
	radial: Model::Morse { center: 1.41, D: 100., k: 400. },
	angular: Model::Quadratic { center: (120.*PI/180.), k: 400. },
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
		flail_step_limit: Some(10),
		turn_condition: ::fire::TurnCondition::Potential,
//		turn_condition: ::fire::TurnCondition::FDotV,
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

pub mod mains {
	use super::*;
	use ::std::fs::File;

	pub fn dla() {
		let tree = ::dla_run();
		::serde_json::to_writer(&mut File::create("xyz-debug/tree.json").unwrap(), &tree).unwrap();
	}

	pub fn hex_test() { ::hexagon_nucleus(PBC); }

	pub fn gen_test() { ::test_outputs(); }

	pub fn reruns() {
		let path = ::std::env::args().nth(1).unwrap_or("xyz-debug/tree.json".to_string());
		::run_reruns_on(&path);
	}
}

fn dla_run() -> Tree {
	let PER_STEP = 2;

	let mut tree = hexagon_nucleus(PBC);

	let mut rng = rand::weak_rng();
	let mut timer = Timer::new(30);
	let mut debug_file = cond_file!(DEBUG, "debug");

	let nbr_radius = BROWNIAN_STOP_RADIUS;
	let final_particles = PER_STEP*NPARTICLE + tree.len();

	for dla_step in 0..NPARTICLE {
		err!("Particle {:8} of {:8}: ", dla_step, NPARTICLE);

		let mut finder = NeighborFinder::from_positions(tree.pos.clone(), tree.pbc);

		let mut pos = random_border_position(&mut rng, tree.pbc);

		// move until ready to place
		while finder.neighborhood(pos, nbr_radius).is_empty() {
			//errln!("({:4},{:4},{:4})  ({:8?} ms)",
			//	(pbc.0).0, (pos.1).0, (pos.2).0, (precise_time_ns() - start_time)/1000);

			let disp = random_direction(&mut rng).mul_s(MOVE_RADIUS);
			pos = tree.pbc.wrap(pos.add_v(disp));
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

			finder.neighborhood(nbrhood_center, RELAX_NEIGHBORHOOD_RADIUS)
			.into_iter()
			.chain(first_new_index..tree.len()) // the new indices are not yet in state
			.filter(|&i| tree.labels[i] != Label::Si)
			// ascending by distance
			.map(|i| (i, tree.pbc.distance(nbrhood_center, tree.pos[i])))
			.sorted_by(|&(_,a), &(_,b)| a.partial_cmp(&b).unwrap())
			.into_iter().take(RELAX_MAX_PARTICLE_COUNT)
			.map(|(i,_)| i)
			.collect_vec()
		};

		let n_free = free_indices.len();
		let (n_relax_steps, stop_reason) = {
			unsafe { relax_with_files(FORCE_PARAMS, &mut tree, free_indices, &format!("{:06}", dla_step), |md| {
				for file in &mut debug_file {
					writeln!(file, "F {} {} {} {} {}", dla_step, md.nstep, md.alpha, md.timestep, md.cooldown).unwrap();
					for i in 0..md.position.len()/3 {
						let v = (md.velocity[3*i+0], md.velocity[3*i+1], md.velocity[3*i+2]).sqnorm().sqrt();
						let f = (md.force[3*i+0], md.force[3*i+1], md.force[3*i+2]).sqnorm().sqrt();
						writeln!(file, "A {} {} {} {:.6} {:.6}", dla_step, md.nstep, i, v, f).unwrap();
					}
				}
			})}
		};

		// debugging info
		let frac = tree.pbc.frac(tree.pbc.wrap(pos));
		timer.push();
		errln!("({:8.6}, {:8.6}, {:8.6})  ({:5?} ms, avg: {:5?})  (relaxed {:3} in {:6} after {:?})",
			frac.0, frac.1, frac.2, timer.last_ms(), timer.average_ms(), n_free, n_relax_steps, stop_reason
		);

		write_xyz(stdout(), &tree, Some(final_particles));
	}

	{
		let pos = tree.pos.clone().into_iter().map(|x| tree.pbc.wrap(x));
		write_xyz_(stdout(), pos, tree.labels.clone(), None);
	}
	assert_eq!(final_particles, tree.len());

	// copy pasta alert
	{
		err!("Relaxing full structure...");

		let carbons = tree.carbons();
		let (n_steps, reason) = unsafe { relax_with_files(FORCE_PARAMS, &mut tree, carbons, "end-a", |_| {}) };
		write_xyz(stdout(), &tree, None);
		errln!(" done in {} steps after {:?}...", n_steps, reason);
	}

	{
		err!("Relaxing with REBO...");

		let carbons = tree.carbons();
		let (n_steps, reason) = unsafe { relax_with_files(JUST_REBO, &mut tree, carbons, "end-b", |_| {}) };
		write_xyz(stdout(), &tree, None);
		errln!(" done in {} steps after {:?}...", n_steps, reason);
	}
	
	tree
}

fn run_reruns_on(path: &str) {
	let mut tree: Tree = serde_json::from_reader(&mut File::open(path).unwrap()).unwrap();

	let free_indices = tree.carbons();

	let mut params = FORCE_PARAMS;

	let mut file = File::create(&format!("reruns-loop.xyz")).unwrap();

	let semi_inclusive_linspace = |n,a,b| (0..n).map(move |i| a + (b-a) * i as f64/n as f64);

	let n_frames_one_dir = 5;
	let min_k = 0.;
	let max_k = 100.;

//	let ks_dsc = semi_inclusive_linspace(n_frames_one_dir, max_k, min_k);
//	let ks_asc = semi_inclusive_linspace(n_frames_one_dir, min_k, max_k);
	let mut ks = ::std::iter::once(100.).chain(semi_inclusive_linspace(5, 25., 0.)).chain(vec![0.]);

	for (step, k) in ks.enumerate() {
		params.radial.set_spring_constant(k).expect("kek");
		params.angular.set_spring_constant(k).expect("kek");

		let (n_relax_steps, reason) = unsafe {
			relax_with_files(params, &mut tree, free_indices.clone(), &format!("reruns-{:03}", step), |_| {})
		};

		errln!("Step {:03} ended in {} steps after {:?}", step, n_relax_steps, reason);
		write_xyz(&mut file, &tree, None);
	}
}

/// Not reentrant; behavior is undefined if the callback also calls this function.
unsafe fn relax_with_files<C,I>(params: ::force::Params, tree: &mut Tree, free_indices: I, file_id: &str, mut cb: C) -> (usize, ::fire::StopReason)
where C:FnMut(&Fire), I: IntoIterator<Item=usize>,
{
	let labels = tree.labels.clone();
	let force_file = cond_file!(FORCE_DEBUG > ForceDebug::None, "xyz-debug/force-{}", file_id);
	let mut xyz_file = cond_file!(XYZ_DEBUG.is_some(), "xyz-debug/event-{}.xyz", file_id);

	relax_using_fire(params, tree, free_indices, force_file, |md| {
		for file in &mut xyz_file {
			if md.nstep % XYZ_DEBUG.unwrap() == 0 {
				write_xyz_(file, unflatten(&md.position), labels.clone(), None);
			}
		}
		cb(&md);
	})
}

/// Not reentrant; behavior is undefined if the callback also calls this function.
unsafe fn relax_using_fire<C, I, W>(params: ::force::Params, tree: &mut Tree, free_indices: I, ffile: Option<W>, mut cb: C) -> (usize, ::fire::StopReason)
where C:FnMut(&Fire), I: IntoIterator<Item=usize>, W: Write,
{
	let free_indices = free_indices.into_iter().collect();

	let force = ::force::Composite::prepare(params, &tree, &free_indices);

	let ffile = ::std::cell::RefCell::new(ffile);
	let (pos,out) = {
		Fire::init(RELAX_PARAMS, flatten(&tree.pos.clone()))
		// cb to write forces before fire
		.relax(|md| {
			let mut ffile = ffile.borrow_mut();
			for file in &mut *ffile {
				writeln!(file, "STEP {}", md.nstep).unwrap();
			}

			let mut md = zero_forces(md);
			force.tally(&mut md, ffile.as_mut(), &free_indices, tree.pbc);
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
	out
}


// what a mess I've made; how did we accumulate so many dependencies? O_o
extern crate time;
extern crate test;
extern crate rand;
#[macro_use] extern crate homogenous;
#[macro_use] extern crate itertools;
extern crate dla_sys;
extern crate libc;
extern crate num;
extern crate nalgebra;
#[macro_use] extern crate serde_derive;
extern crate serde_json;
extern crate serde;



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
use std::io::stdout;
use std::fs::File;

use std::f64::consts::PI;

use force::Model;
use fire::Fire;

#[derive(Copy,Clone,Eq,PartialEq,Ord,PartialOrd,Hash,Serialize,Deserialize,Debug)]
pub enum Label { C, Si }
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
pub struct Tree {
	labels: Vec<Label>,
	pos: Vec<Trip<Cart>>,
	parents: Vec<usize>,
	pbc: Pbc,
}

impl Tree {
	fn from_two(pbc: Pbc, length: Cart, labels: Pair<Label>) -> Self {
		Tree::from_two_pos(pbc, (CART_ORIGIN, (0., length, 0.)), labels)
	}
	fn from_two_pos(pbc: Pbc, pos: Pair<Trip<Cart>>, labels: Pair<Label>) -> Self {
		Tree {
			labels:  vec![labels.0, labels.1],
			pos:     vec![pos.0, pos.1],
			parents: vec![1, 0],
			pbc: pbc,
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
		let iso = self.pbc.look_at(self.pos[parent], self.pos[self.parents[parent]]);
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

	fn carbons(&self) -> Vec<usize> {
		(0..self.len()).filter(|&i| self.labels[i] == Label::C).collect()
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

fn random_direction<R:Rng>(mut rng: R) -> Trip<Cart> {
	let normal = Normal::new(0.0, 1.0);
	normalize(((),(),()).map(|_| normal.ind_sample(&mut rng) as Float))
}

fn random_border_position<R:Rng>(mut rng: R, pbc: Pbc) -> Trip<Cart> {
	let mut point = pbc.random_point_in_volume(&mut rng);

	// project onto a vacuum face
	loop {
		let k = rng.gen_range(0, 3);
		if pbc.vacuum.into_nth(k) {
			*point.mut_nth(k) = 0.;
			break;
		}
	}
	point
}

// for debugging the isometries; this should be a barbell shape, with
// 4 atoms protruding from each end in 4 diagonal directions
fn barbell_nucleus(pbc: Pbc) -> Tree {
	let mut tree = Tree::from_two(pbc, DIMER_INITIAL_SEP, (Label::Si, Label::C));
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

fn random_barbell_nucleus(pbc: Pbc) -> Tree {
	let dir = random_direction(rand::weak_rng());
	let pos1 = pbc.random_point_in_volume(&mut rand::weak_rng());
	let pos2 = pos1.add_v(dir.mul_s(DIMER_INITIAL_SEP));
	let mut tree = Tree::from_two_pos(pbc, (pos1,pos2), (Label::Si, Label::C));
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

fn one_dimer(pbc: Pbc) -> Tree {
	let mut tree = Tree::from_two(pbc, DIMER_INITIAL_SEP, (Label::Si, Label::Si));
	tree.transform_mut(translate(pbc.center()));
	tree
}

fn hexagon_nucleus(pbc: Pbc) -> Tree {
	// HACK meaning of attachment angle is arbitrary (beyond being fixed on a per-atom basis)
	//  so hardcoded angles are trouble
	let mut tree = Tree::from_two(pbc, DIMER_INITIAL_SEP, (Label::Si, Label::Si));
	let i = tree.attach_new(1, Label::Si, DIMER_INITIAL_SEP, PI*0.5);
	let i = tree.attach_new(i, Label::Si, DIMER_INITIAL_SEP, PI*0.5);
	let i = tree.attach_new(i, Label::Si, DIMER_INITIAL_SEP, PI*0.5);
	let _ = tree.attach_new(i, Label::Si, DIMER_INITIAL_SEP, PI*0.5);
	tree.transform_mut(translate(pbc.center()));
	tree.perturb_mut(0.1);

	let free_indices = 0..tree.len();
	let (_,reason) = unsafe { relax_with_files(FORCE_PARAMS, &mut tree, free_indices, "start", |_| {}) };
	match reason {
		::fire::StopReason::Convergence => tree,
		::fire::StopReason::Flailing => {
			errln!("Warning: Core relaxation ended in flailing!");
			tree
		},
		::fire::StopReason::Timeout => panic!("could not relax core"),
	}
}

fn test_outputs() {
	let doit = |tree, path| {
		write_xyz(File::create(path).unwrap(), &tree, None);
	};

	doit(barbell_nucleus(PBC), "barbell.xyz");
	doit(random_barbell_nucleus(PBC), "barbell-random.xyz");

	doit(hexagon_nucleus(PBC), "hexagon.xyz");
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


