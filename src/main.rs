
// BEWARE OF DOG

#![feature(non_ascii_idents)]
#![feature(field_init_shorthand)]
#![feature(test)]
#![allow(dead_code)]
//#![allow(unused_imports)]
#![allow(non_snake_case)]

extern crate time;
extern crate test;
extern crate rand;
#[macro_use] extern crate homogenous;
#[macro_use] extern crate itertools;
extern crate num;
#[macro_use] extern crate serde_derive;
extern crate serde_json;
extern crate serde;

#[macro_use]
pub mod common;
use common::*;

pub mod grid;
pub mod timer;

use itertools::Itertools;

use std::io::Write;
use std::io::stdout;
use std::fs::File;


const GRID_LATTICE_PARAMS: ::grid::LatticeParams = ::grid::LatticeParams {
	a: 1.41,
	c: 3.4,
};

macro_rules! cond_file {
	($cond:expr, $($fmt_args:tt)+) => {
		if $cond {
			Some(File::create(&format!($($fmt_args)+)).unwrap())
		} else { None }
	};
}

/*
fn main() {
	let lattice = GRID_LATTICE_PARAMS;

	for z in 0..80 {

		let sparse: ::grid::SparseGrid = ::serde_json::from_reader(&mut File::open(&format!("layer-{}.json", z)).unwrap()).unwrap();
		let grid = sparse.to_grid();
		let tree = grid.to_structure(lattice);
		write_xyz(&mut File::create(&format!("layer-{}.xyz", z)).unwrap(), &tree, None);
	}
}
*/

fn main() {
	let out_dir: ::std::path::PathBuf =
		::std::env::args().nth(1).expect("missing output dir name").into();

	let lattice = GRID_LATTICE_PARAMS;

	let mut prev_thread = None::<::std::thread::JoinHandle<()>>;
	let grid = grid::dla_run(lattice, |grid| {
		if let Some(handle) = prev_thread.take() {
			handle.join().unwrap();
		}

		let grid = grid.clone();
		let out_dir = out_dir.clone();
		prev_thread = Some(::std::thread::spawn(move || {
			let ofilepath = |s| out_dir.join(s);
			let ofile = |s| File::create(&ofilepath(s)).unwrap();
			let _ = ::std::fs::create_dir(&out_dir);

			::serde_json::to_writer(&mut ofile("lattice.json.tmp"), &::grid::lattice(GRID_LATTICE_PARAMS)).unwrap();
			::serde_json::to_writer(&mut ofile("grid.json.tmp"), &grid.to_sparse()).unwrap();
			::grid::write_poscar(ofile("grid.vasp.tmp"), lattice, &grid).unwrap();
			let tree = grid.to_structure(lattice);
			write_xyz(ofile("grid.xyz.tmp"), &tree, None);

			::std::fs::rename(&ofilepath("lattice.json.tmp"), &ofilepath("lattice.json")).unwrap();
			::std::fs::rename(&ofilepath("grid.json.tmp"),    &ofilepath("grid.json")).unwrap();
			::std::fs::rename(&ofilepath("grid.vasp.tmp"),    &ofilepath("grid.vasp")).unwrap();
			::std::fs::rename(&ofilepath("grid.xyz.tmp"),     &ofilepath("grid.xyz")).unwrap();
		}));
	});
	let tree = grid.to_structure(lattice);
}

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

fn write_xyz<W: Write>(file: W, tree: &Structure, final_length: Option<usize>) {
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

fn round_to_multiple(x:f64, m:f64) -> f64 { (x / m).round() * m }

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Structure {
	labels: Vec<Label>,
	pos: Vec<Trip<Cart>>,
	dim: Trip<Cart>,
}

impl Structure {
	pub fn len(&self) -> usize { self.pos.len() }
}
