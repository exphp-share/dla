
use std::io::{stderr, Write};
use super::{Trip,Cart};
use libc::c_int;
use libc::c_uchar;
use homogenous::prelude::*;

pub fn calc_potential(free: Trip<f64>, fixed: Vec<Trip<f64>>, dim: Trip<f64>) -> (f64, Trip<f64>) {
	// NOTE: this brazenly assumes that [(f64,f64,f64); n] and #[repr(C)] [f64; 3*n]
	//       have the same representation.

	// no NaNs allowed
	assert!(free.0 == free.0);
	assert!(free.1 == free.1);
	assert!(free.2 == free.2);

	let mut pos = fixed;
	pos.insert(0, free);

	let mut lattice = vec![
		dim.0, 0., 0.,
		0., dim.1, 0.,
		0., 0., dim.2,
	];

	println!("PRE: {:?}", pos);
	// ffi outputs
	let mut grad = vec![(0.,0.,0.); pos.len()];
	let mut potential = 0.;
	let flag = { // scope &mut borrows
		let p_pos = pos.as_mut_ptr() as *mut f64;
		let p_grad = grad.as_mut_ptr() as *mut f64;
		let p_potential = (&mut potential) as *mut f64;
		let p_lattice = lattice.as_mut_ptr();
		let flag = unsafe {
			::dla_sys::calc_potential(pos.len() as c_int, p_pos, p_grad, p_potential, p_lattice)
		};
		flag
	};
	println!("POST: {:?}", pos);

	match flag {
		0 => {},
		1 => {
			writeln!(&mut stderr(), "POS {:?}", pos);
			panic!("");
		},
		_ => panic!("calc_potential: unexpected flag value: {}", flag),
	}

	// get the inserted atom's gradient
	(potential, grad[0])
}

pub fn relax_all(pos: Vec<Trip<Cart>>, dim: Trip<f64>) -> Vec<Trip<Cart>> {
	let n = pos.len();
	relax(pos, vec![false; n], dim)
}

pub fn relax(pos: Vec<Trip<Cart>>, fixed: Vec<bool>, dim: Trip<f64>) -> Vec<Trip<Cart>> {
	// NOTE: this brazenly assumes that [(f64,f64,f64); n] and #[repr(C)] [f64; 3*n]
	//       have the same representation.
	let pos: Vec<_> = pos.into_iter().map(|x: Trip<Cart>| x.map(|x| x.0)).collect();

	assert_eq!(fixed.len(), pos.len());
	let mut lattice = vec![
		dim.0, 0., 0.,
		0., dim.1, 0.,
		0., 0., dim.2,
	];

	let mut pos = pos;
	let mut fixed: Vec<_> = fixed.into_iter().map(|f| match f { true => 1, false => 0 } as c_uchar).collect();

	//println!("PRE: {:?}", pos);
	//println!("FIX: {:?}", fixed);
	let flag = {
		let mut potential = 0.;
		let mut grad = vec![(0.,0.,0.); pos.len()];

		let p_pos = pos.as_mut_ptr() as *mut f64;
		let p_fixed = fixed.as_mut_ptr() as *mut c_uchar;
		let p_grad = grad.as_mut_ptr() as *mut f64;
		let p_potential = (&mut potential) as *mut f64;
		let p_lattice = lattice.as_mut_ptr();
		let flag = unsafe {
			::dla_sys::relax_structure(pos.len() as c_int, p_fixed, p_pos, p_grad, p_potential, p_lattice)
		};
		flag
	};
	//println!("POST: {:?}", pos);

	match flag {
		0 => {},
		1 => {
			writeln!(&mut stderr(), "POS {:?}", pos);
			panic!("");
		},
		_ => panic!("calc_potential: unexpected flag value: {}", flag),
	}

	pos.into_iter().map(|x: Trip<f64>| x.map(|x| Cart(x))).collect()
}
