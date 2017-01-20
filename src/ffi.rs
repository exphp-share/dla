use common::*;

use ::homogenous::prelude::*;
use ::homogenous::numeric::prelude::*;
use ::std::io::Write;
use ::std::io::stderr;
use ::libc::c_int;
use ::libc::c_uchar;

pub fn calc_rebo(pos: Vec<Trip<f64>>, dim: Trip<f64>) -> (f64, Vec<Trip<f64>>) {
	let (pot, grad) = calc_rebo_flat(flatten(&pos), dim);
	(pot, unflatten(&grad))
}


pub fn calc_rebo_flat(mut pos: Vec<f64>, dim: Trip<f64>) -> (f64, Vec<f64>) {
	// no NaNs allowed
	assert!(pos.iter().all(|&p| p == p));
	assert!(pos.len()%3 == 0);

	let mut lattice = vec![
		dim.0, 0., 0.,
		0., dim.1, 0.,
		0., 0., dim.2,
	];

	// ffi outputs
	let mut grad = vec![0.; pos.len()];
	let mut potential = 0.;
	let flag = { // scope &mut borrows
		let c_n = (pos.len() / 3) as c_int;
		let p_pos = pos.as_mut_ptr() as *mut f64;
		let p_grad = grad.as_mut_ptr() as *mut f64;
		let p_potential = (&mut potential) as *mut f64;
		let p_lattice = lattice.as_mut_ptr();
		let flag = unsafe {
			::dla_sys::calc_potential(c_n, p_pos, p_grad, p_potential, p_lattice)
		};
		flag
	};

	match flag {
		0 => {},
		1 => {
			writeln!(&mut stderr(), "POS {:?}", pos).unwrap();
			panic!("");
		},
		_ => panic!("calc_potential: unexpected flag value: {}", flag),
	}

	(potential, grad.into_iter().map(|x| -x).collect())
}

pub fn rebo_relax_all(pos: Vec<Trip<Cart>>, dim: Trip<f64>) -> Vec<Trip<Cart>> {
	let n = pos.len();
	rebo_relax(pos, vec![false; n], dim)
}

pub fn rebo_relax(pos: Vec<Trip<Cart>>, fixed: Vec<bool>, dim: Trip<f64>) -> Vec<Trip<Cart>> {
	// NOTE: this brazenly assumes that [(f64,f64,f64); n] and #[repr(C)] [f64; 3*n]
	//       have the same representation.

	assert_eq!(fixed.len(), pos.len());
	let mut lattice = vec![
		dim.0, 0., 0.,
		0., dim.1, 0.,
		0., 0., dim.2,
	];

	let mut pos = pos;
	let mut fixed: Vec<_> = fixed.into_iter().map(|f| match f { true => 1, false => 0 } as c_uchar).collect();

	let flag = {
		let mut potential = 0.;
		let mut grad = vec![(0.,0.,0.); pos.len()];

		let c_n = pos.len() as c_int;
		let p_pos = pos.as_mut_ptr() as *mut f64;
		let p_fixed = fixed.as_mut_ptr() as *mut c_uchar;
		let p_grad = grad.as_mut_ptr() as *mut f64;
		let p_potential = (&mut potential) as *mut f64;
		let p_lattice = lattice.as_mut_ptr();
		let flag = unsafe {
			::dla_sys::relax_structure(c_n, p_fixed, p_pos, p_grad, p_potential, p_lattice)
		};
		flag
	};

	match flag {
		0 => {},
		1 => {
			writeln!(&mut stderr(), "POS {:?}", pos).unwrap();
			panic!("");
		},
		_ => panic!("calc_potential: unexpected flag value: {}", flag),
	}

	pos
}
