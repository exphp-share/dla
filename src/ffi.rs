use common::*;

use ::homogenous::prelude::*;
use ::homogenous::numeric::prelude::*;
use ::std::io::Write;
use ::libc::c_int;
use ::libc::c_uchar;

pub fn calc_rebo(pos: Vec<Trip<f64>>, pbc: Pbc) -> (f64, Vec<Trip<f64>>) {
	let (pot, grad) = calc_rebo_flat(flatten(&pos), pbc);
	(pot, unflatten(&grad))
}


pub fn calc_rebo_flat(mut pos: Vec<f64>, pbc: Pbc) -> (f64, Vec<f64>) {
	// no NaNs allowed
	assert!(pos.iter().all(|&p| p == p));
	assert!(pos.len()%3 == 0);

	let mut lattice = lattice_matrix(pbc);

	// ffi outputs
	let mut grad = vec![0.; pos.len()];
	let mut potential = 0.;
	validate_flag("calc_rebo_flat", { // scope &mut borrows
		let c_n = (pos.len() / 3) as c_int;
		let p_pos = pos.as_mut_ptr() as *mut f64;
		let p_grad = grad.as_mut_ptr() as *mut f64;
		let p_potential = (&mut potential) as *mut f64;
		let p_lattice = lattice.as_mut_ptr();
		unsafe { ::dla_sys::calc_potential(c_n, p_pos, p_grad, p_potential, p_lattice) }
	});

	(potential, grad.into_iter().map(|x| -x).collect())
}

pub unsafe fn persistent_init(mut pos: Vec<f64>, pbc: Pbc) {
	assert!(pos.iter().all(|&p| p == p));
	assert!(pos.len()%3 == 0);

	let mut lattice = lattice_matrix(pbc);

	validate_flag("persistent_init", { // scope &mut borrows
		let c_n = (pos.len() / 3) as c_int;
		let p_pos = pos.as_mut_ptr() as *mut f64;
		let p_lattice = lattice.as_mut_ptr();
		::dla_sys::calc_potential_persistent_init(c_n, p_pos, p_lattice)
	});
}

pub unsafe fn persistent_calc(mut pos: Vec<f64>) -> (f64, Vec<f64>) {
	assert!(pos.iter().all(|&p| p == p));
	assert!(pos.len()%3 == 0);

	let mut grad = vec![0.; pos.len()];
	let mut potential = 0.;
	validate_flag("persistent_calc", { // scope &mut borrows
		let c_n = (pos.len() / 3) as c_int;
		let p_pos = pos.as_mut_ptr() as *mut f64;
		let p_grad = grad.as_mut_ptr() as *mut f64;
		let p_potential = (&mut potential) as *mut f64;
		::dla_sys::calc_potential_persistent_calc(c_n, p_pos, p_grad, p_potential)
	});

	(potential, grad.into_iter().map(|x| -x).collect())
}

pub fn rebo_relax_all(pos: Vec<Trip<Cart>>, pbc: Pbc) -> Vec<Trip<Cart>> {
	let n = pos.len();
	rebo_relax(pos, vec![false; n], pbc)
}

pub fn rebo_relax(pos: Vec<Trip<Cart>>, fixed: Vec<bool>, pbc: Pbc) -> Vec<Trip<Cart>> {
	// NOTE: this brazenly assumes that [(f64,f64,f64); n] and #[repr(C)] [f64; 3*n]
	//       have the same representation.

	assert_eq!(fixed.len(), pos.len());
	let mut lattice = lattice_matrix(pbc);

	let mut pos = pos;
	let mut fixed: Vec<_> = fixed.into_iter().map(|f| match f { true => 1, false => 0 } as c_uchar).collect();

	validate_flag("rebo_relax", {
		let mut potential = 0.;
		let mut grad = vec![(0.,0.,0.); pos.len()];

		let c_n = pos.len() as c_int;
		let p_pos = pos.as_mut_ptr() as *mut f64;
		let p_fixed = fixed.as_mut_ptr() as *mut c_uchar;
		let p_grad = grad.as_mut_ptr() as *mut f64;
		let p_potential = (&mut potential) as *mut f64;
		let p_lattice = lattice.as_mut_ptr();
		unsafe { ::dla_sys::relax_structure(c_n, p_fixed, p_pos, p_grad, p_potential, p_lattice) }
	});

	pos
}

fn validate_flag(msg: &str, flag: c_uchar) {
	match flag {
		0 => {},
		1 => panic!("{}: failure detected in ffi code", msg),
		_ => panic!("{}: unexpected flag value: {}", msg, flag),
	}
}

fn lattice_matrix(pbc: Pbc) -> Vec<Float> {
	let mut lattice = vec![
		pbc.dim.0, 0., 0.,
		0., pbc.dim.1, 0.,
		0., 0., pbc.dim.2,
	];

	// designate aperiodic axes
	if pbc.vacuum.0 { lattice[0*3 + 0] = 0. }
	if pbc.vacuum.1 { lattice[1*3 + 1] = 0. }
	if pbc.vacuum.2 { lattice[2*3 + 2] = 0. }

	lattice
}
