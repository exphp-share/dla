
use std::io::{stderr, Write};
use super::{Trip,Cart};
use libc::c_int;
use libc::c_uchar;
use homogenous::prelude::*;
use homogenous::numeric::prelude::*;

/*
trait ToRaw {
	type Raw;
	fn to_raw(self) -> Self::Raw;
}
impl ToRaw for f64 {
	type Raw = f64;
	fn to_raw(self) -> Self::Raw { self }
}
impl ToRaw for Cart {
	type Raw = f64;
	fn to_raw(self) -> Self::Raw { self.0 }
}
impl ToRaw for Frac {
	type Raw = f64;
	fn to_raw(self) -> Self::Raw { self.0 }
}
impl<T:ToRaw> ToRaw for Trip<T> {
	type Raw = Trip<T::Raw>;
	fn to_raw(self) -> Self::Raw { self.map(T::to_raw) }
}
impl<'a,T:ToRaw+Clone> ToRaw for &'a [T] {
	type Raw = Vec<T::Raw>;
	fn to_raw(self) -> Self::Raw { self.iter().cloned().map(T::to_raw).collect() }
}
*/

pub fn calc_potential(mut pos: Vec<Trip<f64>>, dim: Trip<f64>) -> (f64, Vec<Trip<f64>>) {
	// NOTE: this brazenly assumes that [(f64,f64,f64); n] and #[repr(C)] [f64; 3*n]
	//       have the same representation.

	// no NaNs allowed
	for &p in &pos {
		assert!(p.0 == p.0);
		assert!(p.1 == p.1);
		assert!(p.2 == p.2);
	}

	let mut lattice = vec![
		dim.0, 0., 0.,
		0., dim.1, 0.,
		0., 0., dim.2,
	];

	// ffi outputs
	let mut grad = vec![(0.,0.,0.); pos.len()];
	let mut potential = 0.;
	let flag = { // scope &mut borrows
		let c_n = pos.len() as c_int;
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
			writeln!(&mut stderr(), "POS {:?}", pos);
			panic!("");
		},
		_ => panic!("calc_potential: unexpected flag value: {}", flag),
	}

	(potential, grad.into_iter().map(|x| x.map(|x| -x)).collect())
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
			writeln!(&mut stderr(), "POS {:?}", pos);
			panic!("");
		},
		_ => panic!("calc_potential: unexpected flag value: {}", flag),
	}

	pos.into_iter().map(|x: Trip<f64>| x.map(|x| Cart(x))).collect()
}



use num::{Float,Zero,One,NumCast};
#[derive(Debug,Clone)]
pub struct Params<F> {
	// as alpha increases, inertia decreases.
	// No idea what to call it.  "ertia"?
	pub alpha_max: F,
	pub alpha_dec: F,
	pub inertia_delay: u32,
	pub timestep_start: F,
	pub timestep_max: F,
	pub timestep_inc: F,
	pub timestep_dec: F,
	pub force_tolerance: Option<F>,
	pub step_limit: Option<usize>,
}

impl<F:Float> Default for Params<F> {
	fn default() -> Params<F> {
		let lit = |x| <F as NumCast>::from(x).unwrap();
		Params {
			inertia_delay: 5,
			timestep_dec: lit(0.5),
			timestep_inc: lit(1.1),
			alpha_dec: lit(0.99),
			alpha_max: lit(0.1),
			timestep_start: lit(::std::f64::NAN),
			timestep_max: lit(::std::f64::NAN),
			force_tolerance: None,
			step_limit: None,
		}
	}
}


// part of a that is parallel to b
fn par(a: Trip<Cart>, b: Trip<Cart>) -> Trip<Cart> {
	let b_norm = Cart(b.dot(b).0.sqrt());
	let b_unit = b.div_s(b_norm);
	b_unit.mul_s(a.dot(b_unit))
}
// part of a that is perpendicular to b
fn perp(a: Trip<Cart>, b: Trip<Cart>) -> Trip<Cart> {
	let c = a.sub_v(par(a,b));
	assert!(Cart(c.dot(a).0.abs()) <= Cart(1e-7));
	c
}

pub struct MdInput<F>  { pub position: Vec<F>, pub velocity: Vec<F>, pub timestep: F }
pub struct MdOutput<F> { pub position: Vec<F>, pub velocity: Vec<F>, pub force: Vec<F>, pub potential: F }

struct State<F> {
	timestep: F, alpha: F, delay: u32,
	velocity: Vec<F>,
}

fn fire<F:Float+::std::fmt::Debug>(params: &Params<F>, force: Vec<F>, mut state: State<F>) -> State<F> {
	let f_dot_v = force.iter().zip(&state.velocity).map(|(&x,&y)| x*y).fold(F::zero(), |x,y| x+y);
	let f_norm = force.iter().map(|&x| x*x).fold(F::zero(), |x,y| x+y).sqrt();
	let v_norm = state.velocity.iter().map(|&x| x*x).fold(F::zero(), |x,y| x+y).sqrt();

	for (v,&f) in state.velocity.iter_mut().zip(&force) {
		*v = (F::one() - state.alpha) * *v + state.alpha * f * v_norm/f_norm;
	}

	// don't go uphill
	if f_dot_v < Zero::zero() {
		state.timestep = state.timestep * params.timestep_dec;
		state.alpha = params.alpha_max;
		state.delay = params.inertia_delay;
		state.velocity.resize(0,           Zero::zero());
		state.velocity.resize(force.len(), Zero::zero());

	// start gaining inertia after a while downhill
	} else {
		if state.delay > 0 { state.delay -= 1; }
		else {
			state.timestep = (state.timestep * params.timestep_inc).min(params.timestep_max);
			state.alpha    = state.alpha * params.alpha_dec;
		}
	}

	state
}

// FIXME _hack is here because F is not constrained otherwise
// (G could theoretically implement FnMut for multiple F's)
pub struct Newton<F,G> { force: G, _hack: ::std::marker::PhantomData<F> }
impl<F:Float+::std::fmt::Debug,G> Newton<F,G> where G: FnMut(Vec<F>) -> (F, Vec<F>) {
	pub fn step(&mut self, MdInput { mut position, mut velocity, timestep }: MdInput<F>) -> MdOutput<F> {
		assert_eq!(position.len(), velocity.len());
		let (pot,force) = (&mut self.force)(position.clone());
		for (p,&v) in position.iter_mut().zip(&velocity) { *p = *p + v*timestep; }
		for (v,&f) in velocity.iter_mut().zip(&force)    { *v = *v + f*timestep; }
		MdOutput { position: position, velocity: velocity, force: force, potential: pot }
	}
}

pub fn run_fire<F:Float+::std::fmt::Debug, G>(params: &Params<F>, position: Vec<F>, /*mut md: M,*/ force: G) -> Vec<F>
where
//	M: FnMut(MdInput<F>) -> MdOutput<F>,
	G: FnMut(Vec<F>) -> (F,Vec<F>),
{
//	writeln!(::std::io::stderr(), "{:?}", params);
	assert!(!params.timestep_start.is_nan());
	assert!(!params.timestep_max.is_nan());
	assert!(params.force_tolerance.is_some() || params.step_limit.is_some(), "no stop condition");
	let mut state = State {
		timestep: params.timestep_start,
		alpha: params.alpha_max,
		delay: 0,
		velocity: vec![Zero::zero(); position.len()],
	};

	// hax; let's just do this here
	let mut md = Newton { force: force, _hack: ::std::marker::PhantomData::<F> };

	let mut position = Some(position);
	for nstep in 0.. {
		let MdOutput { position: new_position, velocity: new_velocity, force, potential }
			= md.step(MdInput { position: position.take().unwrap(), velocity: state.velocity.clone(), timestep: state.timestep });
		position = Some(new_position);
		state.velocity = new_velocity;

		let ttt = force.iter().map(|&x| x*x).fold(F::zero(), ::std::ops::Add::add);
		assert!(ttt == ttt);
		if let Some(tol) = params.force_tolerance {
			if force.iter().map(|&x| x*x).fold(F::zero(), ::std::ops::Add::add) <= tol {
				break;
			}
		}

		if let Some(limit) = params.step_limit {
			if nstep >= limit {
				break;
			}
		}

//		writeln!(::std::io::stderr(), "{} V={:?} dt={:?} ttt={:?} alpha={:?} delay={:?}", nstep, potential, state.timestep, ttt, state.alpha, state.delay);
		state = fire(params, force, state);
	}

	position.unwrap()
}
