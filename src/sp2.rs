
use std::io::{stderr, Write};
use super::{Trip,Cart,flatten,unflatten};
use libc::c_int;
use libc::c_uchar;
use homogenous::prelude::*;
use homogenous::numeric::prelude::*;

pub fn calc_potential(pos: Vec<Trip<f64>>, dim: Trip<f64>) -> (f64, Vec<Trip<f64>>) {
	let (pot, grad) = calc_potential_flat(flatten(&pos), dim);
	(pot, unflatten(&grad))
}

pub fn calc_potential_flat(mut pos: Vec<f64>, dim: Trip<f64>) -> (f64, Vec<f64>) {
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
			writeln!(&mut stderr(), "POS {:?}", pos);
			panic!("");
		},
		_ => panic!("calc_potential: unexpected flag value: {}", flag),
	}

	(potential, grad.into_iter().map(|x| -x).collect())
}

pub fn relax_all(pos: Vec<Trip<Cart>>, dim: Trip<f64>) -> Vec<Trip<Cart>> {
	let n = pos.len();
	relax(pos, vec![false; n], dim)
}

pub fn relax(pos: Vec<Trip<Cart>>, fixed: Vec<bool>, dim: Trip<f64>) -> Vec<Trip<Cart>> {
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
			writeln!(&mut stderr(), "POS {:?}", pos);
			panic!("");
		},
		_ => panic!("calc_potential: unexpected flag value: {}", flag),
	}

	pos
}



#[derive(Debug,Clone)]
pub struct Params {
	// as alpha increases, inertia decreases.
	// No idea what to call it.  "ertia"?
	pub alpha_max: f64,
	pub alpha_dec: f64,
	pub inertia_delay: u32,
	pub timestep_start: f64,
	pub timestep_max: f64,
	pub timestep_inc: f64,
	pub timestep_dec: f64,
	pub force_tolerance: Option<f64>,
	pub step_limit: Option<usize>,
	pub flail_step_limit: Option<usize>,
}

pub const DEFAULT_PARAMS: Params = Params {
	inertia_delay: 5,
	timestep_dec: 0.5,
	timestep_inc: 1.1,
	alpha_dec: 0.99,
	alpha_max: 0.1,
	timestep_start: ::std::f64::NAN,
	timestep_max: ::std::f64::NAN,
	force_tolerance: None,
	step_limit: None,
	flail_step_limit: None,
};
impl Default for Params { fn default() -> Params { DEFAULT_PARAMS } }

// a type that violates every good practice I can possibly think of.
// let's open the doors to an era of exposed and mutable state!
// mayhaps, for once, I'll get something done.
pub struct Relax {
	pub params: Params,
	pub position: Vec<f64>,
	pub velocity: Vec<f64>,
	pub force:    Vec<f64>,
	pub timestep: f64,
	pub alpha:    f64,
	pub cooldown: u32,
	pub nstep:    usize,
	pub potential: f64, // mostly an accumulator for debug, but also part of a stop condition
	pub min_potential: f64,
	pub time_since_min_potential: usize,
	pub debug_f_dot_v: f64, // this is also only saved for debug output
}

#[derive(Debug,Copy,Clone,Eq,PartialEq,PartialOrd,Ord,Hash)]
pub enum StopReason {
	Convergence,
	Timeout,
	Flailing,
}

impl Relax
{
	pub fn init(params: Params, position: Vec<f64>) -> Self {
		assert!(!params.timestep_start.is_nan());
		assert!(!params.timestep_max.is_nan());
		Relax {
			velocity: vec![0.; position.len()],
			force:    vec![0.; position.len()],
			position: position,
			timestep: params.timestep_start,
			alpha: params.alpha_max,
			cooldown: 0,
			nstep: 0,
			params: params,
			potential: 0.,
			min_potential: ::std::f64::INFINITY,
			time_since_min_potential: 0,
			debug_f_dot_v: 0.,
		}
	}


	pub fn relax<G,H>(mut self, mut force_writer: G, mut post_fire: H) -> (Vec<f64>, StopReason)
	where G: FnMut(Self) -> Self, H: FnMut(&Self)
	{
		self.nstep = 0;

		// Let this function compute forces, giving it full domain over this object.
		// It must assign stuff to self.force. If the "flailing" stop condition is used
		// it must also write to self.potential. Beyond that, it can do whatever it wants,
		// such as overwriting parameters and other terrible horrible things.
		// ...huh. This is kind of liberating.
		self = (&mut force_writer)(self);

		loop {
			self.nstep += 1;

			assert_eq!(self.position.len(), self.velocity.len());
			assert_eq!(self.velocity.len(), self.force.len());

			{ // verlet
				let dt = self.timestep;
				for (p, &v, &f) in izip!(&mut self.position, &self.velocity, &self.force) {
					*p += v * dt + 0.5 * f * dt * dt;
				}

				for (v, &f) in izip!(&mut self.velocity, &self.force) { *v += 0.5 * dt * f; }
				self = (&mut force_writer)(self);
				for (v, &f) in izip!(&mut self.velocity, &self.force) { *v += 0.5 * dt * f; }
			}

			// this data is used by one of the stop conditions
			if self.potential < self.min_potential {
				self.min_potential = self.potential;
				self.time_since_min_potential = 0;
			} else {
				self.time_since_min_potential += 1;
			}

			if let Some(reason) = self.stop_reason() {
				return (self.position, reason);
			}

			self.step_fire();

			(&mut post_fire)(&self);
		}
		unreachable!()
	}

	fn stop_reason(&self) -> Option<StopReason> {
		let mut has_stop_cond = false;

		if let Some(tol) = self.params.force_tolerance {
			has_stop_cond = true;
			let fsqnorm: f64 = self.force.iter().map(|&x| x*x).sum();
			assert!(fsqnorm == fsqnorm);
			if fsqnorm <= tol {
				return Some(StopReason::Convergence);
			}
		}

		if let Some(limit) = self.params.step_limit {
			has_stop_cond = true;
			if self.nstep >= limit {
				return Some(StopReason::Timeout);
			}
		}

		if let Some(limit) = self.params.flail_step_limit {
			has_stop_cond = true;
			if self.time_since_min_potential >= limit {
				return Some(StopReason::Flailing);
			}
		}

		assert!(has_stop_cond, "no stop condition");
		None
	}

	fn step_fire(&mut self) {
		let f_dot_v = izip!(&self.force, &self.velocity).map(|(&x,&y)| x*y).sum::<f64>();
		let f_norm = self.force.iter().map(|&x| x*x).sum::<f64>().sqrt();
		let v_norm = self.velocity.iter().map(|&x| x*x).sum::<f64>().sqrt();

		for (v,&f) in izip!(&mut self.velocity, &self.force) {
			*v = (1. - self.alpha) * *v + self.alpha * f * v_norm/f_norm;
		}

		self.debug_f_dot_v = f_dot_v;

		// don't go uphill
		if f_dot_v < 0. {
			self.timestep = self.timestep * self.params.timestep_dec;
			self.alpha = self.params.alpha_max;
			self.cooldown = self.params.inertia_delay;
			self.velocity.resize(0,                0.);
			self.velocity.resize(self.force.len(), 0.);

		// start gaining inertia after a while downhill
		} else {
			if self.cooldown > 0 { self.cooldown -= 1; }
			else {
				self.timestep = (self.timestep * self.params.timestep_inc).min(self.params.timestep_max);
				self.alpha    = self.alpha * self.params.alpha_dec;
			}
		}
	}
}

