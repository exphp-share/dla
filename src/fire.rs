
use super::{Trip,Cart,flatten,unflatten};
use homogenous::prelude::*;
use homogenous::numeric::prelude::*;

#[derive(Debug,Clone)]
pub struct Params {
	// alpha is a sort of "steering" coefficient; as it increases,
	// we steer more strongly towards the force
	pub alpha_max: f64,
	pub alpha_dec: f64,
	pub inertia_delay: u32,
	pub timestep_start: f64,
	pub timestep_max: f64,
	pub timestep_inc: f64,
	pub timestep_dec: f64,
	pub turn_condition: TurnCondition,
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
	turn_condition: TurnCondition::FDotV,
	force_tolerance: None,
	step_limit: None,
	flail_step_limit: None,
};
impl Default for Params { fn default() -> Params { DEFAULT_PARAMS } }

// a type that violates every good practice I can possibly think of.
// let's open the doors to an era of exposed and mutable state!
// mayhaps, for once, I'll get something done.
pub struct Fire {
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
	pub f_dot_v: f64, // this is also only saved for debug output

	// used for TurnCondition::Potential
	pub prev_position:  Vec<f64>,
	pub prev_potential: f64,
}

#[derive(Debug,Copy,Clone,Eq,PartialEq,PartialOrd,Ord,Hash)]
pub enum StopReason { Convergence, Timeout, Flailing }

// Standard FIRE has a "turning" condition defined in terms of F dot v,
// which seems to be at best an approximation of how potential will change.
// The paper says it is resistant to random errors in potential, though I
//  somehow rather feel that is in turn susceptible to random errors in _force._
#[derive(Debug,Copy,Clone,Eq,PartialEq,PartialOrd,Ord,Hash)]
pub enum TurnCondition { FDotV, Potential }

impl Fire
{
	pub fn init(params: Params, position: Vec<f64>) -> Self {
		assert!(!params.timestep_start.is_nan());
		assert!(!params.timestep_max.is_nan());
		Fire {
			velocity: vec![0.; position.len()],
			force:    vec![0.; position.len()],
			position: position.clone(),
			timestep: params.timestep_start,
			alpha: params.alpha_max,
			cooldown: 0,
			nstep: 0,
			params: params,
			potential: ::std::f64::INFINITY,

			// used by turning conditions and debug
			min_potential: ::std::f64::INFINITY,
			time_since_min_potential: 0,
			f_dot_v: 0.,

			// used for TurnCondition::Potential
			prev_position:  position,
			prev_potential: ::std::f64::INFINITY,
		}
	}

	pub fn relax<G,H>(mut self, mut force_writer: G, mut post_fire: H) -> (Vec<f64>, (usize, StopReason))
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

			// one of the turning conditions uses this data
			self.prev_position = self.position.clone();
			self.prev_potential = self.potential;

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
				return (self.position, (self.nstep, reason));
			}

			self.step_fire();

			(&mut post_fire)(&self);
		}
	}

	fn stop_reason(&self) -> Option<StopReason> {
		let mut has_stop_cond = false;

		if let Some(tol) = self.params.force_tolerance {
			has_stop_cond = true;

			let fsqnorm: f64 = dot(&self.force, &self.force);
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
		let f_dot_v = dot(&self.force, &self.velocity);
		let f_norm = norm(&self.force);
		let v_norm = norm(&self.velocity);

		// steer towards the force
		for (v,&f) in izip!(&mut self.velocity, &self.force) {
			*v = (1. - self.alpha) * *v + self.alpha * f * v_norm/f_norm;
		}

		self.f_dot_v = f_dot_v;

		// don't go uphill
		let should_turn = match self.params.turn_condition {
			TurnCondition::FDotV => f_dot_v < 0.,
			TurnCondition::Potential => self.prev_potential < self.potential,
		};

		// stop moving immediately and set up heavy steering towards the force
		if should_turn {
			self.timestep = self.timestep * self.params.timestep_dec;
			self.alpha = self.params.alpha_max;
			self.cooldown = self.params.inertia_delay;
			self.velocity.resize(0,                0.);
			self.velocity.resize(self.force.len(), 0.);

			if self.params.turn_condition == TurnCondition::Potential {
				// ohgodwhatahack
				// (FIRE does not typically even USE position)
				self.position.copy_from_slice(&self.prev_position);
			}

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

fn dot(a: &[f64], b: &[f64]) -> f64 {
	assert_eq!(a.len(), b.len());
	izip!(a,b).map(|(&a,&b)| a*b).sum()
}

fn norm(a: &[f64]) -> f64 { dot(a,a).sqrt() }
