
use common::*;

use fire::Fire;

use ::std::io::Write;
use ::std::collections::HashSet as Set;

use ::homogenous::prelude::*;
use ::homogenous::numeric::prelude::*;

use ::nalgebra::Transformation;

//------------------------------

#[derive(PartialEq,Copy,Clone,Debug)]
pub enum Model {
	Morse { center: f64, D: f64, k: f64, },
	Quadratic { center: f64, k: f64, },
	Zero,
}

#[derive(Copy,Clone,Debug,PartialEq)]
pub struct ForceOut { potential: f64, force: f64 }

impl Model {
	pub fn data(self, x: f64) -> ForceOut {
		let square = |x| x*x;
		match self {
			Model::Quadratic { center, k, } => {
				ForceOut {
					potential: k * square(center - x),
					force:     2. * k * (center - x),
				}
			},
			Model::Morse { center, k, D } => {
				let a = (k/(2. * D)).sqrt();
				let f = (a * (center - x)).exp();
				ForceOut {
					potential: a * D * square(f - 1.),
					force:     2. * a * D * f * (f - 1.),
				}
			},
			Model::Zero => { ForceOut { potential: 0., force: 0. } },
		}
	}

	// signed value of force along +x
	pub fn force(self, x: f64) -> f64 { self.data(x).force }
}

//------------------------------

#[derive(PartialEq,Clone,Debug)]
pub struct Radial {
	model: Model,
	terms: Vec<Pair<usize>>,
}

#[derive(PartialEq,Clone,Debug)]
pub struct Angular {
	model: Model,
	terms: Vec<Trip<usize>>,
}

#[derive(PartialEq,Copy,Clone,Debug)]
pub struct Rebo(pub bool);

impl Rebo {
	pub fn tally<W:Write>(&self, md: &mut Fire, mut ffile: Option<W>, free_indices: &Set<usize>, dim: Trip<Float>) {
		if !self.0 { return; }

		let (potential,force) = ::ffi::calc_rebo_flat(md.position.clone(), dim);

		md.potential += potential;

		for i in free_indices {
			for file in &mut ffile {
				writeln!(file, "REB:{} F= {} {} {}", i, force[3*i], force[3*i+1], force[3*i+2]).unwrap();
			}

			for k in 0..3 {
				md.force[3*i + k] += force[3*i + k];
			}
		}
	}
}

impl Radial {
	pub fn prepare(model: Model, free_indices: &Set<usize>, parents: &[usize]) -> Self {
		let terms = {
			// unique bonds (up to ordering)
			let mut set = Set::new();
			for mut i in 0..parents.len() {
				let mut j = parents[i];
				if j < i {
					::std::mem::swap(&mut i, &mut j);
				}
				set.insert((i,j));
			}

			// eliminate useless ones
			set.into_iter()
				.filter(|&(i,j)| free_indices.contains(&i) || free_indices.contains(&j))
				.collect()
		};

		Radial { model: model, terms: terms, }
	}

	pub fn tally<W:Write>(&self, md: &mut Fire, mut ffile: Option<W>, free_indices: &Set<usize>, dim: Trip<Float>) {
		for &(i, j) in &self.terms {
			assert!(i < j, "uniqueness condition");

			let pi = tup3(&md.position, i);
			let pj = tup3(&md.position, j);
			let dvec = nearest_image_sub(pi, pj, dim);

			let ForceOut { force: signed_force, potential } = self.model.data(dvec.sqnorm().sqrt());
			let f = normalize(dvec).mul_s(signed_force);

			for file in &mut ffile {
				writeln!(file, "RAD:{}:{} V= {} F= {} {} {}", i, j, potential, f.0, f.1, f.2).unwrap();
			}

			// Note to self:
			// Yes, it is correct for the potential to always be added once,
			// regardless of how many of the atoms are fixed.
			md.potential += potential;
			// NOTE: this simulates the behavior of the code prior to the recent refactor
			// FIXME: remove once no longer of interest
			if ::DOUBLE_COUNTED_RADIAL_POTENTIAL && free_indices.contains(&i) && free_indices.contains(&j) {
				md.potential += potential;
			}

			if free_indices.contains(&i) { tup3add(&mut md.force, i, f) }
			if free_indices.contains(&j) { tup3add(&mut md.force, j, f.mul_s(-1.)) }
		}
	}
}

impl Angular {
	pub fn prepare(model: Model, free_indices: &Set<usize>, parents: &[usize]) -> Self {
		let terms = {
			let mut set = Set::new();
			for i in 0..parents.len() {
				let j = parents[i];
				let k = parents[j];
				if i != k {
					set.insert((i,j,k));
					set.insert((k,j,i));
				}
			}

			// Each term potentially affects any of the first two indices (but not the third).
			// Drop those that can't do anything.
			set.into_iter()
				.filter(|&(i,j,_)| free_indices.contains(&i) || free_indices.contains(&j))
				.collect()
		};

		Angular { model: model, terms: terms, }
	}

	pub fn tally<W:Write>(&self, md: &mut Fire, mut ffile: Option<W>, free_indices: &Set<usize>, dim: Trip<Float>) {
		for &(i,j,k) in &self.terms {
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
			let (pi,pj,pk) = (pi,pj,pk).map(|x| applyP(iso, x));

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
			let ForceOut { force: signed_force, potential } = self.model.data(θi);
			let f = θ_hat.mul_s(signed_force);

			// bring force back into cartesian.
			// (note: transformation from  grad' V  to  grad V  is more generally the
			//   transpose matrix of the one that maps x to x'. But for a rotation,
			//   this is also the inverse.)
			let f = applyV(inv, f);

			for file in &mut ffile {
				writeln!(file, "ANG:{}:{}:{} V= {} F= {} {} {}", i, j, k, potential, f.0, f.1, f.2).unwrap();
			}

			// Note to self:
			// Yes, it is correct for the potential to always be added once,
			// regardless of how many of the atoms are fixed.
			md.potential += potential;
			if free_indices.contains(&i) && free_indices.contains(&j) { md.potential += potential }

			// ultimately, the two outer atoms (i, k) get pulled in similar directions,
			// and the middle one (j) receives the opposing forces
			if free_indices.contains(&i) { tup3add(&mut md.force, i, f); }
			if free_indices.contains(&j) { tup3add(&mut md.force, j, f.mul_s(-1.)) };
		}
	}
}

//------------------------------

fn tup3(xs: &[f64], i:usize) -> Trip<f64> { (0,1,2).map(|k| xs[3*i+k]) }
fn tup3set(xs: &mut [f64], i:usize, v:(f64,f64,f64)) { v.enumerate().map(|(k,x)| xs[3*i+k] = x); }
fn tup3add(xs: &mut [f64], i:usize, v:(f64,f64,f64)) {
	let tmp = tup3(&xs, i);
	tup3set(xs, i, tmp.add_v(v));
}

fn applyP(iso: NaIso, v: Trip<f64>) -> Trip<f64> { as_na_point(v, |v| iso * v) }
fn applyV(iso: NaIso, v: Trip<f64>) -> Trip<f64> { as_na_vector(v, |v| iso * v) }
