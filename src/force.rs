
use common::*;

use Tree;
use fire::Fire;

use ::std::io::Write;
use ::std::collections::HashSet as Set;
use ::std::cmp::Ordering;

use ::homogenous::prelude::*;
use ::homogenous::numeric::prelude::*;

use ::nalgebra::Transformation;
use ::nalgebra::ApproxEq;

//------------------------------

#[derive(Copy,Clone,Debug,PartialEq)]
pub struct Params {
	pub radial: Model,
	pub angular: Model,
	pub rebo: bool,
}

// NOTE: Hey you.
// Yes, you. The one who wrote this comment earlier. That's right; the one who's me.
//
// I know what you're thinking.
// Don't you dare do it.
//
// You do not need a generalized "Sum of forces" type.
// This is perfectly fine just the way it is.
#[derive(PartialEq,Debug)]
pub struct Composite {
	radial: Radial,
	angular: Angular,
	rebo: PersistentRebo,
}

impl Composite {
	/// Behavior is undefined if a Composite with an active Rebo is prepared when another
	/// already exists.
	pub unsafe fn prepare(params: Params, tree: &Tree, free_indices: &Set<usize>) -> Self {
		let parents = &tree.parents;
		Composite {
			radial: Radial::prepare(params.radial, free_indices, parents),
			angular: Angular::prepare(params.angular, free_indices, parents),
			rebo: PersistentRebo::prepare(params.rebo, &tree),
		}
	}

	// This horrific function (and its ilk) does all of the following:
	// * Mutate md to add potential and force.
	// * Return the total potential added. (usually for debug output one level up)
	// * Writes detailed stats to debug files.
	// It is _quite decidedly_ not pure.
	pub fn tally<W:Write>(&self, md: &mut Fire, mut ffile: Option<W>, free_indices: &Set<usize>, pbc: Pbc) -> f64 {
		let subtotal_rebo    = self.rebo.tally(md, ffile.as_mut(), free_indices, pbc);
		let subtotal_radial  = self.radial.tally(md, ffile.as_mut(), free_indices, pbc);
		let subtotal_angular = self.angular.tally(md, ffile.as_mut(), free_indices, pbc);

		for file in &mut ffile {
			writeln!(file, "SUBTOTAL REBO    V= {:22.18}", subtotal_rebo).unwrap();
			writeln!(file, "SUBTOTAL RADIAL  V= {:22.18}", subtotal_radial).unwrap();
			writeln!(file, "SUBTOTAL ANGULAR V= {:22.18}", subtotal_angular).unwrap();
		}
		subtotal_rebo + subtotal_radial + subtotal_angular
	}
}

//------------------------------

#[derive(Copy,Clone,Debug,PartialEq)]
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
					potential:
						if ::ERRONEOUS_MORSE_PREFACTOR { a * D * square(f - 1.) }
						else { D * square(f - 1.) }
					,
					force:     2. * a * D * f * (f - 1.),
				}
			},
			Model::Zero => { ForceOut { potential: 0., force: 0. } },
		}
	}

	pub fn set_spring_constant(&mut self, x: f64) -> Result<(),()> {
		match *self {
			Model::Quadratic { ref mut k, .. } => *k = x,
			Model::Morse     { ref mut k, .. } => *k = x,
			_ => return Err(()),
		}
		Ok(())
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
#[derive(PartialEq,Debug)]
pub struct PersistentRebo(bool);

impl Rebo {
	pub fn prepare(active: bool, _tree: &Tree) -> Self { Rebo(active) }

	pub fn tally<W:Write>(&self, md: &mut Fire, mut ffile: Option<W>, free_indices: &Set<usize>, pbc: Pbc) -> f64 {
		if !self.0 { return 0.; }

		let (potential,force) = ::ffi::calc_rebo_flat(md.position.clone(), pbc);

		md.potential += potential;

		for &i in free_indices {
			for file in &mut ffile {
				if ::FORCE_DEBUG == ::ForceDebug::Full {
					writeln!(file, "REB:{} F= {} {} {}", i, force[3*i], force[3*i+1], force[3*i+2]).unwrap();
				}
			}

			for k in 0..3 {
				md.force[3*i + k] += force[3*i + k];
			}
		}

		potential
	}
}

impl PersistentRebo {
	/// Behavior is undefined if this is used to create an active PersistentRebo
	/// when another one already exists.
	pub unsafe fn prepare(active: bool, tree: &Tree) -> Self {
		if active {
			::ffi::persistent_init(flatten(&tree.pos), tree.pbc);
		}
		PersistentRebo(active)
	}

	pub fn tally<W:Write>(&self, md: &mut Fire, mut ffile: Option<W>, free_indices: &Set<usize>, pbc: Pbc) -> f64 {
		if !self.0 { return 0.; }

		let (potential,force) = unsafe { ::ffi::persistent_calc(md.position.clone()) };
		if ::VALIDATE_REBO {
			let (epotential,eforce) = ::ffi::calc_rebo_flat(md.position.clone(), pbc);
			assert!((potential - epotential).abs() < 1e-7, "{} {}", potential, epotential);
			assert!(izip!(&force, &eforce).all(|(&f,&e)| (f-e).abs() < 1e-7), "{:?} {:?}", force, eforce);
		}

		md.potential += potential;

		for &i in free_indices {
			for file in &mut ffile {
				if ::FORCE_DEBUG == ::ForceDebug::Full {
					writeln!(file, "REB:{} F= {} {} {}", i, force[3*i], force[3*i+1], force[3*i+2]).unwrap();
				}
			}

			for k in 0..3 {
				md.force[3*i + k] += force[3*i + k];
			}
		}

		potential
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
				.filter(|&ij| ij.any(|i| free_indices.contains(&i)))
				.collect()
		};

		Radial { model: model, terms: terms, }
	}

	pub fn tally<W:Write>(&self, md: &mut Fire, mut ffile: Option<W>, free_indices: &Set<usize>, pbc: Pbc) -> f64 {
		let mut subtotal = 0.;
		for &(i, j) in &self.terms {
			assert!(i < j, "uniqueness condition");

			let pi = tup3(&md.position, i);
			let pj = tup3(&md.position, j);
			let dvec = pbc.nearest_image_sub(pi, pj);

			let ForceOut { force: signed_force, potential } = self.model.data(dvec.sqnorm().sqrt());
			let f = normalize(dvec).mul_s(signed_force);

			for file in &mut ffile {
				if ::FORCE_DEBUG == ::ForceDebug::Full {
					writeln!(file, "RAD:{}:{} V= {} F= {} {} {}", i, j, potential, f.0, f.1, f.2).unwrap();
				}
			}

			// Note to self:
			// Yes, it is correct for the potential to always be added once,
			// regardless of how many of the atoms are fixed.
			subtotal += potential;
			// NOTE: this simulates the behavior of the code prior to the recent refactor
			// FIXME: remove once no longer of interest
			if ::DOUBLE_COUNTED_RADIAL_POTENTIAL && free_indices.contains(&i) && free_indices.contains(&j) {
				subtotal += potential;
			}

			if free_indices.contains(&i) { tup3add(&mut md.force, i, f) }
			if free_indices.contains(&j) { tup3add(&mut md.force, j, f.mul_s(-1.)) }
		}
		md.potential += subtotal;
		subtotal
	}
}

impl Angular {
	pub fn prepare(model: Model, free_indices: &Set<usize>, parents: &[usize]) -> Self {
		let terms = {
			let mut set = Set::new();
			for i in 0..parents.len() {
				let j = parents[i];
				let k = parents[j];
				// A term produces forces for both (i,j,k) and (k,j,i), so canonicalize.
				match i.cmp(&k) {
					Ordering::Less    => { set.insert((i,j,k)); },
					Ordering::Greater => { set.insert((k,j,i)); },
					Ordering::Equal   => { }, // self-angle, probably at the root of the "tree"
				}
			}

			// Each term potentially affects any of the three indices.
			// Drop those that can't do anything.
			set.into_iter()
				.filter(|&ijk| ijk.any(|i| free_indices.contains(&i)))
				.collect()
		};

		Angular { model: model, terms: terms, }
	}

	pub fn tally<W:Write>(&self, md: &mut Fire, mut ffile: Option<W>, free_indices: &Set<usize>, pbc: Pbc) -> f64 {
		let mut subtotal = 0.;

		for &(i,j,k) in &self.terms {
			assert!(i < k, "uniqueness condition");

			// This is a bit painful;
			// We want to add the potential precisely once, but the forces are more easily
			//  computed by considering each of the endpoint atoms.
			//
			// This closure will:
			//   * write the forces
			//   * RETURN the potentials (for further inspection)
			let datas = ((i,j,k), (k,j,i)).map(|(i,j,k)| {

				let pi = tup3(&md.position, i);
				let pj = tup3(&md.position, j);
				let pk = tup3(&md.position, k);

				let pj = pbc.nearest_image_of(pj, pi);
				let pk = pbc.nearest_image_of(pk, pj);

				// move parent to origin
				// rotate parent's parent to +z
				let inv = pbc.look_at(pj, pk);
				let iso = inv.inverse_transformation();
				let (pi,pj,pk) = (pi,pj,pk).map(|x| applyP(iso, x));

				// are things placed about where we expect?
				assert!(pj.all(|x| x.abs() < 1e-5), "{:?} {:?} {:?}", pj, pi, pk);
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
					if ::FORCE_DEBUG == ::ForceDebug::Full {
						writeln!(file, "ANG:{}:{}:{} V= {} F= {} {} {}", i, j, k, potential, f.0, f.1, f.2).unwrap();
					}
				}

				// ultimately, the two outer atoms (i, k) get pulled in similar (but not parallel)
				// directions and the middle one (j) receives the opposing forces
				if free_indices.contains(&i) { tup3add(&mut md.force, i, f); }
				if free_indices.contains(&j) { tup3add(&mut md.force, j, f.mul_s(-1.)) };
				(potential, θi)
			});
			let potentials = datas.map(|x| x.0);
			let thetas = datas.map(|x| x.1);

			// Both directions should have witnessed the same potential...
			assert!(potentials.0.approx_eq(&potentials.1), "{:?} {:?}", potentials, thetas);
			// ...but we only want to add it once.
			subtotal += potentials.0;

			if ::DOUBLE_COUNTED_ANGULAR_POTENTIAL {
				if free_indices.contains(&j) || (free_indices.contains(&i) && free_indices.contains(&k)) {
					subtotal += potentials.0;
				}
			}
		}
		md.potential += subtotal;
		subtotal
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
