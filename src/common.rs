
use nalgebra as na;

use ::std::f64::INFINITY;

use ::homogenous::prelude::*;
use ::homogenous::numeric::prelude::*;
use ::serde::{Serialize,Deserialize};
use ::rand::Rng;

macro_rules! err {
	($($args:tt)*) => {{
		use ::std::io::Write;
		write!(::std::io::stderr(), $($args)*).unwrap();
	}};
}

macro_rules! errln {
	($fmt:expr) => (err!(concat!($fmt, "\n")));
	($fmt:expr, $($arg:tt)*) => (err!(concat!($fmt, "\n"), $($arg)*));
}

pub const CART_ORIGIN: Trip<Cart> = (0., 0., 0.);

pub type Float = f64;
pub type Pair<T> = (T,T);
pub type Trip<T> = (T,T,T);

pub type NaIso = na::Isometry3<Float>;
pub type NaVector3 = na::Vector3<Float>;
pub type NaPoint3 = na::Point3<Float>;

pub fn identity() -> NaIso { NaIso::new(na::zero(), na::zero()) }
pub fn translate((x,y,z): Trip<Cart>) -> NaIso { NaIso::new(NaVector3{x:x,y:y,z:z}, na::zero()) }
pub fn rotate((x,y,z): Trip<Float>) -> NaIso { NaIso::new(na::zero(), NaVector3{x:x,y:y,z:z}) }
pub fn rotate_x(x: Float) -> NaIso { NaIso::new(na::zero(), na_x(x)) }
pub fn rotate_y(x: Float) -> NaIso { NaIso::new(na::zero(), na_y(x)) }
pub fn rotate_z(x: Float) -> NaIso { NaIso::new(na::zero(), na_z(x)) }
pub fn na_x(r: Float) -> NaVector3 { NaVector3{x:r, ..na::zero()} }
pub fn na_y(r: Float) -> NaVector3 { NaVector3{y:r, ..na::zero()} }
pub fn na_z(r: Float) -> NaVector3 { NaVector3{z:r, ..na::zero()} }

//--------------------
// Make frac coords a newtype which is incompatible with other floats, to help prove that
// fractional/cartesian conversions are handled properly.

pub type Cart = Float;

pub fn to_na_vector((x,y,z): Trip<Cart>) -> NaVector3{ NaVector3 { x: x, y: y, z: z } }
pub fn to_na_point((x,y,z): Trip<Cart>)  -> NaPoint3{ NaPoint3  { x: x, y: y, z: z } }

pub fn from_na_vector(NaVector3 {x,y,z}: NaVector3) -> Trip<Cart> { (x,y,z) }
pub fn from_na_point(NaPoint3 {x,y,z}: NaPoint3) -> Trip<Cart> { (x,y,z) }

pub fn as_na_vector<F:FnMut(NaVector3) -> NaVector3>(p: Trip<Cart>, mut f: F) -> Trip<Cart> { from_na_vector(f(to_na_vector(p))) }
pub fn as_na_point<F:FnMut(NaPoint3) -> NaPoint3>(p: Trip<Cart>, mut f: F) -> Trip<Cart> { from_na_point(f(to_na_point(p))) }

pub fn spherical_from_cart((x,y,z): Trip<Cart>) -> Trip<Float> {
	let ρ = (x*x + y*y).sqrt();
	let r = (ρ*ρ + z*z).sqrt();
	(r, ρ.atan2(z), y.atan2(x))
}

pub fn normalize(p: Trip<Cart>) -> Trip<Cart> {
	let r = p.sqnorm().sqrt();
	p.map(|x| x/r)
}

pub fn unit_θ_from_cart((x,y,z): Trip<Cart>) -> Trip<Float> {
	// rats, would be safer to compute these from spherical
	if x == 0. && y == 0. { (z.signum(),0.,0.) }
	else {
		let ρ = (x*x + y*y).sqrt();
		let r = (ρ*ρ + z*z).sqrt();
		(x*z/ρ/r, y*z/ρ/r, -ρ/r)
	}
}

pub fn cart_from_spherical((r,θ,φ): Trip<Float>) -> Trip<Cart> {
	let (sinθ,cosθ) = (θ.sin(), θ.cos());
	let (sinφ,cosφ) = (φ.sin(), φ.cos());
	(r*sinθ*cosφ, r*sinθ*sinφ, r*cosθ)
}

//-------------------------------------

// output range of [0, b]
fn mod_floor(x: f64, b: f64) -> f64 { x - b * (x/b).floor() }
// output range of [-b/2, b/2]
fn mod_round(x: f64, b: f64) -> f64 { x - b * (x/b).round() }

// a diagonal unit cell
#[derive(Copy,Clone,Debug,PartialEq,Serialize,Deserialize)]
pub struct Pbc {
	pub dim: Trip<f64>,
	pub vacuum: Trip<bool>,
}

impl Pbc {
	// image of point in unit cell
	pub fn wrap(&self, point: Trip<Cart>) -> Trip<Cart> {
		zip_with!((self.dim, point) |dim,x| mod_floor(x, dim))
	}

	// fractional coordinates of point
	pub fn frac(&self, point: Trip<Cart>) -> Trip<Float> {
		zip_with!((self.dim, point) |dim,x| x / dim)
	}

	// nearest image displacement (p - q)
	pub fn nearest_image_sub(&self, p: Trip<Cart>, q: Trip<Cart>) -> Trip<Cart> {
		zip_with!((self.dim, p, q) |dim,p,q| mod_round(p - q, dim))
	}

	// image of "of" nearest "by"
	pub fn nearest_image_of(&self, of: Trip<Cart>, by: Trip<Cart>) -> Trip<Cart> {
		by.add_v(self.nearest_image_sub(of, by))
	}

	// distance to nearest vacuum border
	pub fn vacuum_border_distance(&self, point: Trip<Cart>) -> Cart {
		zip_with!((self.dim, self.vacuum, point) |dim,vacuum,x|
			if vacuum { mod_round(x, dim).abs() }
			else { INFINITY }
		).iter().fold(INFINITY, |a,&b| a.min(b))
	}

	// point-to-point distance (nearest image)
	pub fn distance(&self, p: Trip<Cart>, q: Trip<Cart>) -> f64 { self.nearest_image_sub(p,q).sqnorm().sqrt() }

	// unit cell center position
	pub fn center(&self) -> Trip<Cart> { self.dim.mul_s(0.5) }

	// Get a "look at" isometry; it maps the origin to the eye, and +z towards the target.
	// (The up direction is arbitrarily chosen, without risk of it being invalid)
	pub fn look_at(&self, eye: Trip<Cart>, target: Trip<Cart>) -> NaIso {
		let (_,θ,φ) = spherical_from_cart(self.nearest_image_sub(target, eye));
		translate(eye) * rotate_z(φ) * rotate_y(θ)
	}

	pub fn random_point_in_volume<R:Rng>(&self, mut rng: R) -> Trip<Cart> {
		self.dim.map(|dim| dim * rng.next_f64())
	}
}

//-------------------------------------

pub fn unflatten<T:Copy>(slice: &[T]) -> Vec<Trip<T>> {
	let mut iter = slice.iter().cloned();
	let mut out = vec![];
	loop {
		if let Some(x) = iter.next() {
			let y = iter.next().unwrap();
			let z = iter.next().unwrap();
			out.push((x, y, z));
		} else { break }
	}
	out
}

pub fn flatten<T:Copy>(slice: &[Trip<T>]) -> Vec<T> {
	slice.iter().cloned().flat_map(|x| x.into_iter()).collect()
}

