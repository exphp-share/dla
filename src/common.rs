
use nalgebra as na;

use ::homogenous::prelude::*;
use ::homogenous::numeric::prelude::*;

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

#[derive(Debug,PartialEq,PartialOrd,Copy,Clone)]
pub struct Frac(pub Float);
pub type Cart = Float;
pub trait CartExt { fn frac(self, dimension: Float) -> Frac; }
impl CartExt for Cart { fn frac(self, dimension: Float) -> Frac { Frac(self/dimension) } }

impl Frac { pub fn cart(self, dimension: Float) -> Cart { self.0*dimension } }

// add common binops to eliminate the majority of reasons I might need to
// convert back into floats (which would render the type system useless)
newtype_ops!{ [Frac] arithmetic {:=} {^&}Self {^&}{Self} }

// cart() and frac() methods for triples
pub trait ToCart { fn cart(self, dimension: Trip<Float>) -> Trip<Cart>; }
pub trait ToFrac { fn frac(self, dimension: Trip<Float>) -> Trip<Frac>; }
impl ToCart for Trip<Frac> { fn cart(self, dimension: Trip<Float>) -> Trip<Cart> { zip_with!((self,dimension) |x,d| x.cart(d)) } }
impl ToFrac for Trip<Cart> { fn frac(self, dimension: Trip<Float>) -> Trip<Frac> { zip_with!((self,dimension) |x,d| x.frac(d)) } }
impl ToCart for Trip<Cart> { fn cart(self, _dimension: Trip<Float>) -> Trip<Cart> { self } }
impl ToFrac for Trip<Frac> { fn frac(self, _dimension: Trip<Float>) -> Trip<Frac> { self } }

// nalgebra interop, but strictly for cartesian
pub fn to_na_vector((x,y,z): Trip<Cart>) -> NaVector3{ NaVector3 { x: x, y: y, z: z } }
pub fn to_na_point((x,y,z): Trip<Cart>)  -> NaPoint3{ NaPoint3  { x: x, y: y, z: z } }

pub fn from_na_vector(NaVector3 {x,y,z}: NaVector3) -> Trip<Cart> { (x,y,z) }
pub fn from_na_point(NaPoint3 {x,y,z}: NaPoint3) -> Trip<Cart> { (x,y,z) }

pub fn as_na_vector<F:FnMut(NaVector3) -> NaVector3>(p: Trip<Cart>, mut f: F) -> Trip<Cart> { from_na_vector(f(to_na_vector(p))) }
pub fn as_na_point<F:FnMut(NaPoint3) -> NaPoint3>(p: Trip<Cart>, mut f: F) -> Trip<Cart> { from_na_point(f(to_na_point(p))) }

pub fn reduce_pbc(this: Trip<Frac>) -> Trip<Frac> { this.map(|Frac(x)| Frac((x.fract() + 1.0).fract())) }
pub fn nearest_image_sub<P:ToFrac,Q:ToFrac>(this: P, that: Q, dimension: Trip<Float>) -> Trip<Cart> {
	// assumes a diagonal cell
	let this = this.frac(dimension);
	let that = that.frac(dimension);
	let diff = this.sub_v(that)
		.map(|Frac(x)| Frac(x - x.round())); // range [0.5, -0.5]

	diff.map(|Frac(x)| assert!(-0.5-1e-5 <= x && x <= 0.5+1e-5));
	diff.cart(dimension)
}

pub fn nearest_image_dist_sq<P:ToFrac,Q:ToFrac>(this: P, that: Q, dimension: Trip<Float>) -> Cart {
	nearest_image_sub(this, that, dimension).sqnorm()
}


// Get a "look at" isometry; it maps the origin to the eye, and +z towards the target.
// (The up direction is arbitrarily chosen, without risk of it being invalid)
pub fn look_at_pbc(eye: Trip<Cart>, target: Trip<Cart>, dimension: Trip<Float>) -> NaIso {
	let (_,θ,φ) = spherical_from_cart(nearest_image_sub(target, eye, dimension));
	translate(eye) * rotate_z(φ) * rotate_y(θ)
}

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

