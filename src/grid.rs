
const GRID_DIM: Pos3 = (120, 120, 20);
const NPARTICLE: usize = 1000;

const LATTICE_A: f64 = 1f64;
const LATTICE_C: f64 = 1f64;
const CORE_RADIUS: f64 = 5f64;

use ::Label;
use ::common::*;

use ::std::f64::INFINITY;
use ::rand::Rng;
use ::rand::distributions::{IndependentSample,Range};
use ::homogenous::prelude::*;
use ::homogenous::numeric::prelude::*;
use ::itertools::Itertools;

use ::std::io::prelude::*;
use ::std::f64::consts::PI;

// NOTES:
// * Coordinates are axial; (i,j) are coeffs of two of the three equivalent lattice vectors.
//   The coefficient of the third is (-i-j).
// * In practice this means the vectors corresponding to i and j are effectively
//   (a-b) and (a-c), where (a,b,c) are equivalent under 3fold rotation.
//   Thus there is a 60 degree angle between them, even though the unit cell is obtuse:
//
//    b________
//     ^       \
//      \       \
//       \______>\a
//       0


type Tile = Option<Label>;
type Pos = i32;
type Pos2 = (i32, i32);
type Pos3 = (i32, i32, i32);
#[derive(Clone,PartialEq,Debug)]
pub struct Grid {
	dim: Pos3,
	grid: Vec<Tile>,
	displacements: Vec<Pos3>,
}

impl Grid {
	fn new(dim: Pos3) -> Grid {
		Grid {
			dim: dim,
			grid: vec![None; dim.product() as usize],
			displacements: neighbor_displacements(),
		}
	}

	fn strides(&self) -> Pos3 { (self.dim.1 * self.dim.2, self.dim.2, 1) }

	fn index(&self, pos: Pos3) -> usize {
		assert!(pos.zip(self.dim).all(|(x,d)| 0 <= x && x < d), "{:?} {:?}", pos, self.dim);
		zip_with!((self.strides(), pos) |s,p| s*p).sum() as usize
	}

	fn neighbors(&self, pos: Pos3) -> Vec<Pos3> {
		self.displacements.iter().map(|&disp| self.wrap(pos, disp)).collect()
	}

	fn wrap(&self, pos: Pos3, disp: Pos3) -> Pos3 {
		zip_with!((pos, disp, self.dim) |x,dx,m| mod_floor(x+dx, m))
	}
	fn is_occupied(&self, pos: Pos3) -> bool { self.occupant(pos).is_some() }
	fn occupant(&self, pos: Pos3) -> Option<Label> { self.grid[self.index(pos)] }
	fn set_occupant(&mut self, pos: Pos3, val: Tile) {
		let index = self.index(pos);
		self.grid[index] = val;
	}

	fn occupy(&mut self, pos: Pos3, label: Label) {
		assert_eq!(self.occupant(pos), None);
		self.set_occupant(pos, Some(label));
	}

	// HACK
	// Convert into the continuous format, because some other functions were written to
	//   take this structure.
	// Not all contents will make sense; in particular, all atoms will link to number 0;
	// do NOT relax with radial/angular terms.
	pub fn to_tree(&self, lattice: LatticeParams) -> ::Tree {
		let mut pos = vec![];
		let mut labels = vec![];
		for (i,j,k) in nditer(self.dim) {
			if let Some(label) = self.occupant((i,j,k)) {
				let center = cartesian(lattice, (i,j,k));
				for disp in dimer_disps(lattice) {
					labels.push(label);
					pos.push(center.add_v(disp));
				}
			}
		}

		// NOTE: the hexagonal cell gets reinterpreted here as a (larger) orthorhombic cell;
		// this should not be an issue because those dimensions are vacuum-separated.
		let OrthoData { shift, dim } = {
			let corners =
				iproduct!(vec![0, self.dim.0], vec![0, self.dim.1], vec![0, self.dim.2])
				.map(|ipos| cartesian(lattice, ipos));
			orthogonal_data(corners)
		};

		for p in &mut pos {
			*p = p.add_v(shift);
			// NOTE: I think the OrthoData parameters are still not quite sufficient if
			//       we have particles right at the vacuum border; in this case, though,
			//       an error is well-deserved.
			assert!(p.zip(dim).all(|(x,d)| 0. <= x && x <= d), "{:?} {:?}", p, dim);
		}

		::Tree {
			pos: pos,
			parents: vec![0; labels.len()],
			labels: labels,
			pbc: ::Pbc {
				dim: dim,
				vacuum: (true, true, false),
			},
		}
	}
}

struct OrthoData {
	shift: Trip<f64>,
	dim:   Trip<f64>,
}
// produces the smallest orthogonal cell which contains all given points within
//  its boundaries (inclusive)
fn orthogonal_data<I:IntoIterator<Item=Trip<f64>>>(points: I) -> OrthoData {
	let points = points.into_iter().collect_vec();
	let by_axis = (0,1,2).map(|i| points.iter().map(|&tup| tup.into_nth(i)).collect_vec());
	let mins = by_axis.as_ref().map(|xs| xs.iter().clone().fold( INFINITY, |a,b| a.min(*b)));
	let maxs = by_axis.as_ref().map(|xs| xs.iter().clone().fold(-INFINITY, |a,b| a.max(*b)));
	OrthoData {
		shift: mins.map(|x| -x),
		dim: maxs.sub_v(mins),
	}
}

#[derive(Copy,Clone,Debug,PartialEq,Serialize,Deserialize)]
pub struct LatticeParams { pub a: f64, pub c: f64 }
fn cartesian(lattice: LatticeParams, (i,j,k): Pos3) -> (f64,f64,f64) {
	// 120 degrees; although the actual angle between the i and j vectors
	//  is 60 degrees, the components below are written in terms of
	//  vectors related by R3.
	// The 60 degrees then is actually the angles of those vectors
	//  to the x-axis.
	const CELL_ANGLE: f64 = 2.*PI/6.;
	let (i,j,k) = (i as f64, j as f64, k as f64);
	let (x,y,z) = ((CELL_ANGLE.cos() + 1.)*(i+j), CELL_ANGLE.sin()*(j-i), k);
	(lattice.a * x, lattice.a * y, lattice.c * z)
}

fn dimer_disps(lattice: LatticeParams) -> Vec<Trip<f64>> {
	vec![
		(lattice.a * 0.5, 0., 0.),
		(-lattice.a * 0.5, 0., 0.),
	]
}

fn mod_floor(a: i32, m: i32) -> i32 { ((a % m) + m) % m }

fn output_sparse<W: Write>(grid: &Grid, file: &mut W) -> Result<(),::std::io::Error> {
	writeln!(file, "[")?;
	let mut first = true;

	for (i,j,k) in nditer(grid.dim) {
		if let Some(label) = grid.occupant((i,j,k)) {
			if !first { write!(file, ",\n ")?; }
			write!(file, "[{},{},{},{}]", label.as_str(), i, j, k)?;
			first = false;
		}
	}
	writeln!(file, "]")?;
	Ok(())
}

fn nditer(shape: Trip<i32>) -> ::std::vec::IntoIter<Trip<i32>> {
	let (ri,rj,rk) = shape.map(|d| 0..d);
	iproduct!(ri, rj, rk).collect_vec().into_iter()
}

//---------- DLA

// NOTES:
// * 8 directions; 2 axial, 6 planar
// * All 8 have equal weight; this is pretty unisotropic
fn neighbor_displacements() -> Vec<Pos3> {
	let mut out = vec![];

	// axial movement along Z
	for i in vec![-1, 1] { out.push((0, 0, i)) }

	// hexagonal movement in-plane
	// These three tuples are the triangular lattice vectors in axial coords.
	for (j,k) in vec![(0,1), (1,0), (1,-1)] {
		out.push((j, k, 0));
		out.push((-j, -k, 0));
	}

	out
}

fn random_border_position(grid: &Grid) -> Pos3 {
	// this makes no attempt to be isotropic;
	// the angle distribution is "not quite right",
	// and edges/vertices of the cube are slightly favored
	let mut rng = ::rand::thread_rng();

	// randomly pick a position in 3d space
	let pos = grid.dim.map(|d| rng.gen_range(0,d));

	// project onto either the i=0 or j=0 face of the parallelepiped
	let fixed_axis = rng.gen_range(0, 2);
	let pos = pos.update_nth(fixed_axis, |_| 0);
	pos
}

fn add_nucleation_site(mut grid: Grid) -> Grid {
	// a cylinder
	let (ri,rj,rk) = grid.dim.map(|d| 0..d);
	let center = grid.dim.map(|d| d/2);

	for pos in iproduct!(ri, rj, rk) {
		let (x,y,_) = cartesian(::GRID_LATTICE_PARAMS, zip_with!((pos, center) |a,b| a-b));
		if (x*x + y*y) <= CORE_RADIUS*CORE_RADIUS + 1e-10 {
			grid.occupy(pos, Label::Si);
		}
	}
	grid
}

fn is_fillable(grid: &Grid, pos: Pos3) -> bool {
	let disps_ccw_order = vec![
		( 1, 0, 0), ( 0, 1, 0), (-1, 1, 0),
		(-1, 0, 0), ( 0,-1, 0), ( 1,-1, 0),
	];

	// is any set of 2 contiguous neighbors all filled?
	let neighbors = disps_ccw_order.into_iter().map(|disp| grid.wrap(pos, disp)).collect_vec();
	neighbors.iter()
		.cycle().tuple_windows::<(_,_)>().take(neighbors.len())
		.any(|(&p,&q)| grid.is_occupied(p) && grid.is_occupied(q))
}

pub fn dla_run() -> Grid {
	let grid = Grid::new(GRID_DIM);
	let grid = add_nucleation_site(grid);
	dla_run_(grid)
}

pub fn dla_run_(mut grid: Grid) -> Grid {
	let mut rng = ::rand::weak_rng();

	for n in 0..NPARTICLE {
		err!("Particle {} of {}: ", n, NPARTICLE);
		let mut pos = random_border_position(&grid);

		// move until ready to place
		loop {
			if is_fillable(&grid, pos) { break }

			let valid_moves =
				grid.neighbors(pos).into_iter()
					.filter(|&x| !grid.is_occupied(x))
					.collect_vec();
			pos = *rng.choose(&valid_moves).expect("no possible moves! (this is unexpected!)");
		}

		// place the particle

		// don't want the structure to cross the border
		// (heck, even just touching the border means it is already way too large and
		//  has likely ruined the probability distribution)
		if pos.update_nth(2, |_| 1) // avoid triggering on the z axis
				.any(|x| x == 0) {
			errln!("Warning: Touched border!");
			break;
		}

		grid.occupy(pos, Label::C);
		errln!("{:?}", pos);
	}
	grid
}
