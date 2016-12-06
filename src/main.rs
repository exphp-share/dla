
const GRID_DIM: Pos3 = (120, 120, 40);
const NPARTICLE: usize = 1000;

const LATTICE_A: f64 = 1f64;
const LATTICE_C: f64 = 1f64;
const CORE_RADIUS: f64 = 5f64;

extern crate rand;
#[macro_use(zip_with)]
extern crate homogenous;
use rand::Rng;
use rand::distributions::{IndependentSample,Range};
use homogenous::prelude::*;

use std::io::Write;
use std::f64::consts::PI;

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


type Tile = bool; // true when occupied
type Pos = i32;
type Pos2 = (i32, i32);
type Pos3 = (i32, i32, i32);
struct Grid {
	dim: Pos3,
	grid: Vec<Tile>,
}

impl Grid {
	fn new(dim: Pos3) -> Grid {
		Grid {
			dim: dim,
			grid: vec![false; dim.product() as usize],
		}
	}

	fn strides(&self) -> Pos3 { (self.dim.1 * self.dim.2, self.dim.2, 1) }

	fn index(&self, pos: Pos3) -> usize { self.strides().dot(pos) as usize }

	fn is_occupied(&self, pos: Pos3) -> Tile { self.grid[self.index(pos)] }

	fn set_occupied(&mut self, pos: Pos3) {
		let index = self.index(pos);
		self.grid[index] = true;
	}
}

fn cartesian((i,j,k): Pos3) -> (f64,f64,f64) {
	// 120 degrees; although the actual angle between the i and j vectors
	//  is 60 degrees, the components below are written in terms of
	//  vectors related by R3.
	// The 60 degrees then is actually the angles of those vectors
	//  to the x-axis.
	const CELL_ANGLE: f64 = 2.*PI/6.;
	let (i,j,k) = (i as f64, j as f64, k as f64);
	let (x,y,z) = ((CELL_ANGLE.cos() + 1.)*(i+j), CELL_ANGLE.sin()*(j-i), k);
	(LATTICE_A * x, LATTICE_A * y, LATTICE_C * z)
}

fn mod_floor(a: i32, m: i32) -> i32 { ((a % m) + m) % m }

fn output_sparse<W: Write>(grid: &Grid, file: &mut W) {
	writeln!(file, "[");
	let mut first = true;
	for i in 0..grid.dim.0 {
		for j in 0..grid.dim.1 {
			for k in 0..grid.dim.2 {
				if grid.is_occupied((i,j,k)) {
					if !first { write!(file, ",\n "); }
					write!(file, "[{},{},{}]", i, j, k);
					first = false;
				}
			}
		}
	}
	writeln!(file, "]");
}

//---------- DLA

// NOTES:
// * 8 directions; 2 axial, 6 planar
// * All 8 have equal weight; this is pretty unisotropic
fn neighbor_displacements() -> Vec<Pos3> {
	let mut out = vec![];

	// axial movement along Z
	for i in vec![-1, 1] { out.push((i, 0, 0)) }

	// hexagonal movement in-plane
	// These three tuples are the triangular lattice vectors in axial coords.
	for (j,k) in vec![(0,1), (1,0), (-1,-1)] {
		out.push((0, j, k));
		out.push((0, -j, -k));
	}

	out
}

fn random_border_position(grid: &Grid) -> Pos3 {
	// this makes no attempt to be isotropic;
	// the angle distribution is "not quite right",
	// and edges/vertices of the cube are slightly favored
	let mut rng = rand::thread_rng();

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
	let (ci,cj,_) = grid.dim.map(|d| d/2);

	for i in ri {
	for j in rj.clone() {
		let (x,y,_) = cartesian((i-ci, j-cj, 0));
		if (x*x + y*y) <= CORE_RADIUS*CORE_RADIUS + 1e-10 {
			for k in rk.clone() {
				grid.set_occupied((i,j,k));
			}
		}
	}}
	grid
}

fn dla_run() -> Grid {
	let grid = Grid::new(GRID_DIM);
	let grid = add_nucleation_site(grid);
	let mut grid = grid;

	let displacements = neighbor_displacements();
	let dim = grid.dim; // to avoid capturing grid inside closure
	let collect_neighbors = |pos:Pos3|
		displacements.iter()
			.map(|&disp| zip_with!((pos, disp, dim), |x,dx,m| mod_floor(x+dx, m)))
			.collect::<Vec<_>>();

	let mut rng = rand::weak_rng();
	let disp_distribution = Range::new(0, displacements.len());

	for n in 0..NPARTICLE {
		writeln!(std::io::stderr(), "Particle {} of {}", n, NPARTICLE);
		let mut pos = random_border_position(&grid);

		// is a nearby tile occupied?
		for step in 0.. {
			let neighbors = collect_neighbors(pos);
			if neighbors.iter().any(|&p| grid.is_occupied(p)) { break }

			// no need to worry about moving onto an occupied tile since we attach before
			// we have the chance.
			let disp_index = disp_distribution.ind_sample(&mut rng);
			pos = neighbors[disp_index];
		}

		// don't want the structure to cross the border
		// (heck, even just touching the border means it is already way too large and
		//  has likely ruined the probability distribution)
		if pos.update_nth(2, |_| 1) // avoid triggering on the z axis
				.any(|x| x == 0) {
			writeln!(std::io::stderr(), "Warning: Touched border!");
			break;
		}

		// attach immediately
		grid.set_occupied(pos);
	}
	grid
}

fn main() {
	let mut grid = dla_run();
	output_sparse(&grid, &mut std::io::stdout());
}
