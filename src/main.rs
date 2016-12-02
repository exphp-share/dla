
const GRID_DIM: Pos3 = (120, 120, 120);
const NPARTICLE: usize = 10000;

extern crate rand;
use rand::Rng;
use rand::distributions::{IndependentSample,Range};

use std::io::Write;

// NOTES:
// * Coordinates are axial; (i,j) are coeffs of two of the three equivalent lattice vectors.
//   The coefficient of the third is (-i-j).
// * In practice this means the vectors corresponding to i and j are effectively
//   (a-b) and (a-c), where (a,b,c) are equivalent under 3fold rotation.
//   Thus there is a 60 degree angle between them
// * Unit cell is obtuse:
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
			grid: vec![false; (dim.0 * dim.1 * dim.2) as usize],
		}
	}

	fn strides(&self) -> Pos3 { (self.dim.1 * self.dim.2, self.dim.2, 1) }

	fn index(&self, (p1,p2,p3): Pos3) -> usize {
		let (s1,s2,s3) = self.strides();
		let out = p1*s1 + p2*s2 + p3*s3;
		out as usize
	}

	fn is_occupied(&self, pos: Pos3) -> Tile { self.grid[self.index(pos)] }

	// slice of contiguous elements (i,j,*)
	fn line(&self, (p1,p2): Pos2) -> &[Tile] {
		&self.grid[self.index((p1,p2,0))..self.index((p1,p2+1,0))]
	}

	fn set_occupied(&mut self, pos: Pos3) {
		let index = self.index(pos);
		self.grid[index] = true;
	}

	fn is_border(&self, pos: Pos3) -> bool {
		let too_low = tuple_map(pos, |x| x == 0);
		let too_high = tuple_zip_with(pos, self.dim, |p,d| p == d-1);
		tuple_reduce(
			tuple_zip_with(too_low, too_high, |a,b| a || b),
			|a,b| a || b)
	}
}

fn mod_floor(a: i32, m: i32) -> i32 { ((a % m) + m) % m }

fn output_dense<W: Write>(grid: &Grid, file: &mut W) {
	for i in 0..grid.dim.0 {
		for j in 0..grid.dim.1 {
			let buf = grid.line((i as Pos, j as Pos)).iter()
				.map(|&c| match c {
					true  => b'#',
					false => b'_',
				}).collect::<Vec<_>>();
			file.write_all(&buf);
			writeln!(file, "");
		}
		writeln!(file, "");
	}
}

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

//------------
// HACK: using tuple for Pos was a dumb idea; we can't index it.
// type Pos3 = [i32; 3] would be better
#[inline]
fn tuple_nth(pos: Pos3, i: usize) -> Pos {
	match i {
		0 => pos.0,
		1 => pos.1,
		2 => pos.2,
		_ => panic!("bad tuple index"),
	}
}
#[inline]
fn tuple_replace((a,b,c): Pos3, i: usize, x: Pos) -> Pos3 {
	match i {
		0 => (x,b,c),
		1 => (a,x,c),
		2 => (a,b,x),
		_ => panic!("bad tuple index"),
	}
}

fn tuple_zip_with<A,B,C,F:FnMut(A,B) -> C>((a1,a2,a3): (A,A,A), (b1,b2,b3): (B,B,B), mut f: F) -> (C,C,C) {
	(f(a1,b1), f(a2,b2), f(a3,b3))
}
fn tuple_add(p: Pos3, q: Pos3) -> Pos3 { tuple_zip_with(p, q, |a,b| a+b) }
fn tuple_map<T, B, F:FnMut(T) -> B>((a,b,c): (T,T,T), mut f: F) -> (B,B,B) { (f(a), f(b), f(c)) }
fn tuple_reduce<T, F:FnMut(T,T) -> T>((a,b,c): (T,T,T), mut f: F) -> T { let x = f(a,b); f(x,c) }

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

	// randomly pick one of three cube faces
	// (notice the other three faces are translationally equivalent)
	let fixed_axis = rng.gen_range(0, 3);

	// randomly pick a position on said face
	let mut pos = (0,0,0);
	for axis in 0..3 {
		if axis == fixed_axis { continue }
		let r = rng.gen_range(0, tuple_nth(grid.dim, axis));
		pos = tuple_replace(pos, axis, r);
	}

	pos
}

fn add_nucleation_site(mut grid: Grid) -> Grid {
	// a "cube" of fixed size;
	// not actually a cube though since this is axial coordinates
	let radius = 5;
	let center = tuple_map(grid.dim, |x| x/2);
	let (ri,rj,rk) = tuple_map(center, |x0| (x0-radius)..(x0+radius+1));

	for i in ri {
	for j in rj.clone() {
	for k in rk.clone() {
		grid.set_occupied((i,j,k));
	}}}
	grid
}

fn dla_run() -> Grid {
	let grid = Grid::new(GRID_DIM);
	let grid = add_nucleation_site(grid);
	let mut grid = grid;

	let displacements = neighbor_displacements();
	let dim = grid.dim; // to avoid capturing grid inside closure
	let collect_neighbors = |pos|
		displacements.iter()
			.map(|&disp| tuple_add(disp, pos))
			.map(|pos| tuple_zip_with(pos, dim, mod_floor)) // wrap for PBCs
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

		// attach immediately
		grid.set_occupied(pos);
		// don't want the structure to cross the border
		// (heck, even just touching the border means it is already way too large and
		//  has likely ruined the probability distribution)
		if grid.is_border(pos) {
			writeln!(std::io::stderr(), "Warning: Touched border!");
			break;
		}
	}
	grid
}

fn main() {
	let mut grid = dla_run();
	output_sparse(&grid, &mut std::io::stdout());
}
