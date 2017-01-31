
const GRID_DIM: Pos3 = (100, 100, 160);
const NPARTICLE: usize = 50000;

const CORE_RADIUS: f64 = 5f64;

const EXPANSION_THRESHOLD: f64 = 0.30;
const EXPANSION_FACTOR: f64 = 1.05;

const VETO_THRESHOLD: f64 = 0.20;
const VETO_DELAY: u32 = 100;
const VETO_MAX_CHANCE: f64 = 0.02;

use ::Label;
use ::common::*;

use ::std::f64::INFINITY;
use ::rand::Rng;
use ::homogenous::prelude::*;
use ::homogenous::numeric::prelude::*;
use ::itertools::Itertools;

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
		self.strides().mul_v(pos).sum() as usize
	}

	fn indices(&self) -> Box<Iterator<Item=Pos3>> {
		let (rx,ry,rz) = self.dim.map(|d| 0..d);
		Box::new(iproduct!(rx, ry, rz))
	}

	fn corners(&self) -> Box<Iterator<Item=Pos3>> {
		let (rx,ry,rz) = self.dim.map(|d| vec![0,d]);
		Box::new(iproduct!(rx, ry, rz))
	}

	// NOTE: range is 0.0 (on top of an edge) to 0.5 (at center), and corresponds to the
	// fractional distance to the nearest edge
	fn edge_distance(&self, pos: Pos3) -> f64 {
		let mut edge_dist = zip_with!((pos, self.dim)
			|p,d| (p as f64 + 0.5).min((d-p) as f64 - 0.5) / d as f64);
		edge_dist.2 = ::std::f64::INFINITY; // z doesn't count
		let min = edge_dist.fold1(|a,b| if a < b { a } else { b });
		assert!(0.0 <= min && min < 1.0);
		min
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

	pub fn to_structure(&self, lattice: LatticeParams) -> ::Structure {
		let mut pos = vec![];
		let mut labels = vec![];
		for (i,j,k) in self.indices() {
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
			let corners = self.corners().map(|ipos| cartesian(lattice, ipos));
			orthogonal_data(corners)
		};

		for p in &mut pos {
			*p = p.add_v(shift);
			// NOTE: I think the OrthoData parameters are still not quite sufficient if
			//       we have particles right at the vacuum border; in this case, though,
			//       an error is well-deserved.
			assert!(p.zip(dim).all(|(x,d)| 0. <= x && x <= d), "{:?} {:?}", p, dim);
		}

		::Structure {
			pos: pos,
			labels: labels,
			dim: dim,
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

fn dimer_disps(LatticeParams { a, .. }: LatticeParams) -> Vec<Trip<f64>> {
	vec![(0.5*a, 0.0, 0.0), (-0.5*a, 0.0, 0.0)]
}

fn mod_floor(a: i32, m: i32) -> i32 { ((a % m) + m) % m }

pub fn lattice(lattice: LatticeParams) -> Trip<Trip<f64>> {
	(0,1,2).map(|n| cartesian(lattice, (0i32,0i32,0i32).update_nth(n, |_| 1)))
}

//---------- DLA

// NOTES:
// * 8 directions; 2 axial, 6 planar
// * All 8 have equal weight; this is pretty unisotropic
fn neighbor_displacements() -> Vec<Pos3> {
	// axial movement along Z
	let mut out = vec![(0,0,1), (0,0,-1)];

	// hexagonal movement in-plane
	// These three tuples are the triangular lattice vectors in axial coords.
	for (j,k) in vec![(0,1), (1,0), (1,-1)] {
		out.push((j, k, 0));
		out.push((-j, -k, 0));
	}

	out
}

// this makes no attempt to be isotropic;
// the angle distribution is "not quite right",
// and edges/vertices of the cube are slightly favored
fn random_border_position<R: Rng>(mut rng: R, grid: &Grid) -> Pos3 {
	// randomly pick a position in 3d space
	let pos = grid.dim.map(|d| rng.gen_range(0,d));

	// project onto either the i=0 or j=0 face of the parallelepiped
	let fixed_axis = rng.gen_range(0, 2);
	let pos = pos.update_nth(fixed_axis, |_| 0);
	pos
}

fn add_nucleation_site(mut grid: Grid) -> Grid {
	// a cylinder
	let center = grid.dim.map(|d| d/2);

	for pos in grid.indices() {
		let (x,y,_) = cartesian(::GRID_LATTICE_PARAMS, pos.sub_v(center));
		if (x*x + y*y) <= CORE_RADIUS*CORE_RADIUS + 1e-10 {
			grid.occupy(pos, Label::Si);
		}
	}
	grid
}

fn is_fillable(grid: &Grid, pos: Pos3) -> bool {
	let neighbors_ccw = vec![
		( 1, 0, 0), ( 0, 1, 0), (-1, 1, 0),
		(-1, 0, 0), ( 0,-1, 0), ( 1,-1, 0),
	].into_iter().map(|d| grid.wrap(pos, d)).collect_vec();

	// is any set of 2 contiguous neighbors all filled?
	neighbors_ccw.iter()
		.cycle().tuple_windows::<(_,_)>().take(neighbors_ccw.len())
		.any(|pq| pq.all(|&p| grid.is_occupied(p)))
}

pub fn dla_run() -> Grid {
	let grid = Grid::new(GRID_DIM);
	let grid = add_nucleation_site(grid);
	dla_run_(grid)
}

pub fn dla_run_(mut grid: Grid) -> Grid {
	let mut rng = ::rand::weak_rng();
	let mut timer = ::timer::Timer::new(20);

	for n in 0..NPARTICLE {
		err!("Particle {:8} of {:8}: ", n, NPARTICLE);
		let mut pos = random_border_position(&mut rng, &grid);

		// move until ready to place
		let mut veto = VetoState::new();
		while !is_fillable(&grid, pos) {

			let valid_disps =
				grid.displacements.iter().cloned()
					.filter(|&disp| !grid.is_occupied(grid.wrap(pos, disp)))
					.collect_vec();

			let mut disp = (0,0,0);
			while {
				disp = *rng.choose(&valid_disps).expect("no possible moves! (this is unexpected!)");
				veto.veto(&mut rng, &grid, pos, disp)
			} { }

			pos = grid.wrap(pos, disp);
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
		timer.push();
		errln!("({:3},{:3},{:3})   ({:8} ms, {:8} avg)", pos.0, pos.1, pos.2, timer.last_ms(), timer.average_ms());

		if grid.edge_distance(pos) < EXPANSION_THRESHOLD {
			grid = expand(grid, EXPANSION_FACTOR)
		}
	}
	grid
}

pub fn expand(grid: Grid, factor: f64) -> Grid {
	let mut new_dim = grid.dim.map(|d| (d as f64 * factor).round() as i32);
	//let mut shift = new_dim.sub_v(grid.dim).div_s(2);
	new_dim.2 = grid.dim.2;
	//shift.2 = 0;

	let sites = grid.indices().filter(|&p| grid.is_occupied(p)).collect_vec();
	let mins = sites.iter().cloned().fold(grid.dim, |a,b| zip_with!((a,b) ::std::cmp::min));
	let maxs = sites.iter().cloned().fold((0,0,0),  |a,b| zip_with!((a,b) ::std::cmp::max));

	let shift = new_dim.sub_v(mins).sub_v(maxs).div_s(2);

	errln!("new size!: {:?}", new_dim);
	let mut out = Grid::new(new_dim);
	for pos in grid.indices() {
		if let Some(lbl) = grid.occupant(pos) {
			out.occupy(pos.add_v(shift), lbl);
		}
	}
	out
}

struct VetoState {
	delay: u32,
}

impl VetoState {
	pub fn new() -> Self { VetoState { delay: 0 } }
	pub fn veto<R: Rng>(&mut self, rng: &mut R, grid: &Grid, pos: Pos3, disp: Pos3) -> bool {
		match self.delay {
			0 => {
				if pos.sub_v(grid.dim.div_s(2)).mul_v(disp).sum() < 0 { return false; }
				let prob = (1.0 - grid.edge_distance(pos)) * VETO_MAX_CHANCE;
				rng.next_f64() < prob
			},
			_ => {
				self.delay -= 1;
				false
			},
		}
	}
}
