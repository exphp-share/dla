
const GRID_DIM: Pos3 = (100, 100, 160);
const NPARTICLE: usize = 7500;

const CORE_RADIUS: f64 = 5f64;

/// Higher value <=> less frequent expansion of grid size
/// (note: max value is not 1.0, but rather root(3)/2 (I think))
const EXPANSION_THRESHOLD: f64 = 0.8;
const EXPANSION_FACTOR: f64 = 1.05;

const VETO_THRESHOLD: f64 = 0.20;
const VETO_DELAY: u32 = 100;
const VETO_MAX_CHANCE: f64 = 0.02;

use ::Label;
use ::common::*;

use ::std::f64::INFINITY;
use ::std::io::prelude::*;
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
#[derive(Clone,PartialEq,Debug,Serialize,Deserialize)]
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

	fn center(&self) -> Pos3 { self.dim.div_s(2) }

	fn wrap(&self, pos: Pos3, disp: Pos3) -> Pos3 {
		zip_with!((pos, disp, self.dim) |x,dx,m| fast_mod_floor(x+dx, m))
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

		::Structure { pos, labels, dim }
	}

	pub fn to_sparse(self: &Grid) -> SparseGrid {
		let dim = self.dim;
		let (pos,label): (Vec<_>,Vec<_>) = self.indices()
				.filter(|&p| self.is_occupied(p))
				.map(|p| (p, self.occupant(p).unwrap()))
				.unzip();
		SparseGrid { pos, label, dim }
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

// NOTE: this was indeed tested to be about 10x faster than mod_floor
//       for moduli not known at compile time
fn fast_mod_floor(a: i32, m: i32) -> i32 {
	debug_assert!(0 <= a && a < m, "failed precondition ({} mod {})", a, m);
	match a {
		-1 => m-1,
		x if x == m => 0,
		x => x,
	}
}

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

fn add_nucleation_site(lattice: LatticeParams, mut grid: Grid) -> Grid {
	// a cylinder
	for pos in grid.indices() {
		let (x,y,_) = cartesian(lattice, pos.sub_v(grid.center()));
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

pub fn dla_run(lattice: LatticeParams) -> Grid {
	let grid = Grid::new(GRID_DIM);
	let grid = add_nucleation_site(lattice, grid);
	dla_run_(lattice, grid)
}

pub fn dla_run_(lattice: LatticeParams, mut grid: Grid) -> Grid {
	let mut rng = ::rand::weak_rng();
	let mut timer = ::timer::Timer::new(20);

	for n in 0..NPARTICLE {
		err!("Particle {:8} of {:8}: ", n, NPARTICLE);
		let mut pos = random_border_position(&mut rng, &grid);

		// move until ready to place
		let mut veto = VetoState::new();
		let mut roll_count = 0;
		while !is_fillable(&grid, pos) {
			roll_count += 1;

			let valid_disps =
				grid.displacements.iter().cloned()
					.filter(|&disp| !grid.is_occupied(grid.wrap(pos, disp)))
					.collect_vec();

			let mut disp;
			while {
				disp = *rng.choose(&valid_disps).expect("no possible moves! (this is unexpected!)");
				veto.veto(&mut rng, &grid, lattice, pos, disp)
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
		errln!("({:3},{:3},{:3})   ({:6} ms, {:6} avg) ({:7} steps, {:7} vetos)",
			pos.0, pos.1, pos.2, timer.last_ms(), timer.average_ms(),
			roll_count, veto.count);

		let dist_to_center = cartesian(lattice, pos.sub_v(grid.center())).sqnorm().sqrt();
		if dist_to_center / grid.dim.0 as f64 > EXPANSION_THRESHOLD {
			grid = expand(grid, EXPANSION_FACTOR)
		}
	}
	grid
}

pub fn expand(grid: Grid, factor: f64) -> Grid {
	let mut new_dim = grid.dim.map(|d| (d as f64 * factor).round() as i32);
	new_dim.2 = grid.dim.2;

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
	count: u32,
}

impl VetoState {
	pub fn new() -> Self { VetoState { delay: 0, count: 0 } }
	pub fn veto<R: Rng>(&mut self, rng: &mut R, grid: &Grid, lattice: LatticeParams, pos: Pos3, disp: Pos3) -> bool {
		let get_rsq = |pos: Pos3| cartesian(lattice, pos.sub_v(grid.center())).sqnorm();
		match self.delay {
			0 => {
				// compare distances before and after.
				// don't worry about wrapping; we're really testing the direction, not the destination
				if get_rsq(pos.add_v(disp)) >= get_rsq(pos) {
					if rng.next_f64() < VETO_MAX_CHANCE {
						self.count += 1;
						return true;
					}
				}
			},
			_ => { self.delay -= 1; },
		}
		false
	}
}

/// Like Itertools::group_by, but the groups are not limited to being contiguous.
/// It is strict, as evidenced by the return type.
use ::std::collections::btree_map::BTreeMap as Map;
fn nonlocal_groups_by<V, I:IntoIterator<Item=V>, K: Ord, F: FnMut(&V) -> K>(iter: I, mut key: F) -> Map<K,Vec<V>> {
	iter.into_iter()
		.sorted_by(|a,b| key(a).cmp(&key(b))).into_iter()
		.group_by(key).into_iter()
		.map(|(k,g)| (k, g.collect()))
		.collect()
}

pub fn write_poscar<W:Write>(mut file: W, lattice: LatticeParams, grid: &Grid) -> ::std::io::Result<()> {
	writeln!(file, "blah blah blah")?;
	writeln!(file, "1.0")?;
	zip_with!((self::lattice(lattice), grid.dim) |unit_vec, n| {
		let (x,y,z) = unit_vec.mul_s(n as f64);
		writeln!(file, "  {} {} {}", x, y, z).unwrap();
	});

	let indices = grid.indices().filter(|&p| grid.is_occupied(p));
	let groups = nonlocal_groups_by(indices, |&p| grid.occupant(p).unwrap());

	for (label,_) in &groups { write!(file, " {}", label.as_str())?; }
	writeln!(file)?;

	for (_, idx) in &groups { write!(file, " {}", idx.len())?; }
	writeln!(file)?;

	writeln!(file, "Cartesian")?;

	for (label, idxs) in groups {
		for idx in idxs {
			let (x,y,z) = cartesian(lattice, idx);
			writeln!(file, "{} {} {} {}", x, y, z, label.as_str())?;
		}
	}
	Ok(())
}

// more useful to serialize...
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct SparseGrid {
	pos: Vec<Pos3>,
	label: Vec<Label>,
	dim: Pos3,
}

impl SparseGrid {
	pub fn to_grid(&self) -> Grid {
		let mut grid = Grid::new(self.dim);
		for (&p, &lbl) in izip!(&self.pos, &self.label) {
			grid.occupy(p, lbl);
		}
		grid
	}
}
