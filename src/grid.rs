
//const GRID_DIM: Pos3 = (100, 100, 80);
const GRID_DIM: Dim3 = (Vacuum(500), Vacuum(500), Periodic(80));
const NPARTICLE: usize = 2*75_000;
const CB_FREQUENCY: usize = 5_000;

const CORE_RADIUS: f64 = 150f64;

const INTRODUCTION_RADIUS: f64 = 1000f64;
const ELIMINATION_RADIUS:  f64 = 3000f64;

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
type Pos2 = (Pos,Pos);
type Pos3 = (Pos,Pos,Pos);
type Dim2 = (Dim,Dim);
type Dim3 = (Dim,Dim,Dim);
#[derive(Clone,PartialEq,Debug,Serialize,Deserialize)]
pub struct Grid {
	dim: Dim3,
	grid: Vec<Tile>,
	strides: Pos3,
	displacements: Vec<Pos3>,
}

#[derive(Copy,Clone,PartialEq,Eq,Debug,Serialize,Deserialize)]
pub enum Dim {
	Vacuum(Pos),   // a dimension spanning -d...d and beyond
	Periodic(Pos), // a dimension spanning 0..d periodically
}
use self::Dim::*;


impl Dim {
	fn periodic(self) -> Option<Pos> {
		match self {
			Vacuum(_)   => None,
			Periodic(d) => Some(d),
		}
	}

	fn vacuum(self) -> Option<Pos> {
		match self {
			Vacuum(d)   => Some(d),
			Periodic(_) => None,
		}
	}

	fn range(self) -> ::std::ops::Range<Pos> {
		match self {
			Vacuum(d)   => -d..d+1,
			Periodic(d) => 0..d,
		}
	}

	fn offset(self) -> Pos { -self.range().start }

	fn center(self) -> Option<Pos> { self.vacuum().map(|_| 0) }

	fn len(self) -> Pos { self.range().len() as Pos }

	fn contains(self, x: Pos) -> bool {
		if let Periodic(d) = self { debug_assert!(0 <= x && x < d); }
		self.range().contains(x)
	}
}

impl Grid {
	fn new(dim: Dim3) -> Grid {
		Grid {
			dim: dim,
			grid: vec![None; dim.map(|d| d.len()).product() as usize],
			displacements: neighbor_displacements(),
			strides: (dim.1.len() * dim.2.len(), dim.2.len(), 1),
		}
	}

	fn valid_positions(&self) -> Box<Iterator<Item=Pos3>> {
		let (rx,ry,rz) = self.dim.map(|d| d.range());
		Box::new(iproduct!(rx, ry, rz))
	}

	fn disp_from_center(&self, pos: Pos3) -> Pos3 {
		zip_with!((pos, self.dim) |x,d| d.center().map(|c| x - c).unwrap_or(0))
	}

	fn wrap(&self, pos: Pos3, disp: Pos3) -> Pos3 {
		zip_with!((pos, disp, self.dim) |x, dx, dim|
			match dim {
				Vacuum(_) => x + dx,
				Periodic(dim) => fast_mod_floor(x + dx, dim),
			}
		)
	}

	fn in_bounds(&self, pos: Pos3) -> bool { pos.zip(self.dim).all(|(x,dim)| dim.contains(x)) }

	fn index(&self, pos: Pos3) -> Option<usize> {
		if self.in_bounds(pos) {
			Some(zip_with!((pos, self.dim, self.strides) |x,dim,stride| (x + dim.offset()) * stride).sum() as usize)
		} else { None }
	}

	fn is_occupied(&self, pos: Pos3) -> bool { self.occupant(pos).is_some() }
	fn occupant(&self, pos: Pos3) -> Option<Label> { self.index(pos).and_then(|i| self.grid[i]) }
	fn set_occupant(&mut self, pos: Pos3, val: Tile) {
		let i = self.index(pos).unwrap();
		self.grid[i] = val;
	}

	fn occupy(&mut self, pos: Pos3, label: Label) {
		assert_eq!(self.occupant(pos), None);
		self.set_occupant(pos, Some(label));
	}

	fn accomodate(mut self, pos: Pos3) -> Grid {
		while !self.in_bounds(pos) {
			self = expand(self, EXPANSION_FACTOR);
		}
		self
	}

	pub fn to_structure(&self, lattice: LatticeParams) -> ::Structure {
		let mut pos = vec![];
		let mut labels = vec![];
		for (i,j,k) in self.valid_positions() {
			if let Some(label) = self.occupant((i,j,k)) {
				let center = cartesian(lattice, (i,j,k));
				for disp in dimer_disps(lattice) {
					labels.push(label);
					pos.push(center.add_v(disp));
				}
			}
		}

		::Structure { pos, labels, dim: (0., 0., lattice.c) }
	}

	pub fn to_sparse(self: &Grid) -> SparseGrid {
		let dim = self.dim;
		let (pos,label): (Vec<_>,Vec<_>) = self.valid_positions()
				.filter(|&p| self.is_occupied(p))
				.map(|p| (p, self.occupant(p).unwrap()))
				.unzip();
		SparseGrid { pos, label, dim }
	}

	pub fn center_distance_sq(&self, pos: Pos3, lattice: LatticeParams) -> f64 {
		cartesian(lattice, self.disp_from_center(pos)).sqnorm()
	}
}

#[derive(Copy,Clone,Debug,PartialEq,Serialize,Deserialize)]
pub struct LatticeParams { pub a: f64, pub c: f64 }
fn cartesian(lattice: LatticeParams, pos: Pos3) -> (f64,f64,f64) {
	cartesian_f64(lattice, pos.map(|i| i as f64))
}

fn cartesian_f64(lattice: LatticeParams, (i,j,k): (f64,f64,f64)) -> (f64,f64,f64) {
	// 120 degrees; although the actual angle between the i and j vectors
	//  is 60 degrees, the components below are written in terms of
	//  vectors related by R3.
	// The 60 degrees then is actually the angles of those vectors
	//  to the x-axis.
	const CELL_ANGLE: f64 = 2.*PI/6.;
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
	debug_assert!(-1 <= a && a <= m, "failed precondition ({} mod {})", a, m);
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
fn random_border_position<R: Rng>(lattice: LatticeParams, mut rng: R, grid: &Grid, radius: f64) -> Pos3 {
	// too lazy to generalize this
	debug_assert_eq!(grid.dim.map(|d| d.periodic().is_some()), (false,false,true));

	let φ = rng.next_f64() * 2. * PI;
	let (i,j) = nearest_hex_lattice_point(lattice, (φ.sin()*radius, φ.cos()*radius));

	let n = grid.dim.2.periodic().unwrap();
	(i, j, rng.gen_range(0, n))
}

fn nearest_hex_lattice_point(lattice: LatticeParams, pos: (f64,f64)) -> (Pos,Pos) {
	// multiply by the inverse matrix.
	let (x,y) = pos.map(|x| x/lattice.a);
	let i = (1./3.)*x - (1./3f64).sqrt()*y;
	let j = (1./3.)*x + (1./3f64).sqrt()*y;

	// Assume that the nearest lattice point is a vertex of this cell.
	// NOTE: the correctness of this assumption depends on the choice of unit cell,
	//       though I'm not sure what the precise condition is.
	//       Even if our cell fails to meet this condition though, the error should be negligible.
	iproduct!(vec![i.floor() as Pos, i.ceil() as Pos], vec![j.floor() as Pos, j.ceil() as Pos])
		.min_by_key(|&(i0,j0)| OrdOrDie(cartesian_f64(lattice, (i - i0 as f64, j - j0 as f64, 0.)).sqnorm()))
		.unwrap()
}


fn add_nucleation_site(lattice: LatticeParams, mut grid: Grid) -> Grid {
	// a cylinder
	for pos in grid.valid_positions() {
		let (x,y,_) = cartesian(lattice, grid.disp_from_center(pos));
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

pub fn dla_run<F:FnMut(&Grid)>(lattice: LatticeParams, mut cb: F) -> Grid {
	let grid = Grid::new(GRID_DIM);
	let grid = add_nucleation_site(lattice, grid);
	dla_run_(lattice, grid, cb)
}

pub fn dla_run_<F:FnMut(&Grid)>(lattice: LatticeParams, mut grid: Grid, mut cb: F) -> Grid {
	let mut rng = ::rand::weak_rng();
	let mut timer = ::timer::Timer::new(20);

	'event: for n in 0..NPARTICLE {
		if n % CB_FREQUENCY == 0 {
			cb(&grid);
		}

		err!("Particle {:8} of {:8}: ", n, NPARTICLE);
		let mut pos = random_border_position(lattice, &mut rng, &grid, INTRODUCTION_RADIUS);

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
//				veto.veto(&mut rng, &grid, lattice, pos, disp)
				false
			} { }

			pos = grid.wrap(pos, disp);

			if grid.center_distance_sq(pos, lattice) > ELIMINATION_RADIUS * ELIMINATION_RADIUS {
				continue 'event;
			}
		}

		errln!("({:3},{:3},{:3})   ({:6} ms, {:6} avg) ({:7} steps, {:7} vetos)",
			pos.0, pos.1, pos.2, timer.last_ms(), timer.average_ms(),
			roll_count, veto.count);

		// place the particle
		grid.occupy(pos, Label::C);
		timer.push();
	}
	cb(&grid);
	grid
}

pub fn expand(grid: Grid, factor: f64) -> Grid {
//	let mut new_dim = grid.dim.map(|d| (d as f64 * factor).round() as i32);
//	new_dim.2 = grid.dim.2;
	let new_dim = grid.dim.map(|d| match d {
		Periodic(d) => Periodic(d),
		Vacuum(d) => Vacuum((d as f64 * factor).round() as Pos),
	});

//	let sites = grid.valid_positions().filter(|&p| grid.is_occupied(p)).collect_vec();
//	let mins = sites.iter().cloned().fold(grid.dim, |a,b| zip_with!((a,b) ::std::cmp::min));
//	let maxs = sites.iter().cloned().fold((0,0,0),  |a,b| zip_with!((a,b) ::std::cmp::max));

//	let shift = new_dim.sub_v(mins).sub_v(maxs).div_s(2);

	errln!("new size!: {:?}", new_dim.map(|d| d.len()));
	let mut out = Grid::new(new_dim);
	for pos in grid.valid_positions() {
		if let Some(lbl) = grid.occupant(pos) {
//			out.occupy(pos.add_v(shift), lbl);
			out.occupy(pos, lbl);
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
		let get_rsq = |pos: Pos3| grid.center_distance_sq(pos, lattice);
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
	zip_with!((self::lattice(lattice), grid.dim) |lattice_vec, dim| {
		let length = match dim {
			Vacuum(n) => (n * 10) as f64,
			Periodic(n) => n as f64,
		};

		let (x,y,z) = lattice_vec.mul_s(length);
		writeln!(file, "  {} {} {}", x, y, z).unwrap();
	});

	let indices = grid.valid_positions().filter(|&p| grid.is_occupied(p));
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
	dim: Dim3,
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





#[derive(Hash, PartialOrd)]
#[derive(Debug, Clone, Copy)]
pub struct OrdOrDie<T>(pub T);

impl<T:PartialOrd> Ord for OrdOrDie<T> {
	fn cmp(&self, other: &Self) -> ::std::cmp::Ordering {
		self.partial_cmp(other).expect("OrdOrDie: unorderable values!")
	}
}

impl<T:PartialOrd> PartialEq for OrdOrDie<T> {
	fn eq(&self, other: &Self) -> bool { self.cmp(other) == ::std::cmp::Ordering::Equal }
}
impl<T:PartialOrd> Eq for OrdOrDie<T> { }
