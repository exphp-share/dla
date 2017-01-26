use common::*;

use ::homogenous::prelude::*;
use ::homogenous::numeric::prelude::*;

use ::std::ops::Range;
use ::std::collections::HashSet as Set;
use ::std::iter::FromIterator;

// Incrementally tracks potential neighbors for a single atom traveling
// in the presence of many fixed atoms.
pub struct NeighborFinder {
	positions: Vec<Trip<Cart>>,
	pbc: Pbc,
	// tracks x +/- move_radius index on each axis
	hints: Trip<Range<usize>>,
	// Contains images in the fractional range [-1, 2] along each axis
	sorted: Trip<SortedIndices>,
}

impl NeighborFinder {
	pub fn new(pbc: Pbc) -> Self {
		NeighborFinder {
			positions: vec![],
			pbc: pbc,
			hints: ((),(),()).map(|_| 0..0),
			sorted: ((),(),()).map(|_| {
				// Keys at the far reaches of outer space help simplify edge cases.
				// The corresponding values should never be used.
				let mut set = SortedIndices::default();
				set.insert(::std::f64::NEG_INFINITY, ::std::usize::MAX);
				set.insert(::std::f64::INFINITY, ::std::usize::MAX);
				set
			}),
		}
	}

	pub fn from_positions<I:IntoIterator<Item=Trip<Cart>>>(pos: I, pbc: Pbc) -> Self {
		let mut this = NeighborFinder::new(pbc);
		for x in pos {
			this.insert(pbc.wrap(x));
		}
		this
	}

	fn update_cursors(&mut self, point: Trip<Cart>, radius: Cart) {
		let point = self.pbc.wrap(point);

		zip_with!((point, self.sorted.as_ref(), self.hints.as_mut())
		|x,set,hints| {
			hints.start = update_lower_hint(hints.start, &set.keys, x - radius);
			hints.end   = update_upper_hint(hints.end,   &set.keys, x + radius);
		});
	}

	pub fn insert(&mut self, point: Trip<Cart>) {
		let point = self.pbc.wrap(point);

		let i = self.positions.len();
		self.positions.push(point);

		zip_with!((point, self.sorted.as_mut(), self.pbc.dim) |x, set, dim| {
			// periodic images above and below simplify edge cases
			set.insert(x, i);
			set.insert(x - dim, i);
			set.insert(x + dim, i);
		});
	}

	pub fn cursor_neighborhood(&self) -> Set<usize> {
		// should we even really bother?
		if self.hints.clone().any(|x| x.len() == 0) { return Set::new(); }

		zip_with!((self.sorted.as_ref(), self.hints.clone())
			|set, range| { Set::from_iter(set.values[range].iter().cloned()) }
		).fold1(|a, b| (&a) & (&b))
	}

	pub fn closest_neighbor(&mut self, point: Trip<Cart>, radius: Cart) -> Option<usize> {
		let indices = self.neighborhood(point, radius);
		indices.into_iter()
			.map(|i| (Some(i), self.pbc.distance(point, self.positions[i])))
			// locate tuple with minimum distance; unfortunately, Iterator::min_by is unstable.
			.fold((None,::std::f64::MAX), |(i,p),(k,q)| if p < q { (i,p) } else { (k,q) })
			.0
	}

	fn neighborhood_from_candidates<I: IntoIterator<Item=usize>>(&self, point: Trip<Cart>, radius: Cart, indices: I) -> Vec<usize> {
		indices.into_iter().filter(|&i|
			self.pbc.distance(point, self.positions[i]) <= radius
		).collect()
	}

	// Identify indices of atoms within a sphere of the given radius about a point.
	// Optimized for the case where each call considers a point/radius very similar
	//  to the previous call.
	pub fn neighborhood(&mut self, point: Trip<Cart>, radius: Cart) -> Vec<usize> {
		self.update_cursors(point, radius);
		let candidates = self.cursor_neighborhood();
		self.neighborhood_from_candidates(point, radius, candidates)
	}

	pub fn bruteforce_neighborhood(&self, point: Trip<Cart>, radius: Cart) -> Vec<usize> {
		self.neighborhood_from_candidates(point, radius, 0..self.positions.len())
	}
}

//--------------------
// fulfills two needs which BTreeMap fails to satisfy:
//  * support for PartialOrd
//  * multiple values may have same key
type Key = f64;
type Value = usize;
#[derive(Default)]
struct SortedIndices {keys: Vec<Key>, values: Vec<Value>}
impl SortedIndices {
	fn insert(&mut self, k: Key, v: Value) {
		let i = self.lower_bound(k);
		self.keys.insert(i, k); self.values.insert(i, v);
	}

	fn lower_bound(&self, k: Key) -> usize {
		match self.keys.binary_search_by(|b| b.partial_cmp(&k).unwrap()) {
			Ok(x) => x, Err(x) => x,
		}
	}
}

fn update_lower_hint(mut hint: usize, sorted: &[Cart], needle: Cart) -> usize {
	while needle <= sorted[hint] { hint -= 1; }
	while needle >  sorted[hint] { hint += 1; }
	hint
}

fn update_upper_hint(mut hint: usize, sorted: &[Cart], needle: Cart) -> usize {
	while needle <  sorted[hint] { hint -= 1; }
	while needle >= sorted[hint] { hint += 1; }
	hint
}
