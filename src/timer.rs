
use std::collections::vec_deque::VecDeque;
use time::precise_time_ns;
use std::io::prelude::*;

pub struct Timer { deque: VecDeque<u64> }
impl Timer {
	pub fn new(n: usize) -> Timer {
		let mut this = Timer { deque: VecDeque::new() };
		// Fill solely for ease of implementation (the first few outputs may be inaccurate)
		while this.deque.len() < n { this.deque.push_back(precise_time_ns()) }
		this
	}
	pub fn push(&mut self) {
		self.deque.pop_front();
		self.deque.push_back(precise_time_ns());
	}
	pub fn last_ms(&self) -> u64 {
		(self.deque[self.deque.len()-1] - self.deque[self.deque.len()-2]) / 1_000_000
	}
	pub fn average_ms(&self) -> u64 {
		(self.deque[self.deque.len()-1] - self.deque[0]) / ((self.deque.len() as u64 - 1) * 1_000_000)
	}
}

// primitive profiling helpers
/// Measure execution time in ns.
#[inline(never)]
pub fn time<F:FnOnce()>(func: F) -> u64 {
	let (t,a) = time_ret(func);
	::test::black_box(a);
	t
}

/// Measure execution time in ns and get return value.
pub fn time_ret<A, F:FnOnce() -> A>(func: F) -> (u64, A) {
	// hm... I wonder how the compiler knows it can't reorder these...
	// ........or does it?
	let t = precise_time_ns();
	let result = func();
	let dt = precise_time_ns() - t;
	(dt, result)
}

#[inline(never)]
pub fn time_log<A, F:FnOnce() -> A>(msg: &str, func: F) -> A {
	let (t,a) = time_ret(func);
	errln!("time_log: {}: {}", msg, t);
	a
}
