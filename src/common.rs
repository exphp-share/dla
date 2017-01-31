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

pub type Float = f64;
pub type Pair<T> = (T,T);
pub type Trip<T> = (T,T,T);
pub type Cart = Float;
