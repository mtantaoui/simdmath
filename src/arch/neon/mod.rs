//! NEON backend: 4-lane `f32` / 2-lane `f64` register types and the
//! operations built on them.
//!
//! - [`f32x4`] / [`f64x2`] — the SIMD register types and their `Load` /
//!   `Store` implementations
//! - [`math`] — element-wise math functions (`sin`, `exp`, `pow`, …) and the
//!   register-level `VecMath` implementations
//! - [`matmul`] — BLIS-style blocked matrix multiplication
//! - [`vec`] — `SliceExt` / `VecExt` arithmetic implementations

pub(crate) mod f32x4;
pub(crate) mod f64x2;
pub(crate) mod math;
pub(crate) mod matmul;
pub(crate) mod vec;
