//! AVX-512 backend: 16-lane `f32` / 8-lane `f64` register types and the
//! operations built on them.
//!
//! - [`f32x16`] / [`f64x8`] — the SIMD register types and their `Load` /
//!   `Store` implementations
//! - [`math`] — element-wise math functions (`sin`, `exp`, `pow`, …) and the
//!   register-level `VecMath` implementations
//! - [`matmul`] — BLIS-style blocked matrix multiplication
//! - [`vec`] — `SliceExt` / `VecExt` arithmetic implementations

pub(crate) mod f32x16;
pub(crate) mod f64x8;
pub(crate) mod math;
pub(crate) mod matmul;
pub(crate) mod vec;
