//! # simdmath
//!
//! High-performance SIMD implementations of mathematical functions for Rust.
//!
//! This crate provides vectorized versions of common math functions
//! (trigonometric, exponential, logarithmic, power, etc.) that operate on
//! multiple values simultaneously using hardware SIMD instructions.
//!
//! ## Accuracy
//!
//! The table below is the single source of truth for worst-case ULP error,
//! measured per the methodology described in
//! [`docs/ulp-methodology.md`]. Per-function and per-backend claims must
//! match these numbers; the long-form derivation lives in the
//! [reference book](https://github.com/mtantaoui/simdmath/tree/main/book).
//!
//! | Function | f32 (worst-case ULP) | f64 (worst-case ULP) | Domain                    |
//! |----------|----------------------|----------------------|---------------------------|
//! | `abs`    | 0 (exact)            | 0 (exact)            | all finite + ±∞ + NaN     |
//! | `sqrt`   | ≤ 0.5 (correctly rounded) | ≤ 0.5 (correctly rounded) | `x ≥ 0`           |
//! | `cbrt`   | ≤ 1                  | ≤ 1                  | all finite                |
//! | `sin`    | ≤ 2                  | ≤ 2                  | all finite                |
//! | `cos`    | ≤ 2                  | ≤ 2                  | all finite                |
//! | `tan`    | ≤ 2                  | ≤ 2                  | excluding `π/2 + kπ`      |
//! | `asin`   | ≤ 1                  | ≤ 1                  | `[-1, 1]`                 |
//! | `acos`   | ≤ 1                  | ≤ 1                  | `[-1, 1]`                 |
//! | `atan`   | ≤ 3                  | ≤ 1                  | all finite                |
//! | `atan2`  | ≤ 3                  | ≤ 2                  | all finite × all finite   |
//! | `exp`    | ≤ 2                  | ≤ 2                  | all finite (clamped)      |
//! | `ln`     | ≤ 2                  | ≤ 2                  | `x > 0`                   |
//! | `pow`    | ≤ 2                  | ≤ 2                  | per IEEE 754-2008 special cases |
//!
//! [`docs/ulp-methodology.md`]: https://github.com/mtantaoui/simdmath/blob/main/docs/ulp-methodology.md
//!
//! ## Supported architectures
//!
//! | Architecture | ISA      | f32 lanes | f64 lanes |
//! |--------------|----------|-----------|-----------|
//! | x86_64       | AVX2+FMA | 8         | 4         |
//! | x86_64       | AVX-512  | 16        | 8         |
//! | aarch64      | NEON     | 4         | 2         |
//!
//! The correct backend is selected automatically at compile time based on
//! target features. When no SIMD ISA is enabled, the crate falls back to a
//! correct (but slow) scalar implementation that delegates to `f32`/`f64`
//! methods from the standard library. To get SIMD speeds on `x86_64`, build
//! with e.g. `RUSTFLAGS="-C target-feature=+avx2,+fma"`. Runtime CPU dispatch
//! is planned for a future release.
//!
//! ## Usage
//!
//! The primary interface is the [`prelude::VecMath`] / [`prelude::SliceMath`] traits
//! (math functions) and [`prelude::VecExt`] / [`prelude::SliceExt`]
//! (arithmetic + reductions). All four are re-exported from the crate's
//! [`prelude`]:
//!
//! ```rust,ignore
//! use simdmath::prelude::*;
//!
//! let angles = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
//! let sines = angles.sin();
//! let cosines = angles.cos();
//! ```
//!
//! ## Available functions
//!
//! `abs`, `acos`, `asin`, `atan`, `atan2`, `cbrt`, `cos`, `sin`, `tan`,
//! `exp`, `ln`, `pow`, `sqrt`
//!
//! ## Algorithms
//!
//! Implementations are ported from **musl libc** (fdlibm descent) with exact
//! constants. Techniques include Cody-Waite argument reduction, Padé rational
//! approximations, Dekker compensated arithmetic, and Newton–Raphson refinement.
//! Full per-function derivations live in the
//! [mathematical reference book](https://github.com/mtantaoui/simdmath/tree/main/book).

mod arch;
pub mod linalg;
pub(crate) mod math;
pub(crate) mod ops;

#[cfg(test)]
mod test_utils;

/// Convenience re-exports of the public traits.
///
/// `use simdmath::prelude::*;` brings [`prelude::VecMath`], [`prelude::SliceMath`],
/// [`prelude::VecExt`] and [`prelude::SliceExt`] into scope, which is enough to use
/// the crate idiomatically on either `Vec<T>` or `&[T]`.
pub mod prelude {
    pub use crate::math::{SliceMath, VecMath};
    pub use crate::ops::vec::{SliceExt, VecExt};
}
