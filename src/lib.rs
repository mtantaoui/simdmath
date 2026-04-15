//! # simdmath
//!
//! High-performance SIMD implementations of mathematical functions for Rust.
//!
//! This crate provides vectorized versions of common math functions (trigonometric,
//! exponential, logarithmic, power, etc.) that operate on multiple values simultaneously
//! using hardware SIMD instructions. All implementations achieve **≤ 2 ULP** accuracy.
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
//! target features. SSE-only x86 and scalar fallbacks are not yet implemented.
//!
//! ## Usage
//!
//! The primary interface is the [`math::VecMath`] trait, which extends `Vec<f32>`
//! and `Vec<f64>` with SIMD-accelerated math operations:
//!
//! ```rust
//! use simdmath::math::VecMath;
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

mod arch;
pub mod math;
pub mod ops;
