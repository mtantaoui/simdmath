//! SIMD-accelerated element-wise mathematical operations on `Vec<T>`.
//!
//! This module provides [`VecMath`], a public trait that extends both
//! architecture-specific SIMD register types (e.g. [`F32x8`]) and `Vec<T>`
//! with transcendental and mathematical functions backed by SIMD intrinsics.
//!
//! # Design
//!
//! The trait is implemented at two levels:
//!
//! - **Register level** (`arch/<backend>/math.rs`): implements `VecMath<T>` for
//!   SIMD register structs such as `F32x8` (AVX2) or `F32x16` (AVX-512). These
//!   implementations call raw intrinsic wrappers and return the same register type.
//!
//! - **Vector level** (`math/<backend>.rs`): implements `VecMath<T>` for
//!   `Vec<T>` using [`crate::ops::vec::unary_op`], which chunks the slice into
//!   SIMD registers, applies the register-level method, and reassembles the
//!   result.
//!
//! # Relationship to `VecExt`
//!
//! [`crate::ops::vec::VecExt`] covers arithmetic operators (`+`, `-`, `*`, `/`,
//! `%`) and reductions (`sum`, `min`, `max`). `VecMath` covers mathematical
//! functions that go beyond basic arithmetic.

/// Element-wise mathematical operations, implemented for both SIMD register
/// types and `Vec<T>`.
///
/// - On SIMD register types (e.g. `F32x8`), each method returns the same type
///   with the operation applied to every lane.
/// - On `Vec<T>`, each method allocates and returns a new `Vec<T>` of the same
///   length with the operation applied element-wise.
pub trait VecMath<T> {
    /// Returns the absolute value of every element.
    ///
    /// Clears the IEEE 754 sign bit of each lane using a bitwise ANDNOT with
    /// the sign-bit mask. Both `+0.0` and `-0.0` map to `+0.0`; `NaN`
    /// payloads are preserved (only the sign bit is cleared).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![-1.0f32, 2.0, -3.0, 4.0];
    /// assert_eq!(a.abs(), vec![1.0f32, 2.0, 3.0, 4.0]);
    /// ```
    fn abs(&self) -> Self;

    /// Returns the arc cosine (in radians) of every element.
    ///
    /// Computed via a three-range minimax rational approximation (≤ 1 ULP).
    /// The domain is `[-1, 1]`; values outside this range and `NaN` inputs
    /// produce `NaN` in the corresponding lane.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![1.0f32, 0.0, -1.0, 0.5];
    /// let r = a.acos();
    /// // r ≈ [0.0, π/2, π, π/3]
    /// ```
    fn acos(&self) -> Self;

    /// Returns the arc sine (in radians) of every element.
    ///
    /// Computed via a two-range minimax rational approximation (≤ 1 ULP).
    /// The domain is `[-1, 1]`; values outside this range and `NaN` inputs
    /// produce `NaN` in the corresponding lane.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![0.0f32, 0.5, -0.5, 1.0];
    /// let r = a.asin();
    /// // r ≈ [0.0, π/6, -π/6, π/2]
    /// ```
    fn asin(&self) -> Self;

    /// Returns the arc tangent (in radians) of every element.
    ///
    /// Computed via argument reduction followed by a minimax polynomial.
    /// - **f32**: ≤ 3 ULP accuracy (single-range reduction)
    /// - **f64**: ≤ 1 ULP accuracy (musl 4-range reduction)
    ///
    /// The domain is all real numbers; special values:
    /// - `atan(±0)` = `±0`
    /// - `atan(±∞)` = `±π/2`
    /// - `atan(NaN)` = `NaN`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![0.0f32, 1.0, -1.0, f32::INFINITY];
    /// let r = a.atan();
    /// // r ≈ [0.0, π/4, -π/4, π/2]
    /// ```
    fn atan(&self) -> Self;
}

// ---------------------------------------------------------------------------
// Vec<T> implementations — arch dispatch
// ---------------------------------------------------------------------------

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    target_feature = "avx2"
))]
mod avx2;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod neon;
