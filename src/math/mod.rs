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

    /// Returns the two-argument arc tangent `atan2(self, other)` for every element.
    ///
    /// Computes the angle θ of the point `(other[i], self[i])` in radians,
    /// measured counter-clockwise from the positive x-axis. The result is
    /// in the range `(-π, π]`.
    ///
    /// This is equivalent to `self[i].atan2(other[i])` for each lane.
    ///
    /// # Precision
    ///
    /// - **f32**: ≤ 3 ULP accuracy
    /// - **f64**: ≤ 2 ULP accuracy
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let y = vec![1.0f32, 1.0, -1.0, -1.0];
    /// let x = vec![1.0f32, -1.0, 1.0, -1.0];
    /// let r = y.atan2(&x);
    /// // r ≈ [π/4, 3π/4, -π/4, -3π/4]
    /// ```
    fn atan2(&self, other: &Self) -> Self;

    /// Returns the cube root of every element.
    ///
    /// Computed via a bit-manipulation trick for initial estimate followed by
    /// Newton–Raphson refinement. Handles negative numbers correctly (cube root
    /// of a negative number is the negative of the cube root of its absolute value).
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain including subnormals.
    ///
    /// # Special values
    ///
    /// - `cbrt(±0)` = `±0`
    /// - `cbrt(±∞)` = `±∞`
    /// - `cbrt(NaN)` = `NaN`
    /// - `cbrt(-x)` = `-cbrt(x)` for all x
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![1.0f32, 8.0, 27.0, -8.0];
    /// let r = a.cbrt();
    /// // r ≈ [1.0, 2.0, 3.0, -2.0]
    /// ```
    fn cbrt(&self) -> Self;

    /// Returns the cosine (in radians) of every element.
    ///
    /// Computed via Cody-Waite argument reduction to `[-π/4, π/4]` followed
    /// by minimax polynomial evaluation of cos/sin kernels (musl libc port).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    ///
    /// # Special values
    ///
    /// - `cos(±0)` = `1.0`
    /// - `cos(±∞)` = `NaN`
    /// - `cos(NaN)` = `NaN`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![0.0f32, std::f32::consts::PI, std::f32::consts::FRAC_PI_2];
    /// let r = a.cos();
    /// // r ≈ [1.0, -1.0, 0.0]
    /// ```
    fn cos(&self) -> Self;

    /// Returns the exponential function `e^x` of every element.
    ///
    /// Computed via argument reduction by `ln(2)` followed by a Padé-like
    /// degree-5 minimax polynomial (fdlibm port).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    ///
    /// # Special values
    ///
    /// - `exp(0)` = `1.0`
    /// - `exp(+∞)` = `+∞`
    /// - `exp(-∞)` = `0.0`
    /// - `exp(NaN)` = `NaN`
    /// - `exp(x > ~709.8)` = `+∞` (overflow)
    /// - `exp(x < ~-745)` = `0.0` (underflow)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![0.0f32, 1.0, -1.0];
    /// let r = a.exp();
    /// // r ≈ [1.0, 2.718, 0.368]
    /// ```
    fn exp(&self) -> Self;

    /// Returns the natural logarithm of every element.
    ///
    /// Computed via argument decomposition `x = 2^k * m` and a degree-7
    /// minimax polynomial in `s = f/(2+f)` (fdlibm port).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    ///
    /// # Special values
    ///
    /// - `ln(1)` = `0.0`
    /// - `ln(0)` = `-∞`
    /// - `ln(+∞)` = `+∞`
    /// - `ln(x < 0)` = `NaN`
    /// - `ln(NaN)` = `NaN`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![1.0f32, std::f32::consts::E, 10.0];
    /// let r = a.ln();
    /// // r ≈ [0.0, 1.0, 2.303]
    /// ```
    fn ln(&self) -> Self;

    /// Returns the sine (in radians) of every element.
    ///
    /// Computed via Cody-Waite argument reduction to `[-π/4, π/4]` followed
    /// by minimax polynomial evaluation of sin/cos kernels (musl libc port).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    ///
    /// # Special values
    ///
    /// - `sin(±0)` = `±0`
    /// - `sin(±∞)` = `NaN`
    /// - `sin(NaN)` = `NaN`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![0.0f32, std::f32::consts::FRAC_PI_2, std::f32::consts::PI];
    /// let r = a.sin();
    /// // r ≈ [0.0, 1.0, 0.0]
    /// ```
    fn sin(&self) -> Self;

    /// Returns the tangent (in radians) of every element.
    ///
    /// Computed via Cody-Waite argument reduction to `[-π/4, π/4]` followed
    /// by minimax polynomial evaluation of the tangent kernel. For odd
    /// quadrants, uses the cotangent identity `-1/tan(y)` (musl libc port).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    ///
    /// # Special values
    ///
    /// - `tan(±0)` = `±0`
    /// - `tan(±∞)` = `NaN`
    /// - `tan(NaN)` = `NaN`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![0.0f32, std::f32::consts::FRAC_PI_4];
    /// let r = a.tan();
    /// // r ≈ [0.0, 1.0]
    /// ```
    fn tan(&self) -> Self;

    /// Returns `self` raised to the power `exp` for every element: `self[i].powf(exp[i])`.
    ///
    /// Computed via compensated arithmetic: a high/low split of `ln(|x|)`,
    /// Dekker multiplication by `y`, and a compensated `exp` that folds in
    /// the low-order correction term. This achieves ≤ 2 ULP for both f32
    /// and f64 (the naive `exp(y·ln(x))` loses too much precision for f64).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    ///
    /// # Special values (IEEE 754 / C99 §7.12.7.4)
    ///
    /// - `pow(x, ±0)` = `1` for any `x` (including `NaN`)
    /// - `pow(1, y)` = `1` for any `y` (including `NaN`)
    /// - `pow(x, y)` = `NaN` if `x < 0` and `y` is not an integer
    /// - `pow(±0, y)` = `±∞` / `±0` depending on sign and odd-integer status
    /// - `pow(±∞, y)` follows standard infinity rules
    /// - `pow(x, ±∞)` = `0` or `+∞` depending on `|x|` vs 1
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let bases = vec![2.0f32, 3.0, 10.0, 0.5];
    /// let exps  = vec![3.0f32, 2.0, 0.5, -1.0];
    /// let r = bases.pow(&exps);
    /// // r ≈ [8.0, 9.0, 3.162, 2.0]
    /// ```
    fn pow(&self, exp: &Self) -> Self;

    /// Returns the square root of every element.
    ///
    /// Uses the hardware `sqrt` instruction, which is one of the five
    /// IEEE 754 correctly-rounded basic operations.
    ///
    /// # Precision
    ///
    /// **≤ 0.5 ULP** — hardware correctly-rounded operation.
    ///
    /// # Special values
    ///
    /// - `sqrt(+0)` = `+0`
    /// - `sqrt(-0)` = `-0`
    /// - `sqrt(+∞)` = `+∞`
    /// - `sqrt(x < 0)` = `NaN`
    /// - `sqrt(NaN)` = `NaN`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::VecMath;
    /// let a = vec![1.0f32, 4.0, 9.0, 16.0];
    /// let r = a.sqrt();
    /// // r ≈ [1.0, 2.0, 3.0, 4.0]
    /// ```
    fn sqrt(&self) -> Self;
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
