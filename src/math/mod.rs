//! SIMD-accelerated element-wise mathematical operations on `Vec<T>`.
//!
//! This module provides [`VecMath`], the public trait that extends `Vec<f32>`
//! and `Vec<f64>` with transcendental and mathematical functions backed by
//! SIMD intrinsics.
//!
//! # Design
//!
//! Internally `VecMath` is implemented at two levels:
//!
//! - **Register level** (private, `arch/<backend>/math.rs`): implements
//!   `VecMath<T>` for crate-internal SIMD register structs such as `F32x8`
//!   (AVX2) or `F32x16` (AVX-512). These call raw intrinsic wrappers and
//!   return the same register type. **The register types themselves are not
//!   currently part of the public API** — they may be exposed in a future
//!   release behind an `unstable-register-api` feature flag.
//!
//! - **Vector level** (public, `math/<backend>.rs`): implements `VecMath<T>`
//!   for `Vec<T>` by chunking the slice into SIMD registers, applying the
//!   register-level method, and reassembling the result.
//!
//! # Relationship to `VecExt`
//!
//! [`crate::ops::vec::VecExt`] covers arithmetic operators (`+`, `-`, `*`, `/`,
//! `%`) and reductions (`sum`, `min`, `max`). `VecMath` covers mathematical
//! functions that go beyond basic arithmetic.

/// Element-wise mathematical operations on `Vec<f32>` and `Vec<f64>`.
///
/// Each method allocates and returns a new `Vec<T>` of the same length with
/// the operation applied element-wise. The methods dispatch to a SIMD
/// implementation chosen at compile time based on target features (see crate
/// [overview](crate)).
///
/// # See also
///
/// Each method below has a dedicated chapter in the
/// [mathematical reference book](https://github.com/mtantaoui/simdmath/tree/main/book/src/functions)
/// covering the algorithm, error analysis, and per-backend differences:
///
/// | Method  | Book chapter |
/// |---------|--------------|
/// | [`abs`](Self::abs)       | _(arithmetic-class; see `arithmetic.md`)_ |
/// | [`sin`](Self::sin)       | [`functions/sin`](https://mtantaoui.github.io/simdmath/functions/sin.html) |
/// | [`cos`](Self::cos)       | [`functions/cos`](https://mtantaoui.github.io/simdmath/functions/cos.html) |
/// | [`tan`](Self::tan)       | [`functions/tan`](https://mtantaoui.github.io/simdmath/functions/tan.html) |
/// | [`asin`](Self::asin)     | [`functions/asin`](https://mtantaoui.github.io/simdmath/functions/asin.html) |
/// | [`acos`](Self::acos)     | [`functions/acos`](https://mtantaoui.github.io/simdmath/functions/acos.html) |
/// | [`atan`](Self::atan)     | [`functions/atan`](https://mtantaoui.github.io/simdmath/functions/atan.html) |
/// | [`atan2`](Self::atan2)   | [`functions/atan2`](https://mtantaoui.github.io/simdmath/functions/atan2.html) |
/// | [`exp`](Self::exp)       | [`functions/exp`](https://mtantaoui.github.io/simdmath/functions/exp.html) |
/// | [`ln`](Self::ln)         | [`functions/ln`](https://mtantaoui.github.io/simdmath/functions/ln.html) |
/// | [`pow`](Self::pow)       | [`functions/pow`](https://mtantaoui.github.io/simdmath/functions/pow.html) |
/// | [`sqrt`](Self::sqrt)     | [`functions/sqrt`](https://mtantaoui.github.io/simdmath/functions/sqrt.html) |
/// | [`cbrt`](Self::cbrt)     | [`functions/cbrt`](https://mtantaoui.github.io/simdmath/functions/cbrt.html) |
pub trait VecMath<T> {
    /// Returns the absolute value of every element.
    ///
    /// Clears the IEEE 754 sign bit of each lane using a bitwise ANDNOT with
    /// the sign-bit mask. Both `+0.0` and `-0.0` map to `+0.0`; `NaN`
    /// payloads are preserved (only the sign bit is cleared).
    ///
    /// # Precision
    ///
    /// **0 ULP** — exact (a sign-bit clear is bit-for-bit lossless).
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
    /// Computed via a three-range minimax rational approximation. The domain
    /// is `[-1, 1]`; values outside this range and `NaN` inputs produce
    /// `NaN` in the corresponding lane.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
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
    /// Computed via a two-range minimax rational approximation. The domain
    /// is `[-1, 1]`; values outside this range and `NaN` inputs produce
    /// `NaN` in the corresponding lane.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
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
    /// Computed via argument reduction followed by a minimax polynomial. The
    /// domain is all real numbers.
    ///
    /// # Precision
    ///
    /// - **f32**: **≤ 3 ULP** (single-range reduction).
    /// - **f64**: **≤ 1 ULP** (musl four-range reduction).
    ///
    /// # Special values
    ///
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
    /// # Special values
    ///
    /// Follows IEEE 754 / C99 §F.10.1.4. Highlights:
    ///
    /// - `atan2(±0, +0)` = `±0`
    /// - `atan2(±0, -0)` = `±π`
    /// - `atan2(±0, x > 0)` = `±0`
    /// - `atan2(±0, x < 0)` = `±π`
    /// - `atan2(y > 0, ±0)` = `+π/2`
    /// - `atan2(y < 0, ±0)` = `-π/2`
    /// - `atan2(±∞, +∞)` = `±π/4`
    /// - `atan2(±∞, -∞)` = `±3π/4`
    /// - `atan2(±∞, finite)` = `±π/2`
    /// - `atan2(finite, +∞)` = `±0`
    /// - `atan2(finite, -∞)` = `±π`
    /// - `atan2(NaN, _)` = `atan2(_, NaN)` = `NaN`
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

    /// Returns `self` raised to the power `exp` for every element: each output
    /// element is `self[i].powf(exp[i])`.
    ///
    /// This is the **element-wise vector exponent** form: both `self` and `exp`
    /// must have the same length, and each lane is raised to its own exponent.
    /// For raising every element to the *same* scalar exponent (e.g.
    /// `xs.powf(2.5)`), build an exponent vector via `vec![y; xs.len()]` for
    /// now; a dedicated scalar-exponent variant may be added in a future
    /// release.
    ///
    /// Computed via compensated arithmetic: a high/low split of `ln(|x|)`,
    /// Dekker multiplication by `y`, and a compensated `exp` that folds in
    /// the low-order correction term. This achieves ≤ 2 ULP for both f32
    /// and f64 (the naive `exp(y·ln(x))` loses too much precision for f64).
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != exp.len()`.
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

/// Slice form of [`VecMath`]; same operations on `&[T]` instead of `Vec<T>`,
/// returning a freshly-allocated `Vec<T>`.
///
/// Each method has the same algorithm, precision, and special-value behaviour
/// as the corresponding [`VecMath`] method — only the receiver type differs.
/// Because `[T]` is unsized, every method explicitly returns `Vec<T>` rather
/// than `Self`.
///
/// # See also
///
/// Each method has a dedicated chapter in the
/// [mathematical reference book](https://github.com/mtantaoui/simdmath/tree/main/book/src/functions)
/// covering the algorithm, error analysis, and per-backend differences:
///
/// | Method  | Book chapter |
/// |---------|--------------|
/// | [`abs`](Self::abs)       | _(arithmetic-class; see `arithmetic.md`)_ |
/// | [`sin`](Self::sin)       | [`functions/sin`](https://mtantaoui.github.io/simdmath/functions/sin.html) |
/// | [`cos`](Self::cos)       | [`functions/cos`](https://mtantaoui.github.io/simdmath/functions/cos.html) |
/// | [`tan`](Self::tan)       | [`functions/tan`](https://mtantaoui.github.io/simdmath/functions/tan.html) |
/// | [`asin`](Self::asin)     | [`functions/asin`](https://mtantaoui.github.io/simdmath/functions/asin.html) |
/// | [`acos`](Self::acos)     | [`functions/acos`](https://mtantaoui.github.io/simdmath/functions/acos.html) |
/// | [`atan`](Self::atan)     | [`functions/atan`](https://mtantaoui.github.io/simdmath/functions/atan.html) |
/// | [`atan2`](Self::atan2)   | [`functions/atan2`](https://mtantaoui.github.io/simdmath/functions/atan2.html) |
/// | [`exp`](Self::exp)       | [`functions/exp`](https://mtantaoui.github.io/simdmath/functions/exp.html) |
/// | [`ln`](Self::ln)         | [`functions/ln`](https://mtantaoui.github.io/simdmath/functions/ln.html) |
/// | [`pow`](Self::pow)       | [`functions/pow`](https://mtantaoui.github.io/simdmath/functions/pow.html) |
/// | [`sqrt`](Self::sqrt)     | [`functions/sqrt`](https://mtantaoui.github.io/simdmath/functions/sqrt.html) |
/// | [`cbrt`](Self::cbrt)     | [`functions/cbrt`](https://mtantaoui.github.io/simdmath/functions/cbrt.html) |
pub trait SliceMath<T> {
    /// Returns the absolute value of every element. Same precision as
    /// [`VecMath::abs`] (**0 ULP**, exact).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::math::SliceMath;
    /// let a: &[f32] = &[-1.0, 2.0, -3.0, 4.0];
    /// assert_eq!(a.abs(), vec![1.0f32, 2.0, 3.0, 4.0]);
    /// ```
    fn abs(&self) -> Vec<T>;

    /// Arc cosine of every element. Same precision as [`VecMath::acos`]
    /// (**≤ 1 ULP** on `[-1, 1]`; `NaN` outside the domain).
    fn acos(&self) -> Vec<T>;

    /// Arc sine of every element. Same precision as [`VecMath::asin`]
    /// (**≤ 1 ULP** on `[-1, 1]`; `NaN` outside the domain).
    fn asin(&self) -> Vec<T>;

    /// Arc tangent of every element. Same precision as [`VecMath::atan`]
    /// (f32 ≤ 3 ULP, f64 ≤ 1 ULP).
    fn atan(&self) -> Vec<T>;

    /// Two-argument arc tangent: `atan2(self[i], other[i])` for every lane.
    /// Same precision as [`VecMath::atan2`].
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != other.len()`.
    fn atan2(&self, other: &[T]) -> Vec<T>;

    /// Cube root of every element. Same precision as [`VecMath::cbrt`]
    /// (**≤ 1 ULP**).
    fn cbrt(&self) -> Vec<T>;

    /// Cosine (in radians) of every element. Same precision as
    /// [`VecMath::cos`] (**≤ 2 ULP**).
    fn cos(&self) -> Vec<T>;

    /// Exponential `e^x` of every element. Same precision as
    /// [`VecMath::exp`] (**≤ 2 ULP**).
    fn exp(&self) -> Vec<T>;

    /// Natural logarithm of every element. Same precision as
    /// [`VecMath::ln`] (**≤ 2 ULP**).
    fn ln(&self) -> Vec<T>;

    /// Sine (in radians) of every element. Same precision as
    /// [`VecMath::sin`] (**≤ 2 ULP**).
    fn sin(&self) -> Vec<T>;

    /// Tangent (in radians) of every element. Same precision as
    /// [`VecMath::tan`] (**≤ 2 ULP**).
    fn tan(&self) -> Vec<T>;

    /// Element-wise vector exponent: `self[i].powf(exp[i])` per lane. Same
    /// precision as [`VecMath::pow`] (**≤ 2 ULP**).
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != exp.len()`.
    fn pow(&self, exp: &[T]) -> Vec<T>;

    /// Square root of every element. Same precision as [`VecMath::sqrt`]
    /// (**≤ 0.5 ULP**, hardware correctly-rounded).
    fn sqrt(&self) -> Vec<T>;
}

// ---------------------------------------------------------------------------
// Vec<T> / [T] implementations — arch dispatch
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

// Scalar fallback: used when no SIMD ISA is available, including the SSE-only
// x86_64 case (which currently delegates to scalar) and any non-x86/non-arm
// target.
#[cfg(any(
    all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2"),
    ),
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
mod scalar;
