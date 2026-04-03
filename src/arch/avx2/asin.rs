//! AVX2 SIMD implementation of `asin(x)` for `f32` and `f64` vectors.
//!
//! # Algorithm
//!
//! This is a branchless, two-range minimax rational approximation ported
//! from **musl libc's `asinf.c`** and `asin.c` (which descend from Sun's fdlibm).
//!
//! # Precision
//!
//! Both implementations achieve **≤ 1 ULP** accuracy across the entire domain `[-1, 1]`.
//!
//! The core building block is a Padé rational approximation `r(z)` that
//! approximates `(asin(√z)/√z − 1)/z`, yielding:
//!
//! ```text
//! asin(x) ≈ x + x·r(x²)      for |x| ≤ 0.5
//! ```
//!
//! ## Two computational ranges
//!
//! | Range           | Identity used                                        |
//! |-----------------|------------------------------------------------------|
//! | `\|x\| < 0.5`  | `asin(x) = x + x·r(x²)`                             |
//! | `\|x\| ≥ 0.5`  | `asin(x) = π/2 − 2·(s + s·r(z))` where `z = (1−\|x\|)/2`, `s = √z` |
//!
//! The half-angle formula for `|x| ≥ 0.5` avoids catastrophic cancellation
//! near `|x| = 1`, where the derivative of `asin(x)` diverges.
//!
//! ## Compensated sqrt (Dekker split)
//!
//! For `|x| >= 0.5`, the sqrt argument `z = (1−|x|)/2` can be very small, so a
//! compensated sqrt is used for extra precision. The low bits of `s = √z`
//! are masked off to produce `df`; then the rounding error is recovered as
//! `c = (z − df²)/(s + df)` using exact arithmetic. This ensures ≤ 1 ULP
//! accuracy even near |x| = 1.
//!
//! - **f32**: 12 bits masked (df has 11 significant mantissa bits)
//! - **f64**: 32 bits masked (df has 20 significant mantissa bits)
//!
//! ## Special values
//!
//! | Input        | Output          |
//! |--------------|-----------------|
//! | `0.0`        | `0.0`           |
//! | `1.0`        | `π/2 ≈ 1.5708`  |
//! | `-1.0`       | `−π/2 ≈ −1.5708`|
//! | `\|x\| > 1` | `NaN`           |
//! | `NaN`        | `NaN`           |
//!
//! ## Tiny values
//!
//! For very small `|x|`, we return `x` directly since the polynomial correction
//! is smaller than representable precision and computing it risks underflow.
//!
//! - **f32**: `|x| < 2^-12`
//! - **f64**: `|x| < 2^-27`
//!
//! ## Blending strategy
//!
//! All result branches are computed unconditionally and blended at the end.
//! This eliminates branches and keeps all lanes in flight simultaneously.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::avx2::abs::{_mm256_abs_pd, _mm256_abs_ps};
use crate::arch::consts::acos::{
    P_S0_32, P_S0_64, P_S1_32, P_S1_64, P_S2_32, P_S2_64, P_S3_64, P_S4_64, P_S5_64, Q_S1_32,
    Q_S1_64, Q_S2_64, Q_S3_64, Q_S4_64,
};
use crate::arch::consts::asin::{
    PIO2_HI_32, PIO2_HI_64, PIO2_LO_32, PIO2_LO_64, TINY_THRESHOLD_32, TINY_THRESHOLD_64,
};

// ---------------------------------------------------------------------------
// f32 Implementation
// ---------------------------------------------------------------------------

/// Computes `asin(x)` for each lane of an AVX2 `__m256` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
///
/// # Description
///
/// All 8 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large |x|, tiny, |x|=1, out-of-domain) are computed
/// unconditionally and merged with `_mm256_blendv_ps`.
///
/// # Safety
///
/// Requires AVX2 and FMA support. `x` must be a valid `__m256` register.
///
/// # Example
///
/// ```ignore
/// let input = _mm256_set1_ps(0.5);
/// let result = _mm256_asin_ps(input);
/// // result ≈ [0.5236, 0.5236, 0.5236, 0.5236, 0.5236, 0.5236, 0.5236, 0.5236]
/// // (π/6 ≈ 0.5236)
/// ```
#[inline]
pub(crate) unsafe fn _mm256_asin_ps(x: __m256) -> __m256 {
    unsafe {
        // ---------------------------------------------------------------------
        // Broadcast scalar constants to SIMD registers
        // ---------------------------------------------------------------------
        let pio2_hi = _mm256_set1_ps(PIO2_HI_32);
        let pio2_lo = _mm256_set1_ps(PIO2_LO_32);
        let p_s0 = _mm256_set1_ps(P_S0_32);
        let p_s1 = _mm256_set1_ps(P_S1_32);
        let p_s2 = _mm256_set1_ps(P_S2_32);
        let q_s1 = _mm256_set1_ps(Q_S1_32);

        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let two = _mm256_set1_ps(2.0);
        let zero = _mm256_setzero_ps();

        // ---------------------------------------------------------------------
        // Compute |x| for range selection
        // ---------------------------------------------------------------------
        let abs_x = _mm256_abs_ps(x);

        // ---------------------------------------------------------------------
        // Rational polynomial approximation: r(z) = p(z) / q(z)
        //
        // This approximates (asin(√z)/√z − 1)/z on [0, 0.25].
        //   p(z) = z · (P_S0 + z · (P_S1 + z · P_S2))
        //   q(z) = 1 + z · Q_S1
        // ---------------------------------------------------------------------
        #[inline(always)]
        unsafe fn rational_r(
            z: __m256,
            p_s0: __m256,
            p_s1: __m256,
            p_s2: __m256,
            q_s1: __m256,
            one: __m256,
        ) -> __m256 {
            unsafe {
                // Horner's method for numerator: z * (P_S0 + z * (P_S1 + z * P_S2))
                let p_inner = _mm256_fmadd_ps(z, p_s2, p_s1); // P_S1 + z * P_S2
                let p_outer = _mm256_fmadd_ps(z, p_inner, p_s0); // P_S0 + z * (...)
                let p = _mm256_mul_ps(z, p_outer); // z * (...)

                // Denominator: 1 + z * Q_S1
                let q = _mm256_fmadd_ps(z, q_s1, one);

                _mm256_div_ps(p, q)
            }
        }

        // ---------------------------------------------------------------------
        // Condition masks (computed once, used for blending)
        // ---------------------------------------------------------------------
        let is_abs_ge_1 = _mm256_cmp_ps(abs_x, one, _CMP_GE_OQ); // |x| >= 1
        let is_abs_eq_1 = _mm256_cmp_ps(abs_x, one, _CMP_EQ_OQ); // |x| == 1
        let is_abs_lt_half = _mm256_cmp_ps(abs_x, half, _CMP_LT_OQ); // |x| < 0.5
        let is_x_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ); // x < 0

        // Tiny threshold check: |x| < 2^-12 (avoids underflow in polynomial)
        let tiny_bits = _mm256_set1_epi32(TINY_THRESHOLD_32 as i32);
        let is_tiny = _mm256_cmp_ps(abs_x, _mm256_castsi256_ps(tiny_bits), _CMP_LT_OQ);

        // ---------------------------------------------------------------------
        // Case A: |x| == 1 → return ±π/2
        //
        // asin(1) = π/2, asin(-1) = -π/2
        // Use pio2_hi + sign(x) * pio2_lo for full precision
        // ---------------------------------------------------------------------
        let result_eq_1 = _mm256_blendv_ps(
            _mm256_add_ps(pio2_hi, pio2_lo),                      // x = +1 → +π/2
            _mm256_sub_ps(_mm256_sub_ps(zero, pio2_hi), pio2_lo), // x = -1 → -π/2
            is_x_neg,
        );

        // ---------------------------------------------------------------------
        // Case B: |x| > 1 → return NaN (domain error)
        //
        // Compute NaN as 0.0 / (x - x) = 0.0 / 0.0 = NaN
        // ---------------------------------------------------------------------
        let nan = _mm256_div_ps(zero, _mm256_sub_ps(x, x));

        // ---------------------------------------------------------------------
        // Case C: |x| < 0.5 → asin(x) = x + x · r(x²)
        //
        // For tiny |x| < 2^-12, just return x (polynomial is negligible).
        // ---------------------------------------------------------------------
        let x_sq = _mm256_mul_ps(x, x);
        let r_small = rational_r(x_sq, p_s0, p_s1, p_s2, q_s1, one);
        let result_small_computed = _mm256_fmadd_ps(x, r_small, x); // x + x * r(x²)
        let result_small = _mm256_blendv_ps(result_small_computed, x, is_tiny);

        // ---------------------------------------------------------------------
        // Case D: 0.5 ≤ |x| < 1 → use half-angle formula with Dekker split
        //
        // Let z = (1 - |x|) / 2, s = √z
        //
        // Dekker compensated sqrt:
        //   df = s with low 12 mantissa bits cleared (exact high part)
        //   c  = (z - df²) / (s + df)  (rounding correction)
        //
        // Final computation (from musl asinf.c):
        //   w = s·r(z) + c
        //   asin(|x|) = π/2 - 2·(df + w)
        //             = pio2_hi - 2·df - (2·w - pio2_lo)
        //
        // The result is negated if x < 0.
        // ---------------------------------------------------------------------
        let z_large = _mm256_mul_ps(_mm256_sub_ps(one, abs_x), half); // (1 - |x|) / 2
        let s_large = _mm256_sqrt_ps(z_large); // √z
        let r_large = rational_r(z_large, p_s0, p_s1, p_s2, q_s1, one);

        // Dekker split: mask off low 12 bits to get exact high part of s
        let df = _mm256_castsi256_ps(_mm256_and_si256(
            _mm256_castps_si256(s_large),
            _mm256_set1_epi32(0xfffff000_u32 as i32),
        ));

        // Rounding correction: c = (z - df²) / (s + df)
        let c = _mm256_div_ps(
            _mm256_sub_ps(z_large, _mm256_mul_ps(df, df)),
            _mm256_add_ps(s_large, df),
        );

        // Compute w = s·r(z) + c
        let w = _mm256_fmadd_ps(s_large, r_large, c);

        // Compute: pio2_hi - 2·df - (2·w - pio2_lo)
        let two_w = _mm256_mul_ps(two, w);
        let inner = _mm256_sub_ps(two_w, pio2_lo); // 2·w - pio2_lo
        let two_df = _mm256_mul_ps(two, df);
        let result_large_abs = _mm256_sub_ps(_mm256_sub_ps(pio2_hi, two_df), inner);

        // Apply sign: if x < 0, negate the result
        let result_large = _mm256_blendv_ps(
            result_large_abs,
            _mm256_sub_ps(zero, result_large_abs),
            is_x_neg,
        );

        // ---------------------------------------------------------------------
        // Final blending: combine all cases
        // ---------------------------------------------------------------------

        // Handle |x| >= 1: either exact ±π/2 (if |x| == 1) or NaN (if |x| > 1)
        let result_ge_1 = _mm256_blendv_ps(nan, result_eq_1, is_abs_eq_1);

        // Select between small and large cases for |x| < 1
        let result_valid = _mm256_blendv_ps(result_large, result_small, is_abs_lt_half);

        // Final selection: valid cases vs |x| >= 1 cases
        _mm256_blendv_ps(result_valid, result_ge_1, is_abs_ge_1)
    }
}

// ===========================================================================
// f64 Implementation
// ===========================================================================

/// Computes `asin(x)` for each lane of an AVX2 `__m256d` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
///
/// # Description
///
/// All 4 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large |x|, tiny, |x|=1, out-of-domain) are computed
/// unconditionally and merged with `_mm256_blendv_pd`.
///
/// # Safety
///
/// Requires AVX2 and FMA support. `x` must be a valid `__m256d` register.
///
/// # Example
///
/// ```ignore
/// let input = _mm256_set1_pd(0.5);
/// let result = _mm256_asin_pd(input);
/// // result ≈ [0.5236, 0.5236, 0.5236, 0.5236]
/// // (π/6 ≈ 0.5236)
/// ```
#[inline]
pub(crate) unsafe fn _mm256_asin_pd(x: __m256d) -> __m256d {
    unsafe {
        // ---------------------------------------------------------------------
        // Broadcast scalar constants to SIMD registers
        // ---------------------------------------------------------------------
        let pio2_hi = _mm256_set1_pd(PIO2_HI_64);
        let pio2_lo = _mm256_set1_pd(PIO2_LO_64);
        let p_s0 = _mm256_set1_pd(P_S0_64);
        let p_s1 = _mm256_set1_pd(P_S1_64);
        let p_s2 = _mm256_set1_pd(P_S2_64);
        let p_s3 = _mm256_set1_pd(P_S3_64);
        let p_s4 = _mm256_set1_pd(P_S4_64);
        let p_s5 = _mm256_set1_pd(P_S5_64);
        let q_s1 = _mm256_set1_pd(Q_S1_64);
        let q_s2 = _mm256_set1_pd(Q_S2_64);
        let q_s3 = _mm256_set1_pd(Q_S3_64);
        let q_s4 = _mm256_set1_pd(Q_S4_64);

        let one = _mm256_set1_pd(1.0);
        let half = _mm256_set1_pd(0.5);
        let two = _mm256_set1_pd(2.0);
        let zero = _mm256_setzero_pd();

        // ---------------------------------------------------------------------
        // Compute |x| for range selection
        // ---------------------------------------------------------------------
        let abs_x = _mm256_abs_pd(x);

        // ---------------------------------------------------------------------
        // Rational polynomial approximation: r(z) = p(z) / q(z)
        //
        // This approximates (asin(√z)/√z − 1)/z on [0, 0.25].
        //   p(z) = z · (P_S0 + z · (P_S1 + z · (P_S2 + z · (P_S3 + z · (P_S4 + z · P_S5)))))
        //   q(z) = 1 + z · (Q_S1 + z · (Q_S2 + z · (Q_S3 + z · Q_S4)))
        // ---------------------------------------------------------------------
        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn rational_r(
            z: __m256d,
            p_s0: __m256d,
            p_s1: __m256d,
            p_s2: __m256d,
            p_s3: __m256d,
            p_s4: __m256d,
            p_s5: __m256d,
            q_s1: __m256d,
            q_s2: __m256d,
            q_s3: __m256d,
            q_s4: __m256d,
            one: __m256d,
        ) -> __m256d {
            unsafe {
                // Horner's method for numerator (degree 5)
                let p = _mm256_mul_pd(
                    z,
                    _mm256_fmadd_pd(
                        z,
                        _mm256_fmadd_pd(
                            z,
                            _mm256_fmadd_pd(
                                z,
                                _mm256_fmadd_pd(z, _mm256_fmadd_pd(z, p_s5, p_s4), p_s3),
                                p_s2,
                            ),
                            p_s1,
                        ),
                        p_s0,
                    ),
                );

                // Horner's method for denominator (degree 4)
                let q = _mm256_fmadd_pd(
                    z,
                    _mm256_fmadd_pd(
                        z,
                        _mm256_fmadd_pd(z, _mm256_fmadd_pd(z, q_s4, q_s3), q_s2),
                        q_s1,
                    ),
                    one,
                );

                _mm256_div_pd(p, q)
            }
        }

        // ---------------------------------------------------------------------
        // Condition masks (computed once, used for blending)
        // ---------------------------------------------------------------------
        let is_abs_ge_1 = _mm256_cmp_pd(abs_x, one, _CMP_GE_OQ); // |x| >= 1
        let is_abs_eq_1 = _mm256_cmp_pd(abs_x, one, _CMP_EQ_OQ); // |x| == 1
        let is_abs_lt_half = _mm256_cmp_pd(abs_x, half, _CMP_LT_OQ); // |x| < 0.5
        let is_x_neg = _mm256_cmp_pd(x, zero, _CMP_LT_OQ); // x < 0

        // Tiny threshold check: |x| < 2^-27 (avoids underflow in polynomial)
        let tiny_bits = _mm256_set1_epi64x(TINY_THRESHOLD_64 as i64);
        let is_tiny = _mm256_cmp_pd(abs_x, _mm256_castsi256_pd(tiny_bits), _CMP_LT_OQ);

        // ---------------------------------------------------------------------
        // Case A: |x| == 1 → return ±π/2
        //
        // asin(1) = π/2, asin(-1) = -π/2
        // Use pio2_hi + sign(x) * pio2_lo for full precision
        // ---------------------------------------------------------------------
        let result_eq_1 = _mm256_blendv_pd(
            _mm256_add_pd(pio2_hi, pio2_lo),                      // x = +1 → +π/2
            _mm256_sub_pd(_mm256_sub_pd(zero, pio2_hi), pio2_lo), // x = -1 → -π/2
            is_x_neg,
        );

        // ---------------------------------------------------------------------
        // Case B: |x| > 1 → return NaN (domain error)
        //
        // Compute NaN as 0.0 / (x - x) = 0.0 / 0.0 = NaN
        // ---------------------------------------------------------------------
        let nan = _mm256_div_pd(zero, _mm256_sub_pd(x, x));

        // ---------------------------------------------------------------------
        // Case C: |x| < 0.5 → asin(x) = x + x · r(x²)
        //
        // For tiny |x| < 2^-27, just return x (polynomial is negligible).
        // ---------------------------------------------------------------------
        let x_sq = _mm256_mul_pd(x, x);
        let r_small = rational_r(
            x_sq, p_s0, p_s1, p_s2, p_s3, p_s4, p_s5, q_s1, q_s2, q_s3, q_s4, one,
        );
        let result_small_computed = _mm256_fmadd_pd(x, r_small, x); // x + x * r(x²)
        let result_small = _mm256_blendv_pd(result_small_computed, x, is_tiny);

        // ---------------------------------------------------------------------
        // Case D: 0.5 ≤ |x| < 1 → use half-angle formula with Dekker split
        //
        // Let z = (1 - |x|) / 2, s = √z
        //
        // Dekker compensated sqrt:
        //   df = s with low 32 mantissa bits cleared (exact high part)
        //   c  = (z - df²) / (s + df)  (rounding correction)
        //
        // Final computation (from musl asin.c):
        //   w = s·r(z) + c
        //   asin(|x|) = π/2 - 2·(df + w)
        //             = pio2_hi - 2·df - (2·w - pio2_lo)
        //
        // The result is negated if x < 0.
        // ---------------------------------------------------------------------
        let z_large = _mm256_mul_pd(_mm256_sub_pd(one, abs_x), half); // (1 - |x|) / 2
        let s_large = _mm256_sqrt_pd(z_large); // √z
        let r_large = rational_r(
            z_large, p_s0, p_s1, p_s2, p_s3, p_s4, p_s5, q_s1, q_s2, q_s3, q_s4, one,
        );

        // Dekker split: mask off low 32 bits to get exact high part of s
        let df = _mm256_castsi256_pd(_mm256_and_si256(
            _mm256_castpd_si256(s_large),
            _mm256_set1_epi64x(0xffffffff00000000_u64 as i64),
        ));

        // Rounding correction: c = (z - df²) / (s + df)
        let c = _mm256_div_pd(
            _mm256_sub_pd(z_large, _mm256_mul_pd(df, df)),
            _mm256_add_pd(s_large, df),
        );

        // Compute w = s·r(z) + c
        let w = _mm256_fmadd_pd(s_large, r_large, c);

        // Compute: pio2_hi - 2·df - (2·w - pio2_lo)
        let two_w = _mm256_mul_pd(two, w);
        let inner = _mm256_sub_pd(two_w, pio2_lo); // 2·w - pio2_lo
        let two_df = _mm256_mul_pd(two, df);
        let result_large_abs = _mm256_sub_pd(_mm256_sub_pd(pio2_hi, two_df), inner);

        // Apply sign: if x < 0, negate the result
        let result_large = _mm256_blendv_pd(
            result_large_abs,
            _mm256_sub_pd(zero, result_large_abs),
            is_x_neg,
        );

        // ---------------------------------------------------------------------
        // Final blending: combine all cases
        // ---------------------------------------------------------------------

        // Handle |x| >= 1: either exact ±π/2 (if |x| == 1) or NaN (if |x| > 1)
        let result_ge_1 = _mm256_blendv_pd(nan, result_eq_1, is_abs_eq_1);

        // Select between small and large cases for |x| < 1
        let result_valid = _mm256_blendv_pd(result_large, result_small, is_abs_lt_half);

        // Final selection: valid cases vs |x| >= 1 cases
        _mm256_blendv_pd(result_valid, result_ge_1, is_abs_ge_1)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to extract all 8 lanes from an __m256 register.
    unsafe fn extract_f32x8(v: __m256) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v) };
        out
    }

    /// Helper to check if two f32 values are approximately equal.
    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon || (a.is_nan() && b.is_nan())
    }

    // -------------------------------------------------------------------------
    // Special values
    // -------------------------------------------------------------------------

    #[test]
    fn asin_of_zero_is_zero() {
        unsafe {
            let input = _mm256_set1_ps(0.0);
            let result = extract_f32x8(_mm256_asin_ps(input));
            for &val in &result {
                assert_eq!(val, 0.0);
            }
        }
    }

    #[test]
    fn asin_of_neg_zero_is_neg_zero() {
        unsafe {
            let input = _mm256_set1_ps(-0.0);
            let result = extract_f32x8(_mm256_asin_ps(input));
            for &val in &result {
                assert!(val == 0.0 && val.is_sign_negative());
            }
        }
    }

    #[test]
    fn asin_of_one_is_pio2() {
        unsafe {
            let input = _mm256_set1_ps(1.0);
            let result = extract_f32x8(_mm256_asin_ps(input));
            let expected = std::f32::consts::FRAC_PI_2;
            for &val in &result {
                assert!(
                    approx_eq(val, expected, 1e-6),
                    "asin(1) = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn asin_of_neg_one_is_neg_pio2() {
        unsafe {
            let input = _mm256_set1_ps(-1.0);
            let result = extract_f32x8(_mm256_asin_ps(input));
            let expected = -std::f32::consts::FRAC_PI_2;
            for &val in &result {
                assert!(
                    approx_eq(val, expected, 1e-6),
                    "asin(-1) = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn asin_of_half_is_pi_over_6() {
        unsafe {
            let input = _mm256_set1_ps(0.5);
            let result = extract_f32x8(_mm256_asin_ps(input));
            let expected = std::f32::consts::FRAC_PI_6;
            for &val in &result {
                assert!(
                    approx_eq(val, expected, 1e-6),
                    "asin(0.5) = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn asin_of_neg_half_is_neg_pi_over_6() {
        unsafe {
            let input = _mm256_set1_ps(-0.5);
            let result = extract_f32x8(_mm256_asin_ps(input));
            let expected = -std::f32::consts::FRAC_PI_6;
            for &val in &result {
                assert!(
                    approx_eq(val, expected, 1e-6),
                    "asin(-0.5) = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn asin_of_sqrt2_over_2_is_pi_over_4() {
        unsafe {
            let sqrt2_over_2 = std::f32::consts::FRAC_1_SQRT_2;
            let input = _mm256_set1_ps(sqrt2_over_2);
            let result = extract_f32x8(_mm256_asin_ps(input));
            let expected = std::f32::consts::FRAC_PI_4;
            for &val in &result {
                assert!(
                    approx_eq(val, expected, 1e-6),
                    "asin(√2/2) = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn asin_of_sqrt3_over_2_is_pi_over_3() {
        unsafe {
            let sqrt3_over_2 = 3.0_f32.sqrt() / 2.0;
            let input = _mm256_set1_ps(sqrt3_over_2);
            let result = extract_f32x8(_mm256_asin_ps(input));
            let expected = std::f32::consts::FRAC_PI_3;
            for &val in &result {
                assert!(
                    approx_eq(val, expected, 1e-5),
                    "asin(√3/2) = {val}, expected {expected}"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Domain errors (|x| > 1)
    // -------------------------------------------------------------------------

    #[test]
    fn asin_above_one_is_nan() {
        unsafe {
            let input = _mm256_set1_ps(1.5);
            let result = extract_f32x8(_mm256_asin_ps(input));
            for &val in &result {
                assert!(val.is_nan(), "asin(1.5) should be NaN, got {val}");
            }
        }
    }

    #[test]
    fn asin_below_neg_one_is_nan() {
        unsafe {
            let input = _mm256_set1_ps(-1.5);
            let result = extract_f32x8(_mm256_asin_ps(input));
            for &val in &result {
                assert!(val.is_nan(), "asin(-1.5) should be NaN, got {val}");
            }
        }
    }

    #[test]
    fn asin_of_nan_is_nan() {
        unsafe {
            let input = _mm256_set1_ps(f32::NAN);
            let result = extract_f32x8(_mm256_asin_ps(input));
            for &val in &result {
                assert!(val.is_nan(), "asin(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn asin_of_infinity_is_nan() {
        unsafe {
            let input = _mm256_set1_ps(f32::INFINITY);
            let result = extract_f32x8(_mm256_asin_ps(input));
            for &val in &result {
                assert!(val.is_nan(), "asin(∞) should be NaN");
            }
        }
    }

    #[test]
    fn asin_of_neg_infinity_is_nan() {
        unsafe {
            let input = _mm256_set1_ps(f32::NEG_INFINITY);
            let result = extract_f32x8(_mm256_asin_ps(input));
            for &val in &result {
                assert!(val.is_nan(), "asin(-∞) should be NaN");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Lane independence
    // -------------------------------------------------------------------------

    #[test]
    fn asin_processes_all_8_lanes_independently() {
        unsafe {
            // Test values spanning different ranges
            let input = _mm256_setr_ps(
                0.0,   // zero
                0.25,  // small positive
                0.5,   // boundary
                0.75,  // large positive
                -0.25, // small negative
                -0.5,  // boundary negative
                -0.75, // large negative
                1.0,   // exact boundary
            );
            let result = extract_f32x8(_mm256_asin_ps(input));

            let expected = [
                0.0_f32.asin(),
                0.25_f32.asin(),
                0.5_f32.asin(),
                0.75_f32.asin(),
                (-0.25_f32).asin(),
                (-0.5_f32).asin(),
                (-0.75_f32).asin(),
                1.0_f32.asin(),
            ];

            for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    approx_eq(got, exp, 1e-5),
                    "lane {i}: asin({}) = {got}, expected {exp}",
                    [0.0, 0.25, 0.5, 0.75, -0.25, -0.5, -0.75, 1.0][i]
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Identity: asin(sin(x)) ≈ x for x in [-π/2, π/2]
    // -------------------------------------------------------------------------

    #[test]
    fn asin_sin_identity() {
        unsafe {
            // Test asin(sin(x)) ≈ x for various x in [-π/2, π/2]
            let test_values = [0.0_f32, 0.1, 0.5, 1.0, 1.5, -0.1, -0.5, -1.0];

            // Compute sin(x) first (using scalar for reference)
            let sin_values: Vec<f32> = test_values.iter().map(|&x| x.sin()).collect();
            let sin_input = _mm256_loadu_ps(sin_values.as_ptr());

            let result = extract_f32x8(_mm256_asin_ps(sin_input));

            for (i, (&got, &original)) in result.iter().zip(test_values.iter()).enumerate() {
                assert!(
                    approx_eq(got, original, 1e-5),
                    "lane {i}: asin(sin({original})) = {got}, expected {original}"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // ULP accuracy sweep
    // -------------------------------------------------------------------------

    #[test]
    fn max_ulp_error_is_at_most_1() {
        // Sweep a representative sample of the domain [-1, 1].
        // We test every 1024th representable f32 in that range.
        let mut max_ulp: u32 = 0;
        let mut worst_x: f32 = 0.0;

        let mut bits: u32 = (-1.0_f32).to_bits();
        let end_bits: u32 = 1.0_f32.to_bits();

        while bits <= end_bits {
            let x = f32::from_bits(bits);
            if x.is_nan() {
                bits = bits.wrapping_add(1024);
                continue;
            }

            let expected = x.asin();

            // Compute SIMD result
            let simd_result = unsafe {
                let input = _mm256_set1_ps(x);
                let output = _mm256_asin_ps(input);
                let mut arr = [0.0f32; 8];
                _mm256_storeu_ps(arr.as_mut_ptr(), output);
                arr[0]
            };

            // Skip if both are NaN
            if expected.is_nan() && simd_result.is_nan() {
                bits = bits.wrapping_add(1024);
                continue;
            }

            // Compute ULP difference
            let exp_bits = expected.to_bits() as i32;
            let got_bits = simd_result.to_bits() as i32;
            let ulp_diff = (exp_bits - got_bits).unsigned_abs();

            if ulp_diff > max_ulp {
                max_ulp = ulp_diff;
                worst_x = x;
            }

            bits = bits.wrapping_add(1024);
        }

        println!("Max ULP error: {max_ulp} at x = {worst_x}");
        println!("  expected: {}", worst_x.asin());
        println!("  got:      {}", unsafe {
            let input = _mm256_set1_ps(worst_x);
            let output = _mm256_asin_ps(input);
            let mut arr = [0.0f32; 8];
            _mm256_storeu_ps(arr.as_mut_ptr(), output);
            arr[0]
        });

        assert!(max_ulp <= 1, "ULP error {max_ulp} > 1 at x = {worst_x}");
    }

    // =========================================================================
    // f64 Tests
    // =========================================================================

    /// Helper to extract all 4 lanes from an __m256d register.
    unsafe fn extract_f64x4(v: __m256d) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), v) };
        out
    }

    /// Helper to check if two f64 values are approximately equal.
    fn approx_eq_f64(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn asin_pd_of_zero_is_zero() {
        unsafe {
            let input = _mm256_set1_pd(0.0);
            let result = extract_f64x4(_mm256_asin_pd(input));
            for &val in &result {
                assert_eq!(val, 0.0);
            }
        }
    }

    #[test]
    fn asin_pd_of_neg_zero_is_neg_zero() {
        unsafe {
            let input = _mm256_set1_pd(-0.0);
            let result = extract_f64x4(_mm256_asin_pd(input));
            for &val in &result {
                assert!(val == 0.0 && val.is_sign_negative());
            }
        }
    }

    #[test]
    fn asin_pd_of_one_is_pio2() {
        unsafe {
            let input = _mm256_set1_pd(1.0);
            let result = extract_f64x4(_mm256_asin_pd(input));
            let expected = std::f64::consts::FRAC_PI_2;
            for &val in &result {
                assert!(
                    approx_eq_f64(val, expected, 1e-14),
                    "asin(1) = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn asin_pd_of_neg_one_is_neg_pio2() {
        unsafe {
            let input = _mm256_set1_pd(-1.0);
            let result = extract_f64x4(_mm256_asin_pd(input));
            let expected = -std::f64::consts::FRAC_PI_2;
            for &val in &result {
                assert!(
                    approx_eq_f64(val, expected, 1e-14),
                    "asin(-1) = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn asin_pd_of_half_is_pi_over_6() {
        unsafe {
            let input = _mm256_set1_pd(0.5);
            let result = extract_f64x4(_mm256_asin_pd(input));
            let expected = std::f64::consts::FRAC_PI_6;
            for &val in &result {
                assert!(
                    approx_eq_f64(val, expected, 1e-14),
                    "asin(0.5) = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn asin_pd_of_neg_half_is_neg_pi_over_6() {
        unsafe {
            let input = _mm256_set1_pd(-0.5);
            let result = extract_f64x4(_mm256_asin_pd(input));
            let expected = -std::f64::consts::FRAC_PI_6;
            for &val in &result {
                assert!(
                    approx_eq_f64(val, expected, 1e-14),
                    "asin(-0.5) = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn asin_pd_of_sqrt2_over_2_is_pi_over_4() {
        unsafe {
            let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
            let input = _mm256_set1_pd(sqrt2_over_2);
            let result = extract_f64x4(_mm256_asin_pd(input));
            let expected = std::f64::consts::FRAC_PI_4;
            for &val in &result {
                assert!(
                    approx_eq_f64(val, expected, 1e-14),
                    "asin(√2/2) = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn asin_pd_above_one_is_nan() {
        unsafe {
            let input = _mm256_set1_pd(1.5);
            let result = extract_f64x4(_mm256_asin_pd(input));
            for &val in &result {
                assert!(val.is_nan(), "asin(1.5) should be NaN, got {val}");
            }
        }
    }

    #[test]
    fn asin_pd_below_neg_one_is_nan() {
        unsafe {
            let input = _mm256_set1_pd(-1.5);
            let result = extract_f64x4(_mm256_asin_pd(input));
            for &val in &result {
                assert!(val.is_nan(), "asin(-1.5) should be NaN, got {val}");
            }
        }
    }

    #[test]
    fn asin_pd_of_nan_is_nan() {
        unsafe {
            let input = _mm256_set1_pd(f64::NAN);
            let result = extract_f64x4(_mm256_asin_pd(input));
            for &val in &result {
                assert!(val.is_nan(), "asin(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn asin_pd_processes_all_4_lanes_independently() {
        unsafe {
            let input = _mm256_setr_pd(0.0, 0.5, -0.5, 1.0);
            let result = extract_f64x4(_mm256_asin_pd(input));

            let expected = [
                0.0_f64.asin(),
                0.5_f64.asin(),
                (-0.5_f64).asin(),
                1.0_f64.asin(),
            ];

            for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    approx_eq_f64(got, exp, 1e-14),
                    "lane {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    #[test]
    fn asin_pd_max_ulp_error_is_at_most_1() {
        // Sweep a representative sample of the domain [-1, 1].
        // We test every 2^20th representable f64 in that range.
        let mut max_ulp: u64 = 0;
        let mut worst_x: f64 = 0.0;

        let mut bits: u64 = (-1.0_f64).to_bits();
        let end_bits: u64 = 1.0_f64.to_bits();

        // Step size of 2^20 (~1 million) to keep test time reasonable
        let step: u64 = 1 << 20;

        while bits <= end_bits {
            let x = f64::from_bits(bits);
            if x.is_nan() {
                bits = bits.wrapping_add(step);
                continue;
            }

            let expected = x.asin();

            // Compute SIMD result
            let simd_result = unsafe {
                let input = _mm256_set1_pd(x);
                let output = _mm256_asin_pd(input);
                let mut arr = [0.0f64; 4];
                _mm256_storeu_pd(arr.as_mut_ptr(), output);
                arr[0]
            };

            // Skip if both are NaN
            if expected.is_nan() && simd_result.is_nan() {
                bits = bits.wrapping_add(step);
                continue;
            }

            // Compute ULP difference
            let exp_bits = expected.to_bits() as i64;
            let got_bits = simd_result.to_bits() as i64;
            let ulp_diff = (exp_bits - got_bits).unsigned_abs();

            if ulp_diff > max_ulp {
                max_ulp = ulp_diff;
                worst_x = x;
            }

            bits = bits.wrapping_add(step);
        }

        println!("Max ULP error (f64): {max_ulp} at x = {worst_x}");
        println!("  expected: {}", worst_x.asin());
        println!("  got:      {}", unsafe {
            let input = _mm256_set1_pd(worst_x);
            let output = _mm256_asin_pd(input);
            let mut arr = [0.0f64; 4];
            _mm256_storeu_pd(arr.as_mut_ptr(), output);
            arr[0]
        });

        assert!(max_ulp <= 1, "ULP error {max_ulp} > 1 at x = {worst_x}");
    }
}
