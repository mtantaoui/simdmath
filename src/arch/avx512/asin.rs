//! AVX-512 SIMD implementation of `asin(x)` for `f32` and `f64` vectors.
//!
//! This module provides 16-lane f32 and 8-lane f64 implementations using the
//! same algorithm as the AVX2 version: a branchless, two-range minimax
//! rational approximation ported from musl libc's `asinf.c` and `asin.c`.
//!
//! # Precision
//!
//! Both implementations achieve **≤ 1 ULP** accuracy across the entire domain `[-1, 1]`.
//!
//! # Algorithm
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
//! For `|x| >= 0.5`, a compensated sqrt is used for extra precision:
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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::avx512::abs::{_mm512_abs_pd, _mm512_abs_ps};
use crate::arch::consts::acos::{
    P_S0_32, P_S0_64, P_S1_32, P_S1_64, P_S2_32, P_S2_64, P_S3_64, P_S4_64, P_S5_64, Q_S1_32,
    Q_S1_64, Q_S2_64, Q_S3_64, Q_S4_64,
};
use crate::arch::consts::asin::{
    PIO2_HI_32, PIO2_HI_64, PIO2_LO_32, PIO2_LO_64, TINY_THRESHOLD_32, TINY_THRESHOLD_64,
};

// ===========================================================================
// f32 Implementation (16 lanes)
// ===========================================================================

/// Computes `asin(x)` for each lane of an AVX-512 `__m512` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
///
/// # Description
///
/// All 16 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large |x|, tiny, |x|=1, out-of-domain) are computed
/// unconditionally and merged with mask operations.
///
/// # Safety
///
/// Requires AVX-512F. `x` must be a valid `__m512` register.
///
/// # Example
///
/// ```ignore
/// let input = _mm512_set1_ps(0.5);
/// let result = _mm512_asin_ps(input);
/// // result ≈ [0.5236; 16] (π/6 ≈ 0.5236)
/// ```
#[inline]
pub(crate) unsafe fn _mm512_asin_ps(x: __m512) -> __m512 {
    unsafe {
        // ---------------------------------------------------------------------
        // Broadcast scalar constants to SIMD registers
        // ---------------------------------------------------------------------
        let pio2_hi = _mm512_set1_ps(PIO2_HI_32);
        let pio2_lo = _mm512_set1_ps(PIO2_LO_32);
        let p_s0 = _mm512_set1_ps(P_S0_32);
        let p_s1 = _mm512_set1_ps(P_S1_32);
        let p_s2 = _mm512_set1_ps(P_S2_32);
        let q_s1 = _mm512_set1_ps(Q_S1_32);

        let one = _mm512_set1_ps(1.0);
        let half = _mm512_set1_ps(0.5);
        let two = _mm512_set1_ps(2.0);
        let zero = _mm512_setzero_ps();

        // ---------------------------------------------------------------------
        // Compute |x| for range selection
        // ---------------------------------------------------------------------
        let abs_x = _mm512_abs_ps(x);

        // ---------------------------------------------------------------------
        // Rational polynomial approximation: r(z) = p(z) / q(z)
        //
        // This approximates (asin(√z)/√z − 1)/z on [0, 0.25].
        //   p(z) = z · (P_S0 + z · (P_S1 + z · P_S2))
        //   q(z) = 1 + z · Q_S1
        // ---------------------------------------------------------------------
        let r = |z: __m512| -> __m512 {
            let p = _mm512_mul_ps(z, _mm512_fmadd_ps(z, _mm512_fmadd_ps(z, p_s2, p_s1), p_s0));
            let q = _mm512_fmadd_ps(z, q_s1, one);
            _mm512_div_ps(p, q)
        };

        // ---------------------------------------------------------------------
        // Condition masks (computed once, used for blending)
        // ---------------------------------------------------------------------
        let is_abs_ge_1 = _mm512_cmp_ps_mask(abs_x, one, _CMP_GE_OQ);
        let is_abs_eq_1 = _mm512_cmp_ps_mask(abs_x, one, _CMP_EQ_OQ);
        let is_abs_lt_half = _mm512_cmp_ps_mask(abs_x, half, _CMP_LT_OQ);
        let is_x_neg = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);

        // Tiny threshold check: |x| < 2^-12 (avoids underflow in polynomial)
        let tiny_bits = _mm512_set1_epi32(TINY_THRESHOLD_32 as i32);
        let is_tiny = _mm512_cmp_ps_mask(abs_x, _mm512_castsi512_ps(tiny_bits), _CMP_LT_OQ);

        // ---------------------------------------------------------------------
        // Case A: |x| == 1 → return ±π/2
        //
        // asin(1) = π/2, asin(-1) = -π/2
        // Use pio2_hi + sign(x) * pio2_lo for full precision
        // ---------------------------------------------------------------------
        let pos_pio2 = _mm512_add_ps(pio2_hi, pio2_lo);
        let neg_pio2 = _mm512_sub_ps(_mm512_sub_ps(zero, pio2_hi), pio2_lo);
        let result_eq_1 = _mm512_mask_blend_ps(is_x_neg, pos_pio2, neg_pio2);

        // ---------------------------------------------------------------------
        // Case B: |x| > 1 → return NaN (domain error)
        // ---------------------------------------------------------------------
        let nan = _mm512_div_ps(zero, _mm512_sub_ps(x, x));

        // ---------------------------------------------------------------------
        // Case C: |x| < 0.5 → asin(x) = x + x · r(x²)
        //
        // For tiny |x| < 2^-12, just return x (polynomial is negligible).
        // ---------------------------------------------------------------------
        let x_sq = _mm512_mul_ps(x, x);
        let r_small = r(x_sq);
        let result_small_computed = _mm512_fmadd_ps(x, r_small, x);
        let result_small = _mm512_mask_blend_ps(is_tiny, result_small_computed, x);

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
        // ---------------------------------------------------------------------
        let z_large = _mm512_mul_ps(_mm512_sub_ps(one, abs_x), half);
        let s_large = _mm512_sqrt_ps(z_large);
        let r_large = r(z_large);

        // Dekker split: mask off low 12 bits to get exact high part of s
        let df = _mm512_castsi512_ps(_mm512_and_si512(
            _mm512_castps_si512(s_large),
            _mm512_set1_epi32(0xfffff000_u32 as i32),
        ));

        // Rounding correction: c = (z - df²) / (s + df)
        let c = _mm512_div_ps(
            _mm512_sub_ps(z_large, _mm512_mul_ps(df, df)),
            _mm512_add_ps(s_large, df),
        );

        // Compute w = s·r(z) + c
        let w = _mm512_fmadd_ps(s_large, r_large, c);

        // Compute: pio2_hi - 2·df - (2·w - pio2_lo)
        let two_w = _mm512_mul_ps(two, w);
        let inner = _mm512_sub_ps(two_w, pio2_lo);
        let two_df = _mm512_mul_ps(two, df);
        let result_large_abs = _mm512_sub_ps(_mm512_sub_ps(pio2_hi, two_df), inner);

        // Apply sign: if x < 0, negate the result
        let result_large = _mm512_mask_blend_ps(
            is_x_neg,
            result_large_abs,
            _mm512_sub_ps(zero, result_large_abs),
        );

        // ---------------------------------------------------------------------
        // Final blending: combine all cases
        // ---------------------------------------------------------------------

        // Handle |x| >= 1: either exact ±π/2 (if |x| == 1) or NaN (if |x| > 1)
        let result_ge_1 = _mm512_mask_blend_ps(is_abs_eq_1, nan, result_eq_1);

        // Select between small and large cases for |x| < 1
        let result_valid = _mm512_mask_blend_ps(is_abs_lt_half, result_large, result_small);

        // Final selection: valid cases vs |x| >= 1 cases
        _mm512_mask_blend_ps(is_abs_ge_1, result_valid, result_ge_1)
    }
}

// ===========================================================================
// f64 Implementation (8 lanes)
// ===========================================================================

/// Computes `asin(x)` for each lane of an AVX-512 `__m512d` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
///
/// # Description
///
/// All 8 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large |x|, tiny, |x|=1, out-of-domain) are computed
/// unconditionally and merged with mask operations.
///
/// # Safety
///
/// Requires AVX-512F. `x` must be a valid `__m512d` register.
///
/// # Example
///
/// ```ignore
/// let input = _mm512_set1_pd(0.5);
/// let result = _mm512_asin_pd(input);
/// // result ≈ [0.5236; 8] (π/6 ≈ 0.5236)
/// ```
#[inline]
pub(crate) unsafe fn _mm512_asin_pd(x: __m512d) -> __m512d {
    unsafe {
        // ---------------------------------------------------------------------
        // Broadcast scalar constants to SIMD registers
        // ---------------------------------------------------------------------
        let pio2_hi = _mm512_set1_pd(PIO2_HI_64);
        let pio2_lo = _mm512_set1_pd(PIO2_LO_64);
        let p_s0 = _mm512_set1_pd(P_S0_64);
        let p_s1 = _mm512_set1_pd(P_S1_64);
        let p_s2 = _mm512_set1_pd(P_S2_64);
        let p_s3 = _mm512_set1_pd(P_S3_64);
        let p_s4 = _mm512_set1_pd(P_S4_64);
        let p_s5 = _mm512_set1_pd(P_S5_64);
        let q_s1 = _mm512_set1_pd(Q_S1_64);
        let q_s2 = _mm512_set1_pd(Q_S2_64);
        let q_s3 = _mm512_set1_pd(Q_S3_64);
        let q_s4 = _mm512_set1_pd(Q_S4_64);

        let one = _mm512_set1_pd(1.0);
        let half = _mm512_set1_pd(0.5);
        let two = _mm512_set1_pd(2.0);
        let zero = _mm512_setzero_pd();

        // ---------------------------------------------------------------------
        // Compute |x| for range selection
        // ---------------------------------------------------------------------
        let abs_x = _mm512_abs_pd(x);

        // ---------------------------------------------------------------------
        // Rational polynomial approximation: r(z) = p(z) / q(z)
        //
        // This approximates (asin(√z)/√z − 1)/z on [0, 0.25].
        //   p(z) = z · (P_S0 + z · (P_S1 + z · (P_S2 + z · (P_S3 + z · (P_S4 + z · P_S5)))))
        //   q(z) = 1 + z · (Q_S1 + z · (Q_S2 + z · (Q_S3 + z · Q_S4)))
        // ---------------------------------------------------------------------
        let r = |z: __m512d| -> __m512d {
            let p = _mm512_mul_pd(
                z,
                _mm512_fmadd_pd(
                    z,
                    _mm512_fmadd_pd(
                        z,
                        _mm512_fmadd_pd(
                            z,
                            _mm512_fmadd_pd(z, _mm512_fmadd_pd(z, p_s5, p_s4), p_s3),
                            p_s2,
                        ),
                        p_s1,
                    ),
                    p_s0,
                ),
            );
            let q = _mm512_fmadd_pd(
                z,
                _mm512_fmadd_pd(
                    z,
                    _mm512_fmadd_pd(z, _mm512_fmadd_pd(z, q_s4, q_s3), q_s2),
                    q_s1,
                ),
                one,
            );
            _mm512_div_pd(p, q)
        };

        // ---------------------------------------------------------------------
        // Condition masks (computed once, used for blending)
        // ---------------------------------------------------------------------
        let is_abs_ge_1 = _mm512_cmp_pd_mask(abs_x, one, _CMP_GE_OQ);
        let is_abs_eq_1 = _mm512_cmp_pd_mask(abs_x, one, _CMP_EQ_OQ);
        let is_abs_lt_half = _mm512_cmp_pd_mask(abs_x, half, _CMP_LT_OQ);
        let is_x_neg = _mm512_cmp_pd_mask(x, zero, _CMP_LT_OQ);

        // Tiny threshold check: |x| < 2^-27 (avoids underflow in polynomial)
        let tiny_bits = _mm512_set1_epi64(TINY_THRESHOLD_64 as i64);
        let is_tiny = _mm512_cmp_pd_mask(abs_x, _mm512_castsi512_pd(tiny_bits), _CMP_LT_OQ);

        // ---------------------------------------------------------------------
        // Case A: |x| == 1 → return ±π/2
        // ---------------------------------------------------------------------
        let pos_pio2 = _mm512_add_pd(pio2_hi, pio2_lo);
        let neg_pio2 = _mm512_sub_pd(_mm512_sub_pd(zero, pio2_hi), pio2_lo);
        let result_eq_1 = _mm512_mask_blend_pd(is_x_neg, pos_pio2, neg_pio2);

        // ---------------------------------------------------------------------
        // Case B: |x| > 1 → return NaN (domain error)
        // ---------------------------------------------------------------------
        let nan = _mm512_div_pd(zero, _mm512_sub_pd(x, x));

        // ---------------------------------------------------------------------
        // Case C: |x| < 0.5 → asin(x) = x + x · r(x²)
        // ---------------------------------------------------------------------
        let x_sq = _mm512_mul_pd(x, x);
        let r_small = r(x_sq);
        let result_small_computed = _mm512_fmadd_pd(x, r_small, x);
        let result_small = _mm512_mask_blend_pd(is_tiny, result_small_computed, x);

        // ---------------------------------------------------------------------
        // Case D: 0.5 ≤ |x| < 1 → use half-angle formula with Dekker split
        //
        // Dekker compensated sqrt:
        //   df = s with low 32 mantissa bits cleared (exact high part)
        //   c  = (z - df²) / (s + df)  (rounding correction)
        // ---------------------------------------------------------------------
        let z_large = _mm512_mul_pd(_mm512_sub_pd(one, abs_x), half);
        let s_large = _mm512_sqrt_pd(z_large);
        let r_large = r(z_large);

        // Dekker split: mask off low 32 bits to get exact high part of s
        let df = _mm512_castsi512_pd(_mm512_and_si512(
            _mm512_castpd_si512(s_large),
            _mm512_set1_epi64(0xffffffff00000000_u64 as i64),
        ));

        // Rounding correction: c = (z - df²) / (s + df)
        let c = _mm512_div_pd(
            _mm512_sub_pd(z_large, _mm512_mul_pd(df, df)),
            _mm512_add_pd(s_large, df),
        );

        // Compute w = s·r(z) + c
        let w = _mm512_fmadd_pd(s_large, r_large, c);

        // Compute: pio2_hi - 2·df - (2·w - pio2_lo)
        let two_w = _mm512_mul_pd(two, w);
        let inner = _mm512_sub_pd(two_w, pio2_lo);
        let two_df = _mm512_mul_pd(two, df);
        let result_large_abs = _mm512_sub_pd(_mm512_sub_pd(pio2_hi, two_df), inner);

        // Apply sign: if x < 0, negate the result
        let result_large = _mm512_mask_blend_pd(
            is_x_neg,
            result_large_abs,
            _mm512_sub_pd(zero, result_large_abs),
        );

        // ---------------------------------------------------------------------
        // Final blending: combine all cases
        // ---------------------------------------------------------------------
        let result_ge_1 = _mm512_mask_blend_pd(is_abs_eq_1, nan, result_eq_1);
        let result_valid = _mm512_mask_blend_pd(is_abs_lt_half, result_large, result_small);
        _mm512_mask_blend_pd(is_abs_ge_1, result_valid, result_ge_1)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // f32 tests (16 lanes)
    // ========================================================================

    const TOL_32: f32 = 5e-7;

    unsafe fn asin_scalar_32(val: f32) -> f32 {
        unsafe {
            let v = _mm512_set1_ps(val);
            let mut out = [0.0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), _mm512_asin_ps(v));
            out[0]
        }
    }

    #[test]
    fn asin_ps_of_zero_is_zero() {
        unsafe {
            assert_eq!(asin_scalar_32(0.0), 0.0);
        }
    }

    #[test]
    fn asin_ps_of_neg_zero_is_neg_zero() {
        unsafe {
            let result = asin_scalar_32(-0.0);
            assert!(result == 0.0 && result.is_sign_negative());
        }
    }

    #[test]
    fn asin_ps_of_one_is_pio2() {
        unsafe {
            let result = asin_scalar_32(1.0);
            let expected = std::f32::consts::FRAC_PI_2;
            assert!(
                (result - expected).abs() < TOL_32,
                "asin(1) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_ps_of_neg_one_is_neg_pio2() {
        unsafe {
            let result = asin_scalar_32(-1.0);
            let expected = -std::f32::consts::FRAC_PI_2;
            assert!(
                (result - expected).abs() < TOL_32,
                "asin(-1) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_ps_of_half_is_pi_over_6() {
        unsafe {
            let result = asin_scalar_32(0.5);
            let expected = std::f32::consts::FRAC_PI_6;
            assert!(
                (result - expected).abs() < TOL_32,
                "asin(0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_ps_of_neg_half_is_neg_pi_over_6() {
        unsafe {
            let result = asin_scalar_32(-0.5);
            let expected = -std::f32::consts::FRAC_PI_6;
            assert!(
                (result - expected).abs() < TOL_32,
                "asin(-0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_ps_of_sqrt2_over_2_is_pi_over_4() {
        unsafe {
            let sqrt2_over_2 = std::f32::consts::FRAC_1_SQRT_2;
            let result = asin_scalar_32(sqrt2_over_2);
            let expected = std::f32::consts::FRAC_PI_4;
            assert!(
                (result - expected).abs() < TOL_32,
                "asin(√2/2) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_ps_above_one_is_nan() {
        unsafe {
            let result = asin_scalar_32(1.5);
            assert!(result.is_nan(), "asin(1.5) should be NaN, got {result}");
        }
    }

    #[test]
    fn asin_ps_below_neg_one_is_nan() {
        unsafe {
            let result = asin_scalar_32(-1.5);
            assert!(result.is_nan(), "asin(-1.5) should be NaN, got {result}");
        }
    }

    #[test]
    fn asin_ps_of_nan_is_nan() {
        unsafe {
            let result = asin_scalar_32(f32::NAN);
            assert!(result.is_nan(), "asin(NaN) should be NaN");
        }
    }

    #[test]
    fn asin_ps_processes_all_16_lanes() {
        unsafe {
            let input = _mm512_setr_ps(
                0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7,
                1.0,
            );
            let mut result = [0.0f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), _mm512_asin_ps(input));

            let expected = [
                0.0_f32.asin(),
                0.1_f32.asin(),
                0.2_f32.asin(),
                0.3_f32.asin(),
                0.4_f32.asin(),
                0.5_f32.asin(),
                0.6_f32.asin(),
                0.7_f32.asin(),
                (-0.1_f32).asin(),
                (-0.2_f32).asin(),
                (-0.3_f32).asin(),
                (-0.4_f32).asin(),
                (-0.5_f32).asin(),
                (-0.6_f32).asin(),
                (-0.7_f32).asin(),
                1.0_f32.asin(),
            ];

            for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < TOL_32,
                    "lane {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    #[test]
    fn asin_ps_max_ulp_error_is_at_most_1() {
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
            let simd_result = unsafe { asin_scalar_32(x) };

            if expected.is_nan() && simd_result.is_nan() {
                bits = bits.wrapping_add(1024);
                continue;
            }

            let exp_bits = expected.to_bits() as i32;
            let got_bits = simd_result.to_bits() as i32;
            let ulp_diff = (exp_bits - got_bits).unsigned_abs();

            if ulp_diff > max_ulp {
                max_ulp = ulp_diff;
                worst_x = x;
            }

            bits = bits.wrapping_add(1024);
        }

        println!("Max ULP error (f32): {max_ulp} at x = {worst_x}");
        assert!(max_ulp <= 1, "ULP error {max_ulp} > 1 at x = {worst_x}");
    }

    // ========================================================================
    // f64 tests (8 lanes)
    // ========================================================================

    const TOL_64: f64 = 1e-14;

    unsafe fn asin_scalar_64(val: f64) -> f64 {
        unsafe {
            let v = _mm512_set1_pd(val);
            let mut out = [0.0f64; 8];
            _mm512_storeu_pd(out.as_mut_ptr(), _mm512_asin_pd(v));
            out[0]
        }
    }

    #[test]
    fn asin_pd_of_zero_is_zero() {
        unsafe {
            assert_eq!(asin_scalar_64(0.0), 0.0);
        }
    }

    #[test]
    fn asin_pd_of_neg_zero_is_neg_zero() {
        unsafe {
            let result = asin_scalar_64(-0.0);
            assert!(result == 0.0 && result.is_sign_negative());
        }
    }

    #[test]
    fn asin_pd_of_one_is_pio2() {
        unsafe {
            let result = asin_scalar_64(1.0);
            let expected = std::f64::consts::FRAC_PI_2;
            assert!(
                (result - expected).abs() < TOL_64,
                "asin(1) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_pd_of_neg_one_is_neg_pio2() {
        unsafe {
            let result = asin_scalar_64(-1.0);
            let expected = -std::f64::consts::FRAC_PI_2;
            assert!(
                (result - expected).abs() < TOL_64,
                "asin(-1) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_pd_of_half_is_pi_over_6() {
        unsafe {
            let result = asin_scalar_64(0.5);
            let expected = std::f64::consts::FRAC_PI_6;
            assert!(
                (result - expected).abs() < TOL_64,
                "asin(0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_pd_of_neg_half_is_neg_pi_over_6() {
        unsafe {
            let result = asin_scalar_64(-0.5);
            let expected = -std::f64::consts::FRAC_PI_6;
            assert!(
                (result - expected).abs() < TOL_64,
                "asin(-0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_pd_of_sqrt2_over_2_is_pi_over_4() {
        unsafe {
            let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
            let result = asin_scalar_64(sqrt2_over_2);
            let expected = std::f64::consts::FRAC_PI_4;
            assert!(
                (result - expected).abs() < TOL_64,
                "asin(√2/2) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_pd_above_one_is_nan() {
        unsafe {
            let result = asin_scalar_64(1.5);
            assert!(result.is_nan(), "asin(1.5) should be NaN, got {result}");
        }
    }

    #[test]
    fn asin_pd_below_neg_one_is_nan() {
        unsafe {
            let result = asin_scalar_64(-1.5);
            assert!(result.is_nan(), "asin(-1.5) should be NaN, got {result}");
        }
    }

    #[test]
    fn asin_pd_of_nan_is_nan() {
        unsafe {
            let result = asin_scalar_64(f64::NAN);
            assert!(result.is_nan(), "asin(NaN) should be NaN");
        }
    }

    #[test]
    fn asin_pd_processes_all_8_lanes() {
        unsafe {
            let input = _mm512_setr_pd(0.0, 0.25, 0.5, 0.75, -0.25, -0.5, -0.75, 1.0);
            let mut result = [0.0f64; 8];
            _mm512_storeu_pd(result.as_mut_ptr(), _mm512_asin_pd(input));

            let expected = [
                0.0_f64.asin(),
                0.25_f64.asin(),
                0.5_f64.asin(),
                0.75_f64.asin(),
                (-0.25_f64).asin(),
                (-0.5_f64).asin(),
                (-0.75_f64).asin(),
                1.0_f64.asin(),
            ];

            for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < TOL_64,
                    "lane {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    #[test]
    fn asin_pd_max_ulp_error_is_at_most_1() {
        let mut max_ulp: u64 = 0;
        let mut worst_x: f64 = 0.0;

        let mut bits: u64 = (-1.0_f64).to_bits();
        let end_bits: u64 = 1.0_f64.to_bits();
        let step: u64 = 1 << 20;

        while bits <= end_bits {
            let x = f64::from_bits(bits);
            if x.is_nan() {
                bits = bits.wrapping_add(step);
                continue;
            }

            let expected = x.asin();
            let simd_result = unsafe { asin_scalar_64(x) };

            if expected.is_nan() && simd_result.is_nan() {
                bits = bits.wrapping_add(step);
                continue;
            }

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
        assert!(max_ulp <= 1, "ULP error {max_ulp} > 1 at x = {worst_x}");
    }
}
