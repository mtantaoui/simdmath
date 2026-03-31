//! AVX-512 SIMD implementation of `acos(x)` for `f32` and `f64` vectors.
//!
//! This module provides 16-lane f32 and 8-lane f64 implementations using the
//! same algorithm as the AVX2 version: a branchless, three-range minimax
//! rational approximation ported from musl libc's `acosf.c` and `acos.c`.
//!
//! AVX-512 provides masked operations which could enable more efficient
//! blending, but we use the same branchless approach as AVX2 for simplicity
//! and to maintain ≤1 ULP accuracy.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::avx512::abs::{_mm512_abs_pd, _mm512_abs_ps};
use crate::arch::consts::acos::{
    P_S0_32, P_S0_64, P_S1_32, P_S1_64, P_S2_32, P_S2_64, P_S3_64, P_S4_64, P_S5_64, PIO2_HI_32,
    PIO2_HI_64, PIO2_LO_32, PIO2_LO_64, Q_S1_32, Q_S1_64, Q_S2_64, Q_S3_64, Q_S4_64, X1P_120_32,
};

// ===========================================================================
// f32 Implementation (16 lanes)
// ===========================================================================

/// Computes `acos(x)` for each lane of an AVX-512 `__m512` register.
///
/// All 16 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large positive, large negative, |x|=1, out-of-domain)
/// are computed unconditionally and merged with mask operations.
///
/// # Safety
/// `x` must be a valid `__m512` register. No alignment or memory constraints.
#[inline]
pub(crate) unsafe fn _mm512_acos_ps(x: __m512) -> __m512 {
    unsafe {
        // Broadcast scalar constants to SIMD registers once.
        let pio2_hi = _mm512_set1_ps(PIO2_HI_32);
        let pio2_lo = _mm512_set1_ps(PIO2_LO_32);
        let p_s0 = _mm512_set1_ps(P_S0_32);
        let p_s1 = _mm512_set1_ps(P_S1_32);
        let p_s2 = _mm512_set1_ps(P_S2_32);
        let q_s1 = _mm512_set1_ps(Q_S1_32);
        let x1p_120 = _mm512_set1_ps(X1P_120_32);
        let one = _mm512_set1_ps(1.0);
        let half = _mm512_set1_ps(0.5);
        let two = _mm512_set1_ps(2.0);
        let zero = _mm512_setzero_ps();

        // |x| — used to determine which range each lane falls into.
        let abs_x = _mm512_abs_ps(x);

        // -------------------------------------------------------------------------
        // Padé rational approximation r(z)
        //
        //   p(z) = P_S0 + z·(P_S1 + z·P_S2)   [degree-2 numerator, Horner form]
        //   q(z) = 1 + z·Q_S1                  [degree-1 denominator]
        //   r(z) = z·p(z) / q(z)
        // -------------------------------------------------------------------------
        let r = |z: __m512| -> __m512 {
            let p = _mm512_mul_ps(z, _mm512_fmadd_ps(z, _mm512_fmadd_ps(z, p_s2, p_s1), p_s0));
            let q = _mm512_fmadd_ps(z, q_s1, one);
            _mm512_div_ps(p, q)
        };

        // -------------------------------------------------------------------------
        // Lane classification masks
        // -------------------------------------------------------------------------
        let is_abs_ge_1 = _mm512_cmp_ps_mask(abs_x, one, _CMP_GE_OQ);
        let is_abs_eq_1 = _mm512_cmp_ps_mask(abs_x, one, _CMP_EQ_OQ);
        let is_abs_lt_half = _mm512_cmp_ps_mask(abs_x, half, _CMP_LT_OQ);
        let is_x_neg = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);

        // Lanes with x in (-1, -0.5]: x is negative AND |x| >= 0.5.
        let is_x_neg_large = is_x_neg & !is_abs_lt_half;

        // -------------------------------------------------------------------------
        // Case A — |x| == 1  (exact boundary values)
        // -------------------------------------------------------------------------
        let pi_val = _mm512_add_ps(_mm512_mul_ps(two, pio2_hi), x1p_120);
        let result_eq_1 = _mm512_mask_blend_ps(is_x_neg, zero, pi_val);

        // -------------------------------------------------------------------------
        // Case B — |x| > 1  (out-of-domain → NaN)
        // -------------------------------------------------------------------------
        let nan = _mm512_div_ps(zero, _mm512_sub_ps(x, x));

        // For |x| >= 1: select exact result where |x| == 1, NaN otherwise.
        let result_ge_1 = _mm512_mask_blend_ps(is_abs_eq_1, nan, result_eq_1);

        // -------------------------------------------------------------------------
        // Case C — |x| < 0.5  (small argument)
        // -------------------------------------------------------------------------
        let z_small = _mm512_mul_ps(x, x);
        let result_small = _mm512_sub_ps(
            pio2_hi,
            _mm512_sub_ps(x, _mm512_fnmadd_ps(x, r(z_small), pio2_lo)),
        );

        // -------------------------------------------------------------------------
        // Case D — x < -0.5  (large negative argument)
        // -------------------------------------------------------------------------
        let z_neg = _mm512_mul_ps(_mm512_add_ps(one, x), half);
        let s_neg = _mm512_sqrt_ps(z_neg);
        let w_neg = _mm512_fmsub_ps(r(z_neg), s_neg, pio2_lo);
        let result_neg = _mm512_mul_ps(two, _mm512_sub_ps(pio2_hi, _mm512_add_ps(s_neg, w_neg)));

        // -------------------------------------------------------------------------
        // Case E — x > 0.5  (large positive argument with Dekker split)
        // -------------------------------------------------------------------------
        let z_pos = _mm512_mul_ps(_mm512_sub_ps(one, x), half);
        let s_pos = _mm512_sqrt_ps(z_pos);

        // Mask off the low 12 bits of the f32 bit representation to form df.
        let df = _mm512_castsi512_ps(_mm512_and_si512(
            _mm512_castps_si512(s_pos),
            _mm512_set1_epi32(0xfffff000_u32 as i32),
        ));
        let c_pos = _mm512_div_ps(
            _mm512_sub_ps(z_pos, _mm512_mul_ps(df, df)),
            _mm512_add_ps(s_pos, df),
        );
        let w_pos = _mm512_fmadd_ps(r(z_pos), s_pos, c_pos);
        let result_pos = _mm512_mul_ps(two, _mm512_add_ps(df, w_pos));

        // -------------------------------------------------------------------------
        // Merge: blend all cases using masks
        // -------------------------------------------------------------------------
        let result_large = _mm512_mask_blend_ps(is_x_neg_large, result_pos, result_neg);
        let result_valid = _mm512_mask_blend_ps(is_abs_lt_half, result_large, result_small);
        _mm512_mask_blend_ps(is_abs_ge_1, result_valid, result_ge_1)
    }
}

// ===========================================================================
// f64 Implementation (8 lanes)
// ===========================================================================

/// Computes `acos(x)` for each lane of an AVX-512 `__m512d` register.
///
/// All 8 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large positive, large negative, |x|=1, out-of-domain)
/// are computed unconditionally and merged with mask operations.
///
/// # Safety
/// `x` must be a valid `__m512d` register. No alignment or memory constraints.
#[inline]
pub(crate) unsafe fn _mm512_acos_pd(x: __m512d) -> __m512d {
    unsafe {
        // Broadcast scalar constants to SIMD registers once.
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

        // |x| — used to determine which range each lane falls into.
        let abs_x = _mm512_abs_pd(x);

        // -------------------------------------------------------------------------
        // Padé rational approximation r(z) - degree-5/degree-4 for f64
        // -------------------------------------------------------------------------
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

        // -------------------------------------------------------------------------
        // Lane classification masks
        // -------------------------------------------------------------------------
        let is_abs_ge_1 = _mm512_cmp_pd_mask(abs_x, one, _CMP_GE_OQ);
        let is_abs_eq_1 = _mm512_cmp_pd_mask(abs_x, one, _CMP_EQ_OQ);
        let is_abs_lt_half = _mm512_cmp_pd_mask(abs_x, half, _CMP_LT_OQ);
        let is_x_neg = _mm512_cmp_pd_mask(x, zero, _CMP_LT_OQ);

        // Lanes with x in (-1, -0.5]: x is negative AND |x| >= 0.5.
        let is_x_neg_large = is_x_neg & !is_abs_lt_half;

        // -------------------------------------------------------------------------
        // Case A — |x| == 1  (exact boundary values)
        // -------------------------------------------------------------------------
        let pi_val = _mm512_mul_pd(two, pio2_hi);
        let result_eq_1 = _mm512_mask_blend_pd(is_x_neg, zero, pi_val);

        // -------------------------------------------------------------------------
        // Case B — |x| > 1  (out-of-domain → NaN)
        // -------------------------------------------------------------------------
        let nan = _mm512_div_pd(zero, _mm512_sub_pd(x, x));

        // For |x| >= 1: select exact result where |x| == 1, NaN otherwise.
        let result_ge_1 = _mm512_mask_blend_pd(is_abs_eq_1, nan, result_eq_1);

        // -------------------------------------------------------------------------
        // Case C — |x| < 0.5  (small argument)
        // -------------------------------------------------------------------------
        let z_small = _mm512_mul_pd(x, x);
        let result_small = _mm512_sub_pd(
            pio2_hi,
            _mm512_sub_pd(x, _mm512_fnmadd_pd(x, r(z_small), pio2_lo)),
        );

        // -------------------------------------------------------------------------
        // Case D — x < -0.5  (large negative argument)
        // -------------------------------------------------------------------------
        let z_neg = _mm512_mul_pd(_mm512_add_pd(one, x), half);
        let s_neg = _mm512_sqrt_pd(z_neg);
        let w_neg = _mm512_fmsub_pd(r(z_neg), s_neg, pio2_lo);
        let result_neg = _mm512_mul_pd(two, _mm512_sub_pd(pio2_hi, _mm512_add_pd(s_neg, w_neg)));

        // -------------------------------------------------------------------------
        // Case E — x > 0.5  (large positive argument with Dekker split)
        // -------------------------------------------------------------------------
        let z_pos = _mm512_mul_pd(_mm512_sub_pd(one, x), half);
        let s_pos = _mm512_sqrt_pd(z_pos);

        // Mask off the low 32 bits of the f64 bit representation to form df.
        let df = _mm512_castsi512_pd(_mm512_and_si512(
            _mm512_castpd_si512(s_pos),
            _mm512_set1_epi64(0xffffffff00000000_u64 as i64),
        ));
        let c_pos = _mm512_div_pd(
            _mm512_sub_pd(z_pos, _mm512_mul_pd(df, df)),
            _mm512_add_pd(s_pos, df),
        );
        let w_pos = _mm512_fmadd_pd(r(z_pos), s_pos, c_pos);
        let result_pos = _mm512_mul_pd(two, _mm512_add_pd(df, w_pos));

        // -------------------------------------------------------------------------
        // Merge: blend all cases using masks
        // -------------------------------------------------------------------------
        let result_large = _mm512_mask_blend_pd(is_x_neg_large, result_pos, result_neg);
        let result_valid = _mm512_mask_blend_pd(is_abs_lt_half, result_large, result_small);
        _mm512_mask_blend_pd(is_abs_ge_1, result_valid, result_ge_1)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // f32 tests
    // ========================================================================

    const TOL_32: f32 = 5e-7;

    unsafe fn acos_scalar_32(val: f32) -> f32 {
        unsafe {
            let v = _mm512_set1_ps(val);
            let mut out = [0.0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), _mm512_acos_ps(v));
            out[0]
        }
    }

    #[test]
    fn acos_ps_of_one_is_zero() {
        unsafe {
            assert_eq!(acos_scalar_32(1.0), 0.0);
        }
    }

    #[test]
    fn acos_ps_of_neg_one_is_pi() {
        unsafe {
            let result = acos_scalar_32(-1.0);
            assert!(
                (result - std::f32::consts::PI).abs() < TOL_32,
                "acos(-1) = {result}, expected π"
            );
        }
    }

    #[test]
    fn acos_ps_of_zero_is_pio2() {
        unsafe {
            let result = acos_scalar_32(0.0);
            assert!(
                (result - std::f32::consts::FRAC_PI_2).abs() < TOL_32,
                "acos(0) = {result}"
            );
        }
    }

    #[test]
    fn acos_ps_of_half_is_pi_over_3() {
        unsafe {
            let result = acos_scalar_32(0.5);
            let expected = std::f32::consts::PI / 3.0;
            assert!(
                (result - expected).abs() < TOL_32,
                "acos(0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn acos_ps_above_one_is_nan() {
        unsafe {
            assert!(acos_scalar_32(1.5).is_nan());
        }
    }

    #[test]
    fn acos_ps_of_nan_is_nan() {
        unsafe {
            assert!(acos_scalar_32(f32::NAN).is_nan());
        }
    }

    #[test]
    fn acos_ps_processes_all_16_lanes() {
        let inputs: [f32; 16] = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, 0.9,
        ];
        unsafe {
            let v = _mm512_loadu_ps(inputs.as_ptr());
            let mut out = [0.0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), _mm512_acos_ps(v));

            let expected: [f32; 16] = inputs.map(|x| x.acos());
            for (i, (&r, &e)) in out.iter().zip(&expected).enumerate() {
                assert!(
                    (r - e).abs() < TOL_32,
                    "lane {i}: acos({}) = {r}, expected {e}",
                    inputs[i]
                );
            }
        }
    }

    // ========================================================================
    // f64 tests
    // ========================================================================

    const TOL_64: f64 = 1e-15;

    unsafe fn acos_scalar_64(val: f64) -> f64 {
        unsafe {
            let v = _mm512_set1_pd(val);
            let mut out = [0.0f64; 8];
            _mm512_storeu_pd(out.as_mut_ptr(), _mm512_acos_pd(v));
            out[0]
        }
    }

    #[test]
    fn acos_pd_of_one_is_zero() {
        unsafe {
            assert_eq!(acos_scalar_64(1.0), 0.0);
        }
    }

    #[test]
    fn acos_pd_of_neg_one_is_pi() {
        unsafe {
            let result = acos_scalar_64(-1.0);
            assert!(
                (result - std::f64::consts::PI).abs() < TOL_64,
                "acos(-1) = {result}, expected π"
            );
        }
    }

    #[test]
    fn acos_pd_of_zero_is_pio2() {
        unsafe {
            let result = acos_scalar_64(0.0);
            assert!(
                (result - std::f64::consts::FRAC_PI_2).abs() < TOL_64,
                "acos(0) = {result}"
            );
        }
    }

    #[test]
    fn acos_pd_of_half_is_pi_over_3() {
        unsafe {
            let result = acos_scalar_64(0.5);
            let expected = std::f64::consts::PI / 3.0;
            assert!(
                (result - expected).abs() < TOL_64,
                "acos(0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn acos_pd_above_one_is_nan() {
        unsafe {
            assert!(acos_scalar_64(1.5).is_nan());
        }
    }

    #[test]
    fn acos_pd_of_nan_is_nan() {
        unsafe {
            assert!(acos_scalar_64(f64::NAN).is_nan());
        }
    }

    #[test]
    fn acos_pd_processes_all_8_lanes() {
        let inputs: [f64; 8] = [0.0, 0.5, -0.5, 0.9, -0.9, 0.25, -0.25, 0.75];
        unsafe {
            let v = _mm512_loadu_pd(inputs.as_ptr());
            let mut out = [0.0f64; 8];
            _mm512_storeu_pd(out.as_mut_ptr(), _mm512_acos_pd(v));

            let expected: [f64; 8] = inputs.map(|x| x.acos());
            for (i, (&r, &e)) in out.iter().zip(&expected).enumerate() {
                assert!(
                    (r - e).abs() < TOL_64,
                    "lane {i}: acos({}) = {r}, expected {e}",
                    inputs[i]
                );
            }
        }
    }
}
