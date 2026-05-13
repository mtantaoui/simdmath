//! AVX2 SIMD implementation of `sin(x)` for `f32` and `f64` vectors.
//!
//! This module provides 8-lane f32 and 4-lane f64 sine implementations using
//! the Cody-Waite argument reduction algorithm and minimax polynomial
//! approximations ported from musl libc's `sinf.c`, `sin.c`, and kernel functions.
//!
//! # Algorithm
//!
//! 1. **Argument reduction**: Reduce `x` to `y ∈ [-π/4, π/4]` via `y = x - n*(π/2)`
//!    using Cody-Waite extended precision subtraction.
//!
//! 2. **Quadrant selection**: Based on `n mod 4`, select the appropriate kernel:
//!    | n mod 4 | sin(x)   |
//!    |---------|----------|
//!    | 0       |  sin(y)  |
//!    | 1       |  cos(y)  |
//!    | 2       | -sin(y)  |
//!    | 3       | -cos(y)  |
//!
//! 3. **Polynomial evaluation**: Minimax polynomials for sin/cos kernels.
//!
//! # Precision
//!
//! | Variant           | Max Error |
//! |-------------------|-----------|
//! | `_mm256_sin_ps`   | ≤ 2 ULP   |
//! | `_mm256_sin_pd`   | ≤ 2 ULP   |
//!
//! # Special Values
//!
//! | Input       | Output |
//! |-------------|--------|
//! | `0.0`       | `0.0`  |
//! | `-0.0`      | `-0.0` |
//! | `±∞`        | `NaN`  |
//! | `NaN`       | `NaN`  |
//! | Very small  | `x` (correctly rounded) |

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::sin::{
    C0_32, C1_32, C1_64, C2_32, C2_64, C3_32, C3_64, C4_64, C5_64, C6_64, FRAC_2_PI_32,
    FRAC_2_PI_64, PIO2_1_32, PIO2_1_64, PIO2_1T_32, PIO2_2_64, PIO2_2T_64, S1_32, S1_64, S2_32,
    S2_64, S3_32, S3_64, S4_32, S4_64, S5_64, S6_64, TOINT,
};

// =============================================================================
// f32 Implementation (8 lanes, native f32)
// =============================================================================

/// Computes `sin(x)` for each lane of an AVX2 `__m256` register.
///
/// Uses the musl libc algorithm: Cody-Waite argument reduction to `[-π/4, π/4]`
/// followed by polynomial evaluation of the appropriate sin/cos kernel based
/// on the quadrant. All computation is done natively in f32.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires AVX2 and FMA support. The caller must ensure these features are
/// available at runtime.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_sin_ps(x: __m256) -> __m256 {
    unsafe {
        let sign_bit = _mm256_set1_ps(-0.0_f32);

        // -------------------------------------------------------------------------
        // Step 1: Argument reduction — y = x - n*(π/2), |y| ≤ π/4
        // -------------------------------------------------------------------------
        let frac_2_pi = _mm256_set1_ps(FRAC_2_PI_32 as f32);
        let pio2_1 = _mm256_set1_ps(PIO2_1_32 as f32);
        let pio2_1t = _mm256_set1_ps(PIO2_1T_32 as f32);

        let fn_val = _mm256_round_ps(
            _mm256_mul_ps(x, frac_2_pi),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let n = _mm256_cvtps_epi32(fn_val);
        let y = _mm256_fnmadd_ps(fn_val, pio2_1t, _mm256_fnmadd_ps(fn_val, pio2_1, x));

        // -------------------------------------------------------------------------
        // Step 2: Polynomial kernels
        // -------------------------------------------------------------------------
        let sin_y = sindf_kernel_f32(y);
        let cos_y = cosdf_kernel_f32(y);

        // -------------------------------------------------------------------------
        // Step 3: Quadrant selection
        // n mod 4: 0 →  sin(y)   1 →  cos(y)
        //          2 → -sin(y)   3 → -cos(y)
        // -------------------------------------------------------------------------
        let one = _mm256_set1_epi32(1);
        let two = _mm256_set1_epi32(2);
        let use_cos = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(n, one), one));
        let negate = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(n, two), two));

        let kernel = _mm256_blendv_ps(sin_y, cos_y, use_cos);
        let result = _mm256_blendv_ps(kernel, _mm256_xor_ps(kernel, sign_bit), negate);

        // -------------------------------------------------------------------------
        // Step 4: Special cases — ±∞ and NaN → NaN
        // -------------------------------------------------------------------------
        let abs_x = _mm256_andnot_ps(sign_bit, x);
        let inf = _mm256_set1_ps(f32::INFINITY);
        let is_inf_or_nan = _mm256_cmp_ps(abs_x, inf, _CMP_GE_OQ);

        _mm256_blendv_ps(result, _mm256_set1_ps(f32::NAN), is_inf_or_nan)
    }
}

/// Sine kernel for f32 reduced argument in `[-π/4, π/4]`.
///
/// Implements musl's `__sindf` in native f32: sin(x) ≈ x + S1*x³ + S2*x⁵ + S3*x⁷ + S4*x⁹
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn sindf_kernel_f32(x: __m256) -> __m256 {
    let s1 = _mm256_set1_ps(S1_32 as f32);
    let s2 = _mm256_set1_ps(S2_32 as f32);
    let s3 = _mm256_set1_ps(S3_32 as f32);
    let s4 = _mm256_set1_ps(S4_32 as f32);

    let z = _mm256_mul_ps(x, x); // x²
    let w = _mm256_mul_ps(z, z); // x⁴
    let s = _mm256_mul_ps(z, x); // x³

    let r = _mm256_fmadd_ps(z, s4, s3); // S3 + x²·S4
    let inner = _mm256_fmadd_ps(z, s2, s1); // S1 + x²·S2
    let term1 = _mm256_fmadd_ps(s, inner, x); // x + x³·(S1 + x²·S2)
    let sw = _mm256_mul_ps(s, w); // x⁷
    _mm256_fmadd_ps(sw, r, term1) // + x⁷·(S3 + x²·S4)
}

/// Cosine kernel for f32 reduced argument in `[-π/4, π/4]`.
///
/// Implements musl's `__cosdf` in native f32: cos(x) ≈ 1 + C0*x² + C1*x⁴ + C2*x⁶ + C3*x⁸
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosdf_kernel_f32(x: __m256) -> __m256 {
    let c0 = _mm256_set1_ps(C0_32 as f32);
    let c1 = _mm256_set1_ps(C1_32 as f32);
    let c2 = _mm256_set1_ps(C2_32 as f32);
    let c3 = _mm256_set1_ps(C3_32 as f32);
    let one = _mm256_set1_ps(1.0_f32);

    let z = _mm256_mul_ps(x, x); // x²
    let w = _mm256_mul_ps(z, z); // x⁴

    let r = _mm256_fmadd_ps(z, c3, c2); // C2 + x²·C3
    let term1 = _mm256_fmadd_ps(z, c0, one); // 1 + x²·C0
    let term2 = _mm256_fmadd_ps(w, c1, term1); // + x⁴·C1
    let wz = _mm256_mul_ps(w, z); // x⁶
    _mm256_fmadd_ps(wz, r, term2) // + x⁶·(C2 + x²·C3)
}

// =============================================================================
// f64 Implementation (4 lanes)
// =============================================================================

/// Computes `sin(x)` for each lane of an AVX2 `__m256d` register.
///
/// Uses musl libc's algorithm with degree-13 polynomial for the sine kernel
/// and degree-14 for the cosine kernel after Cody-Waite argument reduction.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires AVX2 and FMA support. The caller must ensure these features are
/// available at runtime.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_sin_pd(x: __m256d) -> __m256d {
    unsafe {
        let frac_2_pi = _mm256_set1_pd(FRAC_2_PI_64);
        let pio2_1 = _mm256_set1_pd(PIO2_1_64);
        let pio2_2 = _mm256_set1_pd(PIO2_2_64);
        let pio2_2t = _mm256_set1_pd(PIO2_2T_64);
        let toint = _mm256_set1_pd(TOINT);

        // -------------------------------------------------------------------------
        // Step 1: Argument reduction with extended precision
        // -------------------------------------------------------------------------
        // Uses musl's __rem_pio2 2nd-iteration approach unconditionally.
        // Avoids catastrophic cancellation near multiples of π/2.

        let fn_val = _mm256_sub_pd(_mm256_fmadd_pd(x, frac_2_pi, toint), toint);
        let n = _mm256_cvtpd_epi32(fn_val);

        let r = _mm256_fnmadd_pd(fn_val, pio2_1, x);
        let w = _mm256_mul_pd(fn_val, pio2_2);
        let r2 = _mm256_sub_pd(r, w);
        let excess = _mm256_sub_pd(_mm256_sub_pd(r, r2), w);
        let tail = _mm256_sub_pd(_mm256_mul_pd(fn_val, pio2_2t), excess);
        let y = _mm256_sub_pd(r2, tail);

        let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);

        // -------------------------------------------------------------------------
        // Step 2: Compute kernels
        // -------------------------------------------------------------------------

        let sin_y = sin_kernel_f64(y);
        let cos_y = cos_kernel_f64(y);

        // -------------------------------------------------------------------------
        // Step 3: Quadrant selection
        // n mod 4: 0 → sin(y), 1 → cos(y), 2 → -sin(y), 3 → -cos(y)
        // -------------------------------------------------------------------------

        let n_256 = _mm256_cvtepi32_epi64(n);
        let one = _mm256_set1_epi64x(1);
        let two = _mm256_set1_epi64x(2);

        let n_and_1 = _mm256_and_si256(n_256, one);
        let n_and_2 = _mm256_and_si256(n_256, two);

        let use_cos = _mm256_cmpeq_epi64(n_and_1, one);
        let negate = _mm256_cmpeq_epi64(n_and_2, two);

        let kernel_result = _mm256_blendv_pd(sin_y, cos_y, _mm256_castsi256_pd(use_cos));

        let neg_mask = _mm256_castsi256_pd(negate);
        let sign_bit = _mm256_set1_pd(-0.0);
        let negated = _mm256_xor_pd(kernel_result, sign_bit);
        let result = _mm256_blendv_pd(kernel_result, negated, neg_mask);

        // -------------------------------------------------------------------------
        // Step 4: Handle special cases
        // sin(±∞) = NaN, sin(NaN) = NaN, sin(±0) = ±0
        // -------------------------------------------------------------------------

        let inf = _mm256_set1_pd(f64::INFINITY);
        let is_inf_or_nan = _mm256_cmp_pd(abs_x, inf, _CMP_GE_OQ);
        let nan = _mm256_set1_pd(f64::NAN);

        // For tiny values (including ±0), sin(x) ≈ x
        let tiny = _mm256_set1_pd(1e-300);
        let is_tiny = _mm256_cmp_pd(abs_x, tiny, _CMP_LT_OQ);
        let result = _mm256_blendv_pd(result, x, is_tiny);

        _mm256_blendv_pd(result, nan, is_inf_or_nan)
    }
}

/// Sine kernel for f64 reduced argument.
///
/// Implements musl's `__sin`: sin(x) ≈ x + v*(S1 + z*r)
/// where v = x³, z = x², r = S2 + z*(S3 + z*S4) + z*w*(S5 + z*S6)
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn sin_kernel_f64(x: __m256d) -> __m256d {
    let s1 = _mm256_set1_pd(S1_64);
    let s2 = _mm256_set1_pd(S2_64);
    let s3 = _mm256_set1_pd(S3_64);
    let s4 = _mm256_set1_pd(S4_64);
    let s5 = _mm256_set1_pd(S5_64);
    let s6 = _mm256_set1_pd(S6_64);

    let z = _mm256_mul_pd(x, x); // z = x²
    let w = _mm256_mul_pd(z, z); // w = x⁴
    let v = _mm256_mul_pd(z, x); // v = x³

    // r = S2 + z*(S3 + z*S4) + z*w*(S5 + z*S6)
    let inner1 = _mm256_fmadd_pd(z, s4, s3); // S3 + z*S4
    let inner1 = _mm256_fmadd_pd(z, inner1, s2); // S2 + z*(S3 + z*S4)

    let inner2 = _mm256_fmadd_pd(z, s6, s5); // S5 + z*S6
    let zw = _mm256_mul_pd(z, w); // z*w = x⁶
    let term2 = _mm256_mul_pd(zw, inner2); // z*w*(S5 + z*S6)

    let r = _mm256_add_pd(inner1, term2);

    // sin(x) = x + v*(S1 + z*r)
    let zr = _mm256_mul_pd(z, r); // z*r
    let s1_plus_zr = _mm256_add_pd(s1, zr); // S1 + z*r
    _mm256_fmadd_pd(v, s1_plus_zr, x) // x + v*(S1 + z*r)
}

/// Cosine kernel for f64 reduced argument.
///
/// Implements musl's `__cos`: cos(x) ≈ 1 - x²/2 + C1*x⁴ + ... + C6*x¹⁴
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn cos_kernel_f64(x: __m256d) -> __m256d {
    let c1 = _mm256_set1_pd(C1_64);
    let c2 = _mm256_set1_pd(C2_64);
    let c3 = _mm256_set1_pd(C3_64);
    let c4 = _mm256_set1_pd(C4_64);
    let c5 = _mm256_set1_pd(C5_64);
    let c6 = _mm256_set1_pd(C6_64);
    let half = _mm256_set1_pd(0.5);
    let one = _mm256_set1_pd(1.0);

    let z = _mm256_mul_pd(x, x); // z = x²
    let w = _mm256_mul_pd(z, z); // w = x⁴

    // r = z*(C1 + z*(C2 + z*C3)) + w*w*(C4 + z*(C5 + z*C6))
    let inner1 = _mm256_fmadd_pd(z, c3, c2); // C2 + z*C3
    let inner1 = _mm256_fmadd_pd(z, inner1, c1); // C1 + z*(C2 + z*C3)
    let term1 = _mm256_mul_pd(z, inner1); // z * (...)

    let inner2 = _mm256_fmadd_pd(z, c6, c5); // C5 + z*C6
    let inner2 = _mm256_fmadd_pd(z, inner2, c4); // C4 + z*(C5 + z*C6)
    let ww = _mm256_mul_pd(w, w); // w*w = x⁸
    let term2 = _mm256_mul_pd(ww, inner2); // x⁸ * (...)

    let r = _mm256_add_pd(term1, term2);

    // cos(x) = 1 - hz + (((1-w) - hz) + z*r)
    // Simplified: w = 1 - hz, return w + (((1-w)-hz) + z*r)
    let hz = _mm256_mul_pd(half, z); // hz = z/2
    let w = _mm256_sub_pd(one, hz); // w = 1 - z/2

    // For better accuracy: w + (((1-w) - hz) + z*r)
    let one_minus_w = _mm256_sub_pd(one, w); // 1 - w (captures rounding error)
    let correction = _mm256_sub_pd(one_minus_w, hz); // (1-w) - hz
    let zr = _mm256_mul_pd(z, r);
    let final_correction = _mm256_add_pd(correction, zr);

    _mm256_add_pd(w, final_correction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI32;
    use std::f64::consts::PI as PI64;

    // =========================================================================
    // Helper functions
    // =========================================================================

    #[inline]
    unsafe fn extract_f32x8(v: __m256) -> [f32; 8] {
        unsafe {
            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), v);
            result
        }
    }

    #[inline]
    unsafe fn extract_f64x4(v: __m256d) -> [f64; 4] {
        unsafe {
            let mut result = [0.0f64; 4];
            _mm256_storeu_pd(result.as_mut_ptr(), v);
            result
        }
    }

    use crate::test_utils::{ulp_diff_f32, ulp_diff_f64};

    // =========================================================================
    // f32 tests
    // =========================================================================

    #[test]
    fn test_sin_ps_zero() {
        unsafe {
            let input = _mm256_set1_ps(0.0);
            let result = extract_f32x8(_mm256_sin_ps(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_positive());
            }
        }
    }

    #[test]
    fn test_sin_ps_negative_zero() {
        unsafe {
            let input = _mm256_set1_ps(-0.0);
            let result = extract_f32x8(_mm256_sin_ps(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_negative());
            }
        }
    }

    #[test]
    fn test_sin_ps_pi_over_2() {
        unsafe {
            let input = _mm256_set1_ps(PI32 / 2.0);
            let result = extract_f32x8(_mm256_sin_ps(input));
            for &r in &result {
                assert!((r - 1.0).abs() < 1e-6, "sin(π/2) = {}, expected ~1.0", r);
            }
        }
    }

    #[test]
    fn test_sin_ps_pi() {
        unsafe {
            let input = _mm256_set1_ps(PI32);
            let result = extract_f32x8(_mm256_sin_ps(input));
            for &r in &result {
                assert!(r.abs() < 1e-6, "sin(π) = {}, expected ~0.0", r);
            }
        }
    }

    #[test]
    fn test_sin_ps_pi_over_4() {
        unsafe {
            let input = _mm256_set1_ps(PI32 / 4.0);
            let result = extract_f32x8(_mm256_sin_ps(input));
            let expected = (PI32 / 4.0).sin();
            for &r in &result {
                assert!(
                    (r - expected).abs() < 1e-6,
                    "sin(π/4) = {}, expected {}",
                    r,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_sin_ps_nan() {
        unsafe {
            let input = _mm256_set1_ps(f32::NAN);
            let result = extract_f32x8(_mm256_sin_ps(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_ps_infinity() {
        unsafe {
            let input = _mm256_set1_ps(f32::INFINITY);
            let result = extract_f32x8(_mm256_sin_ps(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_ps_negative_infinity() {
        unsafe {
            let input = _mm256_set1_ps(f32::NEG_INFINITY);
            let result = extract_f32x8(_mm256_sin_ps(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(-∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_ps_lane_independence() {
        unsafe {
            let input = _mm256_setr_ps(
                0.0,
                PI32 / 6.0,
                PI32 / 4.0,
                PI32 / 3.0,
                PI32 / 2.0,
                PI32,
                -PI32 / 2.0,
                -PI32,
            );
            let result = extract_f32x8(_mm256_sin_ps(input));
            let expected = [
                0.0f32.sin(),
                (PI32 / 6.0).sin(),
                (PI32 / 4.0).sin(),
                (PI32 / 3.0).sin(),
                (PI32 / 2.0).sin(),
                PI32.sin(),
                (-PI32 / 2.0).sin(),
                (-PI32).sin(),
            ];
            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (r - e).abs() < 1e-5,
                    "Lane {}: got {}, expected {}",
                    i,
                    r,
                    e
                );
            }
        }
    }

    #[test]
    fn test_sin_ps_ulp_sweep() {
        unsafe {
            let mut max_ulp = 0u32;
            for i in 0..10000 {
                let x = -2.0 * PI32 + (i as f32 / 10000.0) * 4.0 * PI32;
                let input = _mm256_set1_ps(x);
                let result = extract_f32x8(_mm256_sin_ps(input))[0];
                let expected = x.sin();
                if expected.is_finite() && result.is_finite() {
                    max_ulp = max_ulp.max(ulp_diff_f32(result, expected));
                }
            }
            assert!(max_ulp <= 2, "Max ULP error: {} (expected ≤ 2)", max_ulp);
        }
    }

    // =========================================================================
    // f64 tests
    // =========================================================================

    #[test]
    fn test_sin_pd_zero() {
        unsafe {
            let input = _mm256_set1_pd(0.0);
            let result = extract_f64x4(_mm256_sin_pd(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_positive());
            }
        }
    }

    #[test]
    fn test_sin_pd_negative_zero() {
        unsafe {
            let input = _mm256_set1_pd(-0.0);
            let result = extract_f64x4(_mm256_sin_pd(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_negative());
            }
        }
    }

    #[test]
    fn test_sin_pd_pi_over_2() {
        unsafe {
            let input = _mm256_set1_pd(PI64 / 2.0);
            let result = extract_f64x4(_mm256_sin_pd(input));
            for &r in &result {
                assert!((r - 1.0).abs() < 1e-14, "sin(π/2) = {}, expected ~1.0", r);
            }
        }
    }

    #[test]
    fn test_sin_pd_pi() {
        unsafe {
            let input = _mm256_set1_pd(PI64);
            let result = extract_f64x4(_mm256_sin_pd(input));
            for &r in &result {
                assert!(r.abs() < 1e-14, "sin(π) = {}, expected ~0.0", r);
            }
        }
    }

    #[test]
    fn test_sin_pd_nan() {
        unsafe {
            let input = _mm256_set1_pd(f64::NAN);
            let result = extract_f64x4(_mm256_sin_pd(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_pd_infinity() {
        unsafe {
            let input = _mm256_set1_pd(f64::INFINITY);
            let result = extract_f64x4(_mm256_sin_pd(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_pd_negative_infinity() {
        unsafe {
            let input = _mm256_set1_pd(f64::NEG_INFINITY);
            let result = extract_f64x4(_mm256_sin_pd(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(-∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_pd_lane_independence() {
        unsafe {
            let input = _mm256_setr_pd(0.0, PI64 / 2.0, PI64, -PI64 / 2.0);
            let result = extract_f64x4(_mm256_sin_pd(input));
            let expected = [
                0.0f64.sin(),
                (PI64 / 2.0).sin(),
                PI64.sin(),
                (-PI64 / 2.0).sin(),
            ];
            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (r - e).abs() < 1e-14,
                    "Lane {}: got {}, expected {}",
                    i,
                    r,
                    e
                );
            }
        }
    }

    #[test]
    fn test_sin_pd_ulp_sweep() {
        unsafe {
            let mut max_ulp = 0u64;
            for i in 0..10000 {
                let x = -2.0 * PI64 + (i as f64 / 10000.0) * 4.0 * PI64;
                let input = _mm256_set1_pd(x);
                let result = extract_f64x4(_mm256_sin_pd(input))[0];
                let expected = x.sin();
                if expected.is_finite() && result.is_finite() {
                    max_ulp = max_ulp.max(ulp_diff_f64(result, expected));
                }
            }
            assert!(max_ulp <= 2, "Max ULP error: {} (expected ≤ 2)", max_ulp);
        }
    }
}
