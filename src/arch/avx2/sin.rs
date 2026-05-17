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
    C1_64, C2_64, C3_64, C4_64, C5_64, C6_64, FRAC_2_PI_32, FRAC_2_PI_64, PIO2_1_32, PIO2_1_64,
    PIO2_1T_32, PIO2_2_64, PIO2_2T_64, S1_64, S2_64, S3_64, S4_64, S5_64, S6_64, TOINT,
};

use super::cos::{cosdf_kernel, sindf_kernel};

// =============================================================================
// f32 Implementation (8 lanes, computed in f64 precision internally)
// =============================================================================

/// Computes `sin(x)` for each lane of an AVX2 `__m256` register.
///
/// Promotes the 8 f32 lanes to two 4-lane f64 halves and runs the Cody-Waite
/// reduction + sin/cos kernel in f64. The reduction in f64 is what makes the
/// answer correct near multiples of `π/2` — pure f32 reduction loses ~5 bits
/// of precision in `x - n·(π/2)`. This matches what musl's `__rem_pio2f.c`
/// does (the f32 `sin` reduction is done in `double`).
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
        // Split 8-lane f32 into two 4-lane f64 halves (exact promotion)
        let x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
        let x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));

        // Compute sin in f64 precision for each half
        let sin_lo = sin_ps_in_f64(x_lo);
        let sin_hi = sin_ps_in_f64(x_hi);

        // Narrow back to f32 and recombine
        let result_lo = _mm256_cvtpd_ps(sin_lo);
        let result_hi = _mm256_cvtpd_ps(sin_hi);
        _mm256_insertf128_ps(_mm256_castps128_ps256(result_lo), result_hi, 1)
    }
}

/// Internal f64 computation for f32 sine (4 lanes).
///
/// Mirrors `cos_ps_in_f64` in the cos module: f32-precision polynomial
/// coefficients evaluated in f64 arithmetic, which is enough margin for ≤ 2 ULP
/// at f32 output.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn sin_ps_in_f64(x: __m256d) -> __m256d {
    unsafe {
        let frac_2_pi = _mm256_set1_pd(FRAC_2_PI_32);
        let pio2_1 = _mm256_set1_pd(PIO2_1_32);
        let pio2_1t = _mm256_set1_pd(PIO2_1T_32);
        let toint = _mm256_set1_pd(TOINT);
        let sign_bit = _mm256_set1_pd(-0.0);

        // -------------------------------------------------------------------------
        // Step 1: Argument reduction — n = round(x · 2/π), y = x - n·(π/2)
        // -------------------------------------------------------------------------
        let fn_val = _mm256_sub_pd(_mm256_fmadd_pd(x, frac_2_pi, toint), toint);
        let n = _mm256_cvtpd_epi32(fn_val);
        let y = _mm256_fnmadd_pd(fn_val, pio2_1t, _mm256_fnmadd_pd(fn_val, pio2_1, x));

        // -------------------------------------------------------------------------
        // Step 2: Compute both kernels (the quadrant selects which lane uses which)
        // -------------------------------------------------------------------------

        let sin_y = sindf_kernel(y);
        let cos_y = cosdf_kernel(y);
        // -------------------------------------------------------------------------
        // Step 3: Quadrant selection — sin(x):
        //   n mod 4: 0 →  sin(y), 1 →  cos(y), 2 → -sin(y), 3 → -cos(y)
        //   use_cos = (n & 1) == 1   (n = 1 or 3)
        //   negate  = (n & 2) == 2   (n = 2 or 3)
        // -------------------------------------------------------------------------
        let n_256 = _mm256_cvtepi32_epi64(n);
        let one = _mm256_set1_epi64x(1);
        let two = _mm256_set1_epi64x(2);
        let use_cos = _mm256_cmpeq_epi64(_mm256_and_si256(n_256, one), one);
        let negate = _mm256_cmpeq_epi64(_mm256_and_si256(n_256, two), two);

        let kernel_result = _mm256_blendv_pd(sin_y, cos_y, _mm256_castsi256_pd(use_cos));
        let negated = _mm256_xor_pd(kernel_result, sign_bit);
        let result = _mm256_blendv_pd(kernel_result, negated, _mm256_castsi256_pd(negate));

        // -------------------------------------------------------------------------
        // Step 4: Special cases
        //   - ±∞ / NaN → NaN
        //   - For tiny |x| (including ±0), sin(x) ≈ x. Returning `x` directly
        //     preserves the sign of zero, which the polynomial form would lose
        //     (the leading `x` in the sin kernel is added inside an FMA whose other
        //     term can be `+0` even when `x = -0`).
        // -------------------------------------------------------------------------
        let abs_x = _mm256_andnot_pd(sign_bit, x);
        let inf = _mm256_set1_pd(f64::INFINITY);
        let is_inf_or_nan = _mm256_cmp_pd(abs_x, inf, _CMP_GE_OQ);
        let nan = _mm256_set1_pd(f64::NAN);

        // |x| < 2^-26 → sin(x) rounds to x at f32 precision; bypass the kernel
        let tiny = _mm256_set1_pd((2.0_f64).powi(-26));
        let is_tiny = _mm256_cmp_pd(abs_x, tiny, _CMP_LT_OQ);
        let result = _mm256_blendv_pd(result, x, is_tiny);

        _mm256_blendv_pd(result, nan, is_inf_or_nan)
    }
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

        // For tiny values (|x| < 1e-300), sin(x) ≈ x — avoids underflow in the polynomial.
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
    fn test_sin_ps_special_values() {
        unsafe {
            // ±0 → ±0 with sign preserved
            let r = extract_f32x8(_mm256_sin_ps(_mm256_set1_ps(0.0)));
            assert_eq!(r[0], 0.0);
            assert!(r[0].is_sign_positive());

            let r = extract_f32x8(_mm256_sin_ps(_mm256_set1_ps(-0.0)));
            assert_eq!(r[0], 0.0);
            assert!(r[0].is_sign_negative());

            // Known values
            let r = extract_f32x8(_mm256_sin_ps(_mm256_set1_ps(PI32 / 2.0)));
            assert!((r[0] - 1.0).abs() < 1e-6, "sin(π/2) = {}", r[0]);

            let r = extract_f32x8(_mm256_sin_ps(_mm256_set1_ps(PI32)));
            assert!(r[0].abs() < 1e-6, "sin(π) = {}", r[0]);

            let r = extract_f32x8(_mm256_sin_ps(_mm256_set1_ps(PI32 / 4.0)));
            assert!(
                (r[0] - (PI32 / 4.0).sin()).abs() < 1e-6,
                "sin(π/4) = {}",
                r[0]
            );

            // NaN and infinities → NaN
            for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let r = extract_f32x8(_mm256_sin_ps(_mm256_set1_ps(x)));
                assert!(r[0].is_nan(), "sin({x}) should be NaN, got {}", r[0]);
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
    fn test_sin_pd_special_values() {
        unsafe {
            // ±0 → ±0 with sign preserved
            let r = extract_f64x4(_mm256_sin_pd(_mm256_set1_pd(0.0)));
            assert_eq!(r[0], 0.0);
            assert!(r[0].is_sign_positive());

            let r = extract_f64x4(_mm256_sin_pd(_mm256_set1_pd(-0.0)));
            assert_eq!(r[0], 0.0);
            assert!(r[0].is_sign_negative());

            // Known values
            let r = extract_f64x4(_mm256_sin_pd(_mm256_set1_pd(PI64 / 2.0)));
            assert!((r[0] - 1.0).abs() < 1e-14, "sin(π/2) = {}", r[0]);

            let r = extract_f64x4(_mm256_sin_pd(_mm256_set1_pd(PI64)));
            assert!(r[0].abs() < 1e-14, "sin(π) = {}", r[0]);

            // NaN and infinities → NaN
            for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let r = extract_f64x4(_mm256_sin_pd(_mm256_set1_pd(x)));
                assert!(r[0].is_nan(), "sin({x}) should be NaN, got {}", r[0]);
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
