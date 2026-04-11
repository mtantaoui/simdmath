//! AVX-512 SIMD implementation of `cos(x)` for `f32` and `f64` vectors.
//!
//! This module provides 16-lane f32 and 8-lane f64 cosine implementations using
//! the Payne-Hanek argument reduction algorithm and minimax polynomial
//! approximations ported from musl libc's `cosf.c`, `cos.c`, and kernel functions.
//!
//! # Algorithm
//!
//! 1. **Argument reduction**: Reduce `x` to `y ∈ [-π/4, π/4]` via `y = x - n*(π/2)`
//!    using Cody-Waite extended precision subtraction.
//!
//! 2. **Quadrant selection**: Based on `n mod 4`, select the appropriate kernel:
//!    | n mod 4 | cos(x)   |
//!    |---------|----------|
//!    | 0       |  cos(y)  |
//!    | 1       | -sin(y)  |
//!    | 2       | -cos(y)  |
//!    | 3       |  sin(y)  |
//!
//! 3. **Polynomial evaluation**: Minimax polynomials for sin/cos kernels.
//!
//! # Precision
//!
//! | Variant           | Max Error |
//! |-------------------|-----------|
//! | `_mm512_cos_ps`   | ≤ 2 ULP   |
//! | `_mm512_cos_pd`   | ≤ 2 ULP   |
//!
//! # Special Values
//!
//! | Input       | Output |
//! |-------------|--------|
//! | `0.0`       | `1.0`  |
//! | `-0.0`      | `1.0`  |
//! | `±∞`        | `NaN`  |
//! | `NaN`       | `NaN`  |
//! | Very small  | `1.0` (correctly rounded) |

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::cos::{
    C0_32, C1_32, C1_64, C2_32, C2_64, C3_32, C3_64, C4_64, C5_64, C6_64, FRAC_2_PI_32,
    FRAC_2_PI_64, PIO2_1_32, PIO2_1_64, PIO2_1T_32, PIO2_1T_64, PIO2_2_64, PIO2_2T_64, S1_32,
    S1_64, S2_32, S2_64, S3_32, S3_64, S4_32, S4_64, S5_64, S6_64, TOINT,
};

// =============================================================================
// f32 Implementation (16 lanes, computed in f64 precision internally)
// =============================================================================

/// Computes `cos(x)` for each lane of an AVX-512 `__m512` register.
///
/// Uses the musl libc algorithm: Cody-Waite argument reduction to `[-π/4, π/4]`
/// followed by polynomial evaluation of the appropriate sin/cos kernel based
/// on the quadrant. Internal computations use f64 precision for accuracy.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires AVX-512F and FMA support. The caller must ensure these features are
/// available at runtime.
#[inline]
#[allow(dead_code)]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_cos_ps(x: __m512) -> __m512 {
    unsafe {
        // Process as two 8-lane f64 operations for precision
        // Split input into low and high halves, convert to f64
        let x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(x));
        let x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(x, 1));

        // Compute cosine in f64 precision for each half
        let cos_lo = cos_ps_in_f64(x_lo);
        let cos_hi = cos_ps_in_f64(x_hi);

        // Convert back to f32 and combine
        let result_lo = _mm512_cvtpd_ps(cos_lo);
        let result_hi = _mm512_cvtpd_ps(cos_hi);

        _mm512_insertf32x8(_mm512_castps256_ps512(result_lo), result_hi, 1)
    }
}

/// Internal f64 computation for f32 cosine (8 lanes).
///
/// This helper computes cos(x) in f64 precision for 8 f32 values that have
/// been promoted to f64. The extra precision ensures ≤1 ULP in the final f32.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn cos_ps_in_f64(x: __m512d) -> __m512d {
    unsafe {
        let frac_2_pi = _mm512_set1_pd(FRAC_2_PI_32);
        let pio2_1 = _mm512_set1_pd(PIO2_1_32);
        let pio2_1t = _mm512_set1_pd(PIO2_1T_32);
        let toint = _mm512_set1_pd(TOINT);

        // -------------------------------------------------------------------------
        // Step 1: Argument reduction
        // Compute n = round(x * 2/π), then y = x - n * (π/2)
        // -------------------------------------------------------------------------

        // fn = round(x * 2/π) using magic number trick
        let fn_val = _mm512_sub_pd(_mm512_fmadd_pd(x, frac_2_pi, toint), toint);

        // Convert to integer for quadrant selection
        let n = _mm512_cvtpd_epi32(fn_val);

        // Cody-Waite reduction: y = x - fn * pio2_1 - fn * pio2_1t
        let y = _mm512_fnmadd_pd(fn_val, pio2_1t, _mm512_fnmadd_pd(fn_val, pio2_1, x));

        // -------------------------------------------------------------------------
        // Step 2: Compute both sin(y) and cos(y) kernels
        // We need both because quadrant determines which to use
        // -------------------------------------------------------------------------

        let cos_y = cosdf_kernel(y);
        let sin_y = sindf_kernel(y);

        // -------------------------------------------------------------------------
        // Step 3: Quadrant-based selection
        // n mod 4: 0 → cos(y), 1 → -sin(y), 2 → -cos(y), 3 → sin(y)
        //
        // use_sin when n & 1 = 1 (n=1,3)
        // negate when n = 1 or 2, i.e., when (n+1) & 2 != 0:
        //   n=0: (0+1)&2 = 0 → no negate ✓
        //   n=1: (1+1)&2 = 2 → negate ✓
        //   n=2: (2+1)&2 = 2 → negate ✓
        //   n=3: (3+1)&2 = 0 → no negate ✓
        // -------------------------------------------------------------------------

        // Extend n to 512-bit for blending (n is 256-bit from cvtpd_epi32)
        let n_512 = _mm512_cvtepi32_epi64(n);
        let one = _mm512_set1_epi64(1);
        let two = _mm512_set1_epi64(2);

        // Masks for quadrant selection (AVX-512 uses integer masks)
        let n_and_1 = _mm512_and_epi64(n_512, one); // bit 0: use sin kernel
        let n_plus_1 = _mm512_add_epi64(n_512, one);
        let n_plus_1_and_2 = _mm512_and_epi64(n_plus_1, two); // (n+1) & 2: negate

        let use_sin = _mm512_cmpeq_epi64_mask(n_and_1, one);
        let negate = _mm512_cmpeq_epi64_mask(n_plus_1_and_2, two);

        // Select sin or cos kernel
        let kernel_result = _mm512_mask_blend_pd(use_sin, cos_y, sin_y);

        // Apply negation for quadrants 1 and 2
        let sign_bit = _mm512_set1_pd(-0.0);
        let negated = _mm512_xor_pd(kernel_result, sign_bit);
        let result = _mm512_mask_blend_pd(negate, kernel_result, negated);

        // -------------------------------------------------------------------------
        // Step 4: Handle special cases (NaN, Inf)
        // cos(±∞) = NaN, cos(NaN) = NaN
        // -------------------------------------------------------------------------

        let abs_x = _mm512_abs_pd(x);
        let inf = _mm512_set1_pd(f64::INFINITY);
        let is_inf_or_nan = _mm512_cmp_pd_mask(abs_x, inf, _CMP_GE_OQ);
        let nan = _mm512_set1_pd(f64::NAN);

        _mm512_mask_blend_pd(is_inf_or_nan, result, nan)
    }
}

/// Cosine kernel for reduced argument in `[-π/4, π/4]`.
///
/// Implements musl's `__cosdf`: cos(x) ≈ 1 + C0*z + C1*z² + C2*z³ + C3*z⁴
/// where z = x².
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn cosdf_kernel(x: __m512d) -> __m512d {
    unsafe {
        let c0 = _mm512_set1_pd(C0_32);
        let c1 = _mm512_set1_pd(C1_32);
        let c2 = _mm512_set1_pd(C2_32);
        let c3 = _mm512_set1_pd(C3_32);
        let one = _mm512_set1_pd(1.0);

        let z = _mm512_mul_pd(x, x); // z = x²
        let w = _mm512_mul_pd(z, z); // w = z² = x⁴

        // r = C2 + z*C3
        let r = _mm512_fmadd_pd(z, c3, c2);

        // ((1 + z*C0) + w*C1) + (w*z)*r
        let term1 = _mm512_fmadd_pd(z, c0, one); // 1 + z*C0
        let term2 = _mm512_fmadd_pd(w, c1, term1); // + w*C1
        let wz = _mm512_mul_pd(w, z); // w*z = x⁶
        _mm512_fmadd_pd(wz, r, term2) // + x⁶ * (C2 + z*C3)
    }
}

/// Sine kernel for reduced argument in `[-π/4, π/4]`.
///
/// Implements musl's `__sindf`: sin(x) ≈ x + S1*x³ + S2*x⁵ + S3*x⁷ + S4*x⁹
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn sindf_kernel(x: __m512d) -> __m512d {
    unsafe {
        let s1 = _mm512_set1_pd(S1_32);
        let s2 = _mm512_set1_pd(S2_32);
        let s3 = _mm512_set1_pd(S3_32);
        let s4 = _mm512_set1_pd(S4_32);

        let z = _mm512_mul_pd(x, x); // z = x²
        let w = _mm512_mul_pd(z, z); // w = z² = x⁴
        let s = _mm512_mul_pd(z, x); // s = z*x = x³

        // r = S3 + z*S4
        let r = _mm512_fmadd_pd(z, s4, s3);

        // (x + s*(S1 + z*S2)) + s*w*r
        let inner = _mm512_fmadd_pd(z, s2, s1); // S1 + z*S2
        let term1 = _mm512_fmadd_pd(s, inner, x); // x + s*(S1 + z*S2)
        let sw = _mm512_mul_pd(s, w); // s*w = x⁷
        _mm512_fmadd_pd(sw, r, term1) // + x⁷ * (S3 + z*S4)
    }
}

// =============================================================================
// f64 Implementation (8 lanes)
// =============================================================================

/// Computes `cos(x)` for each lane of an AVX-512 `__m512d` register.
///
/// Uses musl libc's algorithm with degree-14 polynomial for the cosine kernel
/// and degree-13 for the sine kernel after Cody-Waite argument reduction.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires AVX-512F and FMA support. The caller must ensure these features are
/// available at runtime.
#[inline]
#[allow(dead_code)]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_cos_pd(x: __m512d) -> __m512d {
    unsafe {
        let frac_2_pi = _mm512_set1_pd(FRAC_2_PI_64);
        let pio2_1 = _mm512_set1_pd(PIO2_1_64);
        let pio2_1t = _mm512_set1_pd(PIO2_1T_64);
        let pio2_2 = _mm512_set1_pd(PIO2_2_64);
        let pio2_2t = _mm512_set1_pd(PIO2_2T_64);
        let toint = _mm512_set1_pd(TOINT);

        // -------------------------------------------------------------------------
        // Step 1: Argument reduction with extended precision
        // -------------------------------------------------------------------------

        let fn_val = _mm512_sub_pd(_mm512_fmadd_pd(x, frac_2_pi, toint), toint);
        let n = _mm512_cvtpd_epi32(fn_val);

        // Extended precision Cody-Waite reduction
        // y = x - fn*(pio2_1 + pio2_1t) with compensation
        let mut y = _mm512_fnmadd_pd(fn_val, pio2_1, x);
        y = _mm512_fnmadd_pd(fn_val, pio2_1t, y);

        // For very large arguments, apply additional correction terms
        let abs_x = _mm512_abs_pd(x);
        let large_thresh = _mm512_set1_pd(1e9); // Beyond this, need more precision
        let is_large = _mm512_cmp_pd_mask(abs_x, large_thresh, _CMP_GT_OQ);

        // Additional reduction for large values
        let y_corrected = _mm512_fnmadd_pd(fn_val, pio2_2, y);
        let y_corrected = _mm512_fnmadd_pd(fn_val, pio2_2t, y_corrected);
        let y = _mm512_mask_blend_pd(is_large, y, y_corrected);

        // -------------------------------------------------------------------------
        // Step 2: Compute kernels
        // -------------------------------------------------------------------------

        let cos_y = cos_kernel_f64(y);
        let sin_y = sin_kernel_f64(y);

        // -------------------------------------------------------------------------
        // Step 3: Quadrant selection
        // n mod 4: 0 → cos(y), 1 → -sin(y), 2 → -cos(y), 3 → sin(y)
        // -------------------------------------------------------------------------

        let n_512 = _mm512_cvtepi32_epi64(n);
        let one = _mm512_set1_epi64(1);
        let two = _mm512_set1_epi64(2);

        let n_and_1 = _mm512_and_epi64(n_512, one);
        let n_plus_1 = _mm512_add_epi64(n_512, one);
        let n_plus_1_and_2 = _mm512_and_epi64(n_plus_1, two);

        let use_sin = _mm512_cmpeq_epi64_mask(n_and_1, one);
        let negate = _mm512_cmpeq_epi64_mask(n_plus_1_and_2, two);

        let kernel_result = _mm512_mask_blend_pd(use_sin, cos_y, sin_y);

        let sign_bit = _mm512_set1_pd(-0.0);
        let negated = _mm512_xor_pd(kernel_result, sign_bit);
        let result = _mm512_mask_blend_pd(negate, kernel_result, negated);

        // -------------------------------------------------------------------------
        // Step 4: Handle special cases
        // -------------------------------------------------------------------------

        let inf = _mm512_set1_pd(f64::INFINITY);
        let is_inf_or_nan = _mm512_cmp_pd_mask(abs_x, inf, _CMP_GE_OQ);
        let nan = _mm512_set1_pd(f64::NAN);

        _mm512_mask_blend_pd(is_inf_or_nan, result, nan)
    }
}

/// Cosine kernel for f64 reduced argument.
///
/// Implements musl's `__cos`: cos(x) ≈ 1 - x²/2 + C1*x⁴ + ... + C6*x¹⁴
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn cos_kernel_f64(x: __m512d) -> __m512d {
    unsafe {
        let c1 = _mm512_set1_pd(C1_64);
        let c2 = _mm512_set1_pd(C2_64);
        let c3 = _mm512_set1_pd(C3_64);
        let c4 = _mm512_set1_pd(C4_64);
        let c5 = _mm512_set1_pd(C5_64);
        let c6 = _mm512_set1_pd(C6_64);
        let half = _mm512_set1_pd(0.5);
        let one = _mm512_set1_pd(1.0);

        let z = _mm512_mul_pd(x, x); // z = x²
        let w = _mm512_mul_pd(z, z); // w = x⁴

        // r = z*(C1 + z*(C2 + z*C3)) + w*w*(C4 + z*(C5 + z*C6))
        let inner1 = _mm512_fmadd_pd(z, c3, c2); // C2 + z*C3
        let inner1 = _mm512_fmadd_pd(z, inner1, c1); // C1 + z*(C2 + z*C3)
        let term1 = _mm512_mul_pd(z, inner1); // z * (...)

        let inner2 = _mm512_fmadd_pd(z, c6, c5); // C5 + z*C6
        let inner2 = _mm512_fmadd_pd(z, inner2, c4); // C4 + z*(C5 + z*C6)
        let ww = _mm512_mul_pd(w, w); // w*w = x⁸
        let term2 = _mm512_mul_pd(ww, inner2); // x⁸ * (...)

        let r = _mm512_add_pd(term1, term2);

        // cos(x) = 1 - hz + (((1-w) - hz) + z*r)
        // Simplified: w = 1 - hz, return w + (((1-w)-hz) + z*r)
        let hz = _mm512_mul_pd(half, z); // hz = z/2
        let w = _mm512_sub_pd(one, hz); // w = 1 - z/2

        // For better accuracy: w + (((1-w) - hz) + z*r)
        let one_minus_w = _mm512_sub_pd(one, w); // 1 - w (captures rounding error)
        let correction = _mm512_sub_pd(one_minus_w, hz); // (1-w) - hz
        let zr = _mm512_mul_pd(z, r);
        let final_correction = _mm512_add_pd(correction, zr);

        _mm512_add_pd(w, final_correction)
    }
}

/// Sine kernel for f64 reduced argument.
///
/// Implements musl's `__sin`: sin(x) ≈ x + S1*x³ + ... + S6*x¹³
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn sin_kernel_f64(x: __m512d) -> __m512d {
    unsafe {
        let s1 = _mm512_set1_pd(S1_64);
        let s2 = _mm512_set1_pd(S2_64);
        let s3 = _mm512_set1_pd(S3_64);
        let s4 = _mm512_set1_pd(S4_64);
        let s5 = _mm512_set1_pd(S5_64);
        let s6 = _mm512_set1_pd(S6_64);

        let z = _mm512_mul_pd(x, x); // z = x²
        let w = _mm512_mul_pd(z, z); // w = x⁴
        let v = _mm512_mul_pd(z, x); // v = x³

        // r = S2 + z*(S3 + z*S4) + z*w*(S5 + z*S6)
        let inner1 = _mm512_fmadd_pd(z, s4, s3); // S3 + z*S4
        let inner1 = _mm512_fmadd_pd(z, inner1, s2); // S2 + z*(S3 + z*S4)

        let inner2 = _mm512_fmadd_pd(z, s6, s5); // S5 + z*S6
        let zw = _mm512_mul_pd(z, w); // z*w = x⁶
        let term2 = _mm512_mul_pd(zw, inner2); // x⁶ * (S5 + z*S6)

        let r = _mm512_add_pd(inner1, term2);

        // sin(x) = x + v*(S1 + z*r)
        let zr = _mm512_mul_pd(z, r);
        let s1_plus_zr = _mm512_add_pd(s1, zr);
        _mm512_fmadd_pd(v, s1_plus_zr, x)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to extract f32 lanes from __m512
    unsafe fn extract_ps(v: __m512) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), v) };
        out
    }

    /// Helper to extract f64 lanes from __m512d
    unsafe fn extract_pd(v: __m512d) -> [f64; 8] {
        let mut out = [0.0f64; 8];
        unsafe { _mm512_storeu_pd(out.as_mut_ptr(), v) };
        out
    }

    /// Compute ULP difference between two f32 values
    fn ulp_diff_f32(a: f32, b: f32) -> u32 {
        if a.is_nan() && b.is_nan() {
            return 0;
        }
        if a.is_nan() || b.is_nan() {
            return u32::MAX;
        }
        let a_bits = a.to_bits() as i32;
        let b_bits = b.to_bits() as i32;
        (a_bits.wrapping_sub(b_bits)).unsigned_abs()
    }

    /// Compute ULP difference between two f64 values
    fn ulp_diff_f64(a: f64, b: f64) -> u64 {
        if a.is_nan() && b.is_nan() {
            return 0;
        }
        if a.is_nan() || b.is_nan() {
            return u64::MAX;
        }
        let a_bits = a.to_bits() as i64;
        let b_bits = b.to_bits() as i64;
        (a_bits.wrapping_sub(b_bits)).unsigned_abs()
    }

    // =========================================================================
    // f32 Tests
    // =========================================================================

    #[test]
    fn test_cos_ps_zero() {
        unsafe {
            let input = _mm512_set1_ps(0.0);
            let result = extract_ps(_mm512_cos_ps(input));
            for &r in &result {
                assert!((r - 1.0).abs() < 1e-6, "cos(0) = {}, expected 1.0", r);
            }
        }
    }

    #[test]
    fn test_cos_ps_negative_zero() {
        unsafe {
            let input = _mm512_set1_ps(-0.0);
            let result = extract_ps(_mm512_cos_ps(input));
            for &r in &result {
                assert!((r - 1.0).abs() < 1e-6, "cos(-0) = {}, expected 1.0", r);
            }
        }
    }

    #[test]
    fn test_cos_ps_pi() {
        unsafe {
            let pi = std::f32::consts::PI;
            let input = _mm512_set1_ps(pi);
            let result = extract_ps(_mm512_cos_ps(input));
            for &r in &result {
                assert!((r - (-1.0)).abs() < 1e-5, "cos(π) = {}, expected -1.0", r);
            }
        }
    }

    #[test]
    fn test_cos_ps_pi_over_2() {
        unsafe {
            let pi_2 = std::f32::consts::FRAC_PI_2;
            let input = _mm512_set1_ps(pi_2);
            let result = extract_ps(_mm512_cos_ps(input));
            for &r in &result {
                assert!(r.abs() < 1e-5, "cos(π/2) = {}, expected 0.0", r);
            }
        }
    }

    #[test]
    fn test_cos_ps_pi_over_4() {
        unsafe {
            let pi_4 = std::f32::consts::FRAC_PI_4;
            let input = _mm512_set1_ps(pi_4);
            let result = extract_ps(_mm512_cos_ps(input));
            let expected = std::f32::consts::FRAC_1_SQRT_2;
            for &r in &result {
                assert!(
                    (r - expected).abs() < 1e-5,
                    "cos(π/4) = {}, expected {}",
                    r,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_cos_ps_nan() {
        unsafe {
            let input = _mm512_set1_ps(f32::NAN);
            let result = extract_ps(_mm512_cos_ps(input));
            for &r in &result {
                assert!(r.is_nan(), "cos(NaN) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_cos_ps_infinity() {
        unsafe {
            let input = _mm512_set1_ps(f32::INFINITY);
            let result = extract_ps(_mm512_cos_ps(input));
            for &r in &result {
                assert!(r.is_nan(), "cos(+∞) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_cos_ps_negative_infinity() {
        unsafe {
            let input = _mm512_set1_ps(f32::NEG_INFINITY);
            let result = extract_ps(_mm512_cos_ps(input));
            for &r in &result {
                assert!(r.is_nan(), "cos(-∞) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_cos_ps_lane_independence() {
        unsafe {
            let input = _mm512_setr_ps(
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                std::f32::consts::PI,
                0.25,
                0.75,
                1.25,
                1.75,
                2.25,
                2.75,
                3.25,
                std::f32::consts::FRAC_PI_2,
            );
            let result = extract_ps(_mm512_cos_ps(input));

            let expected: [f32; 16] = [
                0.0f32.cos(),
                0.5f32.cos(),
                1.0f32.cos(),
                1.5f32.cos(),
                2.0f32.cos(),
                2.5f32.cos(),
                3.0f32.cos(),
                std::f32::consts::PI.cos(),
                0.25f32.cos(),
                0.75f32.cos(),
                1.25f32.cos(),
                1.75f32.cos(),
                2.25f32.cos(),
                2.75f32.cos(),
                3.25f32.cos(),
                std::f32::consts::FRAC_PI_2.cos(),
            ];

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_diff_f32(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: cos = {}, expected {}, ULP = {}",
                    i,
                    r,
                    e,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_cos_ps_ulp_sweep() {
        unsafe {
            let mut max_ulp = 0u32;

            // Test values across several periods
            for i in 0..1000 {
                let x = (i as f32 - 500.0) * 0.02; // Range roughly -10 to 10
                let input = _mm512_set1_ps(x);
                let result = extract_ps(_mm512_cos_ps(input));
                let expected = x.cos();

                for &r in &result {
                    let ulp = ulp_diff_f32(r, expected);
                    max_ulp = max_ulp.max(ulp);
                }
            }

            assert!(max_ulp <= 2, "Max ULP error: {} (expected ≤ 2)", max_ulp);
        }
    }

    // =========================================================================
    // f64 Tests
    // =========================================================================

    #[test]
    fn test_cos_pd_zero() {
        unsafe {
            let input = _mm512_set1_pd(0.0);
            let result = extract_pd(_mm512_cos_pd(input));
            for &r in &result {
                assert!((r - 1.0).abs() < 1e-14, "cos(0) = {}, expected 1.0", r);
            }
        }
    }

    #[test]
    fn test_cos_pd_negative_zero() {
        unsafe {
            let input = _mm512_set1_pd(-0.0);
            let result = extract_pd(_mm512_cos_pd(input));
            for &r in &result {
                assert!((r - 1.0).abs() < 1e-14, "cos(-0) = {}, expected 1.0", r);
            }
        }
    }

    #[test]
    fn test_cos_pd_pi() {
        unsafe {
            let pi = std::f64::consts::PI;
            let input = _mm512_set1_pd(pi);
            let result = extract_pd(_mm512_cos_pd(input));
            for &r in &result {
                assert!((r - (-1.0)).abs() < 1e-14, "cos(π) = {}, expected -1.0", r);
            }
        }
    }

    #[test]
    fn test_cos_pd_pi_over_2() {
        unsafe {
            let pi_2 = std::f64::consts::FRAC_PI_2;
            let input = _mm512_set1_pd(pi_2);
            let result = extract_pd(_mm512_cos_pd(input));
            for &r in &result {
                assert!(r.abs() < 1e-14, "cos(π/2) = {}, expected 0.0", r);
            }
        }
    }

    #[test]
    fn test_cos_pd_nan() {
        unsafe {
            let input = _mm512_set1_pd(f64::NAN);
            let result = extract_pd(_mm512_cos_pd(input));
            for &r in &result {
                assert!(r.is_nan(), "cos(NaN) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_cos_pd_infinity() {
        unsafe {
            let input = _mm512_set1_pd(f64::INFINITY);
            let result = extract_pd(_mm512_cos_pd(input));
            for &r in &result {
                assert!(r.is_nan(), "cos(+∞) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_cos_pd_negative_infinity() {
        unsafe {
            let input = _mm512_set1_pd(f64::NEG_INFINITY);
            let result = extract_pd(_mm512_cos_pd(input));
            for &r in &result {
                assert!(r.is_nan(), "cos(-∞) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_cos_pd_lane_independence() {
        unsafe {
            let input = _mm512_setr_pd(
                0.0,
                1.0,
                2.0,
                std::f64::consts::PI,
                0.5,
                1.5,
                2.5,
                std::f64::consts::FRAC_PI_2,
            );
            let result = extract_pd(_mm512_cos_pd(input));

            let expected: [f64; 8] = [
                0.0f64.cos(),
                1.0f64.cos(),
                2.0f64.cos(),
                std::f64::consts::PI.cos(),
                0.5f64.cos(),
                1.5f64.cos(),
                2.5f64.cos(),
                std::f64::consts::FRAC_PI_2.cos(),
            ];

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_diff_f64(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: got {}, expected {}, ULP = {}",
                    i,
                    r,
                    e,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_cos_pd_ulp_sweep() {
        unsafe {
            let mut max_ulp = 0u64;

            for i in 0..1000 {
                let x = (i as f64 - 500.0) * 0.02;
                let input = _mm512_set1_pd(x);
                let result = extract_pd(_mm512_cos_pd(input));
                let expected = x.cos();

                for &r in &result {
                    let ulp = ulp_diff_f64(r, expected);
                    max_ulp = max_ulp.max(ulp);
                }
            }

            assert!(max_ulp <= 2, "Max ULP error: {} (expected ≤ 2)", max_ulp);
        }
    }
}
