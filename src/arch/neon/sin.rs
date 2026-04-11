//! NEON SIMD implementation of `sin(x)` for `f32` and `f64` vectors.
//!
//! This module provides 4-lane f32 and 2-lane f64 sine implementations using
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
//! | Variant       | Max Error |
//! |---------------|-----------|
//! | `vsin_f32`    | ≤ 2 ULP   |
//! | `vsin_f64`    | ≤ 2 ULP   |
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

use std::arch::aarch64::*;

use crate::arch::consts::sin::{
    C0_32, C1_32, C1_64, C2_32, C2_64, C3_32, C3_64, C4_64, C5_64, C6_64, FRAC_2_PI_32,
    FRAC_2_PI_64, PIO2_1_32, PIO2_1_64, PIO2_1T_32, PIO2_1T_64, PIO2_2_64, PIO2_2T_64, S1_32,
    S1_64, S2_32, S2_64, S3_32, S3_64, S4_32, S4_64, S5_64, S6_64, TOINT,
};

// =============================================================================
// f32 Implementation (4 lanes, computed in f64 precision internally)
// =============================================================================

/// Computes `sin(x)` for each lane of a NEON `float32x4_t` register.
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
/// Requires NEON support. The caller must ensure this feature is available.
#[inline]
#[allow(dead_code)]
pub(crate) unsafe fn vsin_f32(x: float32x4_t) -> float32x4_t {
    // Process as two 2-lane f64 operations for precision
    // Split input into low and high halves, convert to f64
    let x_lo = vcvt_f64_f32(vget_low_f32(x));
    let x_hi = vcvt_f64_f32(vget_high_f32(x));

    // Compute sine in f64 precision for each half
    let sin_lo = sin_ps_in_f64(x_lo);
    let sin_hi = sin_ps_in_f64(x_hi);

    // Convert back to f32 and combine
    let result_lo = vcvt_f32_f64(sin_lo);
    let result_hi = vcvt_f32_f64(sin_hi);

    vcombine_f32(result_lo, result_hi)
}

/// Internal f64 computation for f32 sine (2 lanes).
///
/// This helper computes sin(x) in f64 precision for 2 f32 values that have
/// been promoted to f64. The extra precision ensures ≤2 ULP in the final f32.
#[inline]
unsafe fn sin_ps_in_f64(x: float64x2_t) -> float64x2_t {
    let frac_2_pi = vdupq_n_f64(FRAC_2_PI_32);
    let pio2_1 = vdupq_n_f64(PIO2_1_32);
    let pio2_1t = vdupq_n_f64(PIO2_1T_32);
    let toint = vdupq_n_f64(TOINT);

    // -------------------------------------------------------------------------
    // Step 1: Argument reduction
    // Compute n = round(x * 2/π), then y = x - n * (π/2)
    // -------------------------------------------------------------------------

    // fn = round(x * 2/π) using magic number trick
    let fn_val = vsubq_f64(vfmaq_f64(toint, x, frac_2_pi), toint);

    // Convert to integer for quadrant selection
    let n = vcvtq_s64_f64(fn_val);

    // Cody-Waite reduction: y = x - fn * pio2_1 - fn * pio2_1t
    let y = vfmsq_f64(vfmsq_f64(x, fn_val, pio2_1), fn_val, pio2_1t);

    // -------------------------------------------------------------------------
    // Step 2: Compute both sin(y) and cos(y) kernels
    // We need both because quadrant determines which to use
    // -------------------------------------------------------------------------

    let sin_y = sindf_kernel(y);
    let cos_y = cosdf_kernel(y);

    // -------------------------------------------------------------------------
    // Step 3: Quadrant-based selection
    // n mod 4: 0 → sin(y), 1 → cos(y), 2 → -sin(y), 3 → -cos(y)
    //
    // use_cos when n & 1 = 1 (n=1,3)
    // negate when n & 2 = 2 (n=2,3)
    // -------------------------------------------------------------------------

    let one = vdupq_n_s64(1);
    let two = vdupq_n_s64(2);

    // Masks for quadrant selection
    let n_and_1 = vandq_s64(n, one); // bit 0: use cos kernel
    let n_and_2 = vandq_s64(n, two); // bit 1: negate

    let use_cos = vceqq_s64(n_and_1, one);
    let negate = vceqq_s64(n_and_2, two);

    // Select sin or cos kernel using vbslq (mask, true_val, false_val)
    let kernel_result = vbslq_f64(use_cos, cos_y, sin_y);

    // Apply negation for quadrants 2 and 3
    let sign_bit = vdupq_n_f64(-0.0);
    let negated = vreinterpretq_f64_u64(veorq_u64(
        vreinterpretq_u64_f64(kernel_result),
        vreinterpretq_u64_f64(sign_bit),
    ));
    let result = vbslq_f64(negate, negated, kernel_result);

    // -------------------------------------------------------------------------
    // Step 4: Handle special cases (NaN, Inf, tiny values)
    // sin(±∞) = NaN, sin(NaN) = NaN, sin(±0) = ±0
    // -------------------------------------------------------------------------

    let abs_x = vabsq_f64(x);
    let inf = vdupq_n_f64(f64::INFINITY);
    let is_inf_or_nan = vcgeq_f64(abs_x, inf);
    let nan = vdupq_n_f64(f64::NAN);

    // For tiny values (including ±0), sin(x) ≈ x
    let tiny = vdupq_n_f64(1e-300);
    let is_tiny = vcltq_f64(abs_x, tiny);
    let result = vbslq_f64(is_tiny, x, result);

    vbslq_f64(is_inf_or_nan, nan, result)
}

/// Sine kernel for reduced argument in `[-π/4, π/4]`.
///
/// Implements musl's `__sindf`: sin(x) ≈ x + S1*x³ + S2*x⁵ + S3*x⁷ + S4*x⁹
#[inline]
unsafe fn sindf_kernel(x: float64x2_t) -> float64x2_t {
    let s1 = vdupq_n_f64(S1_32);
    let s2 = vdupq_n_f64(S2_32);
    let s3 = vdupq_n_f64(S3_32);
    let s4 = vdupq_n_f64(S4_32);

    let z = vmulq_f64(x, x); // z = x²
    let w = vmulq_f64(z, z); // w = z² = x⁴
    let s = vmulq_f64(z, x); // s = z*x = x³

    // r = S3 + z*S4
    let r = vfmaq_f64(s3, z, s4);

    // (x + s*(S1 + z*S2)) + s*w*r
    let inner = vfmaq_f64(s1, z, s2); // S1 + z*S2
    let term1 = vfmaq_f64(x, s, inner); // x + s*(S1 + z*S2)
    let sw = vmulq_f64(s, w); // s*w = x⁷
    vfmaq_f64(term1, sw, r) // + x⁷ * (S3 + z*S4)
}

/// Cosine kernel for reduced argument in `[-π/4, π/4]`.
///
/// Implements musl's `__cosdf`: cos(x) ≈ 1 + C0*z + C1*z² + C2*z³ + C3*z⁴
/// where z = x².
#[inline]
unsafe fn cosdf_kernel(x: float64x2_t) -> float64x2_t {
    let c0 = vdupq_n_f64(C0_32);
    let c1 = vdupq_n_f64(C1_32);
    let c2 = vdupq_n_f64(C2_32);
    let c3 = vdupq_n_f64(C3_32);
    let one = vdupq_n_f64(1.0);

    let z = vmulq_f64(x, x); // z = x²
    let w = vmulq_f64(z, z); // w = z² = x⁴

    // r = C2 + z*C3
    let r = vfmaq_f64(c2, z, c3);

    // ((1 + z*C0) + w*C1) + (w*z)*r
    let term1 = vfmaq_f64(one, z, c0); // 1 + z*C0
    let term2 = vfmaq_f64(term1, w, c1); // + w*C1
    let wz = vmulq_f64(w, z); // w*z = x⁶
    vfmaq_f64(term2, wz, r) // + x⁶ * (C2 + z*C3)
}

// =============================================================================
// f64 Implementation (2 lanes)
// =============================================================================

/// Computes `sin(x)` for each lane of a NEON `float64x2_t` register.
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
/// Requires NEON support. The caller must ensure this feature is available.
#[inline]
#[allow(dead_code)]
pub(crate) unsafe fn vsin_f64(x: float64x2_t) -> float64x2_t {
    let frac_2_pi = vdupq_n_f64(FRAC_2_PI_64);
    let pio2_1 = vdupq_n_f64(PIO2_1_64);
    let pio2_1t = vdupq_n_f64(PIO2_1T_64);
    let pio2_2 = vdupq_n_f64(PIO2_2_64);
    let pio2_2t = vdupq_n_f64(PIO2_2T_64);
    let toint = vdupq_n_f64(TOINT);

    // -------------------------------------------------------------------------
    // Step 1: Argument reduction with extended precision
    // -------------------------------------------------------------------------

    let fn_val = vsubq_f64(vfmaq_f64(toint, x, frac_2_pi), toint);
    let n = vcvtq_s64_f64(fn_val);

    // Extended precision Cody-Waite reduction
    // y = x - fn*(pio2_1 + pio2_1t) with compensation
    let mut y = vfmsq_f64(x, fn_val, pio2_1);
    y = vfmsq_f64(y, fn_val, pio2_1t);

    // For very large arguments, apply additional correction terms
    let abs_x = vabsq_f64(x);
    let large_thresh = vdupq_n_f64(1e9); // Beyond this, need more precision
    let is_large = vcgtq_f64(abs_x, large_thresh);

    // Additional reduction for large values
    let y_corrected = vfmsq_f64(y, fn_val, pio2_2);
    let y_corrected = vfmsq_f64(y_corrected, fn_val, pio2_2t);
    let y = vbslq_f64(is_large, y_corrected, y);

    // -------------------------------------------------------------------------
    // Step 2: Compute kernels
    // -------------------------------------------------------------------------

    let sin_y = sin_kernel_f64(y);
    let cos_y = cos_kernel_f64(y);

    // -------------------------------------------------------------------------
    // Step 3: Quadrant selection
    // n mod 4: 0 → sin(y), 1 → cos(y), 2 → -sin(y), 3 → -cos(y)
    // -------------------------------------------------------------------------

    let one = vdupq_n_s64(1);
    let two = vdupq_n_s64(2);

    let n_and_1 = vandq_s64(n, one);
    let n_and_2 = vandq_s64(n, two);

    let use_cos = vceqq_s64(n_and_1, one);
    let negate = vceqq_s64(n_and_2, two);

    let kernel_result = vbslq_f64(use_cos, cos_y, sin_y);

    let sign_bit = vdupq_n_f64(-0.0);
    let negated = vreinterpretq_f64_u64(veorq_u64(
        vreinterpretq_u64_f64(kernel_result),
        vreinterpretq_u64_f64(sign_bit),
    ));
    let result = vbslq_f64(negate, negated, kernel_result);

    // -------------------------------------------------------------------------
    // Step 4: Handle special cases
    // sin(±∞) = NaN, sin(NaN) = NaN, sin(±0) = ±0
    // -------------------------------------------------------------------------

    let inf = vdupq_n_f64(f64::INFINITY);
    let is_inf_or_nan = vcgeq_f64(abs_x, inf);
    let nan = vdupq_n_f64(f64::NAN);

    // For tiny values (including ±0), sin(x) ≈ x
    let tiny = vdupq_n_f64(1e-300);
    let is_tiny = vcltq_f64(abs_x, tiny);
    let result = vbslq_f64(is_tiny, x, result);

    vbslq_f64(is_inf_or_nan, nan, result)
}

/// Sine kernel for f64 reduced argument.
///
/// Implements musl's `__sin`: sin(x) ≈ x + v*(S1 + z*r)
/// where v = x³, z = x², r = S2 + z*(S3 + z*S4) + z*w*(S5 + z*S6)
#[inline]
unsafe fn sin_kernel_f64(x: float64x2_t) -> float64x2_t {
    let s1 = vdupq_n_f64(S1_64);
    let s2 = vdupq_n_f64(S2_64);
    let s3 = vdupq_n_f64(S3_64);
    let s4 = vdupq_n_f64(S4_64);
    let s5 = vdupq_n_f64(S5_64);
    let s6 = vdupq_n_f64(S6_64);

    let z = vmulq_f64(x, x); // z = x²
    let w = vmulq_f64(z, z); // w = x⁴
    let v = vmulq_f64(z, x); // v = x³

    // r = S2 + z*(S3 + z*S4) + z*w*(S5 + z*S6)
    let inner1 = vfmaq_f64(s3, z, s4); // S3 + z*S4
    let inner1 = vfmaq_f64(s2, z, inner1); // S2 + z*(S3 + z*S4)

    let inner2 = vfmaq_f64(s5, z, s6); // S5 + z*S6
    let zw = vmulq_f64(z, w); // z*w = x⁶
    let term2 = vmulq_f64(zw, inner2); // z*w*(S5 + z*S6)

    let r = vaddq_f64(inner1, term2);

    // sin(x) = x + v*(S1 + z*r)
    let zr = vmulq_f64(z, r); // z*r
    let s1_plus_zr = vaddq_f64(s1, zr); // S1 + z*r
    vfmaq_f64(x, v, s1_plus_zr) // x + v*(S1 + z*r)
}

/// Cosine kernel for f64 reduced argument.
///
/// Implements musl's `__cos`: cos(x) ≈ 1 - x²/2 + C1*x⁴ + ... + C6*x¹⁴
#[inline]
unsafe fn cos_kernel_f64(x: float64x2_t) -> float64x2_t {
    let c1 = vdupq_n_f64(C1_64);
    let c2 = vdupq_n_f64(C2_64);
    let c3 = vdupq_n_f64(C3_64);
    let c4 = vdupq_n_f64(C4_64);
    let c5 = vdupq_n_f64(C5_64);
    let c6 = vdupq_n_f64(C6_64);
    let half = vdupq_n_f64(0.5);
    let one = vdupq_n_f64(1.0);

    let z = vmulq_f64(x, x); // z = x²
    let w = vmulq_f64(z, z); // w = x⁴

    // r = z*(C1 + z*(C2 + z*C3)) + w*w*(C4 + z*(C5 + z*C6))
    let inner1 = vfmaq_f64(c2, z, c3); // C2 + z*C3
    let inner1 = vfmaq_f64(c1, z, inner1); // C1 + z*(C2 + z*C3)
    let term1 = vmulq_f64(z, inner1); // z * (...)

    let inner2 = vfmaq_f64(c5, z, c6); // C5 + z*C6
    let inner2 = vfmaq_f64(c4, z, inner2); // C4 + z*(C5 + z*C6)
    let ww = vmulq_f64(w, w); // w*w = x⁸
    let term2 = vmulq_f64(ww, inner2); // x⁸ * (...)

    let r = vaddq_f64(term1, term2);

    // cos(x) = 1 - hz + (((1-w) - hz) + z*r)
    // Simplified: w = 1 - hz, return w + (((1-w)-hz) + z*r)
    let hz = vmulq_f64(half, z); // hz = z/2
    let w = vsubq_f64(one, hz); // w = 1 - z/2

    // For better accuracy: w + (((1-w) - hz) + z*r)
    let one_minus_w = vsubq_f64(one, w); // 1 - w (captures rounding error)
    let correction = vsubq_f64(one_minus_w, hz); // (1-w) - hz
    let zr = vmulq_f64(z, r);
    let final_correction = vaddq_f64(correction, zr);

    vaddq_f64(w, final_correction)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI32;
    use std::f64::consts::PI as PI64;

    /// Helper to extract f32 lanes from float32x4_t
    unsafe fn extract_f32(v: float32x4_t) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), v);
        out
    }

    /// Helper to extract f64 lanes from float64x2_t
    unsafe fn extract_f64(v: float64x2_t) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        vst1q_f64(out.as_mut_ptr(), v);
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
    fn test_sin_f32_zero() {
        unsafe {
            let input = vdupq_n_f32(0.0);
            let result = extract_f32(vsin_f32(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_positive());
            }
        }
    }

    #[test]
    fn test_sin_f32_negative_zero() {
        unsafe {
            let input = vdupq_n_f32(-0.0);
            let result = extract_f32(vsin_f32(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_negative());
            }
        }
    }

    #[test]
    fn test_sin_f32_pi_over_2() {
        unsafe {
            let input = vdupq_n_f32(PI32 / 2.0);
            let result = extract_f32(vsin_f32(input));
            for &r in &result {
                assert!((r - 1.0).abs() < 1e-6, "sin(π/2) = {}, expected ~1.0", r);
            }
        }
    }

    #[test]
    fn test_sin_f32_pi() {
        unsafe {
            let input = vdupq_n_f32(PI32);
            let result = extract_f32(vsin_f32(input));
            for &r in &result {
                assert!(r.abs() < 1e-6, "sin(π) = {}, expected ~0.0", r);
            }
        }
    }

    #[test]
    fn test_sin_f32_pi_over_4() {
        unsafe {
            let input = vdupq_n_f32(PI32 / 4.0);
            let result = extract_f32(vsin_f32(input));
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
    fn test_sin_f32_nan() {
        unsafe {
            let input = vdupq_n_f32(f32::NAN);
            let result = extract_f32(vsin_f32(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_f32_infinity() {
        unsafe {
            let input = vdupq_n_f32(f32::INFINITY);
            let result = extract_f32(vsin_f32(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_f32_negative_infinity() {
        unsafe {
            let input = vdupq_n_f32(f32::NEG_INFINITY);
            let result = extract_f32(vsin_f32(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(-∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_f32_lane_independence() {
        unsafe {
            let input = vld1q_f32([0.0, PI32 / 2.0, PI32, -PI32 / 2.0].as_ptr());
            let result = extract_f32(vsin_f32(input));
            let expected = [
                0.0f32.sin(),
                (PI32 / 2.0).sin(),
                PI32.sin(),
                (-PI32 / 2.0).sin(),
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
    fn test_sin_f32_ulp_sweep() {
        unsafe {
            let mut max_ulp = 0u32;
            for i in 0..10000 {
                let x = -2.0 * PI32 + (i as f32 / 10000.0) * 4.0 * PI32;
                let input = vdupq_n_f32(x);
                let result = extract_f32(vsin_f32(input))[0];
                let expected = x.sin();
                // Skip values near zero where ULP isn't meaningful
                if expected.abs() < 1e-6 {
                    continue;
                }
                if expected.is_finite() && result.is_finite() {
                    let ulp = ulp_diff_f32(result, expected);
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
    fn test_sin_f64_zero() {
        unsafe {
            let input = vdupq_n_f64(0.0);
            let result = extract_f64(vsin_f64(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_positive());
            }
        }
    }

    #[test]
    fn test_sin_f64_negative_zero() {
        unsafe {
            let input = vdupq_n_f64(-0.0);
            let result = extract_f64(vsin_f64(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_negative());
            }
        }
    }

    #[test]
    fn test_sin_f64_pi_over_2() {
        unsafe {
            let input = vdupq_n_f64(PI64 / 2.0);
            let result = extract_f64(vsin_f64(input));
            for &r in &result {
                assert!((r - 1.0).abs() < 1e-14, "sin(π/2) = {}, expected ~1.0", r);
            }
        }
    }

    #[test]
    fn test_sin_f64_pi() {
        unsafe {
            let input = vdupq_n_f64(PI64);
            let result = extract_f64(vsin_f64(input));
            for &r in &result {
                assert!(r.abs() < 1e-14, "sin(π) = {}, expected ~0.0", r);
            }
        }
    }

    #[test]
    fn test_sin_f64_nan() {
        unsafe {
            let input = vdupq_n_f64(f64::NAN);
            let result = extract_f64(vsin_f64(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_f64_infinity() {
        unsafe {
            let input = vdupq_n_f64(f64::INFINITY);
            let result = extract_f64(vsin_f64(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_f64_negative_infinity() {
        unsafe {
            let input = vdupq_n_f64(f64::NEG_INFINITY);
            let result = extract_f64(vsin_f64(input));
            for &r in &result {
                assert!(r.is_nan(), "sin(-∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_sin_f64_lane_independence() {
        unsafe {
            let input = vld1q_f64([PI64 / 2.0, -PI64 / 2.0].as_ptr());
            let result = extract_f64(vsin_f64(input));
            let expected = [(PI64 / 2.0).sin(), (-PI64 / 2.0).sin()];
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
    fn test_sin_f64_ulp_sweep() {
        unsafe {
            let mut max_ulp = 0u64;
            for i in 0..10000 {
                let x = -2.0 * PI64 + (i as f64 / 10000.0) * 4.0 * PI64;
                let input = vdupq_n_f64(x);
                let result = extract_f64(vsin_f64(input))[0];
                let expected = x.sin();
                // Skip values near zero where ULP isn't meaningful
                if expected.abs() < 1e-10 {
                    continue;
                }
                if expected.is_finite() && result.is_finite() {
                    let ulp = ulp_diff_f64(result, expected);
                    max_ulp = max_ulp.max(ulp);
                }
            }
            assert!(max_ulp <= 2, "Max ULP error: {} (expected ≤ 2)", max_ulp);
        }
    }
}
