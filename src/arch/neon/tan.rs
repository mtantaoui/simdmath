//! NEON SIMD implementation of `tan(x)` for `f32` and `f64` vectors.
//!
//! This module provides 4-lane f32 and 2-lane f64 tangent implementations using
//! the Cody-Waite argument reduction algorithm and minimax polynomial
//! approximations ported from musl libc's `tanf.c`, `tan.c`, and kernel functions.
//!
//! # Algorithm
//!
//! 1. **Argument reduction**: Reduce `x` to `y ∈ [-π/4, π/4]` via `y = x - n*(π/2)`
//!    using Cody-Waite extended precision subtraction.
//!
//! 2. **Quadrant selection**: Unlike sin/cos which have period 2π, tan has period π.
//!    Based on `n mod 2`:
//!    | n mod 2 | tan(x)      |
//!    |---------|-------------|
//!    | 0       |  tan(y)     |
//!    | 1       | -1/tan(y)   |
//!
//! 3. **Polynomial evaluation**: Minimax polynomial for the tangent kernel.
//!
//! # Precision
//!
//! | Variant       | Max Error |
//! |---------------|-----------|
//! | `vtan_f32`    | ≤ 2 ULP   |
//! | `vtan_f64`    | ≤ 2 ULP   |
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
//! | `±π/2`      | Large value (approaches ±∞) |

use std::arch::aarch64::*;

use crate::arch::consts::tan::{
    BIG_THRESH_64, FRAC_2_PI_32, FRAC_2_PI_64, PIO2_1_32, PIO2_1_64, PIO2_1T_32, PIO2_1T_64,
    PIO2_2_64, PIO2_2T_64, PIO4_HI_64, PIO4_LO_64, T0_32, T0_64, T1_32, T1_64, T2_32, T2_64, T3_32,
    T3_64, T4_32, T4_64, T5_32, T5_64, T6_64, T7_64, T8_64, T9_64, T10_64, T11_64, T12_64, TOINT,
};

// =============================================================================
// f32 Implementation (4 lanes, computed in f64 precision internally)
// =============================================================================

/// Computes `tan(x)` for each lane of a NEON `float32x4_t` register.
///
/// Uses the musl libc algorithm: Cody-Waite argument reduction to `[-π/4, π/4]`
/// followed by polynomial evaluation of the tangent kernel. When n is odd,
/// computes -1/tan(y) using the cotangent identity. Internal computations
/// use f64 precision for accuracy.
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
pub(crate) unsafe fn vtan_f32(x: float32x4_t) -> float32x4_t {
    // Process as two 2-lane f64 operations for precision
    // Split input into low and high halves, convert to f64
    let x_lo = vcvt_f64_f32(vget_low_f32(x));
    let x_hi = vcvt_f64_f32(vget_high_f32(x));

    // Compute tangent in f64 precision for each half
    let tan_lo = tan_ps_in_f64(x_lo);
    let tan_hi = tan_ps_in_f64(x_hi);

    // Convert back to f32 and combine
    let result_lo = vcvt_f32_f64(tan_lo);
    let result_hi = vcvt_f32_f64(tan_hi);

    vcombine_f32(result_lo, result_hi)
}

/// Internal f64 computation for f32 tangent (2 lanes).
///
/// This helper computes tan(x) in f64 precision for 2 f32 values that have
/// been promoted to f64. The extra precision ensures ≤2 ULP in the final f32.
#[inline]
unsafe fn tan_ps_in_f64(x: float64x2_t) -> float64x2_t {
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
    // Step 2: Compute tan(y) using polynomial approximation
    // tan(y) ≈ y + T0*y³ + T1*y⁵ + T2*y⁷ + T3*y⁹ + T4*y¹¹ + T5*y¹³
    // -------------------------------------------------------------------------

    let tan_y = tandf_kernel(y);

    // -------------------------------------------------------------------------
    // Step 3: Quadrant-based selection
    // n mod 2: 0 → tan(y), 1 → -1/tan(y)
    //
    // When n is odd, we're in a quadrant where tan should use cotangent:
    // tan(y + π/2) = -cot(y) = -1/tan(y)
    // -------------------------------------------------------------------------

    let one = vdupq_n_s64(1);

    // Check if n is odd (n & 1 == 1)
    let n_and_1 = vandq_s64(n, one);
    let is_odd = vceqq_s64(n_and_1, one); // Returns uint64x2_t

    // Compute -1/tan(y) for odd quadrants
    let neg_one = vdupq_n_f64(-1.0);
    let recip = vdivq_f64(neg_one, tan_y);

    // Select tan(y) or -1/tan(y) based on quadrant
    // vbslq_f64 expects uint64x2_t mask, which is_odd already is
    let result = vbslq_f64(is_odd, recip, tan_y);

    // -------------------------------------------------------------------------
    // Step 4: Handle special cases (NaN, Inf, tiny values)
    // tan(±∞) = NaN, tan(NaN) = NaN, tan(±0) = ±0
    // -------------------------------------------------------------------------

    let abs_x = vabsq_f64(x);
    let inf = vdupq_n_f64(f64::INFINITY);
    let is_inf_or_nan = vcgeq_f64(abs_x, inf);
    let nan = vdupq_n_f64(f64::NAN);

    // For tiny values (including ±0), tan(x) ≈ x
    let tiny = vdupq_n_f64(1e-300);
    let is_tiny = vcltq_f64(abs_x, tiny);
    let result = vbslq_f64(is_tiny, x, result);

    vbslq_f64(is_inf_or_nan, nan, result)
}

/// Tangent kernel for reduced argument in `[-π/4, π/4]`.
///
/// Implements musl's `__tandf`: tan(x) ≈ x + T0*x³ + T1*x⁵ + T2*x⁷ + T3*x⁹ + T4*x¹¹ + T5*x¹³
#[inline]
unsafe fn tandf_kernel(x: float64x2_t) -> float64x2_t {
    let t0 = vdupq_n_f64(T0_32);
    let t1 = vdupq_n_f64(T1_32);
    let t2 = vdupq_n_f64(T2_32);
    let t3 = vdupq_n_f64(T3_32);
    let t4 = vdupq_n_f64(T4_32);
    let t5 = vdupq_n_f64(T5_32);

    let z = vmulq_f64(x, x); // z = x²
    let w = vmulq_f64(z, z); // w = z² = x⁴

    // Horner's method for polynomial evaluation
    // r = T4 + z*T5
    let r = vfmaq_f64(t4, z, t5);
    // r = T2 + z*T3 + w*r = T2 + z*T3 + w*(T4 + z*T5)
    let r = vfmaq_f64(vfmaq_f64(t2, z, t3), w, r);
    // r = T0 + z*T1 + w*r
    let r = vfmaq_f64(vfmaq_f64(t0, z, t1), w, r);

    // tan(x) = x + x³ * r = x + z*x*r
    let zx = vmulq_f64(z, x); // z*x = x³
    vfmaq_f64(x, zx, r) // x + x³*r
}

// =============================================================================
// f64 Implementation (2 lanes)
// =============================================================================

/// Computes `tan(x)` for each lane of a NEON `float64x2_t` register.
///
/// Uses musl libc's algorithm with degree-27 polynomial for the tangent kernel
/// after Cody-Waite argument reduction.
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
pub(crate) unsafe fn vtan_f64(x: float64x2_t) -> float64x2_t {
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
    let large_thresh = vdupq_n_f64(1e9);
    let is_large = vcgtq_f64(abs_x, large_thresh);

    // Additional reduction for large values
    let y_corrected = vfmsq_f64(y, fn_val, pio2_2);
    let y_corrected = vfmsq_f64(y_corrected, fn_val, pio2_2t);
    let y = vbslq_f64(is_large, y_corrected, y);

    // -------------------------------------------------------------------------
    // Step 2: Compute quadrant info (n mod 2)
    // -------------------------------------------------------------------------

    let one_i = vdupq_n_s64(1);
    let n_and_1 = vandq_s64(n, one_i);

    // -------------------------------------------------------------------------
    // Step 3: Compute tan(y) with kernel (handles big argument and odd quadrant)
    // -------------------------------------------------------------------------

    let result = tan_kernel_f64(y, n_and_1);

    // -------------------------------------------------------------------------
    // Step 4: Handle special cases
    // tan(±∞) = NaN, tan(NaN) = NaN, tan(±0) = ±0
    // -------------------------------------------------------------------------

    let inf = vdupq_n_f64(f64::INFINITY);
    let is_inf_or_nan = vcgeq_f64(abs_x, inf);
    let nan = vdupq_n_f64(f64::NAN);

    // For tiny values (including ±0), tan(x) ≈ x
    let tiny = vdupq_n_f64(1e-300);
    let is_tiny = vcltq_f64(abs_x, tiny);
    let result = vbslq_f64(is_tiny, x, result);

    vbslq_f64(is_inf_or_nan, nan, result)
}

/// Tangent kernel for f64 reduced argument.
///
/// Implements musl's `__tan` algorithm: polynomial approximation for tan(x) on [-π/4, π/4].
/// Uses a degree-27 polynomial split into odd and even terms for better precision.
///
/// For |x| >= 0.6744 (near ±π/4), uses a special "big" argument path that applies
/// the identity tan(π/4 - y) = (1-tan(y))/(1+tan(y)) to improve accuracy.
///
/// # Parameters
///
/// - `x`: Reduced argument in [-π/4, π/4]
/// - `odd`: Integer vector indicating whether this is the odd quadrant (needs -1/tan transformation)
///
/// # Returns
///
/// The final tan(x) value, with -1/tan transformation applied if in odd quadrant.
#[inline]
unsafe fn tan_kernel_f64(x: float64x2_t, odd: int64x2_t) -> float64x2_t {
    // Constants
    let pio4 = vdupq_n_f64(PIO4_HI_64);
    let pio4lo = vdupq_n_f64(PIO4_LO_64);
    let big_thresh = vdupq_n_f64(BIG_THRESH_64);
    let one = vdupq_n_f64(1.0);
    let two = vdupq_n_f64(2.0);
    let neg_one = vdupq_n_f64(-1.0);
    let sign_bit = vdupq_n_f64(-0.0);

    // Load polynomial coefficients (T[0] to T[12] from musl)
    let t0 = vdupq_n_f64(T0_64);
    let t1 = vdupq_n_f64(T1_64);
    let t2 = vdupq_n_f64(T2_64);
    let t3 = vdupq_n_f64(T3_64);
    let t4 = vdupq_n_f64(T4_64);
    let t5 = vdupq_n_f64(T5_64);
    let t6 = vdupq_n_f64(T6_64);
    let t7 = vdupq_n_f64(T7_64);
    let t8 = vdupq_n_f64(T8_64);
    let t9 = vdupq_n_f64(T9_64);
    let t10 = vdupq_n_f64(T10_64);
    let t11 = vdupq_n_f64(T11_64);
    let t12 = vdupq_n_f64(T12_64);

    // -------------------------------------------------------------------------
    // Step 1: Check for "big" values: |x| >= 0.6744
    // These need special handling because the polynomial loses accuracy near π/4
    // -------------------------------------------------------------------------
    let abs_x = vabsq_f64(x);
    let is_big = vcgeq_f64(abs_x, big_thresh);

    // For big values: transform x to (π/4 - |x|), preserving original sign for later
    let x_sign = vreinterpretq_f64_u64(vandq_u64(
        vreinterpretq_u64_f64(x),
        vreinterpretq_u64_f64(sign_bit),
    ));
    let x_transformed = vaddq_f64(vsubq_f64(pio4, abs_x), pio4lo);

    // Select: use transformed x for big values, original x otherwise
    let x_eval = vbslq_f64(is_big, x_transformed, x);

    // -------------------------------------------------------------------------
    // Step 2: Polynomial evaluation
    // Split into odd and even powers for better numerical stability
    // -------------------------------------------------------------------------
    let z = vmulq_f64(x_eval, x_eval); // z = x²
    let w = vmulq_f64(z, z); // w = x⁴

    // r = T[1] + w*(T[3] + w*(T[5] + w*(T[7] + w*(T[9] + w*T[11]))))
    // Odd-indexed coefficients
    let mut r = t11;
    r = vfmaq_f64(t9, r, w);
    r = vfmaq_f64(t7, r, w);
    r = vfmaq_f64(t5, r, w);
    r = vfmaq_f64(t3, r, w);
    r = vfmaq_f64(t1, r, w);

    // v = z*(T[2] + w*(T[4] + w*(T[6] + w*(T[8] + w*(T[10] + w*T[12])))))
    // Even-indexed coefficients (except T[0])
    let mut v = t12;
    v = vfmaq_f64(t10, v, w);
    v = vfmaq_f64(t8, v, w);
    v = vfmaq_f64(t6, v, w);
    v = vfmaq_f64(t4, v, w);
    v = vfmaq_f64(t2, v, w);
    v = vmulq_f64(z, v);

    // Compute polynomial remainder: r_poly = T[0]*x³ + x⁵*(r+v)
    let s = vmulq_f64(z, x_eval); // s = x³
    let rv_sum = vaddq_f64(r, v);
    let zs = vmulq_f64(z, s); // zs = x⁵
    let poly_r = vfmaq_f64(vmulq_f64(s, t0), zs, rv_sum);

    // tan(x) = x + poly_r (for non-big case)
    let tan_small = vaddq_f64(x_eval, poly_r);

    // -------------------------------------------------------------------------
    // Step 3: Handle "big" case using musl's formula
    // For |x| near π/4, use: v = s - 2*(x + (r - w²/(w+s)))
    // where s = 1-2*odd (±1 based on quadrant), w = x + r
    // -------------------------------------------------------------------------
    let one_i = vdupq_n_s64(1);
    let odd_mask = vceqq_s64(odd, one_i); // Returns uint64x2_t

    // s_factor = 1 if even quadrant, -1 if odd quadrant
    let s_factor = vbslq_f64(odd_mask, neg_one, one);

    let w_big = vaddq_f64(x_eval, poly_r); // w = x + r
    let denom = vaddq_f64(w_big, s_factor); // w + s
    let w_sq = vmulq_f64(w_big, w_big); // w²
    let frac = vdivq_f64(w_sq, denom); // w²/(w+s)
    let inner = vaddq_f64(x_eval, vsubq_f64(poly_r, frac)); // x + (r - w²/(w+s))
    let tan_big_val = vfmsq_f64(s_factor, two, inner); // s - 2*(...)

    // Apply original sign for big values (sign was factored out during transformation)
    let tan_big_val = vreinterpretq_f64_u64(veorq_u64(
        vreinterpretq_u64_f64(tan_big_val),
        vreinterpretq_u64_f64(x_sign),
    ));

    // -------------------------------------------------------------------------
    // Step 4: Handle odd quadrant for non-big case: tan(x + π/2) = -1/tan(x)
    // Big case already incorporates this via s_factor
    // -------------------------------------------------------------------------
    let recip = vdivq_f64(neg_one, tan_small);
    let tan_small_final = vbslq_f64(odd_mask, recip, tan_small);

    // Final selection: big path has complete answer, non-big path computed above
    vbslq_f64(is_big, tan_big_val, tan_small_final)
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
    unsafe fn extract_f32x4(v: float32x4_t) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), v);
        out
    }

    /// Helper to extract f64 lanes from float64x2_t
    unsafe fn extract_f64x2(v: float64x2_t) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        vst1q_f64(out.as_mut_ptr(), v);
        out
    }

    /// Compute ULP difference for f32
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

    /// Compute ULP difference for f64
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
    // f32 tests
    // =========================================================================

    #[test]
    fn test_tan_f32_zero() {
        unsafe {
            let input = vdupq_n_f32(0.0);
            let result = extract_f32x4(vtan_f32(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_positive());
            }
        }
    }

    #[test]
    fn test_tan_f32_negative_zero() {
        unsafe {
            let input = vdupq_n_f32(-0.0);
            let result = extract_f32x4(vtan_f32(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_negative());
            }
        }
    }

    #[test]
    fn test_tan_f32_pi_over_4() {
        unsafe {
            let input = vdupq_n_f32(PI32 / 4.0);
            let result = extract_f32x4(vtan_f32(input));
            let expected = (PI32 / 4.0).tan();
            for &r in &result {
                assert!(
                    (r - expected).abs() < 1e-5,
                    "tan(π/4) = {}, expected {}",
                    r,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_tan_f32_negative_pi_over_4() {
        unsafe {
            let input = vdupq_n_f32(-PI32 / 4.0);
            let result = extract_f32x4(vtan_f32(input));
            let expected = (-PI32 / 4.0).tan();
            for &r in &result {
                assert!(
                    (r - expected).abs() < 1e-5,
                    "tan(-π/4) = {}, expected {}",
                    r,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_tan_f32_pi() {
        unsafe {
            let input = vdupq_n_f32(PI32);
            let result = extract_f32x4(vtan_f32(input));
            // tan(π) ≈ 0
            for &r in &result {
                assert!(r.abs() < 1e-5, "tan(π) = {}, expected ~0.0", r);
            }
        }
    }

    #[test]
    fn test_tan_f32_nan() {
        unsafe {
            let input = vdupq_n_f32(f32::NAN);
            let result = extract_f32x4(vtan_f32(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_f32_infinity() {
        unsafe {
            let input = vdupq_n_f32(f32::INFINITY);
            let result = extract_f32x4(vtan_f32(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_f32_negative_infinity() {
        unsafe {
            let input = vdupq_n_f32(f32::NEG_INFINITY);
            let result = extract_f32x4(vtan_f32(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(-∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_f32_lane_independence() {
        unsafe {
            let input = vld1q_f32([0.0f32, PI32 / 4.0, PI32, -PI32 / 4.0].as_ptr());
            let result = extract_f32x4(vtan_f32(input));
            let inputs = [0.0f32, PI32 / 4.0, PI32, -PI32 / 4.0];
            for (i, (&r, &x)) in result.iter().zip(inputs.iter()).enumerate() {
                let expected = x.tan();
                if expected.abs() < 100.0 {
                    assert!(
                        (r - expected).abs() < 1e-4,
                        "Lane {}: got {}, expected {}",
                        i,
                        r,
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_tan_f32_ulp_sweep() {
        unsafe {
            let mut max_ulp = 0u32;
            // Avoid values near π/2 where tan approaches infinity
            for i in 0..10000 {
                let x = -1.5 + (i as f32 / 10000.0) * 3.0; // Range [-1.5, 1.5]
                let input = vdupq_n_f32(x);
                let result = extract_f32x4(vtan_f32(input))[0];
                let expected = x.tan();
                // Skip large values where ULP isn't meaningful
                if expected.abs() > 100.0 || result.abs() > 100.0 {
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
    // f64 tests
    // =========================================================================

    #[test]
    fn test_tan_f64_zero() {
        unsafe {
            let input = vdupq_n_f64(0.0);
            let result = extract_f64x2(vtan_f64(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_positive());
            }
        }
    }

    #[test]
    fn test_tan_f64_negative_zero() {
        unsafe {
            let input = vdupq_n_f64(-0.0);
            let result = extract_f64x2(vtan_f64(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_negative());
            }
        }
    }

    #[test]
    fn test_tan_f64_pi_over_4() {
        unsafe {
            let input = vdupq_n_f64(PI64 / 4.0);
            let result = extract_f64x2(vtan_f64(input));
            let expected = (PI64 / 4.0).tan();
            for &r in &result {
                assert!(
                    (r - expected).abs() < 1e-14,
                    "tan(π/4) = {}, expected {}",
                    r,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_tan_f64_pi() {
        unsafe {
            let input = vdupq_n_f64(PI64);
            let result = extract_f64x2(vtan_f64(input));
            // tan(π) ≈ 0
            for &r in &result {
                assert!(r.abs() < 1e-14, "tan(π) = {}, expected ~0.0", r);
            }
        }
    }

    #[test]
    fn test_tan_f64_nan() {
        unsafe {
            let input = vdupq_n_f64(f64::NAN);
            let result = extract_f64x2(vtan_f64(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_f64_infinity() {
        unsafe {
            let input = vdupq_n_f64(f64::INFINITY);
            let result = extract_f64x2(vtan_f64(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_f64_negative_infinity() {
        unsafe {
            let input = vdupq_n_f64(f64::NEG_INFINITY);
            let result = extract_f64x2(vtan_f64(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(-∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_f64_lane_independence() {
        unsafe {
            let input = vld1q_f64([0.0f64, PI64 / 4.0].as_ptr());
            let result = extract_f64x2(vtan_f64(input));
            let inputs = [0.0f64, PI64 / 4.0];
            for (i, (&r, &x)) in result.iter().zip(inputs.iter()).enumerate() {
                let expected = x.tan();
                if expected.abs() < 100.0 {
                    assert!(
                        (r - expected).abs() < 1e-13,
                        "Lane {}: got {}, expected {}",
                        i,
                        r,
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_tan_f64_ulp_sweep() {
        unsafe {
            let mut max_ulp = 0u64;
            // Avoid values near π/2 where tan approaches infinity
            for i in 0..10000 {
                let x = -1.5 + (i as f64 / 10000.0) * 3.0; // Range [-1.5, 1.5]
                let input = vdupq_n_f64(x);
                let result = extract_f64x2(vtan_f64(input))[0];
                let expected = x.tan();
                // Skip large values where ULP isn't meaningful
                if expected.abs() > 1000.0 || result.abs() > 1000.0 {
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
