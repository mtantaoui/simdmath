//! NEON SIMD implementation of `ln(x)` (natural logarithm) for `f32` and `f64` vectors.
//!
//! This module provides 4-lane f32 and 2-lane f64 natural logarithm implementations
//! using the fdlibm algorithm ported from musl libc's `e_log.c` / `e_logf.c`.
//!
//! # Algorithm
//!
//! The natural logarithm is computed using argument decomposition and polynomial
//! approximation:
//!
//! 1. **Decompose**: Write `x = 2^k * m` where `m ∈ [√2/2, √2]`.
//! 2. **Substitute**: Let `f = m - 1`, `s = f / (2 + f)`.
//!    Since `m` is near 1, `|f| < 0.414` and `|s| < 0.166`.
//! 3. **Polynomial**: Approximate `log(1+f)` via a degree-7 minimax polynomial
//!    in `z = s²`:
//!    ```text
//!    R(z) = Lg1*z + Lg2*z² + Lg3*z³ + Lg4*z⁴ + Lg5*z⁵ + Lg6*z⁶ + Lg7*z⁷
//!    ```
//! 4. **Reconstruct**: `ln(x) = k*ln2_hi - ((hfsq - (s*(hfsq+R) + k*ln2_lo)) - f)`
//!    where `hfsq = 0.5 * f²`.
//!
//! The split `ln(2) = ln2_hi + ln2_lo` ensures that `k * ln2_hi` is exact
//! for small integer `k`.
//!
//! # Precision
//!
//! | Variant      | Max Error |
//! |--------------|-----------|
//! | `vln_f32`    | ≤ 2 ULP   |
//! | `vln_f64`    | ≤ 2 ULP   |
//!
//! # Special Values
//!
//! | Input       | Output   |
//! |-------------|----------|
//! | `0.0`       | `-∞`     |
//! | `-0.0`      | `-∞`     |
//! | `1.0`       | `0.0`    |
//! | `e`         | `1.0`    |
//! | `+∞`        | `+∞`     |
//! | `x < 0`     | `NaN`    |
//! | `NaN`       | `NaN`    |
//!
//! # Implementation Notes
//!
//! - **f32 uses f64 intermediates**: Each 4-lane f32 vector is split into two
//!   2-lane f64 vectors. The shared `ln_core_f64` kernel computes in f64 precision,
//!   then results are converted back and recombined.
//! - **Subnormal handling (f64)**: Subnormal inputs are multiplied by `2^52` to
//!   normalize them, with a corresponding adjustment to the exponent `k`.
//! - **NEON blending**: Uses `vbslq_f64(mask, true_val, false_val)` — note the
//!   argument order differs from x86 blendv.

use std::arch::aarch64::*;

use crate::arch::consts::ln::{
    LG1_64, LG2_64, LG3_64, LG4_64, LG5_64, LG6_64, LG7_64, LN2_HI_64, LN2_LO_64, SQRT2_64,
    TWO52_64,
};

// =============================================================================
// f32 Implementation (4 lanes, computed in f64 precision internally)
// =============================================================================

/// Computes `ln(x)` for each lane of a NEON `float32x4_t` register.
///
/// Uses the fdlibm algorithm: decompose `x = 2^k * m`, then compute
/// `log(m)` via a minimax polynomial in the transformed variable `s = f/(2+f)`.
/// Internal computations use f64 precision for accuracy.
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
pub(crate) unsafe fn vln_f32(x: float32x4_t) -> float32x4_t {
    // Process as two 2-lane f64 operations for precision
    // Split input into low and high halves, convert to f64
    let x_lo = vcvt_f64_f32(vget_low_f32(x));
    let x_hi = vcvt_f64_f32(vget_high_f32(x));

    // Compute ln in f64 precision for each half
    let ln_lo = ln_core_f64(x_lo);
    let ln_hi = ln_core_f64(x_hi);

    // Convert back to f32 and combine
    let result_lo = vcvt_f32_f64(ln_lo);
    let result_hi = vcvt_f32_f64(ln_hi);

    vcombine_f32(result_lo, result_hi)
}

// =============================================================================
// f64 Implementation (2 lanes)
// =============================================================================

/// Computes `ln(x)` for each lane of a NEON `float64x2_t` register.
///
/// Uses the fdlibm algorithm with 7-term minimax polynomial in `s² = (f/(2+f))²`.
/// Handles the full f64 domain including subnormals, zeros, negatives, and infinities.
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
pub(crate) unsafe fn vln_f64(x: float64x2_t) -> float64x2_t {
    ln_core_f64(x)
}

// =============================================================================
// Core f64 ln kernel (shared by both f32 and f64 paths)
// =============================================================================

/// Core natural logarithm kernel operating on 2-lane f64 vectors (NEON).
///
/// Implements the fdlibm `e_log.c` algorithm:
/// 1. Extract exponent `k` and normalize mantissa to `[√2/2, √2]`
/// 2. Let `f = m - 1`, `s = f / (2 + f)`
/// 3. Evaluate minimax polynomial `R(z)` where `z = s²`
/// 4. Reconstruct: `result = k*ln2_hi - ((hfsq - (s*(hfsq+R) + k*ln2_lo)) - f)`
#[inline]
unsafe fn ln_core_f64(x: float64x2_t) -> float64x2_t {
    let zero = vdupq_n_f64(0.0);
    let one = vdupq_n_f64(1.0);
    let half = vdupq_n_f64(0.5);

    // =====================================================================
    // Step 0: Special case detection (save for later)
    // =====================================================================

    let abs_x = vabsq_f64(x);
    let inf = vdupq_n_f64(f64::INFINITY);

    // x < 0 (excluding -0) → NaN
    let is_negative = vcltq_f64(x, zero);

    // x == 0 or x == -0 → -∞
    let is_zero = vceqq_f64(abs_x, zero);

    // x == +∞ → +∞
    let is_pos_inf = vceqq_f64(x, inf);

    // x is NaN → NaN (NaN != NaN)
    let is_not_nan = vceqq_f64(x, x);
    let all_ones = vreinterpretq_u64_s64(vdupq_n_s64(-1));
    let is_nan = veorq_u64(is_not_nan, all_ones);

    // =====================================================================
    // Step 1: Handle subnormals by scaling up
    // =====================================================================

    // Subnormal: 0 < |x| < 2^-1022 (smallest normal)
    let min_normal = vdupq_n_f64(f64::MIN_POSITIVE); // 2^-1022
    let is_subnormal = vandq_u64(vcgtq_f64(abs_x, zero), vcltq_f64(abs_x, min_normal));

    // Scale subnormals by 2^52 to make them normal
    let two52 = vdupq_n_f64(TWO52_64);
    let x_scaled = vmulq_f64(x, two52);
    let x_work = vbslq_f64(is_subnormal, x_scaled, x);

    // Subnormal exponent adjustment: subtract 52 from k later
    let neg_52 = vdupq_n_f64(-52.0);
    let k_adjust = vbslq_f64(is_subnormal, neg_52, zero);

    // =====================================================================
    // Step 2: Extract exponent k and normalize mantissa
    // =====================================================================

    // Get IEEE 754 bit pattern
    let ix = vreinterpretq_s64_f64(x_work);

    // Extract biased exponent: bits [52:62] → shift right 52
    let exp_bits = vshrq_n_s64::<52>(ix);

    // Convert to f64: k = biased_exponent - 1023
    let bias = vdupq_n_f64(1023.0);
    let k = vsubq_f64(vcvtq_f64_s64(exp_bits), bias);

    // Apply subnormal adjustment
    let k = vaddq_f64(k, k_adjust);

    // Normalize mantissa: clear exponent bits, set exponent to 1023 (range [1, 2))
    let mantissa_mask = vdupq_n_s64(0x000FFFFFFFFFFFFF_u64 as i64);
    let exp_1023 = vdupq_n_s64(0x3FF0000000000000_u64 as i64);
    let m_bits = vorrq_s64(vandq_s64(ix, mantissa_mask), exp_1023);
    let m = vreinterpretq_f64_s64(m_bits);

    // If m > √2, halve it (set exponent to 1022 → range [0.5, 1)) and increment k
    let sqrt2 = vdupq_n_f64(SQRT2_64);
    let is_big = vcgtq_f64(m, sqrt2);

    // For big m: divide by 2 (subtract 1 from exponent field)
    let exp_1022 = vdupq_n_s64(0x3FE0000000000000_u64 as i64);
    let m_halved_bits = vorrq_s64(vandq_s64(ix, mantissa_mask), exp_1022);
    let m_halved = vreinterpretq_f64_s64(m_halved_bits);

    let m = vbslq_f64(is_big, m_halved, m);
    let k_inc = vbslq_f64(is_big, one, zero);
    let k = vaddq_f64(k, k_inc); // k++ if big

    // =====================================================================
    // Step 3: Compute f = m - 1, s = f / (2 + f)
    // =====================================================================

    let f = vsubq_f64(m, one);
    let two_plus_f = vaddq_f64(vdupq_n_f64(2.0), f);
    let s = vdivq_f64(f, two_plus_f);

    // hfsq = 0.5 * f * f
    let hfsq = vmulq_f64(half, vmulq_f64(f, f));

    // =====================================================================
    // Step 4: Evaluate minimax polynomial R(z) where z = s²
    // =====================================================================

    let z = vmulq_f64(s, s); // z = s²
    let w = vmulq_f64(z, z); // w = s⁴

    // Split into odd and even powers for better ILP (instruction-level parallelism)
    // Odd terms:  t1 = Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7))
    // Even terms: t2 = Lg2 + w*(Lg4 + w*Lg6)
    let lg1 = vdupq_n_f64(LG1_64);
    let lg2 = vdupq_n_f64(LG2_64);
    let lg3 = vdupq_n_f64(LG3_64);
    let lg4 = vdupq_n_f64(LG4_64);
    let lg5 = vdupq_n_f64(LG5_64);
    let lg6 = vdupq_n_f64(LG6_64);
    let lg7 = vdupq_n_f64(LG7_64);

    // t1 = Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7))
    let t1 = vfmaq_f64(lg5, w, lg7); // Lg5 + w*Lg7
    let t1 = vfmaq_f64(lg3, w, t1); // Lg3 + w*(Lg5 + w*Lg7)
    let t1 = vfmaq_f64(lg1, w, t1); // Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7))

    // t2 = Lg2 + w * (Lg4 + w * Lg6)
    let t2 = vfmaq_f64(lg4, w, lg6); // Lg4 + w*Lg6
    let t2 = vfmaq_f64(lg2, w, t2); // Lg2 + w*(Lg4 + w*Lg6)

    // R = z*(t1 + z*t2)
    let r = vfmaq_f64(t1, z, t2); // t1 + z*t2
    let r = vmulq_f64(z, r); // z*(t1 + z*t2)

    // =====================================================================
    // Step 5: Reconstruct ln(x) = k*ln2_hi - ((hfsq - (s*(hfsq+R) + k*ln2_lo)) - f)
    // =====================================================================

    let ln2_hi = vdupq_n_f64(LN2_HI_64);
    let ln2_lo = vdupq_n_f64(LN2_LO_64);

    // s * (hfsq + R)
    let s_term = vmulq_f64(s, vaddq_f64(hfsq, r));

    // k * ln2_lo
    let k_ln2_lo = vmulq_f64(k, ln2_lo);

    // inner = s*(hfsq+R) + k*ln2_lo
    let inner = vaddq_f64(s_term, k_ln2_lo);

    // hfsq - inner
    let hfsq_minus_inner = vsubq_f64(hfsq, inner);

    // (hfsq - inner) - f  →  this is the negative of the log(1+f) part
    let log_part = vsubq_f64(hfsq_minus_inner, f);

    // result = k*ln2_hi - log_part
    // vfmsq_f64(c, a, b) = c - a*b, but we want k*ln2_hi - log_part
    // = fma(k, ln2_hi, -log_part) = -log_part + k*ln2_hi
    let result = vfmaq_f64(vnegq_f64(log_part), k, ln2_hi);

    // =====================================================================
    // Step 6: Apply special cases
    // =====================================================================

    let nan = vdupq_n_f64(f64::NAN);
    let neg_inf = vdupq_n_f64(f64::NEG_INFINITY);

    // x == 0 or x == -0 → -∞
    let result = vbslq_f64(is_zero, neg_inf, result);

    // x == +∞ → +∞
    let result = vbslq_f64(is_pos_inf, inf, result);

    // x < 0 → NaN
    let result = vbslq_f64(is_negative, nan, result);

    // x is NaN → NaN (propagate)
    let result = vbslq_f64(is_nan, nan, result);

    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_ln_f32_one() {
        unsafe {
            let input = vdupq_n_f32(1.0);
            let result = extract_f32x4(vln_f32(input));
            for &r in &result {
                assert_eq!(r, 0.0, "ln(1.0) should be exactly 0.0");
                assert_eq!(r.to_bits(), 0, "ln(1.0) should be +0.0");
            }
        }
    }

    #[test]
    fn test_ln_f32_zero() {
        unsafe {
            let input = vdupq_n_f32(0.0);
            let result = extract_f32x4(vln_f32(input));
            for &r in &result {
                assert!(
                    r.is_infinite() && r.is_sign_negative(),
                    "ln(0) should be -∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_ln_f32_negative_zero() {
        unsafe {
            let input = vdupq_n_f32(-0.0);
            let result = extract_f32x4(vln_f32(input));
            for &r in &result {
                assert!(
                    r.is_infinite() && r.is_sign_negative(),
                    "ln(-0) should be -∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_ln_f32_negative() {
        unsafe {
            let inputs = [-1.0_f32, -2.0, -0.5, -100.0];
            let input = vld1q_f32(inputs.as_ptr());
            let result = extract_f32x4(vln_f32(input));
            for (i, &r) in result.iter().enumerate() {
                assert!(
                    r.is_nan(),
                    "Lane {}: ln({}) should be NaN, got {}",
                    i,
                    inputs[i],
                    r
                );
            }
        }
    }

    #[test]
    fn test_ln_f32_infinity() {
        unsafe {
            let input = vdupq_n_f32(f32::INFINITY);
            let result = extract_f32x4(vln_f32(input));
            for &r in &result {
                assert!(
                    r.is_infinite() && r.is_sign_positive(),
                    "ln(+∞) should be +∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_ln_f32_nan() {
        unsafe {
            let input = vdupq_n_f32(f32::NAN);
            let result = extract_f32x4(vln_f32(input));
            for &r in &result {
                assert!(r.is_nan(), "ln(NaN) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_ln_f32_known_values() {
        unsafe {
            let inputs = [std::f32::consts::E, 10.0, 0.5, 2.0];
            let expected: [f32; 4] = inputs.map(|v| v.ln());
            let x = vld1q_f32(inputs.as_ptr());
            let result = extract_f32x4(vln_f32(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_diff_f32(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: ln({}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    e,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_ln_f32_powers_of_two() {
        unsafe {
            let inputs = [2.0_f32, 4.0, 8.0, 16.0];
            let expected: [f32; 4] = inputs.map(|v| v.ln());
            let x = vld1q_f32(inputs.as_ptr());
            let result = extract_f32x4(vln_f32(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_diff_f32(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: ln({}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    e,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_ln_f32_lane_independence() {
        unsafe {
            let inputs = [0.5_f32, 1.0, 2.0, 100.0];
            let x = vld1q_f32(inputs.as_ptr());
            let result = extract_f32x4(vln_f32(x));

            for (i, (&r, &input)) in result.iter().zip(inputs.iter()).enumerate() {
                let expected = input.ln();
                let ulp = ulp_diff_f32(r, expected);
                assert!(
                    ulp <= 2,
                    "Lane {}: ln({}) = {}, expected {}, ULP = {}",
                    i,
                    input,
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_ln_f32_ulp_sweep() {
        unsafe {
            let mut max_ulp: u32 = 0;
            let mut worst_input = 0.0_f32;

            let test_values: Vec<f32> = (0..10000)
                .map(|i| {
                    let t = i as f32 / 10000.0;
                    (1e-30_f32).powf(1.0 - t) * (1e30_f32).powf(t)
                })
                .collect();

            for chunk in test_values.chunks(4) {
                if chunk.len() < 4 {
                    continue;
                }
                let mut arr = [0.0_f32; 4];
                arr.copy_from_slice(chunk);

                let x = vld1q_f32(arr.as_ptr());
                let result = extract_f32x4(vln_f32(x));

                for (i, (&r, &input)) in result.iter().zip(arr.iter()).enumerate() {
                    let expected = input.ln();
                    let ulp = ulp_diff_f32(r, expected);
                    if ulp > max_ulp {
                        max_ulp = ulp;
                        worst_input = input;
                    }
                    assert!(
                        ulp <= 2,
                        "Lane {}: ln({}) = {} (bits: {:08x}), expected {} (bits: {:08x}), ULP = {}",
                        i,
                        input,
                        r,
                        r.to_bits(),
                        expected,
                        expected.to_bits(),
                        ulp
                    );
                }
            }

            eprintln!(
                "f32 ln ULP sweep (NEON): max ULP = {}, worst input = {}",
                max_ulp, worst_input
            );
        }
    }

    // =========================================================================
    // f64 tests
    // =========================================================================

    #[test]
    fn test_ln_f64_one() {
        unsafe {
            let input = vdupq_n_f64(1.0);
            let result = extract_f64x2(vln_f64(input));
            for &r in &result {
                assert_eq!(r, 0.0, "ln(1.0) should be exactly 0.0");
                assert_eq!(r.to_bits(), 0, "ln(1.0) should be +0.0");
            }
        }
    }

    #[test]
    fn test_ln_f64_zero() {
        unsafe {
            let input = vdupq_n_f64(0.0);
            let result = extract_f64x2(vln_f64(input));
            for &r in &result {
                assert!(
                    r.is_infinite() && r.is_sign_negative(),
                    "ln(0) should be -∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_ln_f64_negative_zero() {
        unsafe {
            let input = vdupq_n_f64(-0.0);
            let result = extract_f64x2(vln_f64(input));
            for &r in &result {
                assert!(
                    r.is_infinite() && r.is_sign_negative(),
                    "ln(-0) should be -∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_ln_f64_negative() {
        unsafe {
            let inputs = [-1.0_f64, -100.0];
            let input = vld1q_f64(inputs.as_ptr());
            let result = extract_f64x2(vln_f64(input));
            for (i, &r) in result.iter().enumerate() {
                assert!(
                    r.is_nan(),
                    "Lane {}: ln({}) should be NaN, got {}",
                    i,
                    inputs[i],
                    r
                );
            }
        }
    }

    #[test]
    fn test_ln_f64_infinity() {
        unsafe {
            let input = vdupq_n_f64(f64::INFINITY);
            let result = extract_f64x2(vln_f64(input));
            for &r in &result {
                assert!(
                    r.is_infinite() && r.is_sign_positive(),
                    "ln(+∞) should be +∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_ln_f64_nan() {
        unsafe {
            let input = vdupq_n_f64(f64::NAN);
            let result = extract_f64x2(vln_f64(input));
            for &r in &result {
                assert!(r.is_nan(), "ln(NaN) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_ln_f64_known_values() {
        unsafe {
            let inputs = [std::f64::consts::E, 10.0];
            let expected: [f64; 2] = inputs.map(|v| v.ln());
            let x = vld1q_f64(inputs.as_ptr());
            let result = extract_f64x2(vln_f64(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_diff_f64(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: ln({}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    e,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_ln_f64_powers_of_two() {
        unsafe {
            let inputs = [2.0_f64, 8.0];
            let expected: [f64; 2] = inputs.map(|v| v.ln());
            let x = vld1q_f64(inputs.as_ptr());
            let result = extract_f64x2(vln_f64(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_diff_f64(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: ln({}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    e,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_ln_f64_lane_independence() {
        unsafe {
            let inputs = [0.01_f64, 1000.0];
            let x = vld1q_f64(inputs.as_ptr());
            let result = extract_f64x2(vln_f64(x));

            for (i, (&r, &input)) in result.iter().zip(inputs.iter()).enumerate() {
                let expected = input.ln();
                let ulp = ulp_diff_f64(r, expected);
                assert!(
                    ulp <= 2,
                    "Lane {}: ln({}) = {}, expected {}, ULP = {}",
                    i,
                    input,
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_ln_f64_ulp_sweep() {
        unsafe {
            let mut max_ulp: u64 = 0;
            let mut worst_input = 0.0_f64;

            let test_values: Vec<f64> = (0..10000)
                .map(|i| {
                    let t = i as f64 / 10000.0;
                    (1e-100_f64).powf(1.0 - t) * (1e100_f64).powf(t)
                })
                .collect();

            for chunk in test_values.chunks(2) {
                if chunk.len() < 2 {
                    continue;
                }
                let mut arr = [0.0_f64; 2];
                arr.copy_from_slice(chunk);

                let x = vld1q_f64(arr.as_ptr());
                let result = extract_f64x2(vln_f64(x));

                for (i, (&r, &input)) in result.iter().zip(arr.iter()).enumerate() {
                    let expected = input.ln();
                    let ulp = ulp_diff_f64(r, expected);
                    if ulp > max_ulp {
                        max_ulp = ulp;
                        worst_input = input;
                    }
                    assert!(
                        ulp <= 2,
                        "Lane {}: ln({}) = {} (bits: {:016x}), expected {} (bits: {:016x}), ULP = {}",
                        i,
                        input,
                        r,
                        r.to_bits(),
                        expected,
                        expected.to_bits(),
                        ulp
                    );
                }
            }

            eprintln!(
                "f64 ln ULP sweep (NEON): max ULP = {}, worst input = {}",
                max_ulp, worst_input
            );
        }
    }

    #[test]
    fn test_ln_f64_subnormal() {
        unsafe {
            let tiny = f64::MIN_POSITIVE * 0.5;
            let inputs = [tiny, f64::MIN_POSITIVE];
            let expected: [f64; 2] = inputs.map(|v| v.ln());
            let x = vld1q_f64(inputs.as_ptr());
            let result = extract_f64x2(vln_f64(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_diff_f64(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: ln({:e}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    e,
                    ulp
                );
            }
        }
    }
}
