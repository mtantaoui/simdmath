//! NEON SIMD implementation of `cbrt(x)` (cube root) for `f32` and `f64` vectors.
//!
//! # Algorithm
//!
//! The cube root exploits a clever property of IEEE 754 floating-point:
//! dividing the bit representation by 3 gives a rough approximation to the
//! cube root. This is refined with Newton–Raphson iterations.
//!
//! ## f32 — Two Newton iterations in f64
//!
//! 1. **Bit trick**: `hx/3 + B1` produces ~5 bits of accuracy.
//! 2. **Scale subnormals**: Multiply by `2^24` to normalize, use `B2` instead.
//! 3. **Newton iterations** (in f64 for precision):
//!    `t = t * (x + x + r) / (x + r + r)` where `r = t³`
//! 4. Convert back to f32 (rounding is perfect in round-to-nearest).
//!
//! ## f64 — Polynomial + Newton
//!
//! 1. **Bit trick**: Same approach with 64-bit constants (`B1_64`, `B2_64`).
//! 2. **Polynomial refinement** (~23 bits): `t = t * P(t³/x)` with degree-4 poly.
//! 3. **Round to 23 bits** away from zero (ensures `t*t` is exact).
//! 4. **One Newton iteration** to 53 bits: `t = t + t * (x/t² - t) / (2t + x/t²)`
//!
//! # Precision
//!
//! | Implementation | Accuracy     |
//! |----------------|--------------|
//! | `vcbrt_f32`    | ≤ 1 ULP      |
//! | `vcbrt_f64`    | ≤ 1 ULP      |
//!
//! ## Special values
//!
//! | Input   | Output  |
//! |---------|---------|
//! | `+0.0`  | `+0.0`  |
//! | `-0.0`  | `-0.0`  |
//! | `+∞`    | `+∞`    |
//! | `-∞`    | `-∞`    |
//! | `NaN`   | `NaN`   |
//! | `8.0`   | `2.0`   |
//! | `-8.0`  | `-2.0`  |
//!
//! ## NEON notes
//!
//! - `vbslq` argument order: `vbslq(mask, true_val, false_val)` (differs from x86!)
//! - FMA accumulator first: `vfmaq(c, a, b)` = a*b + c

use std::arch::aarch64::*;

use crate::arch::consts::cbrt::{
    B1_32, B1_64, B2_32, B2_64, P0, P1, P2, P3, P4, ROUND_BIAS_64, ROUND_MASK_64, X1P24_32,
    X1P54_64,
};

// ===========================================================================
// f32 Implementation (4 lanes)
// ===========================================================================

/// Divides each 32-bit unsigned integer lane by 3 using the multiplicative inverse trick.
///
/// Uses `vmull_u32` (widening 2-lane multiply) to get the high 32 bits of
/// `x * 0xAAAAAAAB`, then shifts right by 1.
#[inline]
unsafe fn div_by_3_neon_u32(x: uint32x4_t) -> uint32x4_t {
    unsafe {
        let magic = vdup_n_u32(0xAAAAAAAB);

        // Low 2 lanes: widening multiply, take high 32 bits
        let lo = vget_low_u32(x);
        let prod_lo = vmull_u32(lo, magic);
        let hi_lo = vshrn_n_u64::<32>(prod_lo);

        // High 2 lanes
        let hi = vget_high_u32(x);
        let prod_hi = vmull_u32(hi, magic);
        let hi_hi = vshrn_n_u64::<32>(prod_hi);

        // Combine and shift right by 1
        let combined = vcombine_u32(hi_lo, hi_hi);
        vshrq_n_u32::<1>(combined)
    }
}

/// Computes `cbrt(x)` (cube root) for each lane of a NEON `float32x4_t` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain including subnormals.
///
/// # Description
///
/// Uses a bit-manipulation trick for initial approximation followed by two
/// Newton–Raphson iterations in double precision. All 4 lanes are processed
/// simultaneously.
///
/// # Safety
///
/// `x` must be a valid `float32x4_t` register.
///
/// # Example
///
/// ```ignore
/// let x = vdupq_n_f32(27.0);
/// let result = vcbrt_f32(x);
/// // All 4 lanes ≈ 3.0
/// ```
#[inline]
pub(crate) unsafe fn vcbrt_f32(x: float32x4_t) -> float32x4_t {
    unsafe {
        // -----------------------------------------------------------------------
        // Constants
        // -----------------------------------------------------------------------
        let abs_mask = vdupq_n_u32(0x7fffffff);
        let sign_mask = vdupq_n_u32(0x80000000);
        let inf_threshold = vdupq_n_u32(0x7f800000);
        let subnormal_threshold = vdupq_n_u32(0x00800000);
        let b1 = vdupq_n_u32(B1_32);
        let b2 = vdupq_n_u32(B2_32);
        let zero_u32 = vdupq_n_u32(0);

        // -----------------------------------------------------------------------
        // Extract bit representation and sign
        // -----------------------------------------------------------------------
        let ui = vreinterpretq_u32_f32(x);
        let hx = vandq_u32(ui, abs_mask);
        let sign_bits = vandq_u32(ui, sign_mask);

        // -----------------------------------------------------------------------
        // Special case detection
        // -----------------------------------------------------------------------
        let is_inf_or_nan = vcgeq_u32(hx, inf_threshold);
        let is_zero = vceqq_u32(hx, zero_u32);
        let is_below_normal = vcltq_u32(hx, subnormal_threshold);
        let is_subnormal = vandq_u32(is_below_normal, vmvnq_u32(is_zero));

        // -----------------------------------------------------------------------
        // Initial approximation via bit manipulation
        // -----------------------------------------------------------------------

        // Normal case: hx/3 + B1
        let hx_normal = vaddq_u32(div_by_3_neon_u32(hx), b1);

        // Subnormal case: scale |x| by 2^24, recompute hx, then hx_scaled/3 + B2
        let x_abs = vreinterpretq_f32_u32(hx);
        let x_scaled = vmulq_f32(x_abs, vdupq_n_f32(X1P24_32));
        let hx_scaled = vandq_u32(vreinterpretq_u32_f32(x_scaled), abs_mask);
        let hx_subnormal = vaddq_u32(div_by_3_neon_u32(hx_scaled), b2);

        // Select between normal and subnormal paths
        let hx_approx = vbslq_u32(is_subnormal, hx_subnormal, hx_normal);

        // Restore sign and convert to float
        let t_f32 = vreinterpretq_f32_u32(vorrq_u32(sign_bits, hx_approx));

        // Avoid division by zero in Newton iterations: replace zero lanes with 1.0
        let x_safe = vbslq_f32(is_zero, vdupq_n_f32(1.0), x);

        // -----------------------------------------------------------------------
        // Newton–Raphson iterations in f64 (lower 2 lanes)
        // -----------------------------------------------------------------------
        let x_lo = vcvt_f64_f32(vget_low_f32(x_safe));
        let t_lo = vcvt_f64_f32(vget_low_f32(t_f32));

        // First iteration: t = t * (2x + t³) / (x + 2t³)
        let r_lo = vmulq_f64(vmulq_f64(t_lo, t_lo), t_lo);
        let two_x_lo = vaddq_f64(x_lo, x_lo);
        let t_lo = vmulq_f64(
            t_lo,
            vdivq_f64(
                vaddq_f64(two_x_lo, r_lo),
                vaddq_f64(x_lo, vaddq_f64(r_lo, r_lo)),
            ),
        );

        // Second iteration
        let r_lo = vmulq_f64(vmulq_f64(t_lo, t_lo), t_lo);
        let t_lo = vmulq_f64(
            t_lo,
            vdivq_f64(
                vaddq_f64(two_x_lo, r_lo),
                vaddq_f64(x_lo, vaddq_f64(r_lo, r_lo)),
            ),
        );

        // -----------------------------------------------------------------------
        // Newton–Raphson iterations in f64 (upper 2 lanes)
        // -----------------------------------------------------------------------
        let x_hi = vcvt_high_f64_f32(x_safe);
        let t_hi = vcvt_high_f64_f32(t_f32);

        // First iteration
        let r_hi = vmulq_f64(vmulq_f64(t_hi, t_hi), t_hi);
        let two_x_hi = vaddq_f64(x_hi, x_hi);
        let t_hi = vmulq_f64(
            t_hi,
            vdivq_f64(
                vaddq_f64(two_x_hi, r_hi),
                vaddq_f64(x_hi, vaddq_f64(r_hi, r_hi)),
            ),
        );

        // Second iteration
        let r_hi = vmulq_f64(vmulq_f64(t_hi, t_hi), t_hi);
        let t_hi = vmulq_f64(
            t_hi,
            vdivq_f64(
                vaddq_f64(two_x_hi, r_hi),
                vaddq_f64(x_hi, vaddq_f64(r_hi, r_hi)),
            ),
        );

        // -----------------------------------------------------------------------
        // Combine halves and handle special cases
        // -----------------------------------------------------------------------
        let result_lo = vcvt_f32_f64(t_lo);
        let result_hi = vcvt_f32_f64(t_hi);
        let mut result = vcombine_f32(result_lo, result_hi);

        // Zero: return x (preserves sign of ±0)
        result = vbslq_f32(is_zero, x, result);

        // Inf/NaN: return x + x (propagates NaN, returns ±∞ for ±∞)
        let inf_nan_result = vaddq_f32(x, x);
        result = vbslq_f32(is_inf_or_nan, inf_nan_result, result);

        result
    }
}

// ===========================================================================
// f64 Implementation (2 lanes)
// ===========================================================================

/// Divides each 32-bit value (sitting in the low 32 bits of 2 u64 lanes) by 3.
///
/// Narrows to `uint32x2_t`, uses `vmull_u32` widening multiply with the
/// magic constant, extracts the high 32 bits, and shifts right by 1.
/// Result remains in `uint64x2_t` lanes.
#[inline]
unsafe fn div_by_3_u32_in_u64(hx: uint64x2_t) -> uint64x2_t {
    unsafe {
        let hx_u32 = vmovn_u64(hx); // narrow 2×u64 → 2×u32 (low 32 bits)
        let magic = vdup_n_u32(0xAAAAAAAB);
        let prod = vmull_u32(hx_u32, magic); // 2×u64
        let hi = vshrq_n_u64::<32>(prod); // high 32 bits in u64 lanes
        vshrq_n_u64::<1>(hi) // >> 1
    }
}

/// Computes `cbrt(x)` (cube root) for each lane of a NEON `float64x2_t` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain including subnormals.
///
/// # Description
///
/// Uses musl libc's algorithm: bit trick for initial estimate, polynomial
/// refinement to ~23 bits, rounding step, then one Newton iteration to 53 bits.
///
/// # Safety
///
/// `x` must be a valid `float64x2_t` register.
///
/// # Example
///
/// ```ignore
/// let x = vdupq_n_f64(27.0);
/// let result = vcbrt_f64(x);
/// // Both lanes ≈ 3.0
/// ```
#[inline]
pub(crate) unsafe fn vcbrt_f64(x: float64x2_t) -> float64x2_t {
    unsafe {
        // -----------------------------------------------------------------------
        // Constants
        // -----------------------------------------------------------------------
        let sign_mask_u64 = vdupq_n_u64(1u64 << 63);
        let abs_mask_u32_in_u64 = vdupq_n_u64(0x7fffffff);
        let inf_thresh = vdupq_n_u64(0x7ff00000);
        let subnormal_thresh = vdupq_n_u64(0x00100000);
        let zero_u64 = vdupq_n_u64(0);

        // -----------------------------------------------------------------------
        // Extract bit representation
        // -----------------------------------------------------------------------
        let bits = vreinterpretq_u64_f64(x);

        // Upper 32 bits of each f64 (exponent + high mantissa)
        let hx_u64 = vshrq_n_u64::<32>(bits);

        // Absolute upper 32 bits
        let hx = vandq_u64(hx_u64, abs_mask_u32_in_u64);

        // Sign bit (bit 63)
        let sign = vandq_u64(bits, sign_mask_u64);

        // Absolute value bits for zero check
        let abs_bits = vandq_u64(bits, vdupq_n_u64(0x7fffffffffffffff));

        // -----------------------------------------------------------------------
        // Special case detection
        // -----------------------------------------------------------------------
        let is_inf_or_nan = vcgeq_u64(hx, inf_thresh);
        let is_zero = vceqq_u64(abs_bits, zero_u64);
        let is_below_normal = vcltq_u64(hx, subnormal_thresh);
        let all_ones_u64 = vreinterpretq_u64_s64(vdupq_n_s64(-1));
        let not_zero = veorq_u64(is_zero, all_ones_u64);
        let is_subnormal = vandq_u64(is_below_normal, not_zero);

        // -----------------------------------------------------------------------
        // Initial approximation via bit manipulation
        // -----------------------------------------------------------------------

        // Normal case: hx/3 + B1_64
        let hx_normal = vaddq_u64(div_by_3_u32_in_u64(hx), vdupq_n_u64(B1_64 as u64));

        // Subnormal case: scale |x| by 2^54, recompute hx, then hx_scaled/3 + B2_64
        let x_abs = vreinterpretq_f64_u64(abs_bits);
        let x_scaled = vmulq_f64(x_abs, vdupq_n_f64(X1P54_64));
        let scaled_bits = vreinterpretq_u64_f64(x_scaled);
        let hx_scaled = vandq_u64(vshrq_n_u64::<32>(scaled_bits), abs_mask_u32_in_u64);
        let hx_subnormal = vaddq_u64(div_by_3_u32_in_u64(hx_scaled), vdupq_n_u64(B2_64 as u64));

        // Select between normal and subnormal
        let hx_result = vbslq_u64(is_subnormal, hx_subnormal, hx_normal);

        // Reconstruct initial approximation: sign | (hx_result << 32)
        let approx_bits = vorrq_u64(sign, vshlq_n_u64::<32>(hx_result));
        let t = vreinterpretq_f64_u64(approx_bits);

        // Avoid division by zero: replace zero lanes with 1.0
        let x_safe = vbslq_f64(is_zero, vdupq_n_f64(1.0), x);

        // -----------------------------------------------------------------------
        // Polynomial refinement to ~23 bits: t = t * P(t³/x)
        // -----------------------------------------------------------------------
        let t2 = vmulq_f64(t, t);
        let r = vmulq_f64(t2, vdivq_f64(t, x_safe));

        // P(r) = (P0 + r*(P1 + r*P2)) + r³*(P3 + r*P4)
        let r2 = vmulq_f64(r, r);
        let r3 = vmulq_f64(r2, r);

        // P1 + r*P2  →  vfmaq_f64(P1, r, P2) = r*P2 + P1
        let p12 = vfmaq_f64(vdupq_n_f64(P1), r, vdupq_n_f64(P2));
        // P0 + r*(P1 + r*P2)  →  vfmaq_f64(P0, r, p12) = r*p12 + P0
        let low_part = vfmaq_f64(vdupq_n_f64(P0), r, p12);

        // P3 + r*P4  →  vfmaq_f64(P3, r, P4) = r*P4 + P3
        let p34 = vfmaq_f64(vdupq_n_f64(P3), r, vdupq_n_f64(P4));
        // r³ * (P3 + r*P4)
        let high_part = vmulq_f64(r3, p34);

        let poly = vaddq_f64(low_part, high_part);
        let t = vmulq_f64(t, poly);

        // -----------------------------------------------------------------------
        // Round t to 23 bits (away from zero) for exact t*t
        // -----------------------------------------------------------------------
        let t_bits = vreinterpretq_u64_f64(t);
        let rounded_bits = vandq_u64(
            vaddq_u64(t_bits, vdupq_n_u64(ROUND_BIAS_64)),
            vdupq_n_u64(ROUND_MASK_64),
        );
        let t = vreinterpretq_f64_u64(rounded_bits);

        // -----------------------------------------------------------------------
        // One Newton iteration to 53 bits
        // t = t + t * (r - t) / (w + r)  where s = t*t, r = x/s, w = t+t
        // -----------------------------------------------------------------------
        let s = vmulq_f64(t, t);
        let r = vdivq_f64(x_safe, s);
        let w = vaddq_f64(t, t);
        // t * (r - t) / (w + r)
        let diff = vsubq_f64(r, t);
        let denom = vaddq_f64(w, r);
        let correction = vmulq_f64(t, vdivq_f64(diff, denom));
        let t = vaddq_f64(t, correction);

        // -----------------------------------------------------------------------
        // Handle special cases
        // -----------------------------------------------------------------------
        // Zero: return x (preserves sign of ±0)
        let mut result = vbslq_f64(is_zero, x, t);

        // Inf/NaN: return x + x (propagates NaN, returns ±∞ for ±∞)
        let inf_nan_result = vaddq_f64(x, x);
        result = vbslq_f64(is_inf_or_nan, inf_nan_result, result);

        result
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to extract f32 lanes
    unsafe fn to_array_f32(v: float32x4_t) -> [f32; 4] {
        let mut arr = [0.0_f32; 4];
        unsafe { vst1q_f32(arr.as_mut_ptr(), v) };
        arr
    }

    // Helper to extract f64 lanes
    unsafe fn to_array_f64(v: float64x2_t) -> [f64; 2] {
        let mut arr = [0.0_f64; 2];
        unsafe { vst1q_f64(arr.as_mut_ptr(), v) };
        arr
    }

    // Compute ULP distance for f32
    fn ulp_distance_f32(a: f32, b: f32) -> u32 {
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

    // Compute ULP distance for f64
    fn ulp_distance_f64(a: f64, b: f64) -> u64 {
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

    // -----------------------------------------------------------------------
    // f32 Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cbrt_f32_perfect_cubes() {
        unsafe {
            let inputs = [1.0_f32, 8.0, 27.0, 64.0];
            let expected = [1.0_f32, 2.0, 3.0, 4.0];
            let x = vld1q_f32(inputs.as_ptr());
            let result = to_array_f32(vcbrt_f32(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f32(r, e);
                assert!(
                    ulp <= 1,
                    "Lane {}: cbrt({}) = {}, expected {}, ULP = {}",
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
    fn test_cbrt_f32_negative_values() {
        unsafe {
            let inputs = [-1.0_f32, -8.0, -27.0, -64.0];
            let expected = [-1.0_f32, -2.0, -3.0, -4.0];
            let x = vld1q_f32(inputs.as_ptr());
            let result = to_array_f32(vcbrt_f32(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f32(r, e);
                assert!(
                    ulp <= 1,
                    "Lane {}: cbrt({}) = {}, expected {}, ULP = {}",
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
    fn test_cbrt_f32_zero() {
        unsafe {
            let pos_zero = vdupq_n_f32(0.0);
            let neg_zero = vdupq_n_f32(-0.0);

            let result_pos = to_array_f32(vcbrt_f32(pos_zero));
            let result_neg = to_array_f32(vcbrt_f32(neg_zero));

            for &r in &result_pos {
                assert_eq!(r, 0.0);
                assert!(r.to_bits() == 0, "Should be +0.0");
            }
            for &r in &result_neg {
                assert_eq!(r, 0.0);
                assert!(r.to_bits() == 0x80000000, "Should be -0.0");
            }
        }
    }

    #[test]
    fn test_cbrt_f32_infinity() {
        unsafe {
            let pos_inf = vdupq_n_f32(f32::INFINITY);
            let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);

            let result_pos = to_array_f32(vcbrt_f32(pos_inf));
            let result_neg = to_array_f32(vcbrt_f32(neg_inf));

            for &r in &result_pos {
                assert!(r.is_infinite() && r > 0.0, "Should be +∞");
            }
            for &r in &result_neg {
                assert!(r.is_infinite() && r < 0.0, "Should be -∞");
            }
        }
    }

    #[test]
    fn test_cbrt_f32_nan() {
        unsafe {
            let nan = vdupq_n_f32(f32::NAN);
            let result = to_array_f32(vcbrt_f32(nan));

            for &r in &result {
                assert!(r.is_nan(), "cbrt(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_cbrt_f32_ulp_sweep() {
        unsafe {
            let test_values = [
                0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1000.0, 1e10, 1e20, 1e-10, 1e-20,
            ];

            for &val in &test_values {
                for sign in [1.0_f32, -1.0] {
                    let x_val = sign * val;
                    let x = vdupq_n_f32(x_val);
                    let result = to_array_f32(vcbrt_f32(x));
                    let expected = x_val.cbrt();

                    for &r in &result {
                        let ulp = ulp_distance_f32(r, expected);
                        assert!(
                            ulp <= 1,
                            "cbrt({}) = {}, expected {}, ULP = {}",
                            x_val,
                            r,
                            expected,
                            ulp
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // f64 Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cbrt_f64_perfect_cubes() {
        unsafe {
            let inputs = [1.0_f64, 8.0];
            let expected = [1.0_f64, 2.0];
            let x = vld1q_f64(inputs.as_ptr());
            let result = to_array_f64(vcbrt_f64(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f64(r, e);
                assert!(
                    ulp <= 1,
                    "Lane {}: cbrt({}) = {}, expected {}, ULP = {}",
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
    fn test_cbrt_f64_negative_values() {
        unsafe {
            let inputs = [-1.0_f64, -8.0];
            let expected = [-1.0_f64, -2.0];
            let x = vld1q_f64(inputs.as_ptr());
            let result = to_array_f64(vcbrt_f64(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f64(r, e);
                assert!(
                    ulp <= 1,
                    "Lane {}: cbrt({}) = {}, expected {}, ULP = {}",
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
    fn test_cbrt_f64_zero() {
        unsafe {
            let pos_zero = vdupq_n_f64(0.0);
            let neg_zero = vdupq_n_f64(-0.0);

            let result_pos = to_array_f64(vcbrt_f64(pos_zero));
            let result_neg = to_array_f64(vcbrt_f64(neg_zero));

            for &r in &result_pos {
                assert_eq!(r, 0.0);
                assert!(r.to_bits() == 0, "Should be +0.0");
            }
            for &r in &result_neg {
                assert_eq!(r, 0.0);
                assert!(r.to_bits() == (1_u64 << 63), "Should be -0.0");
            }
        }
    }

    #[test]
    fn test_cbrt_f64_infinity() {
        unsafe {
            let pos_inf = vdupq_n_f64(f64::INFINITY);
            let neg_inf = vdupq_n_f64(f64::NEG_INFINITY);

            let result_pos = to_array_f64(vcbrt_f64(pos_inf));
            let result_neg = to_array_f64(vcbrt_f64(neg_inf));

            for &r in &result_pos {
                assert!(r.is_infinite() && r > 0.0, "Should be +∞");
            }
            for &r in &result_neg {
                assert!(r.is_infinite() && r < 0.0, "Should be -∞");
            }
        }
    }

    #[test]
    fn test_cbrt_f64_nan() {
        unsafe {
            let nan = vdupq_n_f64(f64::NAN);
            let result = to_array_f64(vcbrt_f64(nan));

            for &r in &result {
                assert!(r.is_nan(), "cbrt(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_cbrt_f64_ulp_sweep() {
        unsafe {
            let test_values = [
                0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1000.0, 1e10, 1e50, 1e100,
                1e-10, 1e-50, 1e-100,
            ];

            for &val in &test_values {
                for sign in [1.0_f64, -1.0] {
                    let x_val = sign * val;
                    let x = vdupq_n_f64(x_val);
                    let result = to_array_f64(vcbrt_f64(x));
                    let expected = x_val.cbrt();

                    for &r in &result {
                        let ulp = ulp_distance_f64(r, expected);
                        assert!(
                            ulp <= 1,
                            "cbrt({}) = {}, expected {}, ULP = {}",
                            x_val,
                            r,
                            expected,
                            ulp
                        );
                    }
                }
            }
        }
    }
}
