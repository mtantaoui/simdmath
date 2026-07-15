//! NEON SIMD implementation of `exp(x)` (exponential) for `f32` and `f64` vectors.
//!
//! This module provides 4-lane f32 and 2-lane f64 exponential implementations
//! using the fdlibm algorithm ported from musl libc's `e_exp.c`.
//!
//! # Algorithm
//!
//! The exponential function is computed using argument reduction and polynomial
//! approximation:
//!
//! 1. **Argument reduction**: Decompose `x = k*ln(2) + r` where `k = round(x / ln(2))`
//!    and `|r| ≤ ln(2)/2 ≈ 0.347`. The reduction uses extended-precision
//!    `ln(2) = ln2_hi + ln2_lo` to minimize cancellation.
//!
//! 2. **Polynomial approximation**: Compute a 5-term minimax polynomial `P(r²)`:
//!    ```text
//!    c = r - r²*(P1 + r²*(P2 + r²*(P3 + r²*(P4 + r²*P5))))
//!    exp(r) ≈ 1 + 2r / (2 - c) = 1 + r + r*c/(2 - c)
//!    ```
//!    This Padé-like form is more accurate than a direct Taylor expansion.
//!
//! 3. **Reconstruct**: `exp(x) = 2^k * exp(r)` by adding `k` to the IEEE 754
//!    exponent field.
//!
//! # Precision
//!
//! | Variant      | Max Error |
//! |--------------|-----------|
//! | `vexp_f32`   | ≤ 2 ULP   |
//! | `vexp_f64`   | ≤ 2 ULP   |
//!
//! # Special Values
//!
//! | Input        | Output   |
//! |--------------|----------|
//! | `0.0`        | `1.0`    |
//! | `-0.0`       | `1.0`    |
//! | `+∞`         | `+∞`     |
//! | `-∞`         | `0.0`    |
//! | `NaN`        | `NaN`    |
//! | `x > ~709.8` | `+∞` (overflow)  |
//! | `x < ~-745`  | `0.0` (underflow)|
//!
//! # Implementation Notes
//!
//! - **f32 uses f64 intermediates**: Each 4-lane f32 vector is split into two
//!   2-lane f64 vectors. The shared `exp_core_f64` kernel computes in f64 precision,
//!   then results are converted back and recombined.
//! - **NEON blending**: Uses `vbslq_f64(mask, true_val, false_val)` — note the
//!   argument order differs from x86 blendv.

use std::arch::aarch64::*;

use crate::arch::consts::exp::{
    LN2_HI_64, LN2_INV_64, LN2_LO_64, OVERFLOW_THRESH_64, P1_64, P2_64, P3_64, P4_64, P5_64,
    UNDERFLOW_THRESH_64,
};

// =============================================================================
// f32 Implementation (4 lanes, computed in f64 precision internally)
// =============================================================================

/// Computes `exp(x)` for each lane of a NEON `float32x4_t` register.
///
/// Uses the fdlibm algorithm: argument reduction to `|r| ≤ ln(2)/2` followed
/// by a Padé-like polynomial evaluation. Internal computations use f64 precision
/// for accuracy.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires NEON support. The caller must ensure this feature is available.
#[inline]
pub(crate) unsafe fn vexp_f32(x: float32x4_t) -> float32x4_t {
    unsafe {
        // Process as two 2-lane f64 operations for precision
        // Split input into low and high halves, convert to f64
        let x_lo = vcvt_f64_f32(vget_low_f32(x));
        let x_hi = vcvt_f64_f32(vget_high_f32(x));

        // Compute exp in f64 precision for each half
        let exp_lo = exp_core_f64(x_lo);
        let exp_hi = exp_core_f64(x_hi);

        // Convert back to f32 and combine
        let result_lo = vcvt_f32_f64(exp_lo);
        let result_hi = vcvt_f32_f64(exp_hi);

        vcombine_f32(result_lo, result_hi)
    }
}

// =============================================================================
// f64 Implementation (2 lanes)
// =============================================================================

/// Computes `exp(x)` for each lane of a NEON `float64x2_t` register.
///
/// Uses the fdlibm algorithm with 5-term minimax polynomial after
/// argument reduction by `ln(2)`. Handles overflow, underflow, and special values.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires NEON support. The caller must ensure this feature is available.
#[inline]
pub(crate) unsafe fn vexp_f64(x: float64x2_t) -> float64x2_t {
    unsafe { exp_core_f64(x) }
}

// =============================================================================
// Core f64 exp kernel (shared by both f32 and f64 paths)
// =============================================================================

/// Core exponential kernel operating on 2-lane f64 vectors (NEON).
///
/// Implements the fdlibm `e_exp.c` algorithm:
/// 1. Reduce `x = k*ln(2) + r` where `|r| ≤ ln(2)/2`
/// 2. Compute `c = r - r²*P(r²)` using 5-term minimax polynomial
/// 3. `exp(r) = 1 + r + r*c/(2-c)`
/// 4. Scale by `2^k`
#[inline]
unsafe fn exp_core_f64(x: float64x2_t) -> float64x2_t {
    unsafe {
        let zero = vdupq_n_f64(0.0);
        let one = vdupq_n_f64(1.0);
        let two = vdupq_n_f64(2.0);
        let half = vdupq_n_f64(0.5);

        // =====================================================================
        // Step 0: Special case detection (save for later)
        // =====================================================================

        let inf = vdupq_n_f64(f64::INFINITY);

        // x is NaN → NaN (NaN != NaN)
        let is_not_nan = vceqq_f64(x, x);
        let all_ones = vreinterpretq_u64_s64(vdupq_n_s64(-1));
        let is_nan = veorq_u64(is_not_nan, all_ones);

        // Overflow: x > 709.78... → +∞   (also catches x == +∞)
        let overflow_thresh = vdupq_n_f64(OVERFLOW_THRESH_64);
        let is_overflow = vcgtq_f64(x, overflow_thresh);

        // Underflow: x < -745.13... → 0   (also catches x == -∞)
        let underflow_thresh = vdupq_n_f64(UNDERFLOW_THRESH_64);
        let is_underflow = vcltq_f64(x, underflow_thresh);

        // =====================================================================
        // Step 1: Argument reduction
        // x = k*ln(2) + r, where k = round(x / ln(2)), |r| ≤ ln(2)/2
        // =====================================================================

        let ln2_inv = vdupq_n_f64(LN2_INV_64);
        let ln2_hi = vdupq_n_f64(LN2_HI_64);
        let ln2_lo = vdupq_n_f64(LN2_LO_64);

        // k = round(x / ln(2))
        // Use trunc(x * (1/ln2) + copysign(0.5, x)) to get nearest integer
        let sign_bit = vdupq_n_f64(-0.0);
        let x_sign = vandq_u64(vreinterpretq_u64_f64(x), vreinterpretq_u64_f64(sign_bit));
        let sign_half = vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(half), x_sign));
        let k_f64 = vrndq_f64(vfmaq_f64(sign_half, x, ln2_inv)); // trunc(x/ln2 + copysign(0.5,x))

        // Convert k to integer for later 2^k construction
        let k_i64 = vcvtq_s64_f64(k_f64);

        // r = x - k*ln2_hi - k*ln2_lo (extended precision reduction)
        // vfmsq_f64(a, b, c) = a - b*c
        let r = vfmsq_f64(vfmsq_f64(x, k_f64, ln2_hi), k_f64, ln2_lo);

        // =====================================================================
        // Step 2: Polynomial approximation
        // c = r - r²*(P1 + r²*(P2 + r²*(P3 + r²*(P4 + r²*P5))))
        // exp(r) = 1 - (r*c/(c-2) - r)
        // =====================================================================

        let r2 = vmulq_f64(r, r); // r²

        // Evaluate P(r²) = P1 + r²*(P2 + r²*(P3 + r²*(P4 + r²*P5)))
        // vfmaq_f64(c, a, b) = a*b + c
        let p = vfmaq_f64(vdupq_n_f64(P4_64), r2, vdupq_n_f64(P5_64)); // P4 + r²*P5
        let p = vfmaq_f64(vdupq_n_f64(P3_64), r2, p); // P3 + r²*(...)
        let p = vfmaq_f64(vdupq_n_f64(P2_64), r2, p); // P2 + r²*(...)
        let p = vfmaq_f64(vdupq_n_f64(P1_64), r2, p); // P1 + r²*(...)

        // c = r - r²*P(r²)
        let c = vsubq_f64(r, vmulq_f64(r2, p));

        // exp(r) = 1 - (r*c/(c-2) - r) = 1 + r + r*c/(2-c)
        let rc = vmulq_f64(r, c);
        let c_minus_2 = vsubq_f64(c, two);
        let exp_r = vsubq_f64(one, vsubq_f64(vdivq_f64(rc, c_minus_2), r));

        // =====================================================================
        // Step 3: Reconstruct exp(x) = 2^k * exp(r)
        // =====================================================================

        // 2^k: shift k left by 52 bits and add to 1.0's exponent
        let k_shifted = vshlq_n_s64::<52>(k_i64);
        let one_bits = vdupq_n_s64(0x3FF0000000000000_u64 as i64);
        let scale = vreinterpretq_f64_s64(vaddq_s64(k_shifted, one_bits));

        let result = vmulq_f64(exp_r, scale);

        // =====================================================================
        // Step 4: Apply special cases
        // =====================================================================

        let nan = vdupq_n_f64(f64::NAN);

        // Overflow → +∞, underflow → 0. The two masks are disjoint, so the
        // special value and the combined mask build in parallel and a single
        // select applies both (instead of a serial select per case).
        let special_val = vbslq_f64(is_overflow, inf, zero);
        let is_special = vorrq_u64(is_overflow, is_underflow);
        let result = vbslq_f64(is_special, special_val, result);

        // x is NaN → NaN
        vbslq_f64(is_nan, nan, result)
    }
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
        unsafe { vst1q_f32(out.as_mut_ptr(), v) };
        out
    }

    /// Helper to extract f64 lanes from float64x2_t
    unsafe fn extract_f64x2(v: float64x2_t) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        unsafe { vst1q_f64(out.as_mut_ptr(), v) };
        out
    }

    use crate::test_utils::{ulp_diff_f32, ulp_diff_f64};

    // =========================================================================
    // f32 tests
    // =========================================================================

    #[test]
    fn test_exp_f32_special_values() {
        unsafe {
            // exp(±0) = 1 exactly
            for x in [0.0f32, -0.0] {
                let r = extract_f32x4(vexp_f32(vdupq_n_f32(x)));
                assert_eq!(r[0], 1.0, "exp({x}) should be 1.0");
            }

            // Known finite values (ULP-checked)
            let inputs = [0.0_f32, 1.0, -1.0, 10.0];
            let x = vld1q_f32(inputs.as_ptr());
            let result = extract_f32x4(vexp_f32(x));
            for (i, (&r, &inp)) in result.iter().zip(inputs.iter()).enumerate() {
                let ulp = ulp_diff_f32(r, inp.exp());
                assert!(ulp <= 2, "Lane {i}: exp({inp}) ULP = {ulp}");
            }

            // Overflow and underflow
            let r = extract_f32x4(vexp_f32(vdupq_n_f32(89.0)));
            assert!(
                r[0].is_infinite() && r[0].is_sign_positive(),
                "exp(89) should be +∞"
            );

            let r = extract_f32x4(vexp_f32(vdupq_n_f32(-104.0)));
            assert_eq!(r[0], 0.0, "exp(-104) should be 0.0");

            // exp(+∞) = +∞, exp(-∞) = 0, exp(NaN) = NaN
            let r = extract_f32x4(vexp_f32(vdupq_n_f32(f32::INFINITY)));
            assert!(
                r[0].is_infinite() && r[0].is_sign_positive(),
                "exp(+∞) should be +∞"
            );

            let r = extract_f32x4(vexp_f32(vdupq_n_f32(f32::NEG_INFINITY)));
            assert_eq!(r[0], 0.0, "exp(-∞) should be 0.0");

            let r = extract_f32x4(vexp_f32(vdupq_n_f32(f32::NAN)));
            assert!(r[0].is_nan(), "exp(NaN) should be NaN");
        }
    }

    #[test]
    fn test_exp_f32_lane_independence() {
        unsafe {
            let inputs = [0.0_f32, -1.0, 1.0, 5.0];
            let x = vld1q_f32(inputs.as_ptr());
            let result = extract_f32x4(vexp_f32(x));

            for (i, (&r, &input)) in result.iter().zip(inputs.iter()).enumerate() {
                let expected = input.exp();
                let ulp = ulp_diff_f32(r, expected);
                assert!(
                    ulp <= 2,
                    "Lane {}: exp({}) = {}, expected {}, ULP = {}",
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
    fn test_exp_f32_ulp_sweep() {
        unsafe {
            let mut max_ulp: u32 = 0;
            let mut worst_input = 0.0_f32;

            let test_values: Vec<f32> = (0..10000)
                .map(|i| {
                    let t = i as f32 / 10000.0;
                    -87.0 + t * 175.0
                })
                .collect();

            for chunk in test_values.chunks(4) {
                if chunk.len() < 4 {
                    continue;
                }
                let mut arr = [0.0_f32; 4];
                arr.copy_from_slice(chunk);

                let x = vld1q_f32(arr.as_ptr());
                let result = extract_f32x4(vexp_f32(x));

                for (i, (&r, &input)) in result.iter().zip(arr.iter()).enumerate() {
                    let expected = input.exp();
                    if expected.is_infinite() || expected == 0.0 {
                        continue;
                    }
                    let ulp = ulp_diff_f32(r, expected);
                    if ulp > max_ulp {
                        max_ulp = ulp;
                        worst_input = input;
                    }
                    assert!(
                        ulp <= 2,
                        "Lane {}: exp({}) = {} (bits: {:08x}), expected {} (bits: {:08x}), ULP = {}",
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
                "f32 exp ULP sweep (NEON): max ULP = {}, worst input = {}",
                max_ulp, worst_input
            );
        }
    }

    // =========================================================================
    // f64 tests
    // =========================================================================

    #[test]
    fn test_exp_f64_special_values() {
        unsafe {
            // exp(±0) = 1 exactly
            for x in [0.0f64, -0.0] {
                let r = extract_f64x2(vexp_f64(vdupq_n_f64(x)));
                assert_eq!(r[0], 1.0, "exp({x}) should be 1.0");
            }

            // Known finite values (ULP-checked), including near-zero
            for inp in [1.0f64, -1.0, 1e-15, -1e-15] {
                let r = extract_f64x2(vexp_f64(vdupq_n_f64(inp)));
                let ulp = ulp_diff_f64(r[0], inp.exp());
                assert!(ulp <= 2, "exp({inp}) ULP = {ulp}");
            }

            // Overflow and underflow
            let r = extract_f64x2(vexp_f64(vdupq_n_f64(710.0)));
            assert!(
                r[0].is_infinite() && r[0].is_sign_positive(),
                "exp(710) should be +∞"
            );

            let r = extract_f64x2(vexp_f64(vdupq_n_f64(-746.0)));
            assert_eq!(r[0], 0.0, "exp(-746) should be 0.0");

            // exp(+∞) = +∞, exp(-∞) = 0, exp(NaN) = NaN
            let r = extract_f64x2(vexp_f64(vdupq_n_f64(f64::INFINITY)));
            assert!(
                r[0].is_infinite() && r[0].is_sign_positive(),
                "exp(+∞) should be +∞"
            );

            let r = extract_f64x2(vexp_f64(vdupq_n_f64(f64::NEG_INFINITY)));
            assert_eq!(r[0], 0.0, "exp(-∞) should be 0.0");

            let r = extract_f64x2(vexp_f64(vdupq_n_f64(f64::NAN)));
            assert!(r[0].is_nan(), "exp(NaN) should be NaN");
        }
    }

    #[test]
    fn test_exp_f64_lane_independence() {
        unsafe {
            let inputs = [0.5_f64, -10.0];
            let x = vld1q_f64(inputs.as_ptr());
            let result = extract_f64x2(vexp_f64(x));

            for (i, (&r, &input)) in result.iter().zip(inputs.iter()).enumerate() {
                let expected = input.exp();
                let ulp = ulp_diff_f64(r, expected);
                assert!(
                    ulp <= 2,
                    "Lane {}: exp({}) = {}, expected {}, ULP = {}",
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
    fn test_exp_f64_ulp_sweep() {
        unsafe {
            let mut max_ulp: u64 = 0;
            let mut worst_input = 0.0_f64;

            let test_values: Vec<f64> = (0..10000)
                .map(|i| {
                    let t = i as f64 / 10000.0;
                    -708.0 + t * 1417.0
                })
                .collect();

            for chunk in test_values.chunks(2) {
                if chunk.len() < 2 {
                    continue;
                }
                let mut arr = [0.0_f64; 2];
                arr.copy_from_slice(chunk);

                let x = vld1q_f64(arr.as_ptr());
                let result = extract_f64x2(vexp_f64(x));

                for (i, (&r, &input)) in result.iter().zip(arr.iter()).enumerate() {
                    let expected = input.exp();
                    if expected.is_infinite() || expected == 0.0 {
                        continue;
                    }
                    let ulp = ulp_diff_f64(r, expected);
                    if ulp > max_ulp {
                        max_ulp = ulp;
                        worst_input = input;
                    }
                    assert!(
                        ulp <= 2,
                        "Lane {}: exp({}) = {} (bits: {:016x}), expected {} (bits: {:016x}), ULP = {}",
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
                "f64 exp ULP sweep (NEON): max ULP = {}, worst input = {}",
                max_ulp, worst_input
            );
        }
    }
}
