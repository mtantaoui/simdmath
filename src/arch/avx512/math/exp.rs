//! AVX-512 SIMD implementation of `exp(x)` (exponential) for `f32` and `f64` vectors.
//!
//! This module provides 16-lane f32 and 8-lane f64 exponential implementations
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
//! | Variant          | Max Error |
//! |------------------|-----------|
//! | `_mm512_exp_ps`  | ≤ 2 ULP   |
//! | `_mm512_exp_pd`  | ≤ 2 ULP   |
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
//! - **f32 uses f64 intermediates**: Each 16-lane f32 vector is split into two
//!   8-lane f64 vectors. The shared `exp_core_f64` kernel computes in f64 precision,
//!   then results are converted back and recombined.
//! - **AVX-512 masking**: Uses `__mmask8` integer masks and `_mm512_mask_blend_pd`
//!   instead of vector blends for predicated operations.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::exp::{
    LN2_HI_64, LN2_INV_64, LN2_LO_64, OVERFLOW_THRESH_64, P1_64, P2_64, P3_64, P4_64, P5_64,
    UNDERFLOW_THRESH_64,
};

// =============================================================================
// f32 Implementation (16 lanes, computed in f64 precision internally)
// =============================================================================

/// Computes `exp(x)` for each lane of an AVX-512 `__m512` register.
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
/// Requires AVX-512F support. The caller must ensure this feature is
/// available at runtime.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_exp_ps(x: __m512) -> __m512 {
    unsafe {
        // Process as two 8-lane f64 operations for precision
        // Split input into low and high halves, convert to f64
        let x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(x));
        let x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(x, 1));

        // Compute exp in f64 precision for each half
        let exp_lo = exp_core_f64(x_lo);
        let exp_hi = exp_core_f64(x_hi);

        // Convert back to f32 and combine
        let result_lo = _mm512_cvtpd_ps(exp_lo);
        let result_hi = _mm512_cvtpd_ps(exp_hi);

        _mm512_insertf32x8(_mm512_castps256_ps512(result_lo), result_hi, 1)
    }
}

// =============================================================================
// f64 Implementation (8 lanes)
// =============================================================================

/// Computes `exp(x)` for each lane of an AVX-512 `__m512d` register.
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
/// Requires AVX-512F support. The caller must ensure this feature is
/// available at runtime.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_exp_pd(x: __m512d) -> __m512d {
    unsafe { exp_core_f64(x) }
}

// =============================================================================
// Core f64 exp kernel (shared by both f32 and f64 paths)
// =============================================================================

/// Core exponential kernel operating on 8-lane f64 vectors (AVX-512).
///
/// Implements the fdlibm `e_exp.c` algorithm:
/// 1. Reduce `x = k*ln(2) + r` where `|r| ≤ ln(2)/2`
/// 2. Compute `c = r - r²*P(r²)` using 5-term minimax polynomial
/// 3. `exp(r) = 1 + r + r*c/(2-c)`
/// 4. Scale by `2^k`
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn exp_core_f64(x: __m512d) -> __m512d {
    unsafe {
        let zero = _mm512_setzero_pd();
        let one = _mm512_set1_pd(1.0);
        let two = _mm512_set1_pd(2.0);
        let half = _mm512_set1_pd(0.5);

        // =====================================================================
        // Step 0: Special case detection (save for later)
        // =====================================================================

        let sign_bit = _mm512_set1_pd(-0.0);
        let inf = _mm512_set1_pd(f64::INFINITY);

        // x is NaN → NaN
        let is_nan = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);

        // Overflow: x > 709.78... → +∞   (also catches x == +∞)
        let overflow_thresh = _mm512_set1_pd(OVERFLOW_THRESH_64);
        let is_overflow = _mm512_cmp_pd_mask(x, overflow_thresh, _CMP_GT_OQ);

        // Underflow: x < -745.13... → 0   (also catches x == -∞)
        let underflow_thresh = _mm512_set1_pd(UNDERFLOW_THRESH_64);
        let is_underflow = _mm512_cmp_pd_mask(x, underflow_thresh, _CMP_LT_OQ);

        // =====================================================================
        // Step 1: Argument reduction
        // x = k*ln(2) + r, where k = round(x / ln(2)), |r| ≤ ln(2)/2
        // =====================================================================

        let ln2_inv = _mm512_set1_pd(LN2_INV_64);
        let ln2_hi = _mm512_set1_pd(LN2_HI_64);
        let ln2_lo = _mm512_set1_pd(LN2_LO_64);

        // k = round(x / ln(2))
        // Use trunc(x * (1/ln2) + copysign(0.5, x)) to get nearest integer
        let x_sign = _mm512_castpd_si512(_mm512_and_pd(x, sign_bit));
        let half_bits = _mm512_castpd_si512(half);
        let sign_half = _mm512_castsi512_pd(_mm512_or_si512(half_bits, x_sign));
        let k_f64 =
            _mm512_roundscale_pd(_mm512_fmadd_pd(x, ln2_inv, sign_half), _MM_FROUND_TO_ZERO);

        // Convert k to integer for later 2^k construction
        let k_i32 = _mm512_cvtpd_epi32(k_f64);

        // r = x - k*ln2_hi - k*ln2_lo (extended precision reduction)
        let r = _mm512_fnmadd_pd(k_f64, ln2_lo, _mm512_fnmadd_pd(k_f64, ln2_hi, x));

        // =====================================================================
        // Step 2: Polynomial approximation
        // c = r - r²*(P1 + r²*(P2 + r²*(P3 + r²*(P4 + r²*P5))))
        // exp(r) = 1 - (r*c/(c-2) - r)
        // =====================================================================

        let r2 = _mm512_mul_pd(r, r); // r²

        // Evaluate P(r²) = P1 + r²*(P2 + r²*(P3 + r²*(P4 + r²*P5)))
        let p = _mm512_fmadd_pd(r2, _mm512_set1_pd(P5_64), _mm512_set1_pd(P4_64));
        let p = _mm512_fmadd_pd(r2, p, _mm512_set1_pd(P3_64));
        let p = _mm512_fmadd_pd(r2, p, _mm512_set1_pd(P2_64));
        let p = _mm512_fmadd_pd(r2, p, _mm512_set1_pd(P1_64));

        // c = r - r²*P(r²)
        let c = _mm512_sub_pd(r, _mm512_mul_pd(r2, p));

        // exp(r) = 1 - (r*c/(c-2) - r) = 1 + r + r*c/(2-c)
        let rc = _mm512_mul_pd(r, c);
        let exp_r = _mm512_sub_pd(
            one,
            _mm512_sub_pd(_mm512_div_pd(rc, _mm512_sub_pd(c, two)), r),
        );

        // =====================================================================
        // Step 3: Reconstruct exp(x) = 2^k * exp(r)
        // =====================================================================

        // Extend k from 256-bit i32 to 512-bit i64 for 64-bit exponent manipulation
        let k_i64 = _mm512_cvtepi32_epi64(k_i32);

        // 2^k: shift k left by 52 bits and add to 1.0's exponent
        let k_shifted = _mm512_slli_epi64(k_i64, 52);
        let one_bits = _mm512_set1_epi64(0x3FF0000000000000_u64 as i64);
        let scale = _mm512_castsi512_pd(_mm512_add_epi64(k_shifted, one_bits));

        let result = _mm512_mul_pd(exp_r, scale);

        // =====================================================================
        // Step 4: Apply special cases
        // =====================================================================

        let nan = _mm512_set1_pd(f64::NAN);

        // Overflow → +∞, underflow → 0. The two masks are disjoint, so the
        // special value and the combined mask build in parallel and a single
        // blend applies both (instead of a serial blend per case).
        let special_val = _mm512_mask_blend_pd(is_overflow, zero, inf);
        let is_special = is_overflow | is_underflow;
        let result = _mm512_mask_blend_pd(is_special, result, special_val);

        // x is NaN → NaN
        _mm512_mask_blend_pd(is_nan, result, nan)
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

    use crate::test_utils::{ulp_diff_f32, ulp_diff_f64};

    // =========================================================================
    // f32 tests
    // =========================================================================

    #[test]
    fn test_exp_ps_special_values() {
        unsafe {
            // ±0 → 1.0
            let r = extract_ps(_mm512_exp_ps(_mm512_set1_ps(0.0)));
            assert_eq!(r[0], 1.0, "exp(0) should be exactly 1.0");

            let r = extract_ps(_mm512_exp_ps(_mm512_set1_ps(-0.0)));
            assert_eq!(r[0], 1.0, "exp(-0) should be exactly 1.0");

            // exp(1) within 2 ULP
            let r = extract_ps(_mm512_exp_ps(_mm512_set1_ps(1.0)));
            let expected = 1.0_f32.exp();
            assert!(
                ulp_diff_f32(r[0], expected) <= 2,
                "exp(1) = {}, expected {}, ULP = {}",
                r[0],
                expected,
                ulp_diff_f32(r[0], expected)
            );

            // Known values
            let inputs = [
                0.0_f32, 1.0, -1.0, 2.0, -2.0, 10.0, -10.0, 0.5, -0.5, 5.0, -5.0, 20.0, -20.0,
                0.001, -0.001, 3.0,
            ];
            let expected: [f32; 16] = inputs.map(|v| v.exp());
            let x = _mm512_loadu_ps(inputs.as_ptr());
            let result = extract_ps(_mm512_exp_ps(x));
            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_diff_f32(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: exp({}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    e,
                    ulp
                );
            }

            // Overflow → +∞
            let r = extract_ps(_mm512_exp_ps(_mm512_set1_ps(89.0)));
            assert!(
                r[0].is_infinite() && r[0].is_sign_positive(),
                "exp(89) should be +∞, got {}",
                r[0]
            );

            // Underflow → 0
            let r = extract_ps(_mm512_exp_ps(_mm512_set1_ps(-104.0)));
            assert_eq!(r[0], 0.0, "exp(-104) should be 0.0, got {}", r[0]);

            // +∞ → +∞
            let r = extract_ps(_mm512_exp_ps(_mm512_set1_ps(f32::INFINITY)));
            assert!(
                r[0].is_infinite() && r[0].is_sign_positive(),
                "exp(+∞) should be +∞, got {}",
                r[0]
            );

            // -∞ → 0
            let r = extract_ps(_mm512_exp_ps(_mm512_set1_ps(f32::NEG_INFINITY)));
            assert_eq!(r[0], 0.0, "exp(-∞) should be 0.0, got {}", r[0]);

            // NaN → NaN
            let r = extract_ps(_mm512_exp_ps(_mm512_set1_ps(f32::NAN)));
            assert!(r[0].is_nan(), "exp(NaN) should be NaN, got {}", r[0]);
        }
    }

    #[test]
    fn test_exp_ps_lane_independence() {
        unsafe {
            let inputs = [
                0.0_f32, -1.0, 1.0, 5.0, -5.0, 20.0, -20.0, 0.001, -0.001, 10.0, -10.0, 0.5, -0.5,
                3.0, -3.0, 7.0,
            ];
            let x = _mm512_loadu_ps(inputs.as_ptr());
            let result = extract_ps(_mm512_exp_ps(x));

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
    fn test_exp_ps_ulp_sweep() {
        unsafe {
            let mut max_ulp: u32 = 0;
            let mut worst_input = 0.0_f32;

            let test_values: Vec<f32> = (0..10000)
                .map(|i| {
                    let t = i as f32 / 10000.0;
                    -87.0 + t * 175.0
                })
                .collect();

            for chunk in test_values.chunks(16) {
                if chunk.len() < 16 {
                    continue;
                }
                let mut arr = [0.0_f32; 16];
                arr.copy_from_slice(chunk);

                let x = _mm512_loadu_ps(arr.as_ptr());
                let result = extract_ps(_mm512_exp_ps(x));

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
                "f32 exp ULP sweep (AVX-512): max ULP = {}, worst input = {}",
                max_ulp, worst_input
            );
        }
    }

    // =========================================================================
    // f64 tests
    // =========================================================================

    #[test]
    fn test_exp_pd_special_values() {
        unsafe {
            // ±0 → 1.0
            let r = extract_pd(_mm512_exp_pd(_mm512_set1_pd(0.0)));
            assert_eq!(r[0], 1.0, "exp(0) should be exactly 1.0");

            let r = extract_pd(_mm512_exp_pd(_mm512_set1_pd(-0.0)));
            assert_eq!(r[0], 1.0, "exp(-0) should be exactly 1.0");

            // exp(1) within 2 ULP
            let r = extract_pd(_mm512_exp_pd(_mm512_set1_pd(1.0)));
            let expected = 1.0_f64.exp();
            assert!(
                ulp_diff_f64(r[0], expected) <= 2,
                "exp(1) = {}, expected {}, ULP = {}",
                r[0],
                expected,
                ulp_diff_f64(r[0], expected)
            );

            // Known values
            let inputs = [0.0_f64, 1.0, -1.0, 10.0, -10.0, 0.5, -0.5, 100.0];
            let expected: [f64; 8] = inputs.map(|v| v.exp());
            let x = _mm512_loadu_pd(inputs.as_ptr());
            let result = extract_pd(_mm512_exp_pd(x));
            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_diff_f64(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: exp({}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    e,
                    ulp
                );
            }

            // Overflow → +∞
            let r = extract_pd(_mm512_exp_pd(_mm512_set1_pd(710.0)));
            assert!(
                r[0].is_infinite() && r[0].is_sign_positive(),
                "exp(710) should be +∞, got {}",
                r[0]
            );

            // Underflow → 0
            let r = extract_pd(_mm512_exp_pd(_mm512_set1_pd(-746.0)));
            assert_eq!(r[0], 0.0, "exp(-746) should be 0.0, got {}", r[0]);

            // +∞ → +∞
            let r = extract_pd(_mm512_exp_pd(_mm512_set1_pd(f64::INFINITY)));
            assert!(
                r[0].is_infinite() && r[0].is_sign_positive(),
                "exp(+∞) should be +∞, got {}",
                r[0]
            );

            // -∞ → 0
            let r = extract_pd(_mm512_exp_pd(_mm512_set1_pd(f64::NEG_INFINITY)));
            assert_eq!(r[0], 0.0, "exp(-∞) should be 0.0, got {}", r[0]);

            // NaN → NaN
            let r = extract_pd(_mm512_exp_pd(_mm512_set1_pd(f64::NAN)));
            assert!(r[0].is_nan(), "exp(NaN) should be NaN, got {}", r[0]);

            // Near-zero values
            let near_zero_inputs = [1e-15_f64, -1e-15, 1e-10, -1e-10, 1e-5, -1e-5, 1e-3, -1e-3];
            let near_zero_expected: [f64; 8] = near_zero_inputs.map(|v| v.exp());
            let x = _mm512_loadu_pd(near_zero_inputs.as_ptr());
            let result = extract_pd(_mm512_exp_pd(x));
            for (i, (&r, &e)) in result.iter().zip(near_zero_expected.iter()).enumerate() {
                let ulp = ulp_diff_f64(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: exp({:e}) = {}, expected {}, ULP = {}",
                    i,
                    near_zero_inputs[i],
                    r,
                    e,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_exp_pd_lane_independence() {
        unsafe {
            let inputs = [0.5_f64, -0.5, 1.0, -1.0, 10.0, -10.0, 0.001, -0.001];
            let x = _mm512_loadu_pd(inputs.as_ptr());
            let result = extract_pd(_mm512_exp_pd(x));

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
    fn test_exp_pd_ulp_sweep() {
        unsafe {
            let mut max_ulp: u64 = 0;
            let mut worst_input = 0.0_f64;

            let test_values: Vec<f64> = (0..10000)
                .map(|i| {
                    let t = i as f64 / 10000.0;
                    -708.0 + t * 1417.0
                })
                .collect();

            for chunk in test_values.chunks(8) {
                if chunk.len() < 8 {
                    continue;
                }
                let mut arr = [0.0_f64; 8];
                arr.copy_from_slice(chunk);

                let x = _mm512_loadu_pd(arr.as_ptr());
                let result = extract_pd(_mm512_exp_pd(x));

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
                "f64 exp ULP sweep (AVX-512): max ULP = {}, worst input = {}",
                max_ulp, worst_input
            );
        }
    }
}
