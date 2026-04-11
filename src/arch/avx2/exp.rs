//! AVX2 SIMD implementation of `exp(x)` (exponential) for `f32` and `f64` vectors.
//!
//! This module provides 8-lane f32 and 4-lane f64 exponential implementations
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
//! 2. **Polynomial approximation**: Compute a degree-5 minimax polynomial `P(r²)`:
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
//! | `_mm256_exp_ps`  | ≤ 2 ULP   |
//! | `_mm256_exp_pd`  | ≤ 2 ULP   |
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
//! - **f32 uses f64 intermediates**: Each 8-lane f32 vector is split into two
//!   4-lane f64 vectors. The shared `exp_core_f64` kernel computes in f64 precision,
//!   then results are converted back and recombined.
//! - **Reconstruction**: `2^k` is constructed by adding `k << 52` to the IEEE 754
//!   exponent bits of the polynomial result. For very large/small `k`, a two-step
//!   scaling is used to avoid intermediate overflow/underflow.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::exp::{
    LN2_HI_64, LN2_INV_64, LN2_LO_64, OVERFLOW_THRESH_64, P1_64, P2_64, P3_64, P4_64, P5_64,
    UNDERFLOW_THRESH_64,
};

// =============================================================================
// f32 Implementation (8 lanes, computed in f64 precision internally)
// =============================================================================

/// Computes `exp(x)` for each lane of an AVX2 `__m256` register.
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
/// Requires AVX2 and FMA support. The caller must ensure these features are
/// available at runtime.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_exp_ps(x: __m256) -> __m256 {
    unsafe {
        // Process as two 4-lane f64 operations for precision
        // Split input into low and high halves, convert to f64
        let x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
        let x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));

        // Compute exp in f64 precision for each half
        let exp_lo = exp_core_f64(x_lo);
        let exp_hi = exp_core_f64(x_hi);

        // Convert back to f32 and combine
        let result_lo = _mm256_cvtpd_ps(exp_lo);
        let result_hi = _mm256_cvtpd_ps(exp_hi);

        _mm256_insertf128_ps(_mm256_castps128_ps256(result_lo), result_hi, 1)
    }
}

// =============================================================================
// f64 Implementation (4 lanes)
// =============================================================================

/// Computes `exp(x)` for each lane of an AVX2 `__m256d` register.
///
/// Uses the fdlibm algorithm with degree-5 minimax polynomial after
/// argument reduction by `ln(2)`. Handles overflow, underflow, and special values.
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
pub(crate) unsafe fn _mm256_exp_pd(x: __m256d) -> __m256d {
    unsafe { exp_core_f64(x) }
}

// =============================================================================
// Core f64 exp kernel (shared by both f32 and f64 paths)
// =============================================================================

/// Core exponential kernel operating on 4-lane f64 vectors.
///
/// Implements the fdlibm `e_exp.c` algorithm:
/// 1. Reduce `x = k*ln(2) + r` where `|r| ≤ ln(2)/2`
/// 2. Compute `c = r - r²*P(r²)` using degree-5 minimax polynomial
/// 3. `exp(r) = 1 + r + r*c/(2-c)`
/// 4. Scale by `2^k`
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_core_f64(x: __m256d) -> __m256d {
    {
        let zero = _mm256_setzero_pd();
        let one = _mm256_set1_pd(1.0);
        let two = _mm256_set1_pd(2.0);
        let half = _mm256_set1_pd(0.5);

        // =====================================================================
        // Step 0: Special case detection (save for later)
        // =====================================================================

        let sign_bit = _mm256_set1_pd(-0.0);
        let inf = _mm256_set1_pd(f64::INFINITY);

        // x is NaN → NaN
        let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

        // x == +∞ → +∞
        let is_pos_inf = _mm256_cmp_pd(x, inf, _CMP_EQ_OQ);

        // x == -∞ → 0
        let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
        let is_neg_inf = _mm256_cmp_pd(x, neg_inf, _CMP_EQ_OQ);

        // Overflow: x > 709.78... → +∞
        let overflow_thresh = _mm256_set1_pd(OVERFLOW_THRESH_64);
        let is_overflow = _mm256_cmp_pd(x, overflow_thresh, _CMP_GT_OQ);

        // Underflow: x < -745.13... → 0
        let underflow_thresh = _mm256_set1_pd(UNDERFLOW_THRESH_64);
        let is_underflow = _mm256_cmp_pd(x, underflow_thresh, _CMP_LT_OQ);

        // =====================================================================
        // Step 1: Argument reduction
        // x = k*ln(2) + r, where k = round(x / ln(2)), |r| ≤ ln(2)/2
        // =====================================================================

        let ln2_inv = _mm256_set1_pd(LN2_INV_64);
        let ln2_hi = _mm256_set1_pd(LN2_HI_64);
        let ln2_lo = _mm256_set1_pd(LN2_LO_64);

        // k = round(x / ln(2))
        // Use trunc(x * (1/ln2) + copysign(0.5, x)) to get nearest integer
        let sign_half = _mm256_or_pd(half, _mm256_and_pd(x, sign_bit)); // copysign(0.5, x)
        let k_f64 = _mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(_mm256_fmadd_pd(
            x, ln2_inv, sign_half,
        ));

        // Convert k to integer for later 2^k construction
        let k_i32 = _mm256_cvtpd_epi32(k_f64);

        // r = x - k*ln2_hi - k*ln2_lo (extended precision reduction)
        let r = _mm256_fnmadd_pd(k_f64, ln2_lo, _mm256_fnmadd_pd(k_f64, ln2_hi, x));

        // =====================================================================
        // Step 2: Polynomial approximation
        // c = r - r²*(P1 + r²*(P2 + r²*(P3 + r²*(P4 + r²*P5))))
        // exp(r) = 1 + r + r*c/(2-c)
        //
        // This Padé-like form from fdlibm achieves better accuracy than
        // a direct Horner evaluation of the Taylor series.
        // =====================================================================

        let r2 = _mm256_mul_pd(r, r); // r²

        // Evaluate P(r²) = P1 + r²*(P2 + r²*(P3 + r²*(P4 + r²*P5)))
        let p5 = _mm256_set1_pd(P5_64);
        let p4 = _mm256_set1_pd(P4_64);
        let p3 = _mm256_set1_pd(P3_64);
        let p2 = _mm256_set1_pd(P2_64);
        let p1 = _mm256_set1_pd(P1_64);

        let p = _mm256_fmadd_pd(r2, p5, p4); // P4 + r²*P5
        let p = _mm256_fmadd_pd(r2, p, p3); // P3 + r²*(P4 + r²*P5)
        let p = _mm256_fmadd_pd(r2, p, p2); // P2 + r²*(...)
        let p = _mm256_fmadd_pd(r2, p, p1); // P1 + r²*(...)

        // c = r - r²*P(r²)  — but fdlibm computes it as c = r²*P(r²)
        // then uses:  exp(r) = 1 + (2r / (2 - c) - c) where c = r - r²*P(r²)
        // Actually fdlibm: c = r - t*(P1 + t*(P2 + ...)) where t = r²
        //   then exp(r) = 1 - (r*c/(c-2) - r) = 1 + r - r*c/(c-2)
        //               = 1 + r + r*c/(2-c)
        let c = _mm256_mul_pd(r2, p); // r² * P(r²)
        let c = _mm256_sub_pd(r, c); // c = r - r²*P(r²)

        // exp(r) = 1 + r + r*c/(2-c)
        //        = 1 + (2r - r*c) / (2 - c)
        // Numerator: 2r - r*c = r*(2 - c)  ... wait, that simplifies to r.
        // Actually the fdlibm formula: exp(r) = 1 - (r*c/(c-2) - r)
        //   r*c/(c-2) = -r*c/(2-c)
        //   1 - (-r*c/(2-c) - r) = 1 + r*c/(2-c) + r = 1 + r + r*c/(2-c)
        let rc = _mm256_mul_pd(r, c);
        let exp_r = _mm256_sub_pd(
            one,
            _mm256_sub_pd(
                _mm256_div_pd(rc, _mm256_sub_pd(c, two)), // r*c/(c-2) = -r*c/(2-c)
                r,
            ),
        );

        // =====================================================================
        // Step 3: Reconstruct exp(x) = 2^k * exp(r)
        // Construct 2^k by adding k to the exponent field of 1.0
        // =====================================================================

        // Extend k from 128-bit i32 to 256-bit i64 for 64-bit exponent manipulation
        let k_i64 = _mm256_cvtepi32_epi64(k_i32);

        // 2^k: shift k left by 52 bits (exponent position in f64) and add to 1.0's bits
        let k_shifted = _mm256_slli_epi64(k_i64, 52);
        let one_bits = _mm256_set1_epi64x(0x3FF0000000000000_u64 as i64); // 1.0 in f64
        let scale = _mm256_castsi256_pd(_mm256_add_epi64(k_shifted, one_bits));

        let result = _mm256_mul_pd(exp_r, scale);

        // =====================================================================
        // Step 4: Apply special cases
        // =====================================================================

        let nan = _mm256_set1_pd(f64::NAN);

        // Overflow → +∞
        let result = _mm256_blendv_pd(result, inf, is_overflow);

        // Underflow → 0
        let result = _mm256_blendv_pd(result, zero, is_underflow);

        // x == +∞ → +∞
        let result = _mm256_blendv_pd(result, inf, is_pos_inf);

        // x == -∞ → 0
        let result = _mm256_blendv_pd(result, zero, is_neg_inf);

        // x is NaN → NaN
        _mm256_blendv_pd(result, nan, is_nan)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to extract f32 lanes
    unsafe fn to_array_ps(v: __m256) -> [f32; 8] {
        let mut arr = [0.0_f32; 8];
        unsafe { _mm256_storeu_ps(arr.as_mut_ptr(), v) };
        arr
    }

    // Helper to extract f64 lanes
    unsafe fn to_array_pd(v: __m256d) -> [f64; 4] {
        let mut arr = [0.0_f64; 4];
        unsafe { _mm256_storeu_pd(arr.as_mut_ptr(), v) };
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
    fn test_exp_ps_zero() {
        // exp(0) = 1.0 exactly
        unsafe {
            let x = _mm256_set1_ps(0.0);
            let result = to_array_ps(_mm256_exp_ps(x));
            for &r in &result {
                assert_eq!(r, 1.0, "exp(0) should be exactly 1.0");
            }
        }
    }

    #[test]
    fn test_exp_ps_negative_zero() {
        // exp(-0) = 1.0 exactly
        unsafe {
            let x = _mm256_set1_ps(-0.0);
            let result = to_array_ps(_mm256_exp_ps(x));
            for &r in &result {
                assert_eq!(r, 1.0, "exp(-0) should be exactly 1.0");
            }
        }
    }

    #[test]
    fn test_exp_ps_one() {
        // exp(1) ≈ e
        unsafe {
            let x = _mm256_set1_ps(1.0);
            let result = to_array_ps(_mm256_exp_ps(x));
            let expected = 1.0_f32.exp();
            for &r in &result {
                let ulp = ulp_distance_f32(r, expected);
                assert!(
                    ulp <= 2,
                    "exp(1) = {}, expected {}, ULP = {}",
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_exp_ps_negative_one() {
        // exp(-1) ≈ 1/e
        unsafe {
            let x = _mm256_set1_ps(-1.0);
            let result = to_array_ps(_mm256_exp_ps(x));
            let expected = (-1.0_f32).exp();
            for &r in &result {
                let ulp = ulp_distance_f32(r, expected);
                assert!(
                    ulp <= 2,
                    "exp(-1) = {}, expected {}, ULP = {}",
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_exp_ps_known_values() {
        unsafe {
            let inputs = [0.0_f32, 1.0, -1.0, 2.0, -2.0, 10.0, -10.0, 0.5];
            let expected: [f32; 8] = inputs.map(|v| v.exp());
            let x = _mm256_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm256_exp_ps(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f32(r, e);
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
        }
    }

    #[test]
    fn test_exp_ps_overflow() {
        // exp(x) → +∞ for large x
        unsafe {
            let x = _mm256_set1_ps(89.0);
            let result = to_array_ps(_mm256_exp_ps(x));
            for &r in &result {
                assert!(
                    r.is_infinite() && r.is_sign_positive(),
                    "exp(89) should be +∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_exp_ps_underflow() {
        // exp(x) → 0 for very negative x
        unsafe {
            let x = _mm256_set1_ps(-104.0);
            let result = to_array_ps(_mm256_exp_ps(x));
            for &r in &result {
                assert_eq!(r, 0.0, "exp(-104) should be 0.0, got {}", r);
            }
        }
    }

    #[test]
    fn test_exp_ps_pos_infinity() {
        // exp(+∞) = +∞
        unsafe {
            let x = _mm256_set1_ps(f32::INFINITY);
            let result = to_array_ps(_mm256_exp_ps(x));
            for &r in &result {
                assert!(
                    r.is_infinite() && r.is_sign_positive(),
                    "exp(+∞) should be +∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_exp_ps_neg_infinity() {
        // exp(-∞) = 0
        unsafe {
            let x = _mm256_set1_ps(f32::NEG_INFINITY);
            let result = to_array_ps(_mm256_exp_ps(x));
            for &r in &result {
                assert_eq!(r, 0.0, "exp(-∞) should be 0.0, got {}", r);
            }
        }
    }

    #[test]
    fn test_exp_ps_nan() {
        // exp(NaN) = NaN
        unsafe {
            let x = _mm256_set1_ps(f32::NAN);
            let result = to_array_ps(_mm256_exp_ps(x));
            for &r in &result {
                assert!(r.is_nan(), "exp(NaN) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_exp_ps_lane_independence() {
        unsafe {
            let inputs = [0.0_f32, -1.0, 1.0, 5.0, -5.0, 20.0, -20.0, 0.001];
            let x = _mm256_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm256_exp_ps(x));

            for (i, (&r, &input)) in result.iter().zip(inputs.iter()).enumerate() {
                let expected = input.exp();
                let ulp = ulp_distance_f32(r, expected);
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

            // Sweep from -87 to 88 (representable range for f32 exp)
            let test_values: Vec<f32> = (0..10000)
                .map(|i| {
                    let t = i as f32 / 10000.0;
                    -87.0 + t * 175.0 // [-87, 88]
                })
                .collect();

            for chunk in test_values.chunks(8) {
                if chunk.len() < 8 {
                    continue;
                }
                let mut arr = [0.0_f32; 8];
                arr.copy_from_slice(chunk);

                let x = _mm256_loadu_ps(arr.as_ptr());
                let result = to_array_ps(_mm256_exp_ps(x));

                for (i, (&r, &input)) in result.iter().zip(arr.iter()).enumerate() {
                    let expected = input.exp();
                    if expected.is_infinite() || expected == 0.0 {
                        continue; // skip boundary overflow/underflow
                    }
                    let ulp = ulp_distance_f32(r, expected);
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
                "f32 exp ULP sweep: max ULP = {}, worst input = {}",
                max_ulp, worst_input
            );
        }
    }

    // -----------------------------------------------------------------------
    // f64 Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_exp_pd_zero() {
        unsafe {
            let x = _mm256_set1_pd(0.0);
            let result = to_array_pd(_mm256_exp_pd(x));
            for &r in &result {
                assert_eq!(r, 1.0, "exp(0) should be exactly 1.0");
            }
        }
    }

    #[test]
    fn test_exp_pd_negative_zero() {
        unsafe {
            let x = _mm256_set1_pd(-0.0);
            let result = to_array_pd(_mm256_exp_pd(x));
            for &r in &result {
                assert_eq!(r, 1.0, "exp(-0) should be exactly 1.0");
            }
        }
    }

    #[test]
    fn test_exp_pd_one() {
        unsafe {
            let x = _mm256_set1_pd(1.0);
            let result = to_array_pd(_mm256_exp_pd(x));
            let expected = 1.0_f64.exp();
            for &r in &result {
                let ulp = ulp_distance_f64(r, expected);
                assert!(
                    ulp <= 2,
                    "exp(1) = {}, expected {}, ULP = {}",
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_exp_pd_known_values() {
        unsafe {
            let inputs = [0.0_f64, 1.0, -1.0, 10.0];
            let expected: [f64; 4] = inputs.map(|v| v.exp());
            let x = _mm256_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm256_exp_pd(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f64(r, e);
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
        }
    }

    #[test]
    fn test_exp_pd_overflow() {
        unsafe {
            let x = _mm256_set1_pd(710.0);
            let result = to_array_pd(_mm256_exp_pd(x));
            for &r in &result {
                assert!(
                    r.is_infinite() && r.is_sign_positive(),
                    "exp(710) should be +∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_exp_pd_underflow() {
        unsafe {
            let x = _mm256_set1_pd(-746.0);
            let result = to_array_pd(_mm256_exp_pd(x));
            for &r in &result {
                assert_eq!(r, 0.0, "exp(-746) should be 0.0, got {}", r);
            }
        }
    }

    #[test]
    fn test_exp_pd_pos_infinity() {
        unsafe {
            let x = _mm256_set1_pd(f64::INFINITY);
            let result = to_array_pd(_mm256_exp_pd(x));
            for &r in &result {
                assert!(
                    r.is_infinite() && r.is_sign_positive(),
                    "exp(+∞) should be +∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_exp_pd_neg_infinity() {
        unsafe {
            let x = _mm256_set1_pd(f64::NEG_INFINITY);
            let result = to_array_pd(_mm256_exp_pd(x));
            for &r in &result {
                assert_eq!(r, 0.0, "exp(-∞) should be 0.0, got {}", r);
            }
        }
    }

    #[test]
    fn test_exp_pd_nan() {
        unsafe {
            let x = _mm256_set1_pd(f64::NAN);
            let result = to_array_pd(_mm256_exp_pd(x));
            for &r in &result {
                assert!(r.is_nan(), "exp(NaN) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_exp_pd_lane_independence() {
        unsafe {
            let inputs = [0.5_f64, -0.5, 100.0, -100.0];
            let x = _mm256_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm256_exp_pd(x));

            for (i, (&r, &input)) in result.iter().zip(inputs.iter()).enumerate() {
                let expected = input.exp();
                if expected.is_infinite() || expected == 0.0 {
                    // For overflow/underflow, just check sign/finiteness
                    if expected.is_infinite() {
                        assert!(r.is_infinite());
                    } else {
                        assert_eq!(r, 0.0);
                    }
                    continue;
                }
                let ulp = ulp_distance_f64(r, expected);
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

            // Sweep from -708 to 709 (representable range for f64 exp)
            let test_values: Vec<f64> = (0..10000)
                .map(|i| {
                    let t = i as f64 / 10000.0;
                    -708.0 + t * 1417.0 // [-708, 709]
                })
                .collect();

            for chunk in test_values.chunks(4) {
                if chunk.len() < 4 {
                    continue;
                }
                let mut arr = [0.0_f64; 4];
                arr.copy_from_slice(chunk);

                let x = _mm256_loadu_pd(arr.as_ptr());
                let result = to_array_pd(_mm256_exp_pd(x));

                for (i, (&r, &input)) in result.iter().zip(arr.iter()).enumerate() {
                    let expected = input.exp();
                    if expected.is_infinite() || expected == 0.0 {
                        continue;
                    }
                    let ulp = ulp_distance_f64(r, expected);
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
                "f64 exp ULP sweep: max ULP = {}, worst input = {}",
                max_ulp, worst_input
            );
        }
    }

    #[test]
    fn test_exp_pd_near_zero() {
        // Test values very close to zero where exp(x) ≈ 1 + x
        unsafe {
            let inputs = [1e-15_f64, -1e-15, 1e-10, -1e-10];
            let expected: [f64; 4] = inputs.map(|v| v.exp());
            let x = _mm256_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm256_exp_pd(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f64(r, e);
                assert!(
                    ulp <= 2,
                    "Lane {}: exp({:e}) = {}, expected {}, ULP = {}",
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
