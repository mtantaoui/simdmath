//! AVX2 SIMD implementation of `ln(x)` (natural logarithm) for `f32` and `f64` vectors.
//!
//! This module provides 8-lane f32 and 4-lane f64 natural logarithm implementations
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
//! | Variant          | Max Error |
//! |------------------|-----------|
//! | `_mm256_ln_ps`   | ≤ 2 ULP   |
//! | `_mm256_ln_pd`   | ≤ 2 ULP   |
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
//! - **f32 uses f64 intermediates**: Each 8-lane f32 vector is split into two
//!   4-lane f64 vectors. The shared `ln_core_f64` kernel computes in f64 precision,
//!   then results are converted back and recombined.
//! - **Subnormal handling (f64)**: Subnormal inputs are multiplied by `2^52` to
//!   normalize them, with a corresponding adjustment to the exponent `k`.
//! - **Exponent extraction**: Uses integer shifts and permutes to extract the
//!   biased exponent from IEEE 754 bit patterns.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::ln::{
    LG1_64, LG2_64, LG3_64, LG4_64, LG5_64, LG6_64, LG7_64, LN2_HI_64, LN2_LO_64, SQRT2_64,
    TWO52_64,
};

// =============================================================================
// f32 Implementation (8 lanes, computed in f64 precision internally)
// =============================================================================

/// Computes `ln(x)` for each lane of an AVX2 `__m256` register.
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
/// Requires AVX2 and FMA support. The caller must ensure these features are
/// available at runtime.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_ln_ps(x: __m256) -> __m256 {
    unsafe {
        // Process as two 4-lane f64 operations for precision
        // Split input into low and high halves, convert to f64
        let x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
        let x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));

        // Compute ln in f64 precision for each half
        let ln_lo = ln_core_f64(x_lo);
        let ln_hi = ln_core_f64(x_hi);

        // Convert back to f32 and combine
        let result_lo = _mm256_cvtpd_ps(ln_lo);
        let result_hi = _mm256_cvtpd_ps(ln_hi);

        _mm256_insertf128_ps(_mm256_castps128_ps256(result_lo), result_hi, 1)
    }
}

// =============================================================================
// f64 Implementation (4 lanes)
// =============================================================================

/// Computes `ln(x)` for each lane of an AVX2 `__m256d` register.
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
/// Requires AVX2 and FMA support. The caller must ensure these features are
/// available at runtime.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_ln_pd(x: __m256d) -> __m256d {
    unsafe { ln_core_f64(x) }
}

// =============================================================================
// Core f64 ln kernel (shared by both f32 and f64 paths)
// =============================================================================

/// Core natural logarithm kernel operating on 4-lane f64 vectors.
///
/// Implements the fdlibm `e_log.c` algorithm:
/// 1. Extract exponent `k` and normalize mantissa to `[√2/2, √2]`
/// 2. Let `f = m - 1`, `s = f / (2 + f)`
/// 3. Evaluate minimax polynomial `R(z)` where `z = s²`
/// 4. Reconstruct: `result = k*ln2_hi - ((hfsq - (s*(hfsq+R) + k*ln2_lo)) - f)`
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn ln_core_f64(x: __m256d) -> __m256d {
    {
        let zero = _mm256_setzero_pd();
        let one = _mm256_set1_pd(1.0);
        let half = _mm256_set1_pd(0.5);

        // =====================================================================
        // Step 0: Special case detection (save for later)
        // =====================================================================

        let sign_bit = _mm256_set1_pd(-0.0);
        let abs_x = _mm256_andnot_pd(sign_bit, x);
        let inf = _mm256_set1_pd(f64::INFINITY);

        // x < 0 (excluding -0) → NaN
        let is_negative = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);

        // x == 0 or x == -0 → -∞
        let is_zero = _mm256_cmp_pd(abs_x, zero, _CMP_EQ_OQ);

        // x == +∞ → +∞
        let is_pos_inf = _mm256_cmp_pd(x, inf, _CMP_EQ_OQ);

        // x is NaN → NaN (unordered comparison: true if either is NaN)
        let is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

        // =====================================================================
        // Step 1: Handle subnormals by scaling up
        // =====================================================================

        // Subnormal: 0 < |x| < 2^-1022 (smallest normal)
        let min_normal = _mm256_set1_pd(f64::MIN_POSITIVE); // 2^-1022
        let is_subnormal = _mm256_and_pd(
            _mm256_cmp_pd(abs_x, zero, _CMP_GT_OQ),
            _mm256_cmp_pd(abs_x, min_normal, _CMP_LT_OQ),
        );

        // Scale subnormals by 2^52 to make them normal
        let two52 = _mm256_set1_pd(TWO52_64);
        let x_scaled = _mm256_mul_pd(x, two52);
        let x_work = _mm256_blendv_pd(x, x_scaled, is_subnormal);

        // Subnormal exponent adjustment: subtract 52 from k later
        let k_adjust = _mm256_and_pd(
            _mm256_set1_pd(f64::from_bits(
                // -52.0 as f64 bits
                (-52.0_f64).to_bits(),
            )),
            is_subnormal,
        );

        // =====================================================================
        // Step 2: Extract exponent k and normalize mantissa
        // =====================================================================

        // Get IEEE 754 bit pattern
        let ix = _mm256_castpd_si256(x_work);

        // Extract biased exponent: bits [52:62] → shift right 52
        let exp_bits = _mm256_srli_epi64(ix, 52);

        // Pack 4×i64 exponents to 4×i32 for conversion
        // Indices: pick lanes 0, 2, 4, 6 from the 8×i32 view
        let pack_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
        let exp_i32 = _mm256_permutevar8x32_epi32(exp_bits, pack_idx);

        // Convert to f64: k = biased_exponent - 1023
        let bias = _mm256_set1_pd(1023.0);
        let k = _mm256_sub_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(exp_i32)), bias);
        // Apply subnormal adjustment
        let k = _mm256_add_pd(k, k_adjust);

        // Normalize mantissa: clear exponent bits, set exponent to 1023 (range [1, 2))
        let mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF_u64 as i64);
        let exp_1023 = _mm256_set1_epi64x(0x3FF0000000000000_u64 as i64);
        let m_bits = _mm256_or_si256(_mm256_and_si256(ix, mantissa_mask), exp_1023);
        let m = _mm256_castsi256_pd(m_bits);

        // If m > √2, halve it (set exponent to 1022 → range [0.5, 1)) and increment k
        let sqrt2 = _mm256_set1_pd(SQRT2_64);
        let is_big = _mm256_cmp_pd(m, sqrt2, _CMP_GT_OQ);

        // For big m: divide by 2 (subtract 1 from exponent field)
        let exp_1022 = _mm256_set1_epi64x(0x3FE0000000000000_u64 as i64);
        let m_halved_bits = _mm256_or_si256(_mm256_and_si256(ix, mantissa_mask), exp_1022);
        let m_halved = _mm256_castsi256_pd(m_halved_bits);

        let m = _mm256_blendv_pd(m, m_halved, is_big);
        let k = _mm256_add_pd(k, _mm256_and_pd(one, is_big)); // k++ if big

        // =====================================================================
        // Step 3: Compute f = m - 1, s = f / (2 + f)
        // =====================================================================

        let f = _mm256_sub_pd(m, one);
        let two_plus_f = _mm256_add_pd(_mm256_set1_pd(2.0), f);
        let s = _mm256_div_pd(f, two_plus_f);

        // hfsq = 0.5 * f * f
        let hfsq = _mm256_mul_pd(half, _mm256_mul_pd(f, f));

        // =====================================================================
        // Step 4: Evaluate minimax polynomial R(z) where z = s²
        // =====================================================================

        let z = _mm256_mul_pd(s, s); // z = s²
        let w = _mm256_mul_pd(z, z); // w = s⁴

        // Split into odd and even powers for better ILP (instruction-level parallelism)
        // Odd terms:  t1 = Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7))
        // Even terms: t2 = Lg2 + w*(Lg4 + w*Lg6)
        let lg1 = _mm256_set1_pd(LG1_64);
        let lg2 = _mm256_set1_pd(LG2_64);
        let lg3 = _mm256_set1_pd(LG3_64);
        let lg4 = _mm256_set1_pd(LG4_64);
        let lg5 = _mm256_set1_pd(LG5_64);
        let lg6 = _mm256_set1_pd(LG6_64);
        let lg7 = _mm256_set1_pd(LG7_64);

        // t1 = Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7))
        let t1 = _mm256_fmadd_pd(w, lg7, lg5); // Lg5 + w*Lg7
        let t1 = _mm256_fmadd_pd(w, t1, lg3); // Lg3 + w*(Lg5 + w*Lg7)
        let t1 = _mm256_fmadd_pd(w, t1, lg1); // Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7))

        // t2 = Lg2 + w * (Lg4 + w * Lg6)
        let t2 = _mm256_fmadd_pd(w, lg6, lg4); // Lg4 + w*Lg6
        let t2 = _mm256_fmadd_pd(w, t2, lg2); // Lg2 + w*(Lg4 + w*Lg6)

        // R = z*t1 + t2  (combine odd and even)
        // R = t1 * z + t2 * w (since even terms have one extra z factor)
        // Actually: R(z) = t1*z + t2*z² = z*(t1 + z*t2)
        let r = _mm256_fmadd_pd(z, t2, t1); // t1 + z*t2
        let r = _mm256_mul_pd(z, r); // z*(t1 + z*t2)

        // =====================================================================
        // Step 5: Reconstruct ln(x) = k*ln2_hi - ((hfsq - (s*(hfsq+R) + k*ln2_lo)) - f)
        // =====================================================================

        let ln2_hi = _mm256_set1_pd(LN2_HI_64);
        let ln2_lo = _mm256_set1_pd(LN2_LO_64);

        // s * (hfsq + R)
        let s_term = _mm256_mul_pd(s, _mm256_add_pd(hfsq, r));

        // k * ln2_lo
        let k_ln2_lo = _mm256_mul_pd(k, ln2_lo);

        // inner = s*(hfsq+R) + k*ln2_lo
        let inner = _mm256_add_pd(s_term, k_ln2_lo);

        // hfsq - inner
        let hfsq_minus_inner = _mm256_sub_pd(hfsq, inner);

        // (hfsq - inner) - f  →  this is the negative of the log(1+f) part
        let log_part = _mm256_sub_pd(hfsq_minus_inner, f);

        // result = k*ln2_hi - log_part
        let result = _mm256_fmsub_pd(k, ln2_hi, log_part);

        // =====================================================================
        // Step 6: Apply special cases
        // =====================================================================

        let nan = _mm256_set1_pd(f64::NAN);
        let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);

        // x == 0 or x == -0 → -∞
        let result = _mm256_blendv_pd(result, neg_inf, is_zero);

        // x == +∞ → +∞
        let result = _mm256_blendv_pd(result, inf, is_pos_inf);

        // x < 0 → NaN
        let result = _mm256_blendv_pd(result, nan, is_negative);

        // x is NaN → NaN (propagate)
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
    fn test_ln_ps_known_values() {
        unsafe {
            let inputs = [
                1.0_f32,
                std::f32::consts::E,
                std::f32::consts::E * std::f32::consts::E,
                10.0,
                100.0,
                0.5,
                2.0,
                0.1,
            ];
            let expected: [f32; 8] = inputs.map(|v| v.ln());
            let x = _mm256_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm256_ln_ps(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f32(r, e);
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
    fn test_ln_ps_one() {
        // ln(1) should be exactly 0.0
        unsafe {
            let x = _mm256_set1_ps(1.0);
            let result = to_array_ps(_mm256_ln_ps(x));
            for &r in &result {
                assert_eq!(r, 0.0, "ln(1.0) should be exactly 0.0");
                assert_eq!(r.to_bits(), 0, "ln(1.0) should be +0.0");
            }
        }
    }

    #[test]
    fn test_ln_ps_zero() {
        // ln(0) = -∞, ln(-0) = -∞
        unsafe {
            let pos_zero = _mm256_set1_ps(0.0);
            let neg_zero = _mm256_set1_ps(-0.0);

            let result_pos = to_array_ps(_mm256_ln_ps(pos_zero));
            let result_neg = to_array_ps(_mm256_ln_ps(neg_zero));

            for &r in &result_pos {
                assert!(
                    r.is_infinite() && r.is_sign_negative(),
                    "ln(0) should be -∞, got {}",
                    r
                );
            }
            for &r in &result_neg {
                assert!(
                    r.is_infinite() && r.is_sign_negative(),
                    "ln(-0) should be -∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_ln_ps_negative() {
        // ln(x < 0) = NaN
        unsafe {
            let inputs = [
                -1.0_f32,
                -2.0,
                -0.5,
                -100.0,
                -1e-10,
                -f32::INFINITY,
                -1.0,
                -42.0,
            ];
            let x = _mm256_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm256_ln_ps(x));

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
    fn test_ln_ps_infinity() {
        // ln(+∞) = +∞
        unsafe {
            let x = _mm256_set1_ps(f32::INFINITY);
            let result = to_array_ps(_mm256_ln_ps(x));
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
    fn test_ln_ps_nan() {
        // ln(NaN) = NaN
        unsafe {
            let x = _mm256_set1_ps(f32::NAN);
            let result = to_array_ps(_mm256_ln_ps(x));
            for &r in &result {
                assert!(r.is_nan(), "ln(NaN) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_ln_ps_powers_of_two() {
        // ln(2^k) = k * ln(2) — tests exponent extraction
        unsafe {
            let inputs = [2.0_f32, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
            let expected: [f32; 8] = inputs.map(|v| v.ln());
            let x = _mm256_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm256_ln_ps(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f32(r, e);
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
    fn test_ln_ps_small_values() {
        // Test values close to but not equal to 1
        unsafe {
            let inputs = [0.999_f32, 1.001, 0.99, 1.01, 0.9, 1.1, 0.5, 1.5];
            let expected: [f32; 8] = inputs.map(|v| v.ln());
            let x = _mm256_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm256_ln_ps(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f32(r, e);
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
    fn test_ln_ps_lane_independence() {
        // Verify all lanes are processed independently
        unsafe {
            let inputs = [0.5_f32, 1.0, 2.0, 3.0, 10.0, 100.0, 1000.0, 0.01];
            let x = _mm256_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm256_ln_ps(x));

            for (i, (&r, &input)) in result.iter().zip(inputs.iter()).enumerate() {
                let expected = input.ln();
                let ulp = ulp_distance_f32(r, expected);
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
    fn test_ln_ps_ulp_sweep() {
        // Sweep many values and verify ≤ 2 ULP
        unsafe {
            let mut max_ulp: u32 = 0;
            let mut worst_input = 0.0_f32;

            // Sweep positive values across several decades
            let test_values: Vec<f32> = (0..10000)
                .map(|i| {
                    let t = i as f32 / 10000.0;
                    // Map [0, 1) → [1e-30, 1e30] logarithmically
                    (1e-30_f32).powf(1.0 - t) * (1e30_f32).powf(t)
                })
                .collect();

            for chunk in test_values.chunks(8) {
                if chunk.len() < 8 {
                    continue;
                }
                let mut arr = [0.0_f32; 8];
                arr.copy_from_slice(chunk);

                let x = _mm256_loadu_ps(arr.as_ptr());
                let result = to_array_ps(_mm256_ln_ps(x));

                for (i, (&r, &input)) in result.iter().zip(arr.iter()).enumerate() {
                    let expected = input.ln();
                    let ulp = ulp_distance_f32(r, expected);
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
                "f32 ln ULP sweep: max ULP = {}, worst input = {}",
                max_ulp, worst_input
            );
        }
    }

    // -----------------------------------------------------------------------
    // f64 Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ln_pd_known_values() {
        unsafe {
            let inputs = [1.0_f64, std::f64::consts::E, 10.0, 0.5];
            let expected: [f64; 4] = inputs.map(|v| v.ln());
            let x = _mm256_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm256_ln_pd(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f64(r, e);
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
    fn test_ln_pd_one() {
        unsafe {
            let x = _mm256_set1_pd(1.0);
            let result = to_array_pd(_mm256_ln_pd(x));
            for &r in &result {
                assert_eq!(r, 0.0, "ln(1.0) should be exactly 0.0");
                assert_eq!(r.to_bits(), 0, "ln(1.0) should be +0.0");
            }
        }
    }

    #[test]
    fn test_ln_pd_zero() {
        unsafe {
            let pos_zero = _mm256_set1_pd(0.0);
            let neg_zero = _mm256_set1_pd(-0.0);

            let result_pos = to_array_pd(_mm256_ln_pd(pos_zero));
            let result_neg = to_array_pd(_mm256_ln_pd(neg_zero));

            for &r in &result_pos {
                assert!(
                    r.is_infinite() && r.is_sign_negative(),
                    "ln(0) should be -∞, got {}",
                    r
                );
            }
            for &r in &result_neg {
                assert!(
                    r.is_infinite() && r.is_sign_negative(),
                    "ln(-0) should be -∞, got {}",
                    r
                );
            }
        }
    }

    #[test]
    fn test_ln_pd_negative() {
        unsafe {
            let inputs = [-1.0_f64, -2.0, -0.5, -100.0];
            let x = _mm256_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm256_ln_pd(x));

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
    fn test_ln_pd_infinity() {
        unsafe {
            let x = _mm256_set1_pd(f64::INFINITY);
            let result = to_array_pd(_mm256_ln_pd(x));
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
    fn test_ln_pd_nan() {
        unsafe {
            let x = _mm256_set1_pd(f64::NAN);
            let result = to_array_pd(_mm256_ln_pd(x));
            for &r in &result {
                assert!(r.is_nan(), "ln(NaN) should be NaN, got {}", r);
            }
        }
    }

    #[test]
    fn test_ln_pd_powers_of_two() {
        unsafe {
            let inputs = [2.0_f64, 4.0, 8.0, 16.0];
            let expected: [f64; 4] = inputs.map(|v| v.ln());
            let x = _mm256_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm256_ln_pd(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f64(r, e);
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
    fn test_ln_pd_lane_independence() {
        unsafe {
            let inputs = [0.01_f64, 1.0, 100.0, 1e10];
            let x = _mm256_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm256_ln_pd(x));

            for (i, (&r, &input)) in result.iter().zip(inputs.iter()).enumerate() {
                let expected = input.ln();
                let ulp = ulp_distance_f64(r, expected);
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
    fn test_ln_pd_ulp_sweep() {
        unsafe {
            let mut max_ulp: u64 = 0;
            let mut worst_input = 0.0_f64;

            let test_values: Vec<f64> = (0..10000)
                .map(|i| {
                    let t = i as f64 / 10000.0;
                    (1e-100_f64).powf(1.0 - t) * (1e100_f64).powf(t)
                })
                .collect();

            for chunk in test_values.chunks(4) {
                if chunk.len() < 4 {
                    continue;
                }
                let mut arr = [0.0_f64; 4];
                arr.copy_from_slice(chunk);

                let x = _mm256_loadu_pd(arr.as_ptr());
                let result = to_array_pd(_mm256_ln_pd(x));

                for (i, (&r, &input)) in result.iter().zip(arr.iter()).enumerate() {
                    let expected = input.ln();
                    let ulp = ulp_distance_f64(r, expected);
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
                "f64 ln ULP sweep: max ULP = {}, worst input = {}",
                max_ulp, worst_input
            );
        }
    }

    #[test]
    fn test_ln_pd_subnormal() {
        // Test subnormal f64 inputs
        unsafe {
            let tiny = f64::MIN_POSITIVE * 0.5; // subnormal
            let inputs = [tiny, tiny * 0.1, tiny * 10.0, f64::MIN_POSITIVE];
            let expected: [f64; 4] = inputs.map(|v| v.ln());
            let x = _mm256_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm256_ln_pd(x));

            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                let ulp = ulp_distance_f64(r, e);
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
