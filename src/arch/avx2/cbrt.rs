//! AVX2 SIMD implementation of `cbrt(x)` (cube root) for `f32` and `f64` vectors.
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
//! | Implementation   | Accuracy     |
//! |------------------|--------------|
//! | `_mm256_cbrt_ps` | ≤ 1 ULP      |
//! | `_mm256_cbrt_pd` | ≤ 1 ULP      |
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
//! # Implementation Notes
//!
//! - **Integer division by 3**: Uses `_mm256_div_epi32` which requires AVX2.
//! - **f32 uses f64 intermediates**: Each 8-lane f32 vector is split into two
//!   4-lane f64 vectors for Newton iterations to maintain precision.
//! - **Subnormal handling**: Detected and scaled before processing to avoid
//!   underflow in the bit manipulation.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::cbrt::{
    B1_32, B1_64, B2_32, B2_64, P0, P1, P2, P3, P4, ROUND_BIAS_64, ROUND_MASK_64, X1P24_32,
    X1P54_64,
};

// ===========================================================================
// f32 Implementation
// ===========================================================================

/// Divides each 32-bit integer lane by 3.
///
/// AVX2 lacks integer division, so we extract to array, divide, and repack.
/// This is used only for the initial approximation (8 divisions), after which
/// all heavy computation is in floating-point.
#[inline]
unsafe fn div_by_3_epi32(x: __m256i) -> __m256i {
    let mut arr = [0_i32; 8];
    unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, x) };
    for val in &mut arr {
        *val = (*val as u32 / 3) as i32;
    }
    unsafe { _mm256_loadu_si256(arr.as_ptr() as *const __m256i) }
}

/// Computes `cbrt(x)` (cube root) for each lane of an AVX2 `__m256` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain including subnormals.
///
/// # Description
///
/// Uses a bit-manipulation trick for initial approximation followed by two
/// Newton–Raphson iterations in double precision. All 8 lanes are processed
/// simultaneously using SIMD blending for special cases.
///
/// # Safety
///
/// Requires AVX2 support. `x` must be a valid `__m256` register.
///
/// # Example
///
/// ```ignore
/// let x = _mm256_set1_ps(27.0);
/// let result = _mm256_cbrt_ps(x);
/// // result ≈ [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
/// ```
#[inline]
pub(crate) unsafe fn _mm256_cbrt_ps(x: __m256) -> __m256 {
    unsafe {
        // -----------------------------------------------------------------------
        // Constants
        // -----------------------------------------------------------------------
        let x1p24 = _mm256_set1_ps(X1P24_32);
        let abs_mask = _mm256_set1_epi32(0x7fffffff_u32 as i32);
        let sign_mask = _mm256_set1_epi32(0x80000000_u32 as i32);
        let inf_threshold = _mm256_set1_epi32(0x7f800000_u32 as i32);
        let subnormal_threshold = _mm256_set1_epi32(0x00800000_u32 as i32);
        let b1 = _mm256_set1_epi32(B1_32 as i32);
        let b2 = _mm256_set1_epi32(B2_32 as i32);

        // -----------------------------------------------------------------------
        // Extract bit representation and sign
        // -----------------------------------------------------------------------
        let ui = _mm256_castps_si256(x);
        let hx = _mm256_and_si256(ui, abs_mask); // |x| as integer bits
        let sign_bits = _mm256_and_si256(ui, sign_mask);

        // -----------------------------------------------------------------------
        // Special case detection
        // -----------------------------------------------------------------------

        // Inf or NaN: hx >= 0x7f800000
        let is_inf_or_nan =
            _mm256_cmpgt_epi32(hx, _mm256_sub_epi32(inf_threshold, _mm256_set1_epi32(1)));

        // Zero: hx == 0
        let is_zero = _mm256_cmpeq_epi32(hx, _mm256_setzero_si256());

        // Subnormal: 0 < hx < 0x00800000
        let is_subnormal =
            _mm256_andnot_si256(is_zero, _mm256_cmpgt_epi32(subnormal_threshold, hx));

        // -----------------------------------------------------------------------
        // Initial approximation via bit manipulation
        // -----------------------------------------------------------------------

        // Normal case: hx/3 + B1
        let hx_normal = _mm256_add_epi32(div_by_3_epi32(hx), b1);

        // Subnormal case: scale by 2^24, then (hx_scaled/3 + B2)
        let x_scaled = _mm256_mul_ps(_mm256_castsi256_ps(_mm256_or_si256(sign_bits, hx)), x1p24);
        let ui_scaled = _mm256_castps_si256(x_scaled);
        let hx_scaled = _mm256_and_si256(ui_scaled, abs_mask);
        let hx_subnormal = _mm256_add_epi32(div_by_3_epi32(hx_scaled), b2);

        // Select between normal and subnormal paths
        let hx_approx = _mm256_blendv_epi8(hx_normal, hx_subnormal, is_subnormal);

        // Restore sign and convert to float
        let ui_approx = _mm256_or_si256(sign_bits, hx_approx);
        let t_f32 = _mm256_castsi256_ps(ui_approx);

        // -----------------------------------------------------------------------
        // Newton–Raphson iterations in f64 (lower 4 lanes)
        // -----------------------------------------------------------------------
        let x_low_f32 = _mm256_castps256_ps128(x);
        let t_low_f32 = _mm256_castps256_ps128(t_f32);
        let x_low = _mm256_cvtps_pd(x_low_f32);
        let t_low = _mm256_cvtps_pd(t_low_f32);

        // First iteration: t = t * (2x + t³) / (x + 2t³)
        let r_low = _mm256_mul_pd(t_low, _mm256_mul_pd(t_low, t_low)); // r = t³
        let two_x_low = _mm256_add_pd(x_low, x_low);
        let t_low = _mm256_mul_pd(
            t_low,
            _mm256_div_pd(
                _mm256_add_pd(two_x_low, r_low),                   // 2x + r
                _mm256_add_pd(x_low, _mm256_add_pd(r_low, r_low)), // x + 2r
            ),
        );

        // Second iteration
        let r_low = _mm256_mul_pd(t_low, _mm256_mul_pd(t_low, t_low));
        let t_low_final = _mm256_mul_pd(
            t_low,
            _mm256_div_pd(
                _mm256_add_pd(two_x_low, r_low),
                _mm256_add_pd(x_low, _mm256_add_pd(r_low, r_low)),
            ),
        );

        // -----------------------------------------------------------------------
        // Newton–Raphson iterations in f64 (upper 4 lanes)
        // -----------------------------------------------------------------------
        let x_high_f32 = _mm256_extractf128_ps(x, 1);
        let t_high_f32 = _mm256_extractf128_ps(t_f32, 1);
        let x_high = _mm256_cvtps_pd(x_high_f32);
        let t_high = _mm256_cvtps_pd(t_high_f32);

        // First iteration
        let r_high = _mm256_mul_pd(t_high, _mm256_mul_pd(t_high, t_high));
        let two_x_high = _mm256_add_pd(x_high, x_high);
        let t_high = _mm256_mul_pd(
            t_high,
            _mm256_div_pd(
                _mm256_add_pd(two_x_high, r_high),
                _mm256_add_pd(x_high, _mm256_add_pd(r_high, r_high)),
            ),
        );

        // Second iteration
        let r_high = _mm256_mul_pd(t_high, _mm256_mul_pd(t_high, t_high));
        let t_high_final = _mm256_mul_pd(
            t_high,
            _mm256_div_pd(
                _mm256_add_pd(two_x_high, r_high),
                _mm256_add_pd(x_high, _mm256_add_pd(r_high, r_high)),
            ),
        );

        // -----------------------------------------------------------------------
        // Combine halves and handle special cases
        // -----------------------------------------------------------------------
        let result_low_f32 = _mm256_cvtpd_ps(t_low_final);
        let result_high_f32 = _mm256_cvtpd_ps(t_high_final);
        let mut result =
            _mm256_insertf128_ps(_mm256_castps128_ps256(result_low_f32), result_high_f32, 1);

        // Zero: return x (preserves sign of ±0)
        result = _mm256_blendv_ps(result, x, _mm256_castsi256_ps(is_zero));

        // Inf/NaN: return x + x (propagates NaN, returns ±∞ for ±∞)
        let inf_nan_result = _mm256_add_ps(x, x);
        result = _mm256_blendv_ps(result, inf_nan_result, _mm256_castsi256_ps(is_inf_or_nan));

        result
    }
}

// ===========================================================================
// f64 Implementation
// ===========================================================================

/// Computes `cbrt(x)` (cube root) for each lane of an AVX2 `__m256d` register.
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
/// Requires AVX2 and FMA support. `x` must be a valid `__m256d` register.
///
/// # Example
///
/// ```ignore
/// let x = _mm256_set1_pd(27.0);
/// let result = _mm256_cbrt_pd(x);
/// // result ≈ [3.0, 3.0, 3.0, 3.0]
/// ```
#[inline]
pub(crate) unsafe fn _mm256_cbrt_pd(x: __m256d) -> __m256d {
    // -----------------------------------------------------------------------
    // Process each lane scalar (f64 cbrt requires complex bit manipulation
    // that AVX2 doesn't handle well for 64-bit integers)
    // -----------------------------------------------------------------------
    let mut x_arr = [0.0_f64; 4];
    unsafe { _mm256_storeu_pd(x_arr.as_mut_ptr(), x) };

    let mut result_arr = [0.0_f64; 4];

    for i in 0..4 {
        let xi = x_arr[i];
        let ui = xi.to_bits();
        let mut hx = ((ui >> 32) as u32) & 0x7fffffff;

        // Special case: Inf or NaN → return x + x
        if hx >= 0x7ff00000 {
            result_arr[i] = xi + xi;
            continue;
        }

        // Special case: zero → return x (preserves sign)
        if hx == 0 && (ui & 0xffffffff) == 0 {
            result_arr[i] = xi;
            continue;
        }

        // Subnormal: scale by 2^54
        let mut u_bits = ui;
        if hx < 0x00100000 {
            let scaled = xi * X1P54_64;
            u_bits = scaled.to_bits();
            hx = ((u_bits >> 32) as u32) & 0x7fffffff;
            if hx == 0 {
                result_arr[i] = xi;
                continue;
            }
            hx = hx / 3 + B2_64;
        } else {
            hx = hx / 3 + B1_64;
        }

        // Construct initial approximation (~5 bits)
        u_bits = (u_bits & (1_u64 << 63)) | ((hx as u64) << 32);
        let mut t = f64::from_bits(u_bits);

        // Polynomial refinement to ~23 bits: t = t * P(t³/x)
        let r = (t * t) * (t / xi);
        t *= (P0 + r * (P1 + r * P2)) + ((r * r) * r) * (P3 + r * P4);

        // Round t to 23 bits (away from zero) for exact t*t
        let t_bits = t.to_bits();
        let t_rounded_bits = (t_bits.wrapping_add(ROUND_BIAS_64)) & ROUND_MASK_64;
        t = f64::from_bits(t_rounded_bits);

        // One Newton iteration to 53 bits
        let s = t * t; // exact
        let r = xi / s; // error ≤ 0.5 ulp
        let w = t + t; // exact
        t = t + t * (r - t) / (w + r);

        result_arr[i] = t;
    }

    unsafe { _mm256_loadu_pd(result_arr.as_ptr()) }
}

// ===========================================================================
// Tests
// ===========================================================================

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
    fn test_cbrt_ps_perfect_cubes() {
        unsafe {
            let inputs = [1.0_f32, 8.0, 27.0, 64.0, 125.0, 343.0, 729.0, 1000.0];
            let expected = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 10.0];
            let x = _mm256_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm256_cbrt_ps(x));

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
    fn test_cbrt_ps_negative_values() {
        unsafe {
            let inputs = [
                -1.0_f32, -8.0, -27.0, -64.0, -125.0, -343.0, -729.0, -1000.0,
            ];
            let expected = [-1.0_f32, -2.0, -3.0, -4.0, -5.0, -7.0, -9.0, -10.0];
            let x = _mm256_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm256_cbrt_ps(x));

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
    fn test_cbrt_ps_zero() {
        unsafe {
            let pos_zero = _mm256_set1_ps(0.0);
            let neg_zero = _mm256_set1_ps(-0.0);

            let result_pos = to_array_ps(_mm256_cbrt_ps(pos_zero));
            let result_neg = to_array_ps(_mm256_cbrt_ps(neg_zero));

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
    fn test_cbrt_ps_infinity() {
        unsafe {
            let pos_inf = _mm256_set1_ps(f32::INFINITY);
            let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);

            let result_pos = to_array_ps(_mm256_cbrt_ps(pos_inf));
            let result_neg = to_array_ps(_mm256_cbrt_ps(neg_inf));

            for &r in &result_pos {
                assert!(r.is_infinite() && r > 0.0, "Should be +∞");
            }
            for &r in &result_neg {
                assert!(r.is_infinite() && r < 0.0, "Should be -∞");
            }
        }
    }

    #[test]
    fn test_cbrt_ps_nan() {
        unsafe {
            let nan = _mm256_set1_ps(f32::NAN);
            let result = to_array_ps(_mm256_cbrt_ps(nan));

            for &r in &result {
                assert!(r.is_nan(), "cbrt(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_cbrt_ps_subnormals() {
        unsafe {
            // Smallest positive subnormal: 2^-149
            let tiny = f32::from_bits(1);
            let x = _mm256_set1_ps(tiny);
            let result = to_array_ps(_mm256_cbrt_ps(x));
            let expected = tiny.cbrt();

            for &r in &result {
                let ulp = ulp_distance_f32(r, expected);
                assert!(
                    ulp <= 1,
                    "Subnormal cbrt failed: got {}, expected {}, ULP = {}",
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_cbrt_ps_ulp_sweep() {
        unsafe {
            let test_values = [
                0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1000.0, 1e10, 1e20, 1e-10, 1e-20,
            ];

            for &val in &test_values {
                for sign in [1.0_f32, -1.0] {
                    let x_val = sign * val;
                    let x = _mm256_set1_ps(x_val);
                    let result = to_array_ps(_mm256_cbrt_ps(x));
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

    #[test]
    fn test_cbrt_ps_lane_independence() {
        unsafe {
            let x = _mm256_set_ps(1.0, -8.0, 27.0, f32::INFINITY, 0.0, -0.0, f32::NAN, 64.0);
            let result = to_array_ps(_mm256_cbrt_ps(x));

            // Lane 0: cbrt(64) ≈ 4
            assert!((result[0] - 4.0).abs() < 1e-5);
            // Lane 1: cbrt(NaN) = NaN
            assert!(result[1].is_nan());
            // Lane 2: cbrt(-0) = -0
            assert_eq!(result[2].to_bits(), 0x80000000);
            // Lane 3: cbrt(0) = 0
            assert_eq!(result[3], 0.0);
            // Lane 4: cbrt(∞) = ∞
            assert!(result[4].is_infinite() && result[4] > 0.0);
            // Lane 5: cbrt(27) = 3
            assert!((result[5] - 3.0).abs() < 1e-5);
            // Lane 6: cbrt(-8) = -2
            assert!((result[6] - (-2.0)).abs() < 1e-5);
            // Lane 7: cbrt(1) = 1
            assert!((result[7] - 1.0).abs() < 1e-5);
        }
    }

    // -----------------------------------------------------------------------
    // f64 Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cbrt_pd_perfect_cubes() {
        unsafe {
            let inputs = [1.0_f64, 8.0, 27.0, 125.0];
            let expected = [1.0_f64, 2.0, 3.0, 5.0];
            let x = _mm256_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm256_cbrt_pd(x));

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
    fn test_cbrt_pd_negative_values() {
        unsafe {
            let inputs = [-1.0_f64, -8.0, -27.0, -125.0];
            let expected = [-1.0_f64, -2.0, -3.0, -5.0];
            let x = _mm256_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm256_cbrt_pd(x));

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
    fn test_cbrt_pd_zero() {
        unsafe {
            let pos_zero = _mm256_set1_pd(0.0);
            let neg_zero = _mm256_set1_pd(-0.0);

            let result_pos = to_array_pd(_mm256_cbrt_pd(pos_zero));
            let result_neg = to_array_pd(_mm256_cbrt_pd(neg_zero));

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
    fn test_cbrt_pd_infinity() {
        unsafe {
            let pos_inf = _mm256_set1_pd(f64::INFINITY);
            let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);

            let result_pos = to_array_pd(_mm256_cbrt_pd(pos_inf));
            let result_neg = to_array_pd(_mm256_cbrt_pd(neg_inf));

            for &r in &result_pos {
                assert!(r.is_infinite() && r > 0.0, "Should be +∞");
            }
            for &r in &result_neg {
                assert!(r.is_infinite() && r < 0.0, "Should be -∞");
            }
        }
    }

    #[test]
    fn test_cbrt_pd_nan() {
        unsafe {
            let nan = _mm256_set1_pd(f64::NAN);
            let result = to_array_pd(_mm256_cbrt_pd(nan));

            for &r in &result {
                assert!(r.is_nan(), "cbrt(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_cbrt_pd_subnormals() {
        unsafe {
            // Smallest positive subnormal for f64
            let tiny = f64::from_bits(1);
            let x = _mm256_set1_pd(tiny);
            let result = to_array_pd(_mm256_cbrt_pd(x));
            let expected = tiny.cbrt();

            for &r in &result {
                let ulp = ulp_distance_f64(r, expected);
                assert!(
                    ulp <= 1,
                    "Subnormal cbrt failed: got {}, expected {}, ULP = {}",
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_cbrt_pd_ulp_sweep() {
        unsafe {
            let test_values = [
                0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1000.0, 1e10, 1e50, 1e100,
                1e-10, 1e-50, 1e-100,
            ];

            for &val in &test_values {
                for sign in [1.0_f64, -1.0] {
                    let x_val = sign * val;
                    let x = _mm256_set1_pd(x_val);
                    let result = to_array_pd(_mm256_cbrt_pd(x));
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

    #[test]
    fn test_cbrt_pd_lane_independence() {
        unsafe {
            let x = _mm256_set_pd(f64::NAN, 0.0, -8.0, 27.0);
            let result = to_array_pd(_mm256_cbrt_pd(x));

            // Lane 0: cbrt(27) = 3
            assert!((result[0] - 3.0).abs() < 1e-10);
            // Lane 1: cbrt(-8) = -2
            assert!((result[1] - (-2.0)).abs() < 1e-10);
            // Lane 2: cbrt(0) = 0
            assert_eq!(result[2], 0.0);
            // Lane 3: cbrt(NaN) = NaN
            assert!(result[3].is_nan());
        }
    }
}
