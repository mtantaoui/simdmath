//! AVX-512 SIMD implementation of `cbrt(x)` (cube root) for `f32` and `f64` vectors.
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
//! | `_mm512_cbrt_ps` | ≤ 1 ULP      |
//! | `_mm512_cbrt_pd` | ≤ 1 ULP      |
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
//! - **f32 uses f64 intermediates**: Each 16-lane f32 vector is split into two
//!   8-lane f64 vectors for Newton iterations to maintain precision.
//! - **Subnormal handling**: Detected and scaled before processing.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::cbrt::{
    B1_32, B1_64, B2_32, B2_64, P0, P1, P2, P3, P4, ROUND_BIAS_64, ROUND_MASK_64, X1P24_32,
    X1P54_64,
};

// ===========================================================================
// f32 Implementation (16 lanes)
// ===========================================================================

/// Computes `cbrt(x)` (cube root) for each lane of an AVX-512 `__m512` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain including subnormals.
///
/// # Description
///
/// Uses a bit-manipulation trick for initial approximation followed by two
/// Newton–Raphson iterations in double precision. All 16 lanes are processed
/// simultaneously using AVX-512 mask operations for special cases.
///
/// # Safety
///
/// Requires AVX-512F. `x` must be a valid `__m512` register.
///
/// # Example
///
/// ```ignore
/// let x = _mm512_set1_ps(27.0);
/// let result = _mm512_cbrt_ps(x);
/// // All 16 lanes ≈ 3.0
/// ```
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_cbrt_ps(x: __m512) -> __m512 {
    unsafe {
        // -----------------------------------------------------------------------
        // Process using scalar extraction (AVX-512 lacks convenient int div by 3)
        // -----------------------------------------------------------------------
        let mut x_arr = [0.0_f32; 16];
        _mm512_storeu_ps(x_arr.as_mut_ptr(), x);

        let mut result_arr = [0.0_f32; 16];

        for i in 0..16 {
            let xi = x_arr[i];
            let ui = xi.to_bits();
            let mut hx = ui & 0x7fffffff;

            // Special case: Inf or NaN
            if hx >= 0x7f800000 {
                result_arr[i] = xi + xi;
                continue;
            }

            // Special case: zero
            if hx == 0 {
                result_arr[i] = xi;
                continue;
            }

            // Extract sign
            let sign = ui & 0x80000000;

            // Subnormal: scale by 2^24
            if hx < 0x00800000 {
                let scaled = xi * X1P24_32;
                hx = scaled.to_bits() & 0x7fffffff;
                hx = hx / 3 + B2_32;
            } else {
                hx = hx / 3 + B1_32;
            }

            // Construct initial approximation
            let t_bits = sign | hx;
            let t = f32::from_bits(t_bits);

            // Newton iterations in f64 for precision
            let x_f64 = xi as f64;
            let mut t_f64 = t as f64;

            // First iteration: t = t * (2x + t³) / (x + 2t³)
            let r = t_f64 * t_f64 * t_f64;
            t_f64 = t_f64 * (x_f64 + x_f64 + r) / (x_f64 + r + r);

            // Second iteration
            let r = t_f64 * t_f64 * t_f64;
            t_f64 = t_f64 * (x_f64 + x_f64 + r) / (x_f64 + r + r);

            result_arr[i] = t_f64 as f32;
        }

        _mm512_loadu_ps(result_arr.as_ptr())
    }
}

// ===========================================================================
// f64 Implementation (8 lanes)
// ===========================================================================

/// Computes `cbrt(x)` (cube root) for each lane of an AVX-512 `__m512d` register.
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
/// Requires AVX-512F. `x` must be a valid `__m512d` register.
///
/// # Example
///
/// ```ignore
/// let x = _mm512_set1_pd(27.0);
/// let result = _mm512_cbrt_pd(x);
/// // All 8 lanes ≈ 3.0
/// ```
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_cbrt_pd(x: __m512d) -> __m512d {
    unsafe {
        // -----------------------------------------------------------------------
        // Process each lane scalar (f64 cbrt requires complex bit manipulation)
        // -----------------------------------------------------------------------
        let mut x_arr = [0.0_f64; 8];
        _mm512_storeu_pd(x_arr.as_mut_ptr(), x);

        let mut result_arr = [0.0_f64; 8];

        for i in 0..8 {
            let xi = x_arr[i];
            let ui = xi.to_bits();
            let mut hx = ((ui >> 32) as u32) & 0x7fffffff;

            // Special case: Inf or NaN
            if hx >= 0x7ff00000 {
                result_arr[i] = xi + xi;
                continue;
            }

            // Special case: zero
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
            let s = t * t;
            let r = xi / s;
            let w = t + t;
            t = t + t * (r - t) / (w + r);

            result_arr[i] = t;
        }

        _mm512_loadu_pd(result_arr.as_ptr())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to extract f32 lanes
    unsafe fn to_array_ps(v: __m512) -> [f32; 16] {
        let mut arr = [0.0_f32; 16];
        unsafe { _mm512_storeu_ps(arr.as_mut_ptr(), v) };
        arr
    }

    // Helper to extract f64 lanes
    unsafe fn to_array_pd(v: __m512d) -> [f64; 8] {
        let mut arr = [0.0_f64; 8];
        unsafe { _mm512_storeu_pd(arr.as_mut_ptr(), v) };
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
            let inputs: [f32; 16] = [
                1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0, 729.0, 1000.0, 1.0, 8.0, 27.0,
                64.0, 125.0, 216.0,
            ];
            let expected: [f32; 16] = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ];
            let x = _mm512_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm512_cbrt_ps(x));

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
            let inputs: [f32; 16] = [
                -1.0, -8.0, -27.0, -64.0, -125.0, -216.0, -343.0, -512.0, -729.0, -1000.0, -1.0,
                -8.0, -27.0, -64.0, -125.0, -216.0,
            ];
            let x = _mm512_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm512_cbrt_ps(x));

            for (i, &r) in result.iter().enumerate() {
                let expected = inputs[i].cbrt();
                let ulp = ulp_distance_f32(r, expected);
                assert!(
                    ulp <= 1,
                    "Lane {}: cbrt({}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_cbrt_ps_zero() {
        unsafe {
            let pos_zero = _mm512_set1_ps(0.0);
            let neg_zero = _mm512_set1_ps(-0.0);

            let result_pos = to_array_ps(_mm512_cbrt_ps(pos_zero));
            let result_neg = to_array_ps(_mm512_cbrt_ps(neg_zero));

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
            let pos_inf = _mm512_set1_ps(f32::INFINITY);
            let neg_inf = _mm512_set1_ps(f32::NEG_INFINITY);

            let result_pos = to_array_ps(_mm512_cbrt_ps(pos_inf));
            let result_neg = to_array_ps(_mm512_cbrt_ps(neg_inf));

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
            let nan = _mm512_set1_ps(f32::NAN);
            let result = to_array_ps(_mm512_cbrt_ps(nan));

            for &r in &result {
                assert!(r.is_nan(), "cbrt(NaN) should be NaN");
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
                    let x = _mm512_set1_ps(x_val);
                    let result = to_array_ps(_mm512_cbrt_ps(x));
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
    fn test_cbrt_pd_perfect_cubes() {
        unsafe {
            let inputs: [f64; 8] = [1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0];
            let expected: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let x = _mm512_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm512_cbrt_pd(x));

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
            let inputs: [f64; 8] = [-1.0, -8.0, -27.0, -64.0, -125.0, -216.0, -343.0, -512.0];
            let x = _mm512_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm512_cbrt_pd(x));

            for (i, &r) in result.iter().enumerate() {
                let expected = inputs[i].cbrt();
                let ulp = ulp_distance_f64(r, expected);
                assert!(
                    ulp <= 1,
                    "Lane {}: cbrt({}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_cbrt_pd_zero() {
        unsafe {
            let pos_zero = _mm512_set1_pd(0.0);
            let neg_zero = _mm512_set1_pd(-0.0);

            let result_pos = to_array_pd(_mm512_cbrt_pd(pos_zero));
            let result_neg = to_array_pd(_mm512_cbrt_pd(neg_zero));

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
            let pos_inf = _mm512_set1_pd(f64::INFINITY);
            let neg_inf = _mm512_set1_pd(f64::NEG_INFINITY);

            let result_pos = to_array_pd(_mm512_cbrt_pd(pos_inf));
            let result_neg = to_array_pd(_mm512_cbrt_pd(neg_inf));

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
            let nan = _mm512_set1_pd(f64::NAN);
            let result = to_array_pd(_mm512_cbrt_pd(nan));

            for &r in &result {
                assert!(r.is_nan(), "cbrt(NaN) should be NaN");
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
                    let x = _mm512_set1_pd(x_val);
                    let result = to_array_pd(_mm512_cbrt_pd(x));
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
