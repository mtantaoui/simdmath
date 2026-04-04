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
        // Process using scalar extraction (NEON lacks convenient int div by 3)
        // -----------------------------------------------------------------------
        let mut x_arr = [0.0_f32; 4];
        vst1q_f32(x_arr.as_mut_ptr(), x);

        let mut result_arr = [0.0_f32; 4];

        for i in 0..4 {
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

        vld1q_f32(result_arr.as_ptr())
    }
}

// ===========================================================================
// f64 Implementation (2 lanes)
// ===========================================================================

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
        // Process each lane scalar (f64 cbrt requires complex bit manipulation)
        // -----------------------------------------------------------------------
        let mut x_arr = [0.0_f64; 2];
        vst1q_f64(x_arr.as_mut_ptr(), x);

        let mut result_arr = [0.0_f64; 2];

        for i in 0..2 {
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

        vld1q_f64(result_arr.as_ptr())
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
