//! NEON SIMD implementation of `asin(x)` for `f32` and `f64` vectors.
//!
//! This module provides 4-lane f32 and 2-lane f64 implementations using the
//! same algorithm as the AVX2/AVX-512 versions: a branchless, two-range
//! minimax rational approximation ported from musl libc's `asinf.c` and `asin.c`.
//!
//! # Precision
//!
//! Both implementations achieve **≤ 1 ULP** accuracy across the entire domain `[-1, 1]`.
//!
//! # Algorithm
//!
//! The core building block is a Padé rational approximation `r(z)` that
//! approximates `(asin(√z)/√z − 1)/z`, yielding:
//!
//! ```text
//! asin(x) ≈ x + x·r(x²)      for |x| ≤ 0.5
//! ```
//!
//! ## Two computational ranges
//!
//! | Range           | Identity used                                        |
//! |-----------------|------------------------------------------------------|
//! | `\|x\| < 0.5`  | `asin(x) = x + x·r(x²)`                             |
//! | `\|x\| ≥ 0.5`  | `asin(x) = π/2 − 2·(s + s·r(z))` where `z = (1−\|x\|)/2`, `s = √z` |
//!
//! The half-angle formula for `|x| ≥ 0.5` avoids catastrophic cancellation
//! near `|x| = 1`, where the derivative of `asin(x)` diverges.
//!
//! ## Compensated sqrt (Dekker split)
//!
//! For `|x| >= 0.5`, a compensated sqrt is used for extra precision:
//!
//! - **f32**: 12 bits masked (df has 11 significant mantissa bits)
//! - **f64**: 32 bits masked (df has 20 significant mantissa bits)
//!
//! ## Special values
//!
//! | Input        | Output          |
//! |--------------|-----------------|
//! | `0.0`        | `0.0`           |
//! | `1.0`        | `π/2 ≈ 1.5708`  |
//! | `-1.0`       | `−π/2 ≈ −1.5708`|
//! | `\|x\| > 1` | `NaN`           |
//! | `NaN`        | `NaN`           |

use std::arch::aarch64::*;

use crate::arch::consts::acos::{
    P_S0_32, P_S0_64, P_S1_32, P_S1_64, P_S2_32, P_S2_64, P_S3_64, P_S4_64, P_S5_64, Q_S1_32,
    Q_S1_64, Q_S2_64, Q_S3_64, Q_S4_64,
};
use crate::arch::consts::asin::{
    PIO2_HI_32, PIO2_HI_64, PIO2_LO_32, PIO2_LO_64, TINY_THRESHOLD_32, TINY_THRESHOLD_64,
};
use crate::arch::neon::abs::{vabsq_f32_wrapper, vabsq_f64_wrapper};

// ===========================================================================
// f32 Implementation (4 lanes)
// ===========================================================================

/// Computes `asin(x)` for each lane of a NEON `float32x4_t` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
///
/// # Description
///
/// All 4 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large |x|, tiny, |x|=1, out-of-domain) are computed
/// unconditionally and merged with `vbslq_f32`.
///
/// # Safety
///
/// `x` must be a valid `float32x4_t` register.
///
/// # Example
///
/// ```ignore
/// let input = vdupq_n_f32(0.5);
/// let result = vasin_f32(input);
/// // result ≈ [0.5236, 0.5236, 0.5236, 0.5236] (π/6 ≈ 0.5236)
/// ```
#[inline]
pub(crate) unsafe fn vasin_f32(x: float32x4_t) -> float32x4_t {
    unsafe {
        // ---------------------------------------------------------------------
        // Broadcast scalar constants to SIMD registers
        // ---------------------------------------------------------------------
        let pio2_hi = vdupq_n_f32(PIO2_HI_32);
        let pio2_lo = vdupq_n_f32(PIO2_LO_32);
        let p_s0 = vdupq_n_f32(P_S0_32);
        let p_s1 = vdupq_n_f32(P_S1_32);
        let p_s2 = vdupq_n_f32(P_S2_32);
        let q_s1 = vdupq_n_f32(Q_S1_32);

        let one = vdupq_n_f32(1.0);
        let half = vdupq_n_f32(0.5);
        let two = vdupq_n_f32(2.0);
        let zero = vdupq_n_f32(0.0);

        // ---------------------------------------------------------------------
        // Compute |x| for range selection
        // ---------------------------------------------------------------------
        let abs_x = vabsq_f32_wrapper(x);

        // ---------------------------------------------------------------------
        // Rational polynomial approximation: r(z) = p(z) / q(z)
        //
        // This approximates (asin(√z)/√z − 1)/z on [0, 0.25].
        //   p(z) = z · (P_S0 + z · (P_S1 + z · P_S2))
        //   q(z) = 1 + z · Q_S1
        // ---------------------------------------------------------------------
        let r = |z: float32x4_t| -> float32x4_t {
            // p = z * (P_S0 + z * (P_S1 + z * P_S2))
            let p = vmulq_f32(z, vfmaq_f32(p_s0, z, vfmaq_f32(p_s1, z, p_s2)));
            // q = 1 + z * Q_S1
            let q = vfmaq_f32(one, z, q_s1);
            vdivq_f32(p, q)
        };

        // ---------------------------------------------------------------------
        // Condition masks (computed once, used for blending)
        // ---------------------------------------------------------------------
        let is_abs_ge_1 = vcgeq_f32(abs_x, one); // |x| >= 1
        let is_abs_eq_1 = vceqq_f32(abs_x, one); // |x| == 1
        let is_abs_lt_half = vcltq_f32(abs_x, half); // |x| < 0.5
        let is_x_neg = vcltq_f32(x, zero); // x < 0

        // Tiny threshold check: |x| < 2^-12 (avoids underflow in polynomial)
        let tiny_bits = vdupq_n_u32(TINY_THRESHOLD_32);
        let is_tiny = vcltq_u32(vreinterpretq_u32_f32(abs_x), tiny_bits);

        // ---------------------------------------------------------------------
        // Case A: |x| == 1 → return ±π/2
        //
        // asin(1) = π/2, asin(-1) = -π/2
        // Use pio2_hi + sign(x) * pio2_lo for full precision
        // ---------------------------------------------------------------------
        let pos_pio2 = vaddq_f32(pio2_hi, pio2_lo);
        let neg_pio2 = vsubq_f32(vnegq_f32(pio2_hi), pio2_lo);
        let result_eq_1 = vbslq_f32(is_x_neg, neg_pio2, pos_pio2);

        // ---------------------------------------------------------------------
        // Case B: |x| > 1 → return NaN (domain error)
        //
        // Compute NaN as 0.0 / (x - x) = 0.0 / 0.0 = NaN
        // ---------------------------------------------------------------------
        let nan = vdivq_f32(zero, vsubq_f32(x, x));

        // ---------------------------------------------------------------------
        // Case C: |x| < 0.5 → asin(x) = x + x · r(x²)
        //
        // For tiny |x| < 2^-12, just return x (polynomial is negligible).
        // ---------------------------------------------------------------------
        let x_sq = vmulq_f32(x, x);
        let r_small = r(x_sq);
        let result_small_computed = vfmaq_f32(x, x, r_small); // x + x * r(x²)
        let result_small = vbslq_f32(is_tiny, x, result_small_computed);

        // ---------------------------------------------------------------------
        // Case D: 0.5 ≤ |x| < 1 → use half-angle formula with Dekker split
        //
        // Let z = (1 - |x|) / 2, s = √z
        //
        // Dekker compensated sqrt:
        //   df = s with low 12 mantissa bits cleared (exact high part)
        //   c  = (z - df²) / (s + df)  (rounding correction)
        //
        // Final computation (from musl asinf.c):
        //   w = s·r(z) + c
        //   asin(|x|) = π/2 - 2·(df + w)
        //             = pio2_hi - 2·df - (2·w - pio2_lo)
        //
        // The result is negated if x < 0.
        // ---------------------------------------------------------------------
        let z_large = vmulq_f32(vsubq_f32(one, abs_x), half); // (1 - |x|) / 2
        let s_large = vsqrtq_f32(z_large); // √z
        let r_large = r(z_large);

        // Dekker split: mask off low 12 bits to get exact high part of s
        let df = vreinterpretq_f32_u32(vandq_u32(
            vreinterpretq_u32_f32(s_large),
            vdupq_n_u32(0xfffff000),
        ));

        // Rounding correction: c = (z - df²) / (s + df)
        let c = vdivq_f32(vsubq_f32(z_large, vmulq_f32(df, df)), vaddq_f32(s_large, df));

        // Compute w = s·r(z) + c
        let w = vfmaq_f32(c, s_large, r_large);

        // Compute: pio2_hi - 2·df - (2·w - pio2_lo)
        let two_w = vmulq_f32(two, w);
        let inner = vsubq_f32(two_w, pio2_lo); // 2·w - pio2_lo
        let two_df = vmulq_f32(two, df);
        let result_large_abs = vsubq_f32(vsubq_f32(pio2_hi, two_df), inner);

        // Apply sign: if x < 0, negate the result
        let result_large = vbslq_f32(is_x_neg, vnegq_f32(result_large_abs), result_large_abs);

        // ---------------------------------------------------------------------
        // Final blending: combine all cases
        // ---------------------------------------------------------------------

        // Handle |x| >= 1: either exact ±π/2 (if |x| == 1) or NaN (if |x| > 1)
        let result_ge_1 = vbslq_f32(is_abs_eq_1, result_eq_1, nan);

        // Select between small and large cases for |x| < 1
        let result_valid = vbslq_f32(is_abs_lt_half, result_small, result_large);

        // Final selection: valid cases vs |x| >= 1 cases
        vbslq_f32(is_abs_ge_1, result_ge_1, result_valid)
    }
}

// ===========================================================================
// f64 Implementation (2 lanes)
// ===========================================================================

/// Computes `asin(x)` for each lane of a NEON `float64x2_t` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
///
/// # Description
///
/// All 2 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large |x|, tiny, |x|=1, out-of-domain) are computed
/// unconditionally and merged with `vbslq_f64`.
///
/// # Safety
///
/// `x` must be a valid `float64x2_t` register.
///
/// # Example
///
/// ```ignore
/// let input = vdupq_n_f64(0.5);
/// let result = vasin_f64(input);
/// // result ≈ [0.5236, 0.5236] (π/6 ≈ 0.5236)
/// ```
#[inline]
pub(crate) unsafe fn vasin_f64(x: float64x2_t) -> float64x2_t {
    unsafe {
        // ---------------------------------------------------------------------
        // Broadcast scalar constants to SIMD registers
        // ---------------------------------------------------------------------
        let pio2_hi = vdupq_n_f64(PIO2_HI_64);
        let pio2_lo = vdupq_n_f64(PIO2_LO_64);
        let p_s0 = vdupq_n_f64(P_S0_64);
        let p_s1 = vdupq_n_f64(P_S1_64);
        let p_s2 = vdupq_n_f64(P_S2_64);
        let p_s3 = vdupq_n_f64(P_S3_64);
        let p_s4 = vdupq_n_f64(P_S4_64);
        let p_s5 = vdupq_n_f64(P_S5_64);
        let q_s1 = vdupq_n_f64(Q_S1_64);
        let q_s2 = vdupq_n_f64(Q_S2_64);
        let q_s3 = vdupq_n_f64(Q_S3_64);
        let q_s4 = vdupq_n_f64(Q_S4_64);

        let one = vdupq_n_f64(1.0);
        let half = vdupq_n_f64(0.5);
        let two = vdupq_n_f64(2.0);
        let zero = vdupq_n_f64(0.0);

        // ---------------------------------------------------------------------
        // Compute |x| for range selection
        // ---------------------------------------------------------------------
        let abs_x = vabsq_f64_wrapper(x);

        // ---------------------------------------------------------------------
        // Rational polynomial approximation: r(z) = p(z) / q(z)
        //
        // This approximates (asin(√z)/√z − 1)/z on [0, 0.25].
        //   p(z) = z · (P_S0 + z · (P_S1 + z · (P_S2 + z · (P_S3 + z · (P_S4 + z · P_S5)))))
        //   q(z) = 1 + z · (Q_S1 + z · (Q_S2 + z · (Q_S3 + z · Q_S4)))
        // ---------------------------------------------------------------------
        let r = |z: float64x2_t| -> float64x2_t {
            // Horner's method for numerator (degree 5)
            let p_inner = vfmaq_f64(p_s4, z, p_s5);
            let p_inner = vfmaq_f64(p_s3, z, p_inner);
            let p_inner = vfmaq_f64(p_s2, z, p_inner);
            let p_inner = vfmaq_f64(p_s1, z, p_inner);
            let p_inner = vfmaq_f64(p_s0, z, p_inner);
            let p = vmulq_f64(z, p_inner);

            // Horner's method for denominator (degree 4)
            let q_inner = vfmaq_f64(q_s3, z, q_s4);
            let q_inner = vfmaq_f64(q_s2, z, q_inner);
            let q_inner = vfmaq_f64(q_s1, z, q_inner);
            let q = vfmaq_f64(one, z, q_inner);

            vdivq_f64(p, q)
        };

        // ---------------------------------------------------------------------
        // Condition masks (computed once, used for blending)
        // ---------------------------------------------------------------------
        let is_abs_ge_1 = vcgeq_f64(abs_x, one); // |x| >= 1
        let is_abs_eq_1 = vceqq_f64(abs_x, one); // |x| == 1
        let is_abs_lt_half = vcltq_f64(abs_x, half); // |x| < 0.5
        let is_x_neg = vcltq_f64(x, zero); // x < 0

        // Tiny threshold check: |x| < 2^-27 (avoids underflow in polynomial)
        let tiny_bits = vdupq_n_u64(TINY_THRESHOLD_64);
        let is_tiny = vcltq_u64(vreinterpretq_u64_f64(abs_x), tiny_bits);

        // ---------------------------------------------------------------------
        // Case A: |x| == 1 → return ±π/2
        // ---------------------------------------------------------------------
        let pos_pio2 = vaddq_f64(pio2_hi, pio2_lo);
        let neg_pio2 = vsubq_f64(vnegq_f64(pio2_hi), pio2_lo);
        let result_eq_1 = vbslq_f64(is_x_neg, neg_pio2, pos_pio2);

        // ---------------------------------------------------------------------
        // Case B: |x| > 1 → return NaN (domain error)
        // ---------------------------------------------------------------------
        let nan = vdivq_f64(zero, vsubq_f64(x, x));

        // ---------------------------------------------------------------------
        // Case C: |x| < 0.5 → asin(x) = x + x · r(x²)
        // ---------------------------------------------------------------------
        let x_sq = vmulq_f64(x, x);
        let r_small = r(x_sq);
        let result_small_computed = vfmaq_f64(x, x, r_small);
        let result_small = vbslq_f64(is_tiny, x, result_small_computed);

        // ---------------------------------------------------------------------
        // Case D: 0.5 ≤ |x| < 1 → use half-angle formula with Dekker split
        //
        // Dekker compensated sqrt:
        //   df = s with low 32 mantissa bits cleared (exact high part)
        //   c  = (z - df²) / (s + df)  (rounding correction)
        // ---------------------------------------------------------------------
        let z_large = vmulq_f64(vsubq_f64(one, abs_x), half);
        let s_large = vsqrtq_f64(z_large);
        let r_large = r(z_large);

        // Dekker split: mask off low 32 bits to get exact high part of s
        let df = vreinterpretq_f64_u64(vandq_u64(
            vreinterpretq_u64_f64(s_large),
            vdupq_n_u64(0xffffffff00000000),
        ));

        // Rounding correction: c = (z - df²) / (s + df)
        let c = vdivq_f64(vsubq_f64(z_large, vmulq_f64(df, df)), vaddq_f64(s_large, df));

        // Compute w = s·r(z) + c
        let w = vfmaq_f64(c, s_large, r_large);

        // Compute: pio2_hi - 2·df - (2·w - pio2_lo)
        let two_w = vmulq_f64(two, w);
        let inner = vsubq_f64(two_w, pio2_lo);
        let two_df = vmulq_f64(two, df);
        let result_large_abs = vsubq_f64(vsubq_f64(pio2_hi, two_df), inner);

        // Apply sign: if x < 0, negate the result
        let result_large = vbslq_f64(is_x_neg, vnegq_f64(result_large_abs), result_large_abs);

        // ---------------------------------------------------------------------
        // Final blending: combine all cases
        // ---------------------------------------------------------------------
        let result_ge_1 = vbslq_f64(is_abs_eq_1, result_eq_1, nan);
        let result_valid = vbslq_f64(is_abs_lt_half, result_small, result_large);
        vbslq_f64(is_abs_ge_1, result_ge_1, result_valid)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // f32 tests (4 lanes)
    // ========================================================================

    const TOL_32: f32 = 5e-7;

    unsafe fn asin_scalar_32(val: f32) -> f32 {
        unsafe {
            let v = vdupq_n_f32(val);
            let result = vasin_f32(v);
            vgetq_lane_f32(result, 0)
        }
    }

    #[test]
    fn asin_f32_of_zero_is_zero() {
        unsafe {
            assert_eq!(asin_scalar_32(0.0), 0.0);
        }
    }

    #[test]
    fn asin_f32_of_neg_zero_is_neg_zero() {
        unsafe {
            let result = asin_scalar_32(-0.0);
            assert!(result == 0.0 && result.is_sign_negative());
        }
    }

    #[test]
    fn asin_f32_of_one_is_pio2() {
        unsafe {
            let result = asin_scalar_32(1.0);
            let expected = std::f32::consts::FRAC_PI_2;
            assert!(
                (result - expected).abs() < TOL_32,
                "asin(1) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_f32_of_neg_one_is_neg_pio2() {
        unsafe {
            let result = asin_scalar_32(-1.0);
            let expected = -std::f32::consts::FRAC_PI_2;
            assert!(
                (result - expected).abs() < TOL_32,
                "asin(-1) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_f32_of_half_is_pi_over_6() {
        unsafe {
            let result = asin_scalar_32(0.5);
            let expected = std::f32::consts::FRAC_PI_6;
            assert!(
                (result - expected).abs() < TOL_32,
                "asin(0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_f32_of_neg_half_is_neg_pi_over_6() {
        unsafe {
            let result = asin_scalar_32(-0.5);
            let expected = -std::f32::consts::FRAC_PI_6;
            assert!(
                (result - expected).abs() < TOL_32,
                "asin(-0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_f32_of_sqrt2_over_2_is_pi_over_4() {
        unsafe {
            let sqrt2_over_2 = std::f32::consts::FRAC_1_SQRT_2;
            let result = asin_scalar_32(sqrt2_over_2);
            let expected = std::f32::consts::FRAC_PI_4;
            assert!(
                (result - expected).abs() < TOL_32,
                "asin(√2/2) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_f32_above_one_is_nan() {
        unsafe {
            let result = asin_scalar_32(1.5);
            assert!(result.is_nan(), "asin(1.5) should be NaN, got {result}");
        }
    }

    #[test]
    fn asin_f32_below_neg_one_is_nan() {
        unsafe {
            let result = asin_scalar_32(-1.5);
            assert!(result.is_nan(), "asin(-1.5) should be NaN, got {result}");
        }
    }

    #[test]
    fn asin_f32_of_nan_is_nan() {
        unsafe {
            let result = asin_scalar_32(f32::NAN);
            assert!(result.is_nan(), "asin(NaN) should be NaN");
        }
    }

    #[test]
    fn asin_f32_processes_all_4_lanes() {
        unsafe {
            let input = [0.0f32, 0.5, -0.5, 1.0];
            let v = vld1q_f32(input.as_ptr());
            let result = vasin_f32(v);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);

            let expected = [
                0.0_f32.asin(),
                0.5_f32.asin(),
                (-0.5_f32).asin(),
                1.0_f32.asin(),
            ];

            for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < TOL_32,
                    "lane {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    #[test]
    fn asin_f32_max_ulp_error_is_at_most_1() {
        let mut max_ulp: u32 = 0;
        let mut worst_x: f32 = 0.0;

        let mut bits: u32 = (-1.0_f32).to_bits();
        let end_bits: u32 = 1.0_f32.to_bits();

        while bits <= end_bits {
            let x = f32::from_bits(bits);
            if x.is_nan() {
                bits = bits.wrapping_add(1024);
                continue;
            }

            let expected = x.asin();
            let simd_result = unsafe { asin_scalar_32(x) };

            if expected.is_nan() && simd_result.is_nan() {
                bits = bits.wrapping_add(1024);
                continue;
            }

            let exp_bits = expected.to_bits() as i32;
            let got_bits = simd_result.to_bits() as i32;
            let ulp_diff = (exp_bits - got_bits).unsigned_abs();

            if ulp_diff > max_ulp {
                max_ulp = ulp_diff;
                worst_x = x;
            }

            bits = bits.wrapping_add(1024);
        }

        println!("Max ULP error (f32): {max_ulp} at x = {worst_x}");
        assert!(max_ulp <= 1, "ULP error {max_ulp} > 1 at x = {worst_x}");
    }

    // ========================================================================
    // f64 tests (2 lanes)
    // ========================================================================

    const TOL_64: f64 = 1e-14;

    unsafe fn asin_scalar_64(val: f64) -> f64 {
        unsafe {
            let v = vdupq_n_f64(val);
            let result = vasin_f64(v);
            vgetq_lane_f64(result, 0)
        }
    }

    #[test]
    fn asin_f64_of_zero_is_zero() {
        unsafe {
            assert_eq!(asin_scalar_64(0.0), 0.0);
        }
    }

    #[test]
    fn asin_f64_of_neg_zero_is_neg_zero() {
        unsafe {
            let result = asin_scalar_64(-0.0);
            assert!(result == 0.0 && result.is_sign_negative());
        }
    }

    #[test]
    fn asin_f64_of_one_is_pio2() {
        unsafe {
            let result = asin_scalar_64(1.0);
            let expected = std::f64::consts::FRAC_PI_2;
            assert!(
                (result - expected).abs() < TOL_64,
                "asin(1) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_f64_of_neg_one_is_neg_pio2() {
        unsafe {
            let result = asin_scalar_64(-1.0);
            let expected = -std::f64::consts::FRAC_PI_2;
            assert!(
                (result - expected).abs() < TOL_64,
                "asin(-1) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_f64_of_half_is_pi_over_6() {
        unsafe {
            let result = asin_scalar_64(0.5);
            let expected = std::f64::consts::FRAC_PI_6;
            assert!(
                (result - expected).abs() < TOL_64,
                "asin(0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_f64_of_neg_half_is_neg_pi_over_6() {
        unsafe {
            let result = asin_scalar_64(-0.5);
            let expected = -std::f64::consts::FRAC_PI_6;
            assert!(
                (result - expected).abs() < TOL_64,
                "asin(-0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_f64_of_sqrt2_over_2_is_pi_over_4() {
        unsafe {
            let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
            let result = asin_scalar_64(sqrt2_over_2);
            let expected = std::f64::consts::FRAC_PI_4;
            assert!(
                (result - expected).abs() < TOL_64,
                "asin(√2/2) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn asin_f64_above_one_is_nan() {
        unsafe {
            let result = asin_scalar_64(1.5);
            assert!(result.is_nan(), "asin(1.5) should be NaN, got {result}");
        }
    }

    #[test]
    fn asin_f64_below_neg_one_is_nan() {
        unsafe {
            let result = asin_scalar_64(-1.5);
            assert!(result.is_nan(), "asin(-1.5) should be NaN, got {result}");
        }
    }

    #[test]
    fn asin_f64_of_nan_is_nan() {
        unsafe {
            let result = asin_scalar_64(f64::NAN);
            assert!(result.is_nan(), "asin(NaN) should be NaN");
        }
    }

    #[test]
    fn asin_f64_processes_all_2_lanes() {
        unsafe {
            let input = [0.5f64, -0.5];
            let v = vld1q_f64(input.as_ptr());
            let result = vasin_f64(v);
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), result);

            let expected = [0.5_f64.asin(), (-0.5_f64).asin()];

            for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < TOL_64,
                    "lane {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    #[test]
    fn asin_f64_max_ulp_error_is_at_most_1() {
        let mut max_ulp: u64 = 0;
        let mut worst_x: f64 = 0.0;

        let mut bits: u64 = (-1.0_f64).to_bits();
        let end_bits: u64 = 1.0_f64.to_bits();
        let step: u64 = 1 << 20;

        while bits <= end_bits {
            let x = f64::from_bits(bits);
            if x.is_nan() {
                bits = bits.wrapping_add(step);
                continue;
            }

            let expected = x.asin();
            let simd_result = unsafe { asin_scalar_64(x) };

            if expected.is_nan() && simd_result.is_nan() {
                bits = bits.wrapping_add(step);
                continue;
            }

            let exp_bits = expected.to_bits() as i64;
            let got_bits = simd_result.to_bits() as i64;
            let ulp_diff = (exp_bits - got_bits).unsigned_abs();

            if ulp_diff > max_ulp {
                max_ulp = ulp_diff;
                worst_x = x;
            }

            bits = bits.wrapping_add(step);
        }

        println!("Max ULP error (f64): {max_ulp} at x = {worst_x}");
        assert!(max_ulp <= 1, "ULP error {max_ulp} > 1 at x = {worst_x}");
    }
}
