//! NEON SIMD implementation of `acos(x)` for `f32` and `f64` vectors.
//!
//! This module provides 4-lane f32 and 2-lane f64 implementations using the
//! same algorithm as the AVX2/AVX-512 versions: a branchless, three-range
//! minimax rational approximation ported from musl libc's `acosf.c` and `acos.c`.
//!
//! NEON uses `vbslq` (bitwise select) for blending, which is equivalent to
//! the `blendv` operations in x86 SIMD.

use std::arch::aarch64::*;

use crate::arch::consts::acos::{
    P_S0_32, P_S0_64, P_S1_32, P_S1_64, P_S2_32, P_S2_64, P_S3_64, P_S4_64, P_S5_64, PIO2_HI_32,
    PIO2_HI_64, PIO2_LO_32, PIO2_LO_64, Q_S1_32, Q_S1_64, Q_S2_64, Q_S3_64, Q_S4_64, X1P_120_32,
};
use crate::arch::neon::abs::{vabsq_f32_wrapper, vabsq_f64_wrapper};

// ===========================================================================
// f32 Implementation (4 lanes)
// ===========================================================================

/// Computes `acos(x)` for each lane of a NEON `float32x4_t` register.
///
/// All 4 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large positive, large negative, |x|=1, out-of-domain)
/// are computed unconditionally and merged with `vbslq_f32`.
///
/// # Safety
/// `x` must be a valid `float32x4_t` register.
#[inline]
pub(crate) unsafe fn vacos_f32(x: float32x4_t) -> float32x4_t {
    unsafe {
        // Broadcast scalar constants to SIMD registers once.
        let pio2_hi = vdupq_n_f32(PIO2_HI_32);
        let pio2_lo = vdupq_n_f32(PIO2_LO_32);
        let p_s0 = vdupq_n_f32(P_S0_32);
        let p_s1 = vdupq_n_f32(P_S1_32);
        let p_s2 = vdupq_n_f32(P_S2_32);
        let q_s1 = vdupq_n_f32(Q_S1_32);
        let x1p_120 = vdupq_n_f32(X1P_120_32);
        let one = vdupq_n_f32(1.0);
        let half = vdupq_n_f32(0.5);
        let two = vdupq_n_f32(2.0);
        let zero = vdupq_n_f32(0.0);

        // |x| — used to determine which range each lane falls into.
        let abs_x = vabsq_f32_wrapper(x);

        // -------------------------------------------------------------------------
        // Padé rational approximation r(z)
        //
        //   p(z) = P_S0 + z·(P_S1 + z·P_S2)   [degree-2 numerator, Horner form]
        //   q(z) = 1 + z·Q_S1                  [degree-1 denominator]
        //   r(z) = z·p(z) / q(z)
        //
        // r(z) ≈ (asin(√z)/√z − 1)/z on [0, 0.25], so asin(x) ≈ x + x·r(x²).
        // -------------------------------------------------------------------------
        let r = |z: float32x4_t| -> float32x4_t {
            // p = z * fma(z, fma(z, p_s2, p_s1), p_s0)
            let p = vmulq_f32(z, vfmaq_f32(p_s0, z, vfmaq_f32(p_s1, z, p_s2)));
            // q = fma(z, q_s1, one) = 1 + z * q_s1
            let q = vfmaq_f32(one, z, q_s1);
            vdivq_f32(p, q)
        };

        // -------------------------------------------------------------------------
        // Lane classification masks (ordered, quiet: NaN → false for all)
        // -------------------------------------------------------------------------
        let is_abs_ge_1 = vcgeq_f32(abs_x, one); // |x| >= 1
        let is_abs_eq_1 = vceqq_f32(abs_x, one); // |x| == 1
        let is_abs_lt_half = vcltq_f32(abs_x, half); // |x| < 0.5
        let is_x_neg = vcltq_f32(x, zero); // x < 0

        // Lanes with x in (-1, -0.5]: x is negative AND |x| >= 0.5.
        // ~is_abs_lt_half & is_x_neg
        let is_x_neg_large = vandq_u32(vmvnq_u32(is_abs_lt_half), is_x_neg);

        // -------------------------------------------------------------------------
        // Case A — |x| == 1  (exact boundary values)
        //
        //   acos( 1) = 0
        //   acos(-1) = 2·pio2_hi + x1p_120  ≈ π
        //              (x1p_120 raises the IEEE 754 inexact flag as mandated)
        // -------------------------------------------------------------------------
        let result_eq_1 = vbslq_f32(
            is_x_neg,
            vaddq_f32(vmulq_f32(two, pio2_hi), x1p_120), // x = -1 → π
            zero,                                        // x = +1 → 0
        );

        // -------------------------------------------------------------------------
        // Case B — |x| > 1  (out-of-domain → NaN)
        //
        // x − x = 0 for all finite x; dividing 0/0 produces NaN. The subtraction
        // prevents compilers from constant-folding the expression.
        // -------------------------------------------------------------------------
        let nan = vdivq_f32(zero, vsubq_f32(x, x));

        // For |x| >= 1: select exact result where |x| == 1, NaN otherwise.
        let result_ge_1 = vbslq_f32(is_abs_eq_1, result_eq_1, nan);

        // -------------------------------------------------------------------------
        // Case C — |x| < 0.5  (small argument)
        //
        // acos(x) = π/2 − asin(x) ≈ π/2 − x − x·r(x²)
        //         = pio2_hi − (x − (pio2_lo − x·r(x²)))
        //
        // The nested subtraction keeps pio2_hi and pio2_lo paired correctly so
        // that precision is not lost through the two-part representation of π/2.
        // -------------------------------------------------------------------------
        let z_small = vmulq_f32(x, x);
        // Case C: pio2_lo − x·r(z) as a single FMA (fnmadd = −a·b + c).
        // NEON: vfmsq_f32(c, a, b) = c - a*b, so we use vfmsq_f32(pio2_lo, x, r(z_small))
        let result_small = vsubq_f32(pio2_hi, vsubq_f32(x, vfmsq_f32(pio2_lo, x, r(z_small))));

        // -------------------------------------------------------------------------
        // Case D — x < -0.5  (large negative argument)
        //
        // Identity: acos(x) = π − 2·asin(√((1+x)/2))
        //         = 2·(π/2 − asin(s))  where s = √z, z = (1+x)/2
        //         = 2·(pio2_hi − (s + w))
        //
        //   w = r(z)·s − pio2_lo   (the pio2_lo term completes the two-sum)
        // -------------------------------------------------------------------------
        let z_neg = vmulq_f32(vaddq_f32(one, x), half);
        let s_neg = vsqrtq_f32(z_neg);
        // Case D: r(z)·s − pio2_lo as a single FMA (fmsub = a·b − c).
        // NEON: We need a*b - c. vfmaq_f32(c, a, b) = c + a*b, so we do:
        // r(z)*s - pio2_lo = -(pio2_lo - r(z)*s) = -vfmsq_f32(pio2_lo, r(z_neg), s_neg)
        // Or simply: vfmsq_f32(neg_pio2_lo, ...) but easier to just compute separately
        let w_neg = vsubq_f32(vmulq_f32(r(z_neg), s_neg), pio2_lo);
        let result_neg = vmulq_f32(two, vsubq_f32(pio2_hi, vaddq_f32(s_neg, w_neg)));

        // -------------------------------------------------------------------------
        // Case E — x > 0.5  (large positive argument)
        //
        // Identity: acos(x) = 2·asin(√((1−x)/2)) = 2·(df + w)
        //
        //   z  = (1−x)/2
        //   s  = √z
        //   df = s with the low 12 mantissa bits cleared   (Dekker high part of s)
        //   c  = (z − df²) / (s + df)                     (rounding correction)
        //   w  = r(z)·s + c
        //
        // Clearing 12 bits leaves df with only 11 significant mantissa bits, so
        // df² is exactly representable in 23 bits — making (z − df²) exact.
        // Then c = (s² − df²)/(s + df) = s − df recovers the discarded low bits.
        // Together, df + c is a full-precision reconstruction of s (Dekker split).
        // -------------------------------------------------------------------------
        let z_pos = vmulq_f32(vsubq_f32(one, x), half);
        let s_pos = vsqrtq_f32(z_pos);

        // Mask off the low 12 bits of the f32 bit representation to form df.
        let df = vreinterpretq_f32_u32(vandq_u32(
            vreinterpretq_u32_f32(s_pos),
            vdupq_n_u32(0xfffff000),
        ));
        // Recover the rounding error introduced by truncating s to df.
        let c_pos = vdivq_f32(vsubq_f32(z_pos, vmulq_f32(df, df)), vaddq_f32(s_pos, df));
        // Case E: r(z)·s + c as a single FMA.
        let w_pos = vfmaq_f32(c_pos, r(z_pos), s_pos);
        let result_pos = vmulq_f32(two, vaddq_f32(df, w_pos));

        // -------------------------------------------------------------------------
        // Merge: blend all cases. Priority (highest → lowest):
        //   |x| >= 1   → result_ge_1  (exact or NaN)
        //   |x| < 0.5  → result_small (Case C)
        //   x ∈ (-1,-0.5] → result_neg (Case D)
        //   x ∈ [0.5, 1)  → result_pos (Case E)
        // -------------------------------------------------------------------------
        let result_large = vbslq_f32(is_x_neg_large, result_neg, result_pos);
        let result_valid = vbslq_f32(is_abs_lt_half, result_small, result_large);
        vbslq_f32(is_abs_ge_1, result_ge_1, result_valid)
    }
}

// ===========================================================================
// f64 Implementation (2 lanes)
// ===========================================================================

/// Computes `acos(x)` for each lane of a NEON `float64x2_t` register.
///
/// All 2 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large positive, large negative, |x|=1, out-of-domain)
/// are computed unconditionally and merged with `vbslq_f64`.
///
/// # Safety
/// `x` must be a valid `float64x2_t` register.
#[inline]
pub(crate) unsafe fn vacos_f64(x: float64x2_t) -> float64x2_t {
    unsafe {
        // Broadcast scalar constants to SIMD registers once.
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

        // |x| — used to determine which range each lane falls into.
        let abs_x = vabsq_f64_wrapper(x);

        // -------------------------------------------------------------------------
        // Padé rational approximation r(z) - degree-5/degree-4 for f64
        //
        //   p(z) = P_S0 + z·(P_S1 + z·(P_S2 + z·(P_S3 + z·(P_S4 + z·P_S5))))
        //   q(z) = 1 + z·(Q_S1 + z·(Q_S2 + z·(Q_S3 + z·Q_S4)))
        //   r(z) = z·p(z) / q(z)
        // -------------------------------------------------------------------------
        let r = |z: float64x2_t| -> float64x2_t {
            let p = vmulq_f64(
                z,
                vfmaq_f64(
                    p_s0,
                    z,
                    vfmaq_f64(
                        p_s1,
                        z,
                        vfmaq_f64(p_s2, z, vfmaq_f64(p_s3, z, vfmaq_f64(p_s4, z, p_s5))),
                    ),
                ),
            );
            let q = vfmaq_f64(
                one,
                z,
                vfmaq_f64(q_s1, z, vfmaq_f64(q_s2, z, vfmaq_f64(q_s3, z, q_s4))),
            );
            vdivq_f64(p, q)
        };

        // -------------------------------------------------------------------------
        // Lane classification masks (ordered, quiet: NaN → false for all)
        // -------------------------------------------------------------------------
        let is_abs_ge_1 = vcgeq_f64(abs_x, one); // |x| >= 1
        let is_abs_eq_1 = vceqq_f64(abs_x, one); // |x| == 1
        let is_abs_lt_half = vcltq_f64(abs_x, half); // |x| < 0.5
        let is_x_neg = vcltq_f64(x, zero); // x < 0

        // Lanes with x in (-1, -0.5]: x is negative AND |x| >= 0.5.
        // ~is_abs_lt_half & is_x_neg
        let is_x_neg_large = vandq_u64(vmvnq_u64(is_abs_lt_half), is_x_neg);

        // -------------------------------------------------------------------------
        // Case A — |x| == 1  (exact boundary values)
        //
        //   acos( 1) = 0
        //   acos(-1) = π = 2·pio2_hi
        // -------------------------------------------------------------------------
        let result_eq_1 = vbslq_f64(
            is_x_neg,
            vmulq_f64(two, pio2_hi), // x = -1 → π
            zero,                    // x = +1 → 0
        );

        // -------------------------------------------------------------------------
        // Case B — |x| > 1  (out-of-domain → NaN)
        //
        // x − x = 0 for all finite x; dividing 0/0 produces NaN. The subtraction
        // prevents compilers from constant-folding the expression.
        // -------------------------------------------------------------------------
        let nan = vdivq_f64(zero, vsubq_f64(x, x));

        // For |x| >= 1: select exact result where |x| == 1, NaN otherwise.
        let result_ge_1 = vbslq_f64(is_abs_eq_1, result_eq_1, nan);

        // -------------------------------------------------------------------------
        // Case C — |x| < 0.5  (small argument)
        //
        // acos(x) = π/2 − asin(x) ≈ π/2 − x − x·r(x²)
        //         = pio2_hi − (x − (pio2_lo − x·r(x²)))
        //
        // The nested subtraction keeps pio2_hi and pio2_lo paired correctly so
        // that precision is not lost through the two-part representation of π/2.
        // -------------------------------------------------------------------------
        let z_small = vmulq_f64(x, x);
        // Case C: pio2_lo − x·r(z) as a single FMA (fnmadd = −a·b + c).
        // NEON: vfmsq_f64(c, a, b) = c - a*b
        let result_small = vsubq_f64(pio2_hi, vsubq_f64(x, vfmsq_f64(pio2_lo, x, r(z_small))));

        // -------------------------------------------------------------------------
        // Case D — x < -0.5  (large negative argument)
        //
        // Identity: acos(x) = π − 2·asin(√((1+x)/2))
        //         = 2·(π/2 − asin(s))  where s = √z, z = (1+x)/2
        //         = 2·(pio2_hi − (s + w))
        //
        //   w = r(z)·s − pio2_lo   (the pio2_lo term completes the two-sum)
        // -------------------------------------------------------------------------
        let z_neg = vmulq_f64(vaddq_f64(one, x), half);
        let s_neg = vsqrtq_f64(z_neg);
        // Case D: r(z)·s − pio2_lo as a single FMA (fmsub = a·b − c).
        let w_neg = vsubq_f64(vmulq_f64(r(z_neg), s_neg), pio2_lo);
        let result_neg = vmulq_f64(two, vsubq_f64(pio2_hi, vaddq_f64(s_neg, w_neg)));

        // -------------------------------------------------------------------------
        // Case E — x > 0.5  (large positive argument)
        //
        // Identity: acos(x) = 2·asin(√((1−x)/2)) = 2·(df + w)
        //
        //   z  = (1−x)/2
        //   s  = √z
        //   df = s with the low 32 mantissa bits cleared   (Dekker high part of s)
        //   c  = (z − df²) / (s + df)                     (rounding correction)
        //   w  = r(z)·s + c
        //
        // Clearing 32 bits leaves df with only 20 significant mantissa bits, so
        // df² is exactly representable — making (z − df²) exact.
        // Then c = (s² − df²)/(s + df) = s − df recovers the discarded low bits.
        // Together, df + c is a full-precision reconstruction of s (Dekker split).
        // -------------------------------------------------------------------------
        let z_pos = vmulq_f64(vsubq_f64(one, x), half);
        let s_pos = vsqrtq_f64(z_pos);

        // Mask off the low 32 bits of the f64 bit representation to form df.
        let df = vreinterpretq_f64_u64(vandq_u64(
            vreinterpretq_u64_f64(s_pos),
            vdupq_n_u64(0xffffffff00000000),
        ));
        // Recover the rounding error introduced by truncating s to df.
        let c_pos = vdivq_f64(vsubq_f64(z_pos, vmulq_f64(df, df)), vaddq_f64(s_pos, df));
        // Case E: r(z)·s + c as a single FMA.
        let w_pos = vfmaq_f64(c_pos, r(z_pos), s_pos);
        let result_pos = vmulq_f64(two, vaddq_f64(df, w_pos));

        // -------------------------------------------------------------------------
        // Merge: blend all cases. Priority (highest → lowest):
        //   |x| >= 1   → result_ge_1  (exact or NaN)
        //   |x| < 0.5  → result_small (Case C)
        //   x ∈ (-1,-0.5] → result_neg (Case D)
        //   x ∈ [0.5, 1)  → result_pos (Case E)
        // -------------------------------------------------------------------------
        let result_large = vbslq_f64(is_x_neg_large, result_neg, result_pos);
        let result_valid = vbslq_f64(is_abs_lt_half, result_small, result_large);
        vbslq_f64(is_abs_ge_1, result_ge_1, result_valid)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // f32 tests
    // ========================================================================

    const TOL_32: f32 = 5e-7;

    unsafe fn acos_scalar_32(val: f32) -> f32 {
        unsafe {
            let v = vdupq_n_f32(val);
            let result = vacos_f32(v);
            vgetq_lane_f32(result, 0)
        }
    }

    #[test]
    fn acos_f32_of_one_is_zero() {
        unsafe {
            assert_eq!(acos_scalar_32(1.0), 0.0);
        }
    }

    #[test]
    fn acos_f32_of_neg_one_is_pi() {
        unsafe {
            let result = acos_scalar_32(-1.0);
            assert!(
                (result - std::f32::consts::PI).abs() < TOL_32,
                "acos(-1) = {result}, expected π"
            );
        }
    }

    #[test]
    fn acos_f32_of_zero_is_pio2() {
        unsafe {
            let result = acos_scalar_32(0.0);
            assert!(
                (result - std::f32::consts::FRAC_PI_2).abs() < TOL_32,
                "acos(0) = {result}"
            );
        }
    }

    #[test]
    fn acos_f32_of_half_is_pi_over_3() {
        unsafe {
            let result = acos_scalar_32(0.5);
            let expected = std::f32::consts::PI / 3.0;
            assert!(
                (result - expected).abs() < TOL_32,
                "acos(0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn acos_f32_above_one_is_nan() {
        unsafe {
            assert!(acos_scalar_32(1.5).is_nan());
        }
    }

    #[test]
    fn acos_f32_of_nan_is_nan() {
        unsafe {
            assert!(acos_scalar_32(f32::NAN).is_nan());
        }
    }

    #[test]
    fn acos_f32_processes_all_4_lanes() {
        let inputs: [f32; 4] = [0.0, 0.5, -0.5, 0.9];
        unsafe {
            let v = vld1q_f32(inputs.as_ptr());
            let result = vacos_f32(v);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);

            let expected: [f32; 4] = inputs.map(|x| x.acos());
            for (i, (&r, &e)) in out.iter().zip(&expected).enumerate() {
                assert!(
                    (r - e).abs() < TOL_32,
                    "lane {i}: acos({}) = {r}, expected {e}",
                    inputs[i]
                );
            }
        }
    }

    // ========================================================================
    // f64 tests
    // ========================================================================

    const TOL_64: f64 = 1e-15;

    unsafe fn acos_scalar_64(val: f64) -> f64 {
        unsafe {
            let v = vdupq_n_f64(val);
            let result = vacos_f64(v);
            vgetq_lane_f64(result, 0)
        }
    }

    #[test]
    fn acos_f64_of_one_is_zero() {
        unsafe {
            assert_eq!(acos_scalar_64(1.0), 0.0);
        }
    }

    #[test]
    fn acos_f64_of_neg_one_is_pi() {
        unsafe {
            let result = acos_scalar_64(-1.0);
            assert!(
                (result - std::f64::consts::PI).abs() < TOL_64,
                "acos(-1) = {result}, expected π"
            );
        }
    }

    #[test]
    fn acos_f64_of_zero_is_pio2() {
        unsafe {
            let result = acos_scalar_64(0.0);
            assert!(
                (result - std::f64::consts::FRAC_PI_2).abs() < TOL_64,
                "acos(0) = {result}"
            );
        }
    }

    #[test]
    fn acos_f64_of_half_is_pi_over_3() {
        unsafe {
            let result = acos_scalar_64(0.5);
            let expected = std::f64::consts::PI / 3.0;
            assert!(
                (result - expected).abs() < TOL_64,
                "acos(0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn acos_f64_above_one_is_nan() {
        unsafe {
            assert!(acos_scalar_64(1.5).is_nan());
        }
    }

    #[test]
    fn acos_f64_of_nan_is_nan() {
        unsafe {
            assert!(acos_scalar_64(f64::NAN).is_nan());
        }
    }

    #[test]
    fn acos_f64_processes_all_2_lanes() {
        let inputs: [f64; 2] = [0.0, 0.5];
        unsafe {
            let v = vld1q_f64(inputs.as_ptr());
            let result = vacos_f64(v);
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), result);

            let expected: [f64; 2] = inputs.map(|x| x.acos());
            for (i, (&r, &e)) in out.iter().zip(&expected).enumerate() {
                assert!(
                    (r - e).abs() < TOL_64,
                    "lane {i}: acos({}) = {r}, expected {e}",
                    inputs[i]
                );
            }
        }
    }
}
