//! NEON SIMD implementation of `atan(x)` for `f32` and `f64` vectors.
//!
//! # Algorithm
//!
//! Both implementations use a branchless argument reduction followed by a
//! minimax polynomial approximation, identical to the AVX2/AVX-512 versions
//! but adapted for ARM NEON (4 lanes for f32, 2 lanes for f64).
//!
//! ## f32 — Single-range reduction
//!
//! For `|x| > 1`: `atan(x) = π/2 - atan(1/x)`, reducing to `[-1, 1]`.
//! A degree-9 odd polynomial approximates `atan` on that range.
//!
//! ## f64 — Four-range reduction (musl libc)
//!
//! The domain is split at the breakpoints `{7/16, 11/16, 19/16, 39/16}` with
//! a dedicated two-sum offset `atanhi[i] + atanlo[i]` per range for full
//! precision compensation. A degree-11 polynomial (split into odd/even parts)
//! is applied to the reduced argument.
//!
//! # Precision
//!
//! | Implementation | Accuracy    |
//! |----------------|-------------|
//! | `vatan_f32`    | ≤ 3 ULP     |
//! | `vatan_f64`    | ≤ 1 ULP     |
//!
//! ## Special values
//!
//! | Input   | Output         |
//! |---------|----------------|
//! | `0.0`   | `0.0`          |
//! | `-0.0`  | `-0.0`         |
//! | `1.0`   | `π/4`          |
//! | `-1.0`  | `-π/4`         |
//! | `+∞`    | `π/2`          |
//! | `-∞`    | `-π/2`         |
//! | `NaN`   | `NaN`          |
//!
//! ## Blending strategy
//!
//! All branches are computed unconditionally per lane; results are merged with
//! `vbslq_f32` / `vbslq_f64`. This keeps all SIMD lanes active simultaneously
//! and avoids scalar fall-backs.
//!
//! ## NEON notes
//!
//! - `vbslq` argument order: `vbslq(mask, true_val, false_val)` (differs from x86!)
//! - FMA accumulator first: `vfmaq(c, a, b)` = a*b + c

use std::arch::aarch64::*;

use crate::arch::consts::atan::{
    AT0, AT1, AT2, AT3, AT4, AT5, AT6, AT7, AT8, AT9, AT10, ATAN_P0_32, ATAN_P1_32, ATAN_P2_32,
    ATAN_P3_32, ATAN_P4_32, ATAN_P5_32, ATAN_P6_32, ATAN_P7_32, ATAN_P8_32, ATAN_THRESH_0,
    ATAN_THRESH_1, ATAN_THRESH_2, ATAN_THRESH_3, ATANHI_0, ATANHI_1, ATANHI_2, ATANHI_3, ATANLO_0,
    ATANLO_1, ATANLO_2, ATANLO_3, FRAC_PI_2_32,
};
use crate::arch::neon::abs::{vabsq_f32_wrapper, vabsq_f64_wrapper};

// ---------------------------------------------------------------------------
// f32 Implementation (4 lanes)
// ---------------------------------------------------------------------------

/// Computes `atan(x)` for each lane of a NEON `float32x4_t` register.
///
/// # Precision
///
/// **≤ 3 ULP** error across the entire domain.
///
/// # Description
///
/// All 4 lanes are processed simultaneously without branches. The algorithm
/// uses argument reduction for `|x| > 1` and a degree-9 minimax polynomial
/// for the core approximation.
///
/// # Safety
///
/// `x` must be a valid `float32x4_t` register.
///
/// # Example
///
/// ```ignore
/// let input = vdupq_n_f32(1.0);
/// let result = vatan_f32(input);
/// // All 4 lanes ≈ π/4 ≈ 0.7854
/// ```
#[inline]
pub(crate) unsafe fn vatan_f32(x: float32x4_t) -> float32x4_t {
    unsafe {
    // ---------------------------------------------------------------------
    // Broadcast constants to SIMD registers
    // ---------------------------------------------------------------------
    let one = vdupq_n_f32(1.0);
    let frac_pi_2 = vdupq_n_f32(FRAC_PI_2_32);

    // Polynomial coefficients
    let p0 = vdupq_n_f32(ATAN_P0_32);
    let p1 = vdupq_n_f32(ATAN_P1_32);
    let p2 = vdupq_n_f32(ATAN_P2_32);
    let p3 = vdupq_n_f32(ATAN_P3_32);
    let p4 = vdupq_n_f32(ATAN_P4_32);
    let p5 = vdupq_n_f32(ATAN_P5_32);
    let p6 = vdupq_n_f32(ATAN_P6_32);
    let p7 = vdupq_n_f32(ATAN_P7_32);
    let p8 = vdupq_n_f32(ATAN_P8_32);

    // ---------------------------------------------------------------------
    // Extract sign and compute |x|
    //
    // We extract the sign bit directly to preserve -0.0 correctly.
    // atan(-x) = -atan(x), so we work with |x| and restore sign at end.
    // ---------------------------------------------------------------------
    // Sign mask: 0x80000000 in each lane
    let sign_mask_u32 = vdupq_n_u32(0x80000000);
    let sign_bits = vandq_u32(vreinterpretq_u32_f32(x), sign_mask_u32);
    let abs_x = vabsq_f32_wrapper(x);

    // ---------------------------------------------------------------------
    // Argument reduction
    //
    // For |x| > 1: atan(|x|) = π/2 - atan(1/|x|)
    // ---------------------------------------------------------------------
    let needs_reduction = vcgtq_f32(abs_x, one);

    // Reduced argument: use 1/|x| if |x| > 1, otherwise |x|
    let recip = vdivq_f32(one, abs_x);
    // NEON vbslq: vbslq(mask, true_val, false_val)
    let t = vbslq_f32(needs_reduction, recip, abs_x);

    // ---------------------------------------------------------------------
    // Polynomial evaluation using Horner's method
    //
    // atan(t) ≈ t · (P0 + t² · (P1 + t² · (P2 + t² · (P3 + ...))))
    //
    // We evaluate the polynomial in t² to exploit the odd symmetry of atan.
    // NEON vfmaq: vfmaq(c, a, b) = a*b + c
    // ---------------------------------------------------------------------
    let t2 = vmulq_f32(t, t);

    // Horner's method from highest to lowest coefficient
    let mut u = p8;
    u = vfmaq_f32(p7, u, t2); // P7 + t² · P8
    u = vfmaq_f32(p6, u, t2); // P6 + t² · (P7 + ...)
    u = vfmaq_f32(p5, u, t2); // P5 + ...
    u = vfmaq_f32(p4, u, t2); // P4 + ...
    u = vfmaq_f32(p3, u, t2); // P3 + ...
    u = vfmaq_f32(p2, u, t2); // P2 + ...
    u = vfmaq_f32(p1, u, t2); // P1 + ...
    u = vfmaq_f32(p0, u, t2); // P0 + ...

    // Multiply by t to get the final polynomial value
    let poly_result = vmulq_f32(u, t);

    // ---------------------------------------------------------------------
    // Apply argument reduction correction
    //
    // If |x| > 1: atan(|x|) = π/2 - atan(1/|x|) = π/2 - poly_result
    // Otherwise: atan(|x|) = poly_result
    // ---------------------------------------------------------------------
    let reduced_result = vsubq_f32(frac_pi_2, poly_result);
    let abs_result = vbslq_f32(needs_reduction, reduced_result, poly_result);

    // ---------------------------------------------------------------------
    // Restore sign: atan(-x) = -atan(x)
    //
    // XOR the result with the original sign bits to restore the sign.
    // This correctly handles -0.0 → -0.0.
    // ---------------------------------------------------------------------
    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(abs_result), sign_bits))
    }
}

// ===========================================================================
// f64 Implementation (2 lanes)
// ===========================================================================

/// Computes `atan(x)` for each lane of a NEON `float64x2_t` register (2 × f64).
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain, ported from **musl libc `atan.c`**.
///
/// # Description
///
/// Uses a 4-range argument reduction with two-sum offsets `atanhi + atanlo`
/// per range (breakpoints at 7/16, 11/16, 19/16, 39/16), followed by an
/// 11-term polynomial split into odd/even parts for efficiency.
///
/// All 2 lanes are processed simultaneously without branches.
///
/// # Safety
///
/// `x` must be a valid `float64x2_t` register.
///
/// # Example
///
/// ```ignore
/// let input = vdupq_n_f64(1.0);
/// let result = vatan_f64(input);
/// // Both lanes ≈ π/4 ≈ 0.7854
/// ```
#[inline]
pub(crate) unsafe fn vatan_f64(x: float64x2_t) -> float64x2_t {
    unsafe {
    // ---------------------------------------------------------------------
    // Broadcast constants
    // ---------------------------------------------------------------------
    let one = vdupq_n_f64(1.0);
    let two = vdupq_n_f64(2.0);
    let three = vdupq_n_f64(3.0);

    // Polynomial coefficients (musl aT[])
    let at0 = vdupq_n_f64(AT0);
    let at1 = vdupq_n_f64(AT1);
    let at2 = vdupq_n_f64(AT2);
    let at3 = vdupq_n_f64(AT3);
    let at4 = vdupq_n_f64(AT4);
    let at5 = vdupq_n_f64(AT5);
    let at6 = vdupq_n_f64(AT6);
    let at7 = vdupq_n_f64(AT7);
    let at8 = vdupq_n_f64(AT8);
    let at9 = vdupq_n_f64(AT9);
    let at10 = vdupq_n_f64(AT10);

    // Two-sum offsets for each reduction range
    let hi0 = vdupq_n_f64(ATANHI_0);
    let lo0 = vdupq_n_f64(ATANLO_0);
    let hi1 = vdupq_n_f64(ATANHI_1);
    let lo1 = vdupq_n_f64(ATANLO_1);
    let hi2 = vdupq_n_f64(ATANHI_2);
    let lo2 = vdupq_n_f64(ATANLO_2);
    let hi3 = vdupq_n_f64(ATANHI_3);
    let lo3 = vdupq_n_f64(ATANLO_3);

    // Range thresholds
    let thr0 = vdupq_n_f64(ATAN_THRESH_0); // 7/16 = 0.4375
    let thr1 = vdupq_n_f64(ATAN_THRESH_1); // 11/16 = 0.6875
    let thr2 = vdupq_n_f64(ATAN_THRESH_2); // 19/16 = 1.1875
    let thr3 = vdupq_n_f64(ATAN_THRESH_3); // 39/16 = 2.4375

    // ---------------------------------------------------------------------
    // Extract sign and compute |x|
    // ---------------------------------------------------------------------
    // Sign mask: 0x8000000000000000 in each lane
    let sign_mask_u64 = vdupq_n_u64(0x8000000000000000);
    let sign_bits = vandq_u64(vreinterpretq_u64_f64(x), sign_mask_u64);
    let abs_x = vabsq_f64_wrapper(x);

    // ---------------------------------------------------------------------
    // Compute range masks (from smallest to largest)
    // ---------------------------------------------------------------------
    let is_lt_thr0 = vcltq_f64(abs_x, thr0); // |x| < 7/16
    let is_lt_thr1 = vcltq_f64(abs_x, thr1); // |x| < 11/16
    let is_lt_thr2 = vcltq_f64(abs_x, thr2); // |x| < 19/16
    let is_lt_thr3 = vcltq_f64(abs_x, thr3); // |x| < 39/16

    // ---------------------------------------------------------------------
    // Compute all 4 reduced arguments unconditionally
    //
    //  id=-1 (|x| < 7/16):  t = abs_x  (no reduction)
    //  id=0  (7/16..11/16): t = (2·|x| − 1) / (2 + |x|)
    //  id=1  (11/16..19/16):t = (|x| − 1) / (|x| + 1)
    //  id=2  (19/16..39/16):t = (2·|x| − 3) / (2 + 3·|x|)
    //  id=3  (|x| ≥ 39/16): t = −1 / |x|   (gives π/2 − atan(1/|x|))
    // ---------------------------------------------------------------------

    // id=0: (2·|x| − 1) / (2 + |x|)
    let t0 = vdivq_f64(
        vsubq_f64(vmulq_f64(two, abs_x), one),
        vaddq_f64(two, abs_x),
    );

    // id=1: (|x| − 1) / (|x| + 1)
    let t1 = vdivq_f64(vsubq_f64(abs_x, one), vaddq_f64(abs_x, one));

    // id=2: (2·|x| − 3) / (2 + 3·|x|)
    let t2 = vdivq_f64(
        vsubq_f64(vmulq_f64(two, abs_x), three),
        vaddq_f64(two, vmulq_f64(three, abs_x)),
    );

    // id=3: −1 / |x|  (atan(|x|) = π/2 + atan(−1/|x|))
    let neg_one = vnegq_f64(one);
    let t3 = vdivq_f64(neg_one, abs_x);

    // ---------------------------------------------------------------------
    // Select the reduced argument by cascaded blending.
    //
    // Priority (highest first): id=3, id=2, id=1, id=0, id=-1
    // NEON vbslq: vbslq(mask, true_val, false_val)
    // ---------------------------------------------------------------------
    // Start from id=3, blend down toward id=-1
    let t = {
        let t = vbslq_f64(is_lt_thr3, t2, t3); // id=3 or id=2
        let t = vbslq_f64(is_lt_thr2, t1, t); // …or id=1
        let t = vbslq_f64(is_lt_thr1, t0, t); // …or id=0
        vbslq_f64(is_lt_thr0, abs_x, t) // …or no reduction
    };

    // ---------------------------------------------------------------------
    // Select the matching hi/lo offsets by cascaded blending
    //
    // id=-1 → hi=0, lo=0
    // ---------------------------------------------------------------------
    let zero_pd = vdupq_n_f64(0.0);
    let hi = {
        let hi = vbslq_f64(is_lt_thr3, hi2, hi3);
        let hi = vbslq_f64(is_lt_thr2, hi1, hi);
        let hi = vbslq_f64(is_lt_thr1, hi0, hi);
        vbslq_f64(is_lt_thr0, zero_pd, hi)
    };
    let lo = {
        let lo = vbslq_f64(is_lt_thr3, lo2, lo3);
        let lo = vbslq_f64(is_lt_thr2, lo1, lo);
        let lo = vbslq_f64(is_lt_thr1, lo0, lo);
        vbslq_f64(is_lt_thr0, zero_pd, lo)
    };

    // ---------------------------------------------------------------------
    // Polynomial evaluation (musl split odd/even Horner scheme)
    //
    // With z = t² and w = z²:
    //   s1 = z · (aT[0] + w·(aT[2] + w·(aT[4] + w·(aT[6] + w·(aT[8] + w·aT[10])))))
    //   s2 = w · (aT[1] + w·(aT[3] + w·(aT[5] + w·(aT[7] + w·aT[9]))))
    //
    // Splitting into odd/even improves instruction-level parallelism because
    // s1 and s2 can be computed simultaneously.
    //
    // NEON vfmaq: vfmaq(c, a, b) = a*b + c
    // ---------------------------------------------------------------------
    let z = vmulq_f64(t, t); // t²
    let w = vmulq_f64(z, z); // t⁴

    // s1: odd-indexed coefficients (aT[0], aT[2], aT[4], aT[6], aT[8], aT[10])
    let s1 = vmulq_f64(
        z,
        vfmaq_f64(
            at0,
            w,
            vfmaq_f64(at2, w, vfmaq_f64(at4, w, vfmaq_f64(at6, w, vfmaq_f64(at8, w, at10)))),
        ),
    );

    // s2: even-indexed coefficients (aT[1], aT[3], aT[5], aT[7], aT[9])
    let s2 = vmulq_f64(
        w,
        vfmaq_f64(at1, w, vfmaq_f64(at3, w, vfmaq_f64(at5, w, vfmaq_f64(at7, w, at9)))),
    );

    // ---------------------------------------------------------------------
    // Combine: result = hi + lo + t − t·(s1 + s2)
    //
    // Expanding the musl formula: atanhi + atanlo + t*(1 − (s1+s2))
    // Written as:
    //   correction = t * (s1 + s2)
    //   result     = hi + lo + t − correction
    // ---------------------------------------------------------------------
    let sum_s = vaddq_f64(s1, s2);
    let correction = vmulq_f64(t, sum_s); // t·(s1+s2)

    // hi + lo + t − correction  (order matters for cancellation)
    let result_abs = vaddq_f64(vaddq_f64(hi, lo), vsubq_f64(t, correction));

    // ---------------------------------------------------------------------
    // Restore sign: atan(−x) = −atan(x)
    //
    // XOR with original sign bits correctly propagates −0.0 → −0.0.
    // ---------------------------------------------------------------------
    vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(result_abs), sign_bits))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2 as FRAC_PI_2_F32;
    use std::f32::consts::FRAC_PI_4 as FRAC_PI_4_F32;
    use std::f64::consts::FRAC_PI_2 as FRAC_PI_2_F64;
    use std::f64::consts::FRAC_PI_4 as FRAC_PI_4_F64;

    const TOL_F32: f32 = 1e-5;
    const TOL_F64: f64 = 1e-14;

    // Helper to extract f32 lanes from float32x4_t
    unsafe fn extract_f32(v: float32x4_t) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        unsafe { vst1q_f32(out.as_mut_ptr(), v) };
        out
    }

    // Helper to extract f64 lanes from float64x2_t
    unsafe fn extract_f64(v: float64x2_t) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        unsafe { vst1q_f64(out.as_mut_ptr(), v) };
        out
    }

    // =========================================================================
    // f32 tests
    // =========================================================================

    #[test]
    fn atan_f32_zero_returns_zero() {
        unsafe {
            let input = vdupq_n_f32(0.0);
            let result = extract_f32(vatan_f32(input));
            assert!(result.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn atan_f32_negative_zero_returns_negative_zero() {
        unsafe {
            let input = vdupq_n_f32(-0.0);
            let result = extract_f32(vatan_f32(input));
            for &x in &result {
                assert!(x == 0.0 && x.is_sign_negative());
            }
        }
    }

    #[test]
    fn atan_f32_one_returns_pi_over_4() {
        unsafe {
            let input = vdupq_n_f32(1.0);
            let result = extract_f32(vatan_f32(input));
            for &x in &result {
                assert!(
                    (x - FRAC_PI_4_F32).abs() < TOL_F32,
                    "got {x}, expected {FRAC_PI_4_F32}"
                );
            }
        }
    }

    #[test]
    fn atan_f32_neg_one_returns_neg_pi_over_4() {
        unsafe {
            let input = vdupq_n_f32(-1.0);
            let result = extract_f32(vatan_f32(input));
            let expected = -FRAC_PI_4_F32;
            for &x in &result {
                assert!(
                    (x - expected).abs() < TOL_F32,
                    "got {x}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn atan_f32_infinity_returns_pi_over_2() {
        unsafe {
            let input = vdupq_n_f32(f32::INFINITY);
            let result = extract_f32(vatan_f32(input));
            for &x in &result {
                assert!(
                    (x - FRAC_PI_2_F32).abs() < TOL_F32,
                    "got {x}, expected {FRAC_PI_2_F32}"
                );
            }
        }
    }

    #[test]
    fn atan_f32_neg_infinity_returns_neg_pi_over_2() {
        unsafe {
            let input = vdupq_n_f32(f32::NEG_INFINITY);
            let result = extract_f32(vatan_f32(input));
            let expected = -FRAC_PI_2_F32;
            for &x in &result {
                assert!(
                    (x - expected).abs() < TOL_F32,
                    "got {x}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn atan_f32_nan_returns_nan() {
        unsafe {
            let input = vdupq_n_f32(f32::NAN);
            let result = extract_f32(vatan_f32(input));
            assert!(result.iter().all(|x| x.is_nan()));
        }
    }

    #[test]
    fn atan_f32_small_values() {
        unsafe {
            let vals = [0.1f32, 0.01, -0.1, -0.01];
            let input = vld1q_f32(vals.as_ptr());
            let result = extract_f32(vatan_f32(input));
            let expected: [f32; 4] = [
                0.1f32.atan(),
                0.01f32.atan(),
                (-0.1f32).atan(),
                (-0.01f32).atan(),
            ];
            for (i, (&r, &e)) in result.iter().zip(&expected).enumerate() {
                assert!((r - e).abs() < TOL_F32, "lane {i}: got {r}, expected {e}");
            }
        }
    }

    #[test]
    fn atan_f32_large_values() {
        unsafe {
            let vals = [10.0f32, 100.0, -10.0, -100.0];
            let input = vld1q_f32(vals.as_ptr());
            let result = extract_f32(vatan_f32(input));
            let expected: [f32; 4] = [
                10.0f32.atan(),
                100.0f32.atan(),
                (-10.0f32).atan(),
                (-100.0f32).atan(),
            ];
            for (i, (&r, &e)) in result.iter().zip(&expected).enumerate() {
                assert!((r - e).abs() < TOL_F32, "lane {i}: got {r}, expected {e}");
            }
        }
    }

    #[test]
    fn atan_f32_all_lanes_independent() {
        unsafe {
            let vals = [-2.0f32, -0.5, 0.5, 2.0];
            let input = vld1q_f32(vals.as_ptr());
            let result = extract_f32(vatan_f32(input));
            let expected: [f32; 4] = [
                (-2.0f32).atan(),
                (-0.5f32).atan(),
                0.5f32.atan(),
                2.0f32.atan(),
            ];
            for (i, (&r, &e)) in result.iter().zip(&expected).enumerate() {
                assert!((r - e).abs() < TOL_F32, "lane {i}: got {r}, expected {e}");
            }
        }
    }

    #[test]
    fn atan_f32_ulp_sweep() {
        unsafe {
            let mut max_ulp: u32 = 0;

            // Sweep across domain
            for i in -100_000i32..=100_000 {
                let x = (i as f32) * 0.0001; // [-10, 10]
                let expected = x.atan();
                if !expected.is_finite() {
                    continue;
                }

                let input = vdupq_n_f32(x);
                let result = extract_f32(vatan_f32(input))[0];

                let ulp = expected.to_bits().abs_diff(result.to_bits());
                max_ulp = max_ulp.max(ulp);
            }

            assert!(max_ulp <= 3, "Max ULP {} exceeds 3", max_ulp);
        }
    }

    // =========================================================================
    // f64 tests
    // =========================================================================

    #[test]
    fn atan_f64_zero_returns_zero() {
        unsafe {
            let result = extract_f64(vatan_f64(vdupq_n_f64(0.0)));
            assert!(result.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn atan_f64_negative_zero_returns_negative_zero() {
        unsafe {
            let result = extract_f64(vatan_f64(vdupq_n_f64(-0.0)));
            for &x in &result {
                assert!(x == 0.0 && x.is_sign_negative());
            }
        }
    }

    #[test]
    fn atan_f64_one_returns_pi_over_4() {
        unsafe {
            let result = extract_f64(vatan_f64(vdupq_n_f64(1.0)));
            let expected = FRAC_PI_4_F64;
            for &x in &result {
                assert!(
                    (x - expected).abs() < TOL_F64,
                    "got {x}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn atan_f64_neg_one_returns_neg_pi_over_4() {
        unsafe {
            let result = extract_f64(vatan_f64(vdupq_n_f64(-1.0)));
            let expected = -FRAC_PI_4_F64;
            for &x in &result {
                assert!(
                    (x - expected).abs() < TOL_F64,
                    "got {x}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn atan_f64_infinity_returns_pi_over_2() {
        unsafe {
            let result = extract_f64(vatan_f64(vdupq_n_f64(f64::INFINITY)));
            let expected = FRAC_PI_2_F64;
            for &x in &result {
                assert!(
                    (x - expected).abs() < TOL_F64,
                    "got {x}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn atan_f64_neg_infinity_returns_neg_pi_over_2() {
        unsafe {
            let result = extract_f64(vatan_f64(vdupq_n_f64(f64::NEG_INFINITY)));
            let expected = -FRAC_PI_2_F64;
            for &x in &result {
                assert!(
                    (x - expected).abs() < TOL_F64,
                    "got {x}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn atan_f64_nan_returns_nan() {
        unsafe {
            let result = extract_f64(vatan_f64(vdupq_n_f64(f64::NAN)));
            assert!(result.iter().all(|x| x.is_nan()));
        }
    }

    #[test]
    fn atan_f64_all_lanes_independent() {
        unsafe {
            let vals = [-2.0f64, 2.0];
            let input = vld1q_f64(vals.as_ptr());
            let result = extract_f64(vatan_f64(input));
            let expected = [(-2.0f64).atan(), 2.0f64.atan()];
            for (i, (&r, &e)) in result.iter().zip(&expected).enumerate() {
                assert!((r - e).abs() < TOL_F64, "lane {i}: got {r}, expected {e}");
            }
        }
    }

    #[test]
    fn atan_f64_ulp_sweep() {
        unsafe {
            let mut max_ulp: u64 = 0;

            // Sweep across a wide domain including all 5 reduction ranges
            for i in 0..2000 {
                let t = (i as f64 / 1999.0) * 20.0 - 10.0; // [-10.0, 10.0]
                let result = extract_f64(vatan_f64(vdupq_n_f64(t)));
                let expected = t.atan();

                for &r in &result {
                    let ulp = if expected == r {
                        0
                    } else {
                        expected.to_bits().abs_diff(r.to_bits())
                    };
                    max_ulp = max_ulp.max(ulp);
                }
            }

            assert!(max_ulp <= 1, "max ULP error {max_ulp} exceeds 1");
        }
    }
}
