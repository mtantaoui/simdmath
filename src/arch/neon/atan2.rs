//! NEON SIMD implementation of `atan2(y, x)` for `f32` and `f64` vectors.
//!
//! # Algorithm
//!
//! Computes the two-argument arctangent, returning the angle θ of the point
//! `(x, y)` in radians, measured counter-clockwise from the positive x-axis.
//! The result is in the range `(-π, π]`.
//!
//! The implementation follows **musl libc's `atan2f.c` / `atan2.c`** (fdlibm descent):
//!
//! 1. **Extract signs** — encode `m = 2·sign(x) + sign(y)` to identify the quadrant
//! 2. **Handle special cases** — NaN, zeros, infinities
//! 3. **Compute `atan(|y/x|)`** — using the existing `vatan_f32` / `vatan_f64`
//! 4. **Apply quadrant correction** — adjust result based on `m`
//!
//! ## Quadrant encoding
//!
//! | m | sign(x) | sign(y) | Quadrant | Result formula                |
//! |---|---------|---------|----------|-------------------------------|
//! | 0 |    +    |    +    |   I      | `atan(y/x)`                   |
//! | 1 |    +    |    -    |   IV     | `-atan(\|y/x\|)`              |
//! | 2 |    -    |    +    |   II     | `π - atan(\|y/x\|)`           |
//! | 3 |    -    |    -    |   III    | `atan(\|y/x\|) - π`           |
//!
//! ## Special values
//!
//! | y       | x       | atan2(y, x)            |
//! |---------|---------|------------------------|
//! | `±0`    | `+0`    | `±0`                   |
//! | `±0`    | `-0`    | `±π`                   |
//! | `±0`    | `x > 0` | `±0`                   |
//! | `±0`    | `x < 0` | `±π`                   |
//! | `y > 0` | `±0`    | `+π/2`                 |
//! | `y < 0` | `±0`    | `-π/2`                 |
//! | `±∞`    | `+∞`    | `±π/4`                 |
//! | `±∞`    | `-∞`    | `±3π/4`                |
//! | `±∞`    | finite  | `±π/2`                 |
//! | finite  | `+∞`    | `±0`                   |
//! | finite  | `-∞`    | `±π`                   |
//! | NaN     | any     | NaN                    |
//! | any     | NaN     | NaN                    |
//!
//! ## Precision
//!
//! - **f32**: ≤ 3 ULP error across the entire domain
//! - **f64**: ≤ 2 ULP error across the entire domain
//!
//! ## Blending strategy
//!
//! All branches are computed unconditionally; results are merged with
//! `vbslq_f32` / `vbslq_f64` in priority order (highest to lowest):
//! NaN → x==1 → y==0 → x==0 → x==∞ → huge ratio → general case
//!
//! ## NEON notes
//!
//! - `vbslq` argument order: `vbslq(mask, true_val, false_val)` (differs from x86!)
//! - No `vmvnq_u64` — emulate with `veorq_u64(x, all_ones)`
//! - FMA accumulator first: `vfmaq(c, a, b)` = a*b + c

use std::arch::aarch64::*;

use crate::arch::consts::atan2::{
    FRAC_3_PI_4_32, FRAC_3_PI_4_64, FRAC_PI_2_32, FRAC_PI_2_64, FRAC_PI_4_32, FRAC_PI_4_64,
    HUGE_RATIO_THRESHOLD_32, HUGE_RATIO_THRESHOLD_64, PI_HI_32, PI_HI_64, PI_LO_32, PI_LO_64,
};
use crate::arch::neon::abs::{vabsq_f32_wrapper, vabsq_f64_wrapper};
use crate::arch::neon::atan::{vatan_f32, vatan_f64};

// ---------------------------------------------------------------------------
// f32 Implementation (4 lanes)
// ---------------------------------------------------------------------------

/// Computes `atan2(y, x)` for each lane of two NEON `float32x4_t` registers (4 × f32).
///
/// # Precision
///
/// **≤ 3 ULP** error across the entire domain.
///
/// # Description
///
/// All 4 lanes are processed simultaneously without branches. Special cases
/// (NaN, zeros, infinities) are handled via branchless blending with `vbslq`.
///
/// # Safety
///
/// `y` and `x` must be valid `float32x4_t` registers.
///
/// # Example
///
/// ```ignore
/// let y = vdupq_n_f32(1.0);
/// let x = vdupq_n_f32(1.0);
/// let result = vatan2_f32(y, x);
/// // result ≈ [0.7854; 4] (π/4)
/// ```
#[inline]
pub(crate) unsafe fn vatan2_f32(y: float32x4_t, x: float32x4_t) -> float32x4_t {
    unsafe {
        // -------------------------------------------------------------------------
        // Broadcast constants
        // -------------------------------------------------------------------------
        let zero = vdupq_n_f32(0.0);
        let neg_zero = vdupq_n_f32(-0.0);
        let pi_hi = vdupq_n_f32(PI_HI_32);
        let pi_lo = vdupq_n_f32(PI_LO_32);
        let pi_over_2 = vdupq_n_f32(FRAC_PI_2_32);
        let pi_over_4 = vdupq_n_f32(FRAC_PI_4_32);
        let three_pi_over_4 = vdupq_n_f32(FRAC_3_PI_4_32);

        // Integer constants for bit manipulation
        let abs_mask = vdupq_n_u32(0x7FFF_FFFF);
        let one_bits = vdupq_n_u32(0x3F80_0000); // 1.0f32
        let inf_bits = vdupq_n_u32(0x7F80_0000); // +∞
        let huge_threshold = vdupq_n_s32(HUGE_RATIO_THRESHOLD_32);

        // -------------------------------------------------------------------------
        // Extract bit representations and absolute values
        // -------------------------------------------------------------------------
        let x_bits = vreinterpretq_u32_f32(x);
        let y_bits = vreinterpretq_u32_f32(y);

        let ix = vandq_u32(x_bits, abs_mask); // |x| as integer bits
        let iy = vandq_u32(y_bits, abs_mask); // |y| as integer bits

        let abs_x = vabsq_f32_wrapper(x);
        let abs_y = vabsq_f32_wrapper(y);

        // -------------------------------------------------------------------------
        // Compute quadrant index: m = 2·sign(x) + sign(y)
        //
        // m ∈ {0, 1, 2, 3} encodes which quadrant (x, y) lies in:
        //   m=0: x≥0, y≥0 (Q1)    m=1: x≥0, y<0 (Q4)
        //   m=2: x<0, y≥0 (Q2)    m=3: x<0, y<0 (Q3)
        // -------------------------------------------------------------------------
        let sign_x = vshrq_n_u32(x_bits, 31); // 0 or 1
        let sign_y = vshrq_n_u32(y_bits, 31); // 0 or 1
        let m = vorrq_u32(vshlq_n_u32(sign_x, 1), sign_y);

        // Precompute masks for each m value
        let m_eq_0 = vceqq_u32(m, vdupq_n_u32(0));
        let m_eq_1 = vceqq_u32(m, vdupq_n_u32(1));
        let m_eq_2 = vceqq_u32(m, vdupq_n_u32(2));
        let m_01 = vorrq_u32(m_eq_0, m_eq_1); // m ∈ {0, 1} → x ≥ 0
        let m_02 = vorrq_u32(m_eq_0, m_eq_2); // m ∈ {0, 2} → y ≥ 0

        // -------------------------------------------------------------------------
        // Condition masks for special cases
        // -------------------------------------------------------------------------
        let is_x_one = vceqq_u32(x_bits, one_bits); // x == 1.0 exactly
        let is_y_zero = vceqq_u32(iy, vdupq_n_u32(0)); // y == ±0
        let is_x_zero = vceqq_u32(ix, vdupq_n_u32(0)); // x == ±0
        let is_x_inf = vceqq_u32(ix, inf_bits); // |x| == ∞
        let is_y_inf = vceqq_u32(iy, inf_bits); // |y| == ∞

        // NaN check: x != x or y != y
        let is_x_nan = vmvnq_u32(vceqq_f32(x, x));
        let is_y_nan = vmvnq_u32(vceqq_f32(y, y));

        // Check if |y/x| > 2^26 (y dominates) — atan2 ≈ ±π/2
        let iy_s = vreinterpretq_s32_u32(iy);
        let ix_s = vreinterpretq_s32_u32(ix);
        let iy_minus_ix = vsubq_s32(iy_s, ix_s);
        let is_huge_ratio = vcgtq_s32(iy_minus_ix, huge_threshold); // returns uint32x4_t
        let huge_or_y_inf = vorrq_u32(is_huge_ratio, is_y_inf);

        // Check if |y/x| < 2^-26 AND x < 0 — atan2 ≈ ±π (y negligible, negative x)
        let iy_plus_threshold = vaddq_s32(iy_s, huge_threshold);
        let is_tiny_ratio = vcgtq_s32(ix_s, iy_plus_threshold); // returns uint32x4_t
        let m_ge_2 = vcgtq_u32(m, vdupq_n_u32(1)); // x < 0
        let tiny_and_x_neg = vandq_u32(is_tiny_ratio, m_ge_2);

        // -------------------------------------------------------------------------
        // Case: NaN — if either input is NaN, return NaN (as x + y)
        // -------------------------------------------------------------------------
        let result_nan = vaddq_f32(x, y);

        // -------------------------------------------------------------------------
        // Case: x == 1.0 — return atan(y) directly
        // -------------------------------------------------------------------------
        let result_x_one = vatan_f32(y);

        // -------------------------------------------------------------------------
        // Case: y == ±0
        //
        //   m=0 (x≥0, y=+0): +0     m=1 (x≥0, y=-0): -0
        //   m=2 (x<0, y=+0): +π     m=3 (x<0, y=-0): -π
        // -------------------------------------------------------------------------
        // NEON vbslq: vbslq(mask, true_val, false_val)
        let result_y_zero = vbslq_f32(
            m_01,
            vbslq_f32(m_eq_0, y, neg_zero),                  // m=0: y, m=1: -0
            vbslq_f32(m_eq_2, pi_hi, vsubq_f32(zero, pi_hi)), // m=2: +π, m=3: -π
        );

        // -------------------------------------------------------------------------
        // Case: x == ±0 (but y ≠ 0)
        //
        //   y > 0: +π/2     y < 0: -π/2
        // -------------------------------------------------------------------------
        let result_x_zero = vbslq_f32(m_02, pi_over_2, vsubq_f32(zero, pi_over_2));

        // -------------------------------------------------------------------------
        // Case: x == ±∞
        //
        // If y is also ±∞:
        //   m=0: +π/4    m=1: -π/4    m=2: +3π/4    m=3: -3π/4
        //
        // If y is finite:
        //   m=0: +0      m=1: -0      m=2: +π       m=3: -π
        // -------------------------------------------------------------------------
        let result_both_inf = vbslq_f32(
            m_01,
            vbslq_f32(m_eq_0, pi_over_4, vsubq_f32(zero, pi_over_4)),
            vbslq_f32(m_eq_2, three_pi_over_4, vsubq_f32(zero, three_pi_over_4)),
        );

        let result_x_inf_y_finite = vbslq_f32(
            m_01,
            vbslq_f32(m_eq_0, zero, neg_zero),
            vbslq_f32(m_eq_2, pi_hi, vsubq_f32(zero, pi_hi)),
        );

        let result_x_inf = vbslq_f32(is_y_inf, result_both_inf, result_x_inf_y_finite);

        // -------------------------------------------------------------------------
        // Case: |y/x| > 2^26 or y == ±∞ — return ±π/2
        // -------------------------------------------------------------------------
        let result_huge = vbslq_f32(m_02, pi_over_2, vsubq_f32(zero, pi_over_2));

        // -------------------------------------------------------------------------
        // General case: compute z = atan(|y/x|), then apply quadrant correction
        //
        // If |y/x| < 2^-26 AND x < 0, use z = 0 (y is negligible)
        //
        // Quadrant corrections:
        //   m=0: +z                    (Q1: result is positive, < π/2)
        //   m=1: -z                    (Q4: result is negative, > -π/2)
        //   m=2: π - (z - π_lo) ≈ π-z (Q2: result is positive, > π/2)
        //   m=3: (z - π_lo) - π ≈ z-π (Q3: result is negative, < -π/2)
        //
        // The π_lo term is a two-sum correction for full precision.
        // -------------------------------------------------------------------------
        let ratio = vdivq_f32(abs_y, abs_x);
        let z_computed = vatan_f32(ratio);
        let z = vbslq_f32(tiny_and_x_neg, zero, z_computed);

        // Quadrant corrections
        let result_m0 = z; // +z
        let result_m1 = vsubq_f32(zero, z); // -z
        let result_m2 = vsubq_f32(pi_hi, vsubq_f32(z, pi_lo)); // π - (z - π_lo)
        let result_m3 = vsubq_f32(vsubq_f32(z, pi_lo), pi_hi); // (z - π_lo) - π

        let result_general = vbslq_f32(
            m_01,
            vbslq_f32(m_eq_0, result_m0, result_m1),
            vbslq_f32(m_eq_2, result_m2, result_m3),
        );

        // -------------------------------------------------------------------------
        // Merge all cases in priority order (lowest to highest)
        // -------------------------------------------------------------------------
        let mut result = result_general;

        result = vbslq_f32(huge_or_y_inf, result_huge, result);
        result = vbslq_f32(is_x_inf, result_x_inf, result);
        result = vbslq_f32(is_x_zero, result_x_zero, result);
        result = vbslq_f32(is_y_zero, result_y_zero, result);
        result = vbslq_f32(is_x_one, result_x_one, result);
        result = vbslq_f32(vorrq_u32(is_x_nan, is_y_nan), result_nan, result);

        result
    }
}

// ===========================================================================
// f64 Implementation (2 lanes)
// ===========================================================================

/// Computes `atan2(y, x)` for each lane of two NEON `float64x2_t` registers (2 × f64).
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Description
///
/// All 2 lanes are processed simultaneously without branches. Special cases
/// (NaN, zeros, infinities) are handled via branchless blending with `vbslq`.
///
/// # Safety
///
/// `y` and `x` must be valid `float64x2_t` registers.
///
/// # Example
///
/// ```ignore
/// let y = vdupq_n_f64(1.0);
/// let x = vdupq_n_f64(1.0);
/// let result = vatan2_f64(y, x);
/// // result ≈ [0.7854; 2] (π/4)
/// ```
#[inline]
pub(crate) unsafe fn vatan2_f64(y: float64x2_t, x: float64x2_t) -> float64x2_t {
    unsafe {
        // -------------------------------------------------------------------------
        // Broadcast constants
        // -------------------------------------------------------------------------
        let zero = vdupq_n_f64(0.0);
        let neg_zero = vdupq_n_f64(-0.0);
        let pi_hi = vdupq_n_f64(PI_HI_64);
        let pi_lo = vdupq_n_f64(PI_LO_64);
        let pi_over_2 = vdupq_n_f64(FRAC_PI_2_64);
        let pi_over_4 = vdupq_n_f64(FRAC_PI_4_64);
        let three_pi_over_4 = vdupq_n_f64(FRAC_3_PI_4_64);

        // Integer constants for bit manipulation (64-bit)
        let abs_mask = vdupq_n_u64(0x7FFF_FFFF_FFFF_FFFF);
        let one_bits = vdupq_n_u64(0x3FF0_0000_0000_0000); // 1.0f64
        let inf_bits = vdupq_n_u64(0x7FF0_0000_0000_0000); // +∞
        let huge_threshold = vdupq_n_s64(HUGE_RATIO_THRESHOLD_64);

        // All-ones mask for NOT emulation (no vmvnq_u64 in NEON)
        let all_ones_u64 = vreinterpretq_u64_s64(vdupq_n_s64(-1));

        // -------------------------------------------------------------------------
        // Extract bit representations and absolute values
        // -------------------------------------------------------------------------
        let x_bits = vreinterpretq_u64_f64(x);
        let y_bits = vreinterpretq_u64_f64(y);

        let ix = vandq_u64(x_bits, abs_mask); // |x| as integer bits
        let iy = vandq_u64(y_bits, abs_mask); // |y| as integer bits

        let abs_x = vabsq_f64_wrapper(x);
        let abs_y = vabsq_f64_wrapper(y);

        // -------------------------------------------------------------------------
        // Compute quadrant index: m = 2·sign(x) + sign(y)
        //
        // For 64-bit lanes, we shift by 63 to get the sign bit.
        // -------------------------------------------------------------------------
        let sign_x = vshrq_n_u64(x_bits, 63);
        let sign_y = vshrq_n_u64(y_bits, 63);
        let m = vorrq_u64(vshlq_n_u64(sign_x, 1), sign_y);

        // Precompute masks for each m value
        let m_eq_0 = vceqq_u64(m, vdupq_n_u64(0));
        let m_eq_1 = vceqq_u64(m, vdupq_n_u64(1));
        let m_eq_2 = vceqq_u64(m, vdupq_n_u64(2));
        let m_01 = vorrq_u64(m_eq_0, m_eq_1);
        let m_02 = vorrq_u64(m_eq_0, m_eq_2);

        // -------------------------------------------------------------------------
        // Condition masks for special cases
        // -------------------------------------------------------------------------
        let is_x_one = vceqq_u64(x_bits, one_bits);
        let is_y_zero = vceqq_u64(iy, vdupq_n_u64(0));
        let is_x_zero = vceqq_u64(ix, vdupq_n_u64(0));
        let is_x_inf = vceqq_u64(ix, inf_bits);
        let is_y_inf = vceqq_u64(iy, inf_bits);

        // NaN check: x != x or y != y (use XOR with all_ones to emulate NOT)
        let is_x_nan = veorq_u64(vceqq_f64(x, x), all_ones_u64);
        let is_y_nan = veorq_u64(vceqq_f64(y, y), all_ones_u64);

        // Check if |y/x| > 2^60 (y dominates)
        let iy_s = vreinterpretq_s64_u64(iy);
        let ix_s = vreinterpretq_s64_u64(ix);
        let iy_minus_ix = vsubq_s64(iy_s, ix_s);
        let is_huge_ratio = vcgtq_s64(iy_minus_ix, huge_threshold); // returns uint64x2_t
        let huge_or_y_inf = vorrq_u64(is_huge_ratio, is_y_inf);

        // Check if |y/x| < 2^-60 AND x < 0
        let iy_plus_threshold = vaddq_s64(iy_s, huge_threshold);
        let is_tiny_ratio = vcgtq_s64(ix_s, iy_plus_threshold); // returns uint64x2_t
        let m_ge_2 = vcgtq_u64(m, vdupq_n_u64(1));
        let tiny_and_x_neg = vandq_u64(is_tiny_ratio, m_ge_2);

        // -------------------------------------------------------------------------
        // Case: NaN — if either input is NaN, return NaN (as x + y)
        // -------------------------------------------------------------------------
        let result_nan = vaddq_f64(x, y);

        // -------------------------------------------------------------------------
        // Case: x == 1.0 — return atan(y) directly
        // -------------------------------------------------------------------------
        let result_x_one = vatan_f64(y);

        // -------------------------------------------------------------------------
        // Case: y == ±0
        // -------------------------------------------------------------------------
        let result_y_zero = vbslq_f64(
            m_01,
            vbslq_f64(m_eq_0, y, neg_zero),
            vbslq_f64(m_eq_2, pi_hi, vsubq_f64(zero, pi_hi)),
        );

        // -------------------------------------------------------------------------
        // Case: x == ±0 (but y ≠ 0)
        // -------------------------------------------------------------------------
        let result_x_zero = vbslq_f64(m_02, pi_over_2, vsubq_f64(zero, pi_over_2));

        // -------------------------------------------------------------------------
        // Case: x == ±∞
        // -------------------------------------------------------------------------
        let result_both_inf = vbslq_f64(
            m_01,
            vbslq_f64(m_eq_0, pi_over_4, vsubq_f64(zero, pi_over_4)),
            vbslq_f64(m_eq_2, three_pi_over_4, vsubq_f64(zero, three_pi_over_4)),
        );

        let result_x_inf_y_finite = vbslq_f64(
            m_01,
            vbslq_f64(m_eq_0, zero, neg_zero),
            vbslq_f64(m_eq_2, pi_hi, vsubq_f64(zero, pi_hi)),
        );

        let result_x_inf = vbslq_f64(is_y_inf, result_both_inf, result_x_inf_y_finite);

        // -------------------------------------------------------------------------
        // Case: |y/x| > 2^60 or y == ±∞ — return ±π/2
        // -------------------------------------------------------------------------
        let result_huge = vbslq_f64(m_02, pi_over_2, vsubq_f64(zero, pi_over_2));

        // -------------------------------------------------------------------------
        // General case: compute z = atan(|y/x|), then apply quadrant correction
        // -------------------------------------------------------------------------
        let ratio = vdivq_f64(abs_y, abs_x);
        let z_computed = vatan_f64(ratio);
        let z = vbslq_f64(tiny_and_x_neg, zero, z_computed);

        // Quadrant corrections
        let result_m0 = z;
        let result_m1 = vsubq_f64(zero, z);
        let result_m2 = vsubq_f64(pi_hi, vsubq_f64(z, pi_lo));
        let result_m3 = vsubq_f64(vsubq_f64(z, pi_lo), pi_hi);

        let result_general = vbslq_f64(
            m_01,
            vbslq_f64(m_eq_0, result_m0, result_m1),
            vbslq_f64(m_eq_2, result_m2, result_m3),
        );

        // -------------------------------------------------------------------------
        // Merge all cases in priority order (lowest to highest)
        // -------------------------------------------------------------------------
        let mut result = result_general;

        result = vbslq_f64(huge_or_y_inf, result_huge, result);
        result = vbslq_f64(is_x_inf, result_x_inf, result);
        result = vbslq_f64(is_x_zero, result_x_zero, result);
        result = vbslq_f64(is_y_zero, result_y_zero, result);
        result = vbslq_f64(is_x_one, result_x_one, result);
        result = vbslq_f64(vorrq_u64(is_x_nan, is_y_nan), result_nan, result);

        result
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    /// Tolerance: ~2 ULPs at the scale of π.
    const TOL: f32 = 5e-7;

    /// Load 4 copies of `y` and `x`, call `vatan2_f32`, and return lane 0.
    unsafe fn atan2_scalar(y: f32, x: f32) -> f32 {
        unsafe {
            let vy = vdupq_n_f32(y);
            let vx = vdupq_n_f32(x);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vatan2_f32(vy, vx));
            out[0]
        }
    }

    // ---- Special / boundary values: zeros ------------------------------------

    #[test]
    fn atan2_pos_zero_pos_zero() {
        unsafe {
            assert_eq!(atan2_scalar(0.0, 0.0), 0.0);
        }
    }

    #[test]
    fn atan2_neg_zero_pos_zero() {
        unsafe {
            let result = atan2_scalar(-0.0, 0.0);
            assert!(result == 0.0 && result.is_sign_negative(), "expected -0.0, got {result}");
        }
    }

    #[test]
    fn atan2_pos_zero_neg_zero() {
        unsafe {
            let result = atan2_scalar(0.0, -0.0);
            assert!(
                (result - PI).abs() < TOL,
                "atan2(+0, -0) = {result}, expected π"
            );
        }
    }

    #[test]
    fn atan2_neg_zero_neg_zero() {
        unsafe {
            let result = atan2_scalar(-0.0, -0.0);
            assert!(
                (result - (-PI)).abs() < TOL,
                "atan2(-0, -0) = {result}, expected -π"
            );
        }
    }

    #[test]
    fn atan2_zero_positive_x() {
        unsafe {
            assert_eq!(atan2_scalar(0.0, 1.0), 0.0);
            let neg = atan2_scalar(-0.0, 1.0);
            assert!(neg == 0.0 && neg.is_sign_negative());
        }
    }

    #[test]
    fn atan2_zero_negative_x() {
        unsafe {
            let pos = atan2_scalar(0.0, -1.0);
            assert!((pos - PI).abs() < TOL, "atan2(+0, -1) = {pos}, expected π");
            let neg = atan2_scalar(-0.0, -1.0);
            assert!((neg - (-PI)).abs() < TOL, "atan2(-0, -1) = {neg}, expected -π");
        }
    }

    // ---- Special values: x == 0 (vertical axis) ------------------------------

    #[test]
    fn atan2_positive_y_zero_x() {
        unsafe {
            let result = atan2_scalar(1.0, 0.0);
            assert!(
                (result - FRAC_PI_2).abs() < TOL,
                "atan2(1, 0) = {result}, expected π/2"
            );
        }
    }

    #[test]
    fn atan2_negative_y_zero_x() {
        unsafe {
            let result = atan2_scalar(-1.0, 0.0);
            assert!(
                (result - (-FRAC_PI_2)).abs() < TOL,
                "atan2(-1, 0) = {result}, expected -π/2"
            );
        }
    }

    // ---- Special values: infinities ------------------------------------------

    #[test]
    fn atan2_inf_inf() {
        unsafe {
            let r1 = atan2_scalar(f32::INFINITY, f32::INFINITY);
            assert!((r1 - FRAC_PI_4).abs() < TOL, "atan2(+∞, +∞) = {r1}");

            let r2 = atan2_scalar(f32::NEG_INFINITY, f32::INFINITY);
            assert!((r2 - (-FRAC_PI_4)).abs() < TOL, "atan2(-∞, +∞) = {r2}");

            let r3 = atan2_scalar(f32::INFINITY, f32::NEG_INFINITY);
            assert!((r3 - 3.0 * FRAC_PI_4).abs() < TOL, "atan2(+∞, -∞) = {r3}");

            let r4 = atan2_scalar(f32::NEG_INFINITY, f32::NEG_INFINITY);
            assert!((r4 - (-3.0 * FRAC_PI_4)).abs() < TOL, "atan2(-∞, -∞) = {r4}");
        }
    }

    #[test]
    fn atan2_inf_finite() {
        unsafe {
            let r1 = atan2_scalar(f32::INFINITY, 1.0);
            assert!((r1 - FRAC_PI_2).abs() < TOL, "atan2(+∞, 1) = {r1}");

            let r2 = atan2_scalar(f32::NEG_INFINITY, 1.0);
            assert!((r2 - (-FRAC_PI_2)).abs() < TOL, "atan2(-∞, 1) = {r2}");
        }
    }

    #[test]
    fn atan2_finite_inf() {
        unsafe {
            assert_eq!(atan2_scalar(1.0, f32::INFINITY), 0.0);
            let r2 = atan2_scalar(-1.0, f32::INFINITY);
            assert!(r2 == 0.0 && r2.is_sign_negative());

            let r3 = atan2_scalar(1.0, f32::NEG_INFINITY);
            assert!((r3 - PI).abs() < TOL, "atan2(1, -∞) = {r3}");

            let r4 = atan2_scalar(-1.0, f32::NEG_INFINITY);
            assert!((r4 - (-PI)).abs() < TOL, "atan2(-1, -∞) = {r4}");
        }
    }

    // ---- Special values: NaN -------------------------------------------------

    #[test]
    fn atan2_nan_any() {
        unsafe {
            assert!(atan2_scalar(f32::NAN, 1.0).is_nan());
            assert!(atan2_scalar(1.0, f32::NAN).is_nan());
            assert!(atan2_scalar(f32::NAN, f32::NAN).is_nan());
        }
    }

    // ---- Quadrant tests ------------------------------------------------------

    #[test]
    fn atan2_quadrant_1() {
        unsafe {
            let result = atan2_scalar(1.0, 1.0);
            assert!(
                (result - FRAC_PI_4).abs() < TOL,
                "atan2(1, 1) = {result}, expected π/4"
            );
        }
    }

    #[test]
    fn atan2_quadrant_2() {
        unsafe {
            let result = atan2_scalar(1.0, -1.0);
            assert!(
                (result - 3.0 * FRAC_PI_4).abs() < TOL,
                "atan2(1, -1) = {result}, expected 3π/4"
            );
        }
    }

    #[test]
    fn atan2_quadrant_3() {
        unsafe {
            let result = atan2_scalar(-1.0, -1.0);
            assert!(
                (result - (-3.0 * FRAC_PI_4)).abs() < TOL,
                "atan2(-1, -1) = {result}, expected -3π/4"
            );
        }
    }

    #[test]
    fn atan2_quadrant_4() {
        unsafe {
            let result = atan2_scalar(-1.0, 1.0);
            assert!(
                (result - (-FRAC_PI_4)).abs() < TOL,
                "atan2(-1, 1) = {result}, expected -π/4"
            );
        }
    }

    // ---- All 4 lanes processed correctly -------------------------------------

    #[test]
    fn atan2_processes_all_4_lanes_independently() {
        let y_vals = [1.0f32, -1.0, 1.0, -1.0];
        let x_vals = [1.0f32, 1.0, -1.0, -1.0];
        unsafe {
            let vy = vld1q_f32(y_vals.as_ptr());
            let vx = vld1q_f32(x_vals.as_ptr());
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vatan2_f32(vy, vx));

            for i in 0..4 {
                let expected = y_vals[i].atan2(x_vals[i]);
                assert!(
                    (out[i] - expected).abs() < TOL,
                    "lane {i}: atan2({}, {}) = {}, expected {expected}",
                    y_vals[i],
                    x_vals[i],
                    out[i]
                );
            }
        }
    }

    // ---- ULP accuracy sweep --------------------------------------------------

    #[test]
    fn max_ulp_error_is_at_most_3() {
        let mut max_ulp: u32 = 0;
        let mut worst_y: f32 = 0.0;
        let mut worst_x: f32 = 0.0;

        // Sweep angles and radii
        for i in 0..2000 {
            let theta = (i as f32 / 1999.0) * 2.0 * PI - PI;
            for j in 0..100 {
                let r = 0.001 + (j as f32) * 0.5;
                let x = r * theta.cos();
                let y = r * theta.sin();

                let expected = y.atan2(x);
                if !expected.is_finite() {
                    continue;
                }

                let result = unsafe { atan2_scalar(y, x) };
                let ulp = expected.to_bits().abs_diff(result.to_bits());
                if ulp > max_ulp {
                    max_ulp = ulp;
                    worst_y = y;
                    worst_x = x;
                }
            }
        }

        assert!(
            max_ulp <= 3,
            "max ULP {max_ulp} at atan2({worst_y}, {worst_x}) — expected ≤ 3"
        );
    }

    // ===========================================================================
    // f64 tests
    // ===========================================================================

    /// Tolerance: ~2 ULPs at the scale of π.
    const TOL_64: f64 = 1e-15;

    /// Load 2 copies of `y` and `x`, call `vatan2_f64`, and return lane 0.
    unsafe fn atan2_scalar_64(y: f64, x: f64) -> f64 {
        unsafe {
            let vy = vdupq_n_f64(y);
            let vx = vdupq_n_f64(x);
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), vatan2_f64(vy, vx));
            out[0]
        }
    }

    // ---- f64 Special / boundary values: zeros --------------------------------

    #[test]
    fn atan2_f64_pos_zero_pos_zero() {
        unsafe {
            assert_eq!(atan2_scalar_64(0.0, 0.0), 0.0);
        }
    }

    #[test]
    fn atan2_f64_neg_zero_pos_zero() {
        unsafe {
            let result = atan2_scalar_64(-0.0, 0.0);
            assert!(result == 0.0 && result.is_sign_negative(), "expected -0.0, got {result}");
        }
    }

    #[test]
    fn atan2_f64_pos_zero_neg_zero() {
        unsafe {
            let result = atan2_scalar_64(0.0, -0.0);
            assert!(
                (result - std::f64::consts::PI).abs() < TOL_64,
                "atan2(+0, -0) = {result}, expected π"
            );
        }
    }

    #[test]
    fn atan2_f64_neg_zero_neg_zero() {
        unsafe {
            let result = atan2_scalar_64(-0.0, -0.0);
            assert!(
                (result - (-std::f64::consts::PI)).abs() < TOL_64,
                "atan2(-0, -0) = {result}, expected -π"
            );
        }
    }

    // ---- f64 Special values: infinities --------------------------------------

    #[test]
    fn atan2_f64_inf_inf() {
        unsafe {
            let r1 = atan2_scalar_64(f64::INFINITY, f64::INFINITY);
            assert!((r1 - std::f64::consts::FRAC_PI_4).abs() < TOL_64);

            let r2 = atan2_scalar_64(f64::NEG_INFINITY, f64::INFINITY);
            assert!((r2 - (-std::f64::consts::FRAC_PI_4)).abs() < TOL_64);

            let r3 = atan2_scalar_64(f64::INFINITY, f64::NEG_INFINITY);
            assert!((r3 - 3.0 * std::f64::consts::FRAC_PI_4).abs() < TOL_64);

            let r4 = atan2_scalar_64(f64::NEG_INFINITY, f64::NEG_INFINITY);
            assert!((r4 - (-3.0 * std::f64::consts::FRAC_PI_4)).abs() < TOL_64);
        }
    }

    #[test]
    fn atan2_f64_inf_finite() {
        unsafe {
            let r1 = atan2_scalar_64(f64::INFINITY, 1.0);
            assert!((r1 - std::f64::consts::FRAC_PI_2).abs() < TOL_64);

            let r2 = atan2_scalar_64(f64::NEG_INFINITY, 1.0);
            assert!((r2 - (-std::f64::consts::FRAC_PI_2)).abs() < TOL_64);
        }
    }

    #[test]
    fn atan2_f64_finite_inf() {
        unsafe {
            assert_eq!(atan2_scalar_64(1.0, f64::INFINITY), 0.0);
            let r2 = atan2_scalar_64(-1.0, f64::INFINITY);
            assert!(r2 == 0.0 && r2.is_sign_negative());

            let r3 = atan2_scalar_64(1.0, f64::NEG_INFINITY);
            assert!((r3 - std::f64::consts::PI).abs() < TOL_64);

            let r4 = atan2_scalar_64(-1.0, f64::NEG_INFINITY);
            assert!((r4 - (-std::f64::consts::PI)).abs() < TOL_64);
        }
    }

    // ---- f64 Special values: NaN ---------------------------------------------

    #[test]
    fn atan2_f64_nan_any() {
        unsafe {
            assert!(atan2_scalar_64(f64::NAN, 1.0).is_nan());
            assert!(atan2_scalar_64(1.0, f64::NAN).is_nan());
            assert!(atan2_scalar_64(f64::NAN, f64::NAN).is_nan());
        }
    }

    // ---- f64 Quadrant tests --------------------------------------------------

    #[test]
    fn atan2_f64_quadrant_1() {
        unsafe {
            let result = atan2_scalar_64(1.0, 1.0);
            assert!((result - std::f64::consts::FRAC_PI_4).abs() < TOL_64);
        }
    }

    #[test]
    fn atan2_f64_quadrant_2() {
        unsafe {
            let result = atan2_scalar_64(1.0, -1.0);
            assert!((result - 3.0 * std::f64::consts::FRAC_PI_4).abs() < TOL_64);
        }
    }

    #[test]
    fn atan2_f64_quadrant_3() {
        unsafe {
            let result = atan2_scalar_64(-1.0, -1.0);
            assert!((result - (-3.0 * std::f64::consts::FRAC_PI_4)).abs() < TOL_64);
        }
    }

    #[test]
    fn atan2_f64_quadrant_4() {
        unsafe {
            let result = atan2_scalar_64(-1.0, 1.0);
            assert!((result - (-std::f64::consts::FRAC_PI_4)).abs() < TOL_64);
        }
    }

    // ---- f64 All 2 lanes processed correctly ---------------------------------

    #[test]
    fn atan2_f64_processes_all_2_lanes_independently() {
        let y_vals = [1.0f64, -1.0];
        let x_vals = [1.0f64, -1.0];
        unsafe {
            let vy = vld1q_f64(y_vals.as_ptr());
            let vx = vld1q_f64(x_vals.as_ptr());
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), vatan2_f64(vy, vx));

            for i in 0..2 {
                let expected = y_vals[i].atan2(x_vals[i]);
                assert!(
                    (out[i] - expected).abs() < TOL_64,
                    "lane {i}: atan2({}, {}) = {}, expected {expected}",
                    y_vals[i],
                    x_vals[i],
                    out[i]
                );
            }
        }
    }

    // ---- f64 ULP accuracy sweep ----------------------------------------------

    #[test]
    fn atan2_f64_max_ulp_error_is_at_most_2() {
        let mut max_ulp: u64 = 0;
        let mut worst_y: f64 = 0.0;
        let mut worst_x: f64 = 0.0;

        for i in 0..2000 {
            let theta =
                (i as f64 / 1999.0) * 2.0 * std::f64::consts::PI - std::f64::consts::PI;
            for j in 0..100 {
                let r = 0.001 + (j as f64) * 0.5;
                let x = r * theta.cos();
                let y = r * theta.sin();

                let expected = y.atan2(x);
                if !expected.is_finite() {
                    continue;
                }

                let result = unsafe { atan2_scalar_64(y, x) };
                let ulp = expected.to_bits().abs_diff(result.to_bits());
                if ulp > max_ulp {
                    max_ulp = ulp;
                    worst_y = y;
                    worst_x = x;
                }
            }
        }

        assert!(
            max_ulp <= 2,
            "max ULP {max_ulp} at atan2({worst_y}, {worst_x}) — expected ≤ 2"
        );
    }
}
