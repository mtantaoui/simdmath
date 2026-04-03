//! AVX2 SIMD implementation of `atan2(y, x)` for `f32` and `f64` vectors.
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
//! 3. **Compute `atan(|y/x|)`** — using the existing `_mm256_atan_ps` / `_mm256_atan_pd`
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
//! `_mm256_blendv_ps` / `_mm256_blendv_pd` in priority order (highest to lowest):
//! NaN → x==1 → y==0 → x==0 → x==∞ → huge ratio → general case

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::avx2::abs::_mm256_abs_pd;
use crate::arch::avx2::atan::{_mm256_atan_pd, _mm256_atan_ps};
use crate::arch::consts::atan2::{
    FRAC_3_PI_4_32, FRAC_3_PI_4_64, FRAC_PI_2_32, FRAC_PI_2_64, FRAC_PI_4_32, FRAC_PI_4_64,
    HUGE_RATIO_THRESHOLD_32, HUGE_RATIO_THRESHOLD_64, PI_HI_32, PI_HI_64, PI_LO_32, PI_LO_64,
};

// ---------------------------------------------------------------------------
// f32 Implementation
// ---------------------------------------------------------------------------

/// Computes `atan2(y, x)` for each lane of two AVX2 `__m256` registers.
///
/// # Precision
///
/// **≤ 3 ULP** error across the entire domain (inherits from `_mm256_atan_ps`).
///
/// # Description
///
/// All 8 lanes are processed simultaneously without branches. Special cases
/// (NaN, zeros, infinities) are handled via branchless blending.
///
/// # Safety
///
/// Requires AVX2 and FMA support. `y` and `x` must be valid `__m256` registers.
///
/// # Example
///
/// ```ignore
/// let y = _mm256_set1_ps(1.0);
/// let x = _mm256_set1_ps(1.0);
/// let result = _mm256_atan2_ps(y, x);
/// // result ≈ [0.7854; 8] (π/4)
/// ```
#[inline]
pub(crate) unsafe fn _mm256_atan2_ps(y: __m256, x: __m256) -> __m256 {
    unsafe {
        // -------------------------------------------------------------------------
        // Broadcast constants
        // -------------------------------------------------------------------------
        let zero = _mm256_setzero_ps();
        let neg_zero = _mm256_set1_ps(-0.0);
        let pi_hi = _mm256_set1_ps(PI_HI_32);
        let pi_lo = _mm256_set1_ps(PI_LO_32);
        let pi_over_2 = _mm256_set1_ps(FRAC_PI_2_32);
        let pi_over_4 = _mm256_set1_ps(FRAC_PI_4_32);
        let three_pi_over_4 = _mm256_set1_ps(FRAC_3_PI_4_32);

        // Integer masks for bit manipulation
        let abs_mask = _mm256_set1_epi32(0x7FFF_FFFF_u32 as i32);
        let one_bits = _mm256_set1_epi32(0x3F80_0000_u32 as i32); // 1.0f32
        let inf_bits = _mm256_set1_epi32(0x7F80_0000_u32 as i32); // +∞
        let huge_threshold = _mm256_set1_epi32(HUGE_RATIO_THRESHOLD_32);

        // -------------------------------------------------------------------------
        // Extract bit representations and absolute values
        // -------------------------------------------------------------------------
        let x_bits = _mm256_castps_si256(x);
        let y_bits = _mm256_castps_si256(y);

        let ix = _mm256_and_si256(x_bits, abs_mask); // |x| as integer bits
        let iy = _mm256_and_si256(y_bits, abs_mask); // |y| as integer bits

        let abs_x = _mm256_castsi256_ps(ix);
        let abs_y = _mm256_castsi256_ps(iy);

        // -------------------------------------------------------------------------
        // Compute quadrant index: m = 2·sign(x) + sign(y)
        //
        // m ∈ {0, 1, 2, 3} encodes which quadrant (x, y) lies in:
        //   m=0: x≥0, y≥0 (Q1)    m=1: x≥0, y<0 (Q4)
        //   m=2: x<0, y≥0 (Q2)    m=3: x<0, y<0 (Q3)
        // -------------------------------------------------------------------------
        let sign_x = _mm256_srli_epi32(x_bits, 31); // 0 or 1
        let sign_y = _mm256_srli_epi32(y_bits, 31); // 0 or 1
        let m = _mm256_or_si256(_mm256_slli_epi32(sign_x, 1), sign_y);

        // Precompute masks for each m value
        let m_eq_0 = _mm256_cmpeq_epi32(m, _mm256_setzero_si256());
        let m_eq_1 = _mm256_cmpeq_epi32(m, _mm256_set1_epi32(1));
        let m_eq_2 = _mm256_cmpeq_epi32(m, _mm256_set1_epi32(2));
        let m_01 = _mm256_or_si256(m_eq_0, m_eq_1); // m ∈ {0, 1} → x ≥ 0
        let m_02 = _mm256_or_si256(m_eq_0, m_eq_2); // m ∈ {0, 2} → y ≥ 0

        // -------------------------------------------------------------------------
        // Condition masks for special cases
        // -------------------------------------------------------------------------
        let is_x_one = _mm256_cmpeq_epi32(x_bits, one_bits); // x == 1.0 exactly
        let is_y_zero = _mm256_cmpeq_epi32(iy, _mm256_setzero_si256()); // y == ±0
        let is_x_zero = _mm256_cmpeq_epi32(ix, _mm256_setzero_si256()); // x == ±0
        let is_x_inf = _mm256_cmpeq_epi32(ix, inf_bits); // |x| == ∞
        let is_y_inf = _mm256_cmpeq_epi32(iy, inf_bits); // |y| == ∞
        let is_x_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q); // x is NaN
        let is_y_nan = _mm256_cmp_ps(y, y, _CMP_UNORD_Q); // y is NaN

        // Check if |y/x| > 2^26 (y dominates) — atan2 ≈ ±π/2
        let iy_minus_ix = _mm256_sub_epi32(iy, ix);
        let is_huge_ratio = _mm256_cmpgt_epi32(iy_minus_ix, huge_threshold);
        let huge_or_y_inf = _mm256_or_si256(is_huge_ratio, is_y_inf);

        // Check if |y/x| < 2^-26 AND x < 0 — atan2 ≈ ±π (y negligible, negative x)
        let iy_plus_threshold = _mm256_add_epi32(iy, huge_threshold);
        let is_tiny_ratio = _mm256_cmpgt_epi32(ix, iy_plus_threshold);
        let m_ge_2 = _mm256_cmpgt_epi32(m, _mm256_set1_epi32(1)); // x < 0
        let tiny_and_x_neg = _mm256_and_si256(is_tiny_ratio, m_ge_2);

        // -------------------------------------------------------------------------
        // Case: NaN — if either input is NaN, return NaN (as x + y)
        // -------------------------------------------------------------------------
        let result_nan = _mm256_add_ps(x, y);

        // -------------------------------------------------------------------------
        // Case: x == 1.0 — return atan(y) directly
        // -------------------------------------------------------------------------
        let result_x_one = _mm256_atan_ps(y);

        // -------------------------------------------------------------------------
        // Case: y == ±0
        //
        //   m=0 (x≥0, y=+0): +0     m=1 (x≥0, y=-0): -0
        //   m=2 (x<0, y=+0): +π     m=3 (x<0, y=-0): -π
        // -------------------------------------------------------------------------
        let result_y_zero = _mm256_blendv_ps(
            _mm256_blendv_ps(
                _mm256_sub_ps(zero, pi_hi), // m=3: -π
                pi_hi,                      // m=2: +π
                _mm256_castsi256_ps(m_eq_2),
            ),
            _mm256_blendv_ps(neg_zero, y, _mm256_castsi256_ps(m_eq_0)), // m=0: y, m=1: -0
            _mm256_castsi256_ps(m_01),
        );

        // -------------------------------------------------------------------------
        // Case: x == ±0 (but y ≠ 0)
        //
        //   y > 0: +π/2     y < 0: -π/2
        // -------------------------------------------------------------------------
        let result_x_zero = _mm256_blendv_ps(
            _mm256_sub_ps(zero, pi_over_2), // y < 0: -π/2
            pi_over_2,                      // y ≥ 0: +π/2
            _mm256_castsi256_ps(m_02),
        );

        // -------------------------------------------------------------------------
        // Case: x == ±∞
        //
        // If y is also ±∞:
        //   m=0: +π/4    m=1: -π/4    m=2: +3π/4    m=3: -3π/4
        //
        // If y is finite:
        //   m=0: +0      m=1: -0      m=2: +π       m=3: -π
        // -------------------------------------------------------------------------
        let result_both_inf = _mm256_blendv_ps(
            _mm256_blendv_ps(
                _mm256_sub_ps(zero, three_pi_over_4), // m=3: -3π/4
                three_pi_over_4,                      // m=2: +3π/4
                _mm256_castsi256_ps(m_eq_2),
            ),
            _mm256_blendv_ps(
                _mm256_sub_ps(zero, pi_over_4), // m=1: -π/4
                pi_over_4,                      // m=0: +π/4
                _mm256_castsi256_ps(m_eq_0),
            ),
            _mm256_castsi256_ps(m_01),
        );

        let result_x_inf_y_finite = _mm256_blendv_ps(
            _mm256_blendv_ps(
                _mm256_sub_ps(zero, pi_hi), // m=3: -π
                pi_hi,                      // m=2: +π
                _mm256_castsi256_ps(m_eq_2),
            ),
            _mm256_blendv_ps(neg_zero, zero, _mm256_castsi256_ps(m_eq_0)), // m=0: +0, m=1: -0
            _mm256_castsi256_ps(m_01),
        );

        let result_x_inf = _mm256_blendv_ps(
            result_x_inf_y_finite,
            result_both_inf,
            _mm256_castsi256_ps(is_y_inf),
        );

        // -------------------------------------------------------------------------
        // Case: |y/x| > 2^26 or y == ±∞ — return ±π/2
        // -------------------------------------------------------------------------
        let result_huge = _mm256_blendv_ps(
            _mm256_sub_ps(zero, pi_over_2), // y < 0: -π/2
            pi_over_2,                      // y ≥ 0: +π/2
            _mm256_castsi256_ps(m_02),
        );

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
        let ratio = _mm256_div_ps(abs_y, abs_x);
        let z_computed = _mm256_atan_ps(ratio);
        let z = _mm256_blendv_ps(z_computed, zero, _mm256_castsi256_ps(tiny_and_x_neg));

        // Quadrant corrections
        let result_m0 = z; // +z
        let result_m1 = _mm256_sub_ps(zero, z); // -z
        let result_m2 = _mm256_sub_ps(pi_hi, _mm256_sub_ps(z, pi_lo)); // π - (z - π_lo)
        let result_m3 = _mm256_sub_ps(_mm256_sub_ps(z, pi_lo), pi_hi); // (z - π_lo) - π

        let result_general = _mm256_blendv_ps(
            _mm256_blendv_ps(result_m3, result_m2, _mm256_castsi256_ps(m_eq_2)),
            _mm256_blendv_ps(result_m1, result_m0, _mm256_castsi256_ps(m_eq_0)),
            _mm256_castsi256_ps(m_01),
        );

        // -------------------------------------------------------------------------
        // Merge all cases in priority order (lowest to highest)
        // -------------------------------------------------------------------------
        let mut result = result_general;

        result = _mm256_blendv_ps(result, result_huge, _mm256_castsi256_ps(huge_or_y_inf));
        result = _mm256_blendv_ps(result, result_x_inf, _mm256_castsi256_ps(is_x_inf));
        result = _mm256_blendv_ps(result, result_x_zero, _mm256_castsi256_ps(is_x_zero));
        result = _mm256_blendv_ps(result, result_y_zero, _mm256_castsi256_ps(is_y_zero));
        result = _mm256_blendv_ps(result, result_x_one, _mm256_castsi256_ps(is_x_one));
        result = _mm256_blendv_ps(result, result_nan, _mm256_or_ps(is_x_nan, is_y_nan));

        result
    }
}

// ===========================================================================
// f64 Implementation
// ===========================================================================

/// Computes `atan2(y, x)` for each lane of two AVX2 `__m256d` registers (4 × f64).
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Description
///
/// All 4 lanes are processed simultaneously without branches. Special cases
/// (NaN, zeros, infinities) are handled via branchless blending.
///
/// # Safety
///
/// Requires AVX2 and FMA support. `y` and `x` must be valid `__m256d` registers.
///
/// # Example
///
/// ```ignore
/// let y = _mm256_set1_pd(1.0);
/// let x = _mm256_set1_pd(1.0);
/// let result = _mm256_atan2_pd(y, x);
/// // result ≈ [0.7854; 4] (π/4)
/// ```
#[inline]
pub(crate) unsafe fn _mm256_atan2_pd(y: __m256d, x: __m256d) -> __m256d {
    unsafe {
        // -------------------------------------------------------------------------
        // Broadcast constants
        // -------------------------------------------------------------------------
        let zero = _mm256_setzero_pd();
        let neg_zero = _mm256_set1_pd(-0.0);
        let pi_hi = _mm256_set1_pd(PI_HI_64);
        let pi_lo = _mm256_set1_pd(PI_LO_64);
        let pi_over_2 = _mm256_set1_pd(FRAC_PI_2_64);
        let pi_over_4 = _mm256_set1_pd(FRAC_PI_4_64);
        let three_pi_over_4 = _mm256_set1_pd(FRAC_3_PI_4_64);

        // Integer masks for bit manipulation (64-bit lanes)
        let abs_mask = _mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF_u64 as i64);
        let one_bits = _mm256_set1_epi64x(0x3FF0_0000_0000_0000_u64 as i64); // 1.0f64
        let inf_bits = _mm256_set1_epi64x(0x7FF0_0000_0000_0000_u64 as i64); // +∞
        let huge_threshold = _mm256_set1_epi64x(HUGE_RATIO_THRESHOLD_64);

        // -------------------------------------------------------------------------
        // Extract bit representations and absolute values
        // -------------------------------------------------------------------------
        let x_bits = _mm256_castpd_si256(x);
        let y_bits = _mm256_castpd_si256(y);

        let ix = _mm256_and_si256(x_bits, abs_mask); // |x| as integer bits
        let iy = _mm256_and_si256(y_bits, abs_mask); // |y| as integer bits

        let abs_x = _mm256_abs_pd(x);
        let abs_y = _mm256_abs_pd(y);

        // -------------------------------------------------------------------------
        // Compute quadrant index: m = 2·sign(x) + sign(y)
        //
        // For 64-bit lanes, we need to shift by 63 to get the sign bit.
        // -------------------------------------------------------------------------
        let sign_x = _mm256_srli_epi64(x_bits, 63);
        let sign_y = _mm256_srli_epi64(y_bits, 63);
        let m = _mm256_or_si256(_mm256_slli_epi64(sign_x, 1), sign_y);

        // Precompute masks for each m value
        let m_eq_0 = _mm256_cmpeq_epi64(m, _mm256_setzero_si256());
        let m_eq_1 = _mm256_cmpeq_epi64(m, _mm256_set1_epi64x(1));
        let m_eq_2 = _mm256_cmpeq_epi64(m, _mm256_set1_epi64x(2));
        let m_01 = _mm256_or_si256(m_eq_0, m_eq_1);
        let m_02 = _mm256_or_si256(m_eq_0, m_eq_2);

        // -------------------------------------------------------------------------
        // Condition masks for special cases
        // -------------------------------------------------------------------------
        let is_x_one = _mm256_cmpeq_epi64(x_bits, one_bits);
        let is_y_zero = _mm256_cmpeq_epi64(iy, _mm256_setzero_si256());
        let is_x_zero = _mm256_cmpeq_epi64(ix, _mm256_setzero_si256());
        let is_x_inf = _mm256_cmpeq_epi64(ix, inf_bits);
        let is_y_inf = _mm256_cmpeq_epi64(iy, inf_bits);
        let is_x_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
        let is_y_nan = _mm256_cmp_pd(y, y, _CMP_UNORD_Q);

        // Check if |y/x| > 2^60 (y dominates)
        let iy_minus_ix = _mm256_sub_epi64(iy, ix);
        let is_huge_ratio = _mm256_cmpgt_epi64(iy_minus_ix, huge_threshold);
        let huge_or_y_inf = _mm256_or_si256(is_huge_ratio, is_y_inf);

        // Check if |y/x| < 2^-60 AND x < 0
        let iy_plus_threshold = _mm256_add_epi64(iy, huge_threshold);
        let is_tiny_ratio = _mm256_cmpgt_epi64(ix, iy_plus_threshold);
        let m_ge_2 = _mm256_cmpgt_epi64(m, _mm256_set1_epi64x(1));
        let tiny_and_x_neg = _mm256_and_si256(is_tiny_ratio, m_ge_2);

        // -------------------------------------------------------------------------
        // Case: NaN — if either input is NaN, return NaN (as x + y)
        // -------------------------------------------------------------------------
        let result_nan = _mm256_add_pd(x, y);

        // -------------------------------------------------------------------------
        // Case: x == 1.0 — return atan(y) directly
        // -------------------------------------------------------------------------
        let result_x_one = _mm256_atan_pd(y);

        // -------------------------------------------------------------------------
        // Case: y == ±0
        // -------------------------------------------------------------------------
        let result_y_zero = _mm256_blendv_pd(
            _mm256_blendv_pd(
                _mm256_sub_pd(zero, pi_hi),
                pi_hi,
                _mm256_castsi256_pd(m_eq_2),
            ),
            _mm256_blendv_pd(neg_zero, y, _mm256_castsi256_pd(m_eq_0)),
            _mm256_castsi256_pd(m_01),
        );

        // -------------------------------------------------------------------------
        // Case: x == ±0 (but y ≠ 0)
        // -------------------------------------------------------------------------
        let result_x_zero = _mm256_blendv_pd(
            _mm256_sub_pd(zero, pi_over_2),
            pi_over_2,
            _mm256_castsi256_pd(m_02),
        );

        // -------------------------------------------------------------------------
        // Case: x == ±∞
        // -------------------------------------------------------------------------
        let result_both_inf = _mm256_blendv_pd(
            _mm256_blendv_pd(
                _mm256_sub_pd(zero, three_pi_over_4),
                three_pi_over_4,
                _mm256_castsi256_pd(m_eq_2),
            ),
            _mm256_blendv_pd(
                _mm256_sub_pd(zero, pi_over_4),
                pi_over_4,
                _mm256_castsi256_pd(m_eq_0),
            ),
            _mm256_castsi256_pd(m_01),
        );

        let result_x_inf_y_finite = _mm256_blendv_pd(
            _mm256_blendv_pd(
                _mm256_sub_pd(zero, pi_hi),
                pi_hi,
                _mm256_castsi256_pd(m_eq_2),
            ),
            _mm256_blendv_pd(neg_zero, zero, _mm256_castsi256_pd(m_eq_0)),
            _mm256_castsi256_pd(m_01),
        );

        let result_x_inf = _mm256_blendv_pd(
            result_x_inf_y_finite,
            result_both_inf,
            _mm256_castsi256_pd(is_y_inf),
        );

        // -------------------------------------------------------------------------
        // Case: |y/x| > 2^60 or y == ±∞ — return ±π/2
        // -------------------------------------------------------------------------
        let result_huge = _mm256_blendv_pd(
            _mm256_sub_pd(zero, pi_over_2),
            pi_over_2,
            _mm256_castsi256_pd(m_02),
        );

        // -------------------------------------------------------------------------
        // General case: compute z = atan(|y/x|), then apply quadrant correction
        // -------------------------------------------------------------------------
        let ratio = _mm256_div_pd(abs_y, abs_x);
        let z_computed = _mm256_atan_pd(ratio);
        let z = _mm256_blendv_pd(z_computed, zero, _mm256_castsi256_pd(tiny_and_x_neg));

        // Quadrant corrections
        let result_m0 = z;
        let result_m1 = _mm256_sub_pd(zero, z);
        let result_m2 = _mm256_sub_pd(pi_hi, _mm256_sub_pd(z, pi_lo));
        let result_m3 = _mm256_sub_pd(_mm256_sub_pd(z, pi_lo), pi_hi);

        let result_general = _mm256_blendv_pd(
            _mm256_blendv_pd(result_m3, result_m2, _mm256_castsi256_pd(m_eq_2)),
            _mm256_blendv_pd(result_m1, result_m0, _mm256_castsi256_pd(m_eq_0)),
            _mm256_castsi256_pd(m_01),
        );

        // -------------------------------------------------------------------------
        // Merge all cases in priority order (lowest to highest)
        // -------------------------------------------------------------------------
        let mut result = result_general;

        result = _mm256_blendv_pd(result, result_huge, _mm256_castsi256_pd(huge_or_y_inf));
        result = _mm256_blendv_pd(result, result_x_inf, _mm256_castsi256_pd(is_x_inf));
        result = _mm256_blendv_pd(result, result_x_zero, _mm256_castsi256_pd(is_x_zero));
        result = _mm256_blendv_pd(result, result_y_zero, _mm256_castsi256_pd(is_y_zero));
        result = _mm256_blendv_pd(result, result_x_one, _mm256_castsi256_pd(is_x_one));
        result = _mm256_blendv_pd(result, result_nan, _mm256_or_pd(is_x_nan, is_y_nan));

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

    /// Load 8 copies of `y` and `x`, call `_mm256_atan2_ps`, and return lane 0.
    unsafe fn atan2_scalar(y: f32, x: f32) -> f32 {
        unsafe {
            let vy = _mm256_set1_ps(y);
            let vx = _mm256_set1_ps(x);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), _mm256_atan2_ps(vy, vx));
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
            assert!(
                result == 0.0 && result.is_sign_negative(),
                "expected -0.0, got {result}"
            );
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
            assert!(
                (neg - (-PI)).abs() < TOL,
                "atan2(-0, -1) = {neg}, expected -π"
            );
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
            assert!(
                (r4 - (-3.0 * FRAC_PI_4)).abs() < TOL,
                "atan2(-∞, -∞) = {r4}"
            );
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
        // Q1: x > 0, y > 0 → result in (0, π/2)
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
        // Q2: x < 0, y > 0 → result in (π/2, π)
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
        // Q3: x < 0, y < 0 → result in (-π, -π/2)
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
        // Q4: x > 0, y < 0 → result in (-π/2, 0)
        unsafe {
            let result = atan2_scalar(-1.0, 1.0);
            assert!(
                (result - (-FRAC_PI_4)).abs() < TOL,
                "atan2(-1, 1) = {result}, expected -π/4"
            );
        }
    }

    // ---- All 8 lanes processed correctly -------------------------------------

    #[test]
    fn atan2_processes_all_8_lanes_independently() {
        let y_vals = [1.0f32, -1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 2.0];
        let x_vals = [1.0f32, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0];
        unsafe {
            let vy = _mm256_loadu_ps(y_vals.as_ptr());
            let vx = _mm256_loadu_ps(x_vals.as_ptr());
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), _mm256_atan2_ps(vy, vx));

            for i in 0..8 {
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

    /// Load 4 copies of `y` and `x`, call `_mm256_atan2_pd`, and return lane 0.
    unsafe fn atan2_scalar_64(y: f64, x: f64) -> f64 {
        unsafe {
            let vy = _mm256_set1_pd(y);
            let vx = _mm256_set1_pd(x);
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), _mm256_atan2_pd(vy, vx));
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
            assert!(
                result == 0.0 && result.is_sign_negative(),
                "expected -0.0, got {result}"
            );
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

    // ---- f64 All 4 lanes processed correctly ---------------------------------

    #[test]
    fn atan2_f64_processes_all_4_lanes_independently() {
        let y_vals = [1.0f64, -1.0, 1.0, -1.0];
        let x_vals = [1.0f64, 1.0, -1.0, -1.0];
        unsafe {
            let vy = _mm256_loadu_pd(y_vals.as_ptr());
            let vx = _mm256_loadu_pd(x_vals.as_ptr());
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), _mm256_atan2_pd(vy, vx));

            for i in 0..4 {
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
            let theta = (i as f64 / 1999.0) * 2.0 * std::f64::consts::PI - std::f64::consts::PI;
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
