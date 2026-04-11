//! NEON SIMD implementation of `pow(x, y)` for `f32` and `f64` vectors.
//!
//! This module provides 4-lane f32 and 2-lane f64 power function implementations.
//! The core algorithm is `pow(x, y) = exp(y · ln(|x|))` with **compensated
//! arithmetic** to preserve precision across the `ln` → multiply → `exp` chain.
//!
//! # Algorithm
//!
//! 1. **Compensated log**: Compute `ln(|x|) = hi + lo` using the fdlibm
//!    polynomial with the final reconstruction split into a high part (rounded)
//!    and a low part (residual tail carrying ~15 extra bits).
//!
//! 2. **Dekker multiplication**: Compute `y · (hi + lo)` as `(ehi, elo)` using
//!    FMA: `ehi = y*hi`, `elo = fma(y, hi, -ehi) + y*lo`. This preserves the
//!    full double-precision product without losing bits to rounding.
//!
//! 3. **Compensated exp**: Compute `exp(ehi + elo)` by folding `elo` into the
//!    argument-reduction remainder `r`, so the polynomial evaluates `exp(r + elo)`
//!    instead of just `exp(r)`.
//!
//! 4. **Sign correction**: For negative bases with odd integer exponents,
//!    negate the result.
//!
//! 5. **Special-value handling** (IEEE 754 / C99 §7.12.7.4):
//!    - `pow(x, ±0)     = 1` for any `x` (including NaN)
//!    - `pow(1, y)       = 1` for any `y` (including NaN)
//!    - `pow(-1, ±∞)     = 1`
//!    - `pow(x, y)       = NaN` when `x < 0` and `y` is not an integer
//!    - `pow(±0, y)      = ±∞` when `y` is odd integer < 0
//!    - `pow(±0, y)      = +∞` when `y < 0` and `y` is not odd integer
//!    - `pow(±0, y)      = ±0` when `y` is odd integer > 0
//!    - `pow(±0, y)      = +0` when `y > 0` and `y` is not odd integer
//!    - `pow(-∞, y)      = -0` when `y` is odd integer < 0
//!    - `pow(-∞, y)      = +0` when `y < 0` and `y` is not odd integer
//!    - `pow(-∞, y)      = -∞` when `y` is odd integer > 0
//!    - `pow(-∞, y)      = +∞` when `y > 0` and `y` is not odd integer
//!    - `pow(+∞, y)      = +0` when `y < 0`
//!    - `pow(+∞, y)      = +∞` when `y > 0`
//!    - `pow(x, -∞)      = +∞` when `|x| < 1`
//!    - `pow(x, -∞)      = +0` when `|x| > 1`
//!    - `pow(x, +∞)      = +0` when `|x| < 1`
//!    - `pow(x, +∞)      = +∞` when `|x| > 1`
//!
//! # Precision
//!
//! | Variant       | Max Error |
//! |---------------|-----------|
//! | `vpow_f32`    | ≤ 2 ULP  |
//! | `vpow_f64`    | ≤ 2 ULP  |
//!
//! # Special Values
//!
//! | x         | y            | Result   |
//! |-----------|--------------|----------|
//! | any       | `±0`         | `1.0`    |
//! | `1.0`     | any          | `1.0`    |
//! | `< 0`     | non-integer  | `NaN`    |
//! | `±0`      | odd int < 0  | `±∞`     |
//! | `NaN`     | non-zero     | `NaN`    |
//! | non-one   | `NaN`        | `NaN`    |

use std::arch::aarch64::*;

use crate::arch::consts::exp::{
    LN2_HI_64 as EXP_LN2_HI, LN2_INV_64, LN2_LO_64 as EXP_LN2_LO, OVERFLOW_THRESH_64,
    P1_64, P2_64, P3_64, P4_64, P5_64, UNDERFLOW_THRESH_64,
};
use crate::arch::consts::ln::{
    LG1_64, LG2_64, LG3_64, LG4_64, LG5_64, LG6_64, LG7_64, LN2_HI_64, LN2_LO_64, SQRT2_64,
    TWO52_64,
};

// =============================================================================
// f32 Implementation (4 lanes, via f64 promotion)
// =============================================================================

/// Computes `x^y` (power) for each lane of a NEON `float32x4_t` register.
///
/// Promotes `f32` inputs to `f64`, delegates to [`pow_core_f64`], then
/// narrows back to `f32`. This two-step approach preserves accuracy by
/// performing the intermediate `y * ln(|x|)` and `exp(…)` in f64 precision.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires NEON support. The caller must ensure this feature is available.
#[inline]
#[allow(dead_code)]
pub(crate) unsafe fn vpow_f32(x: float32x4_t, y: float32x4_t) -> float32x4_t {
    // Split 4-lane f32 into two 2-lane f64 halves
    let x_lo = vcvt_f64_f32(vget_low_f32(x));
    let x_hi = vcvt_f64_f32(vget_high_f32(x));
    let y_lo = vcvt_f64_f32(vget_low_f32(y));
    let y_hi = vcvt_f64_f32(vget_high_f32(y));

    // Compute pow in f64 precision
    let pow_lo = pow_core_f64(x_lo, y_lo);
    let pow_hi = pow_core_f64(x_hi, y_hi);

    // Convert back to f32 and combine
    let result_lo = vcvt_f32_f64(pow_lo);
    let result_hi = vcvt_f32_f64(pow_hi);

    vcombine_f32(result_lo, result_hi)
}

// =============================================================================
// f64 Implementation (2 lanes)
// =============================================================================

/// Computes `x^y` (power) for each lane of a NEON `float64x2_t` register.
///
/// Delegates to [`pow_core_f64`] which implements the full IEEE 754 semantics
/// for the power function using compensated arithmetic for precision.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires NEON support. The caller must ensure this feature is available.
#[inline]
#[allow(dead_code)]
pub(crate) unsafe fn vpow_f64(x: float64x2_t, y: float64x2_t) -> float64x2_t {
    pow_core_f64(x, y)
}

// =============================================================================
// Compensated ln: returns (hi, lo) where ln(x) ≈ hi + lo
// =============================================================================

/// Computes `ln(x)` as a high/low pair for extra precision.
///
/// Returns `(hi, lo)` such that `ln(x) ≈ hi + lo`, where `lo` carries
/// approximately 15 extra bits of precision beyond what a single `f64` can hold.
/// This is the key building block for precise `pow`: by preserving the tail of
/// the logarithm, the subsequent multiplication `y * ln(x)` avoids catastrophic
/// cancellation.
///
/// Uses the same fdlibm polynomial as [`ln_core_f64`](super::ln), but splits
/// the final reconstruction using Knuth's 2Sum to capture the rounding error.
///
/// # Safety
///
/// Only valid for positive inputs. Zero, negative, infinity, and NaN inputs
/// produce unspecified results (the caller must handle those cases).
#[inline]
unsafe fn ln_hilo(x: float64x2_t) -> (float64x2_t, float64x2_t) {
    let one = vdupq_n_f64(1.0);
    let half = vdupq_n_f64(0.5);
    let zero = vdupq_n_f64(0.0);

    // =================================================================
    // Step 1: Handle subnormals by scaling up
    // =================================================================

    let min_normal = vdupq_n_f64(f64::MIN_POSITIVE);
    let is_subnormal = vandq_u64(vcgtq_f64(x, zero), vcltq_f64(x, min_normal));

    let two52 = vdupq_n_f64(TWO52_64);
    let x_scaled = vmulq_f64(x, two52);
    let x_work = vbslq_f64(is_subnormal, x_scaled, x);

    let neg_52 = vdupq_n_f64(-52.0);
    let k_adjust = vbslq_f64(is_subnormal, neg_52, zero);

    // =================================================================
    // Step 2: Extract exponent k and normalize mantissa to [1, 2)
    // =================================================================

    let ix = vreinterpretq_s64_f64(x_work);

    let exp_bits = vshrq_n_s64::<52>(ix);

    let bias = vdupq_n_f64(1023.0);
    let k = vsubq_f64(vcvtq_f64_s64(exp_bits), bias);
    let k = vaddq_f64(k, k_adjust);

    // Normalize mantissa: clear exponent, set to biased 1023 → [1, 2)
    let mantissa_mask = vdupq_n_s64(0x000FFFFFFFFFFFFF_u64 as i64);
    let exp_1023 = vdupq_n_s64(0x3FF0000000000000_u64 as i64);
    let m_bits = vorrq_s64(vandq_s64(ix, mantissa_mask), exp_1023);
    let m = vreinterpretq_f64_s64(m_bits);

    // If m > √2, halve it and increment k
    let sqrt2 = vdupq_n_f64(SQRT2_64);
    let is_big = vcgtq_f64(m, sqrt2);

    let exp_1022 = vdupq_n_s64(0x3FE0000000000000_u64 as i64);
    let m_halved_bits = vorrq_s64(vandq_s64(ix, mantissa_mask), exp_1022);
    let m_halved = vreinterpretq_f64_s64(m_halved_bits);

    let m = vbslq_f64(is_big, m_halved, m);
    let k_inc = vbslq_f64(is_big, one, zero);
    let k = vaddq_f64(k, k_inc);

    // =================================================================
    // Step 3: f = m - 1, s = f / (2 + f)
    // =================================================================

    let f = vsubq_f64(m, one);
    let two_plus_f = vaddq_f64(vdupq_n_f64(2.0), f);
    let s = vdivq_f64(f, two_plus_f);
    let hfsq = vmulq_f64(half, vmulq_f64(f, f));

    // =================================================================
    // Step 4: Minimax polynomial R(z) where z = s²
    // =================================================================

    let z = vmulq_f64(s, s);
    let w = vmulq_f64(z, z);

    // Odd powers:  t1 = Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7))
    let t1 = vfmaq_f64(vdupq_n_f64(LG5_64), w, vdupq_n_f64(LG7_64));
    let t1 = vfmaq_f64(vdupq_n_f64(LG3_64), w, t1);
    let t1 = vfmaq_f64(vdupq_n_f64(LG1_64), w, t1);

    // Even powers: t2 = Lg2 + w*(Lg4 + w*Lg6)
    let t2 = vfmaq_f64(vdupq_n_f64(LG4_64), w, vdupq_n_f64(LG6_64));
    let t2 = vfmaq_f64(vdupq_n_f64(LG2_64), w, t2);

    // R = z * (t1 + z*t2)
    let r = vfmaq_f64(t1, z, t2);
    let r = vmulq_f64(z, r);

    // =================================================================
    // Step 5: Split result into (hi, lo) using 2Sum
    //
    // ln(x) = f - hfsq + s*(hfsq+R) + k*ln2_hi + k*ln2_lo
    //
    // We split into:
    //   val_hi = f - hfsq
    //   val_lo = (f - val_hi - hfsq) + s*(hfsq+R) + k*ln2_lo
    //
    // Then combine with k*ln2_hi using Knuth's 2Sum:
    //   hi = val_hi + k*ln2_hi   (rounded)
    //   lo = (rounding error of that addition) + val_lo
    // =================================================================

    let ln2_hi = vdupq_n_f64(LN2_HI_64);
    let ln2_lo = vdupq_n_f64(LN2_LO_64);

    // val_hi = f - hfsq (may round)
    let val_hi = vsubq_f64(f, hfsq);

    // val_lo recovers the rounding error of (f - hfsq) and adds the remaining terms
    let val_lo = vsubq_f64(f, val_hi);
    let val_lo = vsubq_f64(val_lo, hfsq);
    let s_term = vmulq_f64(s, vaddq_f64(hfsq, r));
    let val_lo = vaddq_f64(val_lo, s_term);
    let val_lo = vfmaq_f64(val_lo, k, ln2_lo); // val_lo + k*ln2_lo

    // Knuth 2Sum: hi = val_hi + k*ln2_hi, capturing the rounding error
    let k_ln2_hi = vmulq_f64(k, ln2_hi);
    let hi = vaddq_f64(val_hi, k_ln2_hi);
    let b_virt = vsubq_f64(hi, val_hi);
    let a_virt = vsubq_f64(hi, b_virt);
    let b_err = vsubq_f64(k_ln2_hi, b_virt);
    let a_err = vsubq_f64(val_hi, a_virt);
    let lo = vaddq_f64(vaddq_f64(a_err, b_err), val_lo);

    (hi, lo)
}

// =============================================================================
// Compensated exp: exp(ehi + elo) with extra-precision input
// =============================================================================

/// Computes `exp(ehi + elo)` where the true argument is `ehi + elo`.
///
/// This is the standard fdlibm exp algorithm, modified to fold the correction
/// term `elo` into the argument-reduction remainder. After reducing `ehi` to
/// `r = ehi - k·ln2`, we compute `exp(r + elo)` instead of just `exp(r)`,
/// preserving the extra ~15 bits from the compensated logarithm.
///
/// Handles overflow (→ +∞) and underflow (→ 0) from the combined `ehi + elo`.
///
/// # Safety
///
/// Requires NEON support. The caller must ensure this feature is available.
#[inline]
unsafe fn exp_compensated(ehi: float64x2_t, elo: float64x2_t) -> float64x2_t {
    let zero = vdupq_n_f64(0.0);
    let one = vdupq_n_f64(1.0);
    let two = vdupq_n_f64(2.0);
    let half = vdupq_n_f64(0.5);

    // =================================================================
    // Overflow / underflow detection (on ehi, the dominant term)
    // =================================================================

    let inf = vdupq_n_f64(f64::INFINITY);
    let overflow_thresh = vdupq_n_f64(OVERFLOW_THRESH_64);
    let underflow_thresh = vdupq_n_f64(UNDERFLOW_THRESH_64);

    let is_overflow = vcgtq_f64(ehi, overflow_thresh);
    let is_underflow = vcltq_f64(ehi, underflow_thresh);
    let is_not_nan = vceqq_f64(ehi, ehi);
    let all_ones = vreinterpretq_u64_s64(vdupq_n_s64(-1));
    let is_nan = veorq_u64(is_not_nan, all_ones);

    // =================================================================
    // Argument reduction: ehi = k·ln2 + r, then r += elo
    // =================================================================

    let ln2_inv = vdupq_n_f64(LN2_INV_64);
    let ln2_hi = vdupq_n_f64(EXP_LN2_HI);
    let ln2_lo = vdupq_n_f64(EXP_LN2_LO);

    // k = round(ehi / ln2) via trunc(ehi/ln2 + copysign(0.5, ehi))
    let sign_bit = vdupq_n_f64(-0.0);
    let ehi_sign = vandq_u64(vreinterpretq_u64_f64(ehi), vreinterpretq_u64_f64(sign_bit));
    let sign_half = vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(half), ehi_sign));
    let k_f64 = vrndq_f64(vfmaq_f64(sign_half, ehi, ln2_inv));

    // r = ehi - k·ln2_hi - k·ln2_lo + elo (extended-precision reduction with tail)
    // vfmsq_f64(a, b, c) = a - b*c
    let r = vfmsq_f64(vfmsq_f64(ehi, k_f64, ln2_hi), k_f64, ln2_lo);
    let r = vaddq_f64(r, elo); // fold in the compensation tail

    // =================================================================
    // Polynomial: exp(r) ≈ 1 + r + r·c/(2-c)
    // where c = r - r²·P(r²)
    // =================================================================

    let r2 = vmulq_f64(r, r);

    let p = vfmaq_f64(vdupq_n_f64(P4_64), r2, vdupq_n_f64(P5_64));
    let p = vfmaq_f64(vdupq_n_f64(P3_64), r2, p);
    let p = vfmaq_f64(vdupq_n_f64(P2_64), r2, p);
    let p = vfmaq_f64(vdupq_n_f64(P1_64), r2, p);

    let c = vsubq_f64(r, vmulq_f64(r2, p));

    let rc = vmulq_f64(r, c);
    let exp_r = vsubq_f64(
        one,
        vsubq_f64(vdivq_f64(rc, vsubq_f64(c, two)), r),
    );

    // =================================================================
    // Reconstruct: exp(ehi+elo) = 2^k · exp(r+elo)
    // =================================================================

    let k_i64 = vcvtq_s64_f64(k_f64);
    let k_shifted = vshlq_n_s64::<52>(k_i64);
    let one_bits = vdupq_n_s64(0x3FF0000000000000_u64 as i64);
    let scale = vreinterpretq_f64_s64(vaddq_s64(k_shifted, one_bits));

    let result = vmulq_f64(exp_r, scale);

    // =================================================================
    // Apply overflow / underflow / NaN
    // =================================================================

    let nan = vdupq_n_f64(f64::NAN);
    let result = vbslq_f64(is_overflow, inf, result);
    let result = vbslq_f64(is_underflow, zero, result);
    vbslq_f64(is_nan, nan, result)
}

// =============================================================================
// Core f64 pow kernel (shared by both f32 and f64 paths)
// =============================================================================

/// Core power kernel operating on 2-lane f64 vectors.
///
/// Implements `pow(x, y) = exp(y · ln(|x|))` with **compensated arithmetic**
/// and complete IEEE 754 / C99 special-value handling.
///
/// The algorithm:
/// 1. **Classify inputs**: detect zeros, infinities, NaN, negative bases,
///    integer/odd-integer exponents.
/// 2. **Compensated log**: [`ln_hilo`] computes `ln(|x|) = hi + lo`.
/// 3. **Dekker multiplication**: `y · (hi + lo) = ehi + elo` via FMA.
/// 4. **Compensated exp**: [`exp_compensated`] evaluates `exp(ehi + elo)`.
/// 5. **Sign + special cases**: negate for odd-integer exponents on negative
///    bases, then apply IEEE 754 rules.
#[inline]
unsafe fn pow_core_f64(x: float64x2_t, y: float64x2_t) -> float64x2_t {
    let zero = vdupq_n_f64(0.0);
    let one = vdupq_n_f64(1.0);
    let neg_one = vdupq_n_f64(-1.0);
    let inf = vdupq_n_f64(f64::INFINITY);
    let neg_inf = vdupq_n_f64(f64::NEG_INFINITY);
    let nan = vdupq_n_f64(f64::NAN);
    let half = vdupq_n_f64(0.5);
    let sign_bit = vdupq_n_f64(-0.0);

    // =====================================================================
    // Phase 1: Classify inputs
    // =====================================================================

    let x_abs = vabsq_f64(x);
    let all_ones = vreinterpretq_u64_s64(vdupq_n_s64(-1));

    let x_is_nan = veorq_u64(vceqq_f64(x, x), all_ones);
    let x_is_zero = vceqq_f64(x, zero);
    let x_is_one = vceqq_f64(x, one);
    let x_is_neg = vcltq_f64(x, zero);
    let x_is_neg_one = vceqq_f64(x, neg_one);
    let x_is_pos_inf = vceqq_f64(x, inf);
    let x_is_neg_inf = vceqq_f64(x, neg_inf);
    let x_sign = vreinterpretq_f64_u64(vandq_u64(
        vreinterpretq_u64_f64(x),
        vreinterpretq_u64_f64(sign_bit),
    ));

    let x_abs_lt_one = vcltq_f64(x_abs, one);
    let x_abs_gt_one = vcgtq_f64(x_abs, one);

    let y_is_nan = veorq_u64(vceqq_f64(y, y), all_ones);
    let y_is_zero = vceqq_f64(y, zero);
    let y_is_pos = vcgtq_f64(y, zero);
    let y_is_neg = vcltq_f64(y, zero);
    let y_is_pos_inf = vceqq_f64(y, inf);
    let y_is_neg_inf = vceqq_f64(y, neg_inf);
    let y_is_inf = vorrq_u64(y_is_pos_inf, y_is_neg_inf);

    let y_trunc = vrndq_f64(y);
    let y_is_integer = vceqq_f64(y, y_trunc);

    let y_half = vmulq_f64(y, half);
    let y_half_trunc = vrndq_f64(y_half);
    let y_half_is_int = vceqq_f64(y_half, y_half_trunc);
    let y_is_odd_int = vbicq_u64(y_is_integer, y_half_is_int);

    // =====================================================================
    // Phase 2: Compensated exp(y · ln(|x|))
    // =====================================================================

    // Step 2a: ln(|x|) = ln_hi + ln_lo  (compensated log)
    // For non-positive x, ln_hilo produces garbage — those lanes are
    // overwritten by special-case handling in Phase 3.
    let (ln_hi, ln_lo) = ln_hilo(x_abs);

    // Step 2b: Dekker multiplication using FMA
    // ehi = y * ln_hi                       (rounded product)
    // elo = fma(y, ln_hi, -ehi) + y * ln_lo (exact error + low-order term)
    let ehi = vmulq_f64(y, ln_hi);
    let fmsub_term = vfmaq_f64(vnegq_f64(ehi), y, ln_hi); // y*ln_hi - ehi
    let elo = vfmaq_f64(fmsub_term, y, ln_lo); // y*ln_lo + (y*ln_hi - ehi)

    // Step 2c: exp(ehi + elo) with compensation
    let mut result = exp_compensated(ehi, elo);

    // =====================================================================
    // Phase 3: Apply IEEE 754 special cases (order matters!)
    // =====================================================================

    // --- Negative-base sign correction ---
    let should_negate = vandq_u64(x_is_neg, y_is_odd_int);
    let neg_result = vreinterpretq_f64_u64(veorq_u64(
        vreinterpretq_u64_f64(result),
        vreinterpretq_u64_f64(sign_bit),
    ));
    result = vbslq_f64(should_negate, neg_result, result);

    // Negative base with non-integer exponent → NaN (but not -∞)
    let neg_base_non_int = vbicq_u64(x_is_neg, y_is_integer);
    let neg_base_non_int = vbicq_u64(neg_base_non_int, x_is_neg_inf);
    result = vbslq_f64(neg_base_non_int, nan, result);

    // --- pow(±0, y) ---
    let x_zero_y_neg = vandq_u64(x_is_zero, y_is_neg);
    let x_zero_y_neg_odd = vandq_u64(x_zero_y_neg, y_is_odd_int);
    let signed_inf = vreinterpretq_f64_u64(vorrq_u64(
        vreinterpretq_u64_f64(inf),
        vreinterpretq_u64_f64(x_sign),
    ));
    result = vbslq_f64(x_zero_y_neg_odd, signed_inf, result);

    let x_zero_y_neg_not_odd = vbicq_u64(x_zero_y_neg, y_is_odd_int);
    result = vbslq_f64(x_zero_y_neg_not_odd, inf, result);

    let x_zero_y_pos = vandq_u64(x_is_zero, y_is_pos);
    let x_zero_y_pos_odd = vandq_u64(x_zero_y_pos, y_is_odd_int);
    let signed_zero = vreinterpretq_f64_u64(vorrq_u64(
        vreinterpretq_u64_f64(zero),
        vreinterpretq_u64_f64(x_sign),
    ));
    result = vbslq_f64(x_zero_y_pos_odd, signed_zero, result);

    let x_zero_y_pos_not_odd = vbicq_u64(x_zero_y_pos, y_is_odd_int);
    result = vbslq_f64(x_zero_y_pos_not_odd, zero, result);

    // --- pow(+∞, y) ---
    let pos_inf_y_neg = vandq_u64(x_is_pos_inf, y_is_neg);
    result = vbslq_f64(pos_inf_y_neg, zero, result);
    let pos_inf_y_pos = vandq_u64(x_is_pos_inf, y_is_pos);
    result = vbslq_f64(pos_inf_y_pos, inf, result);

    // --- pow(-∞, y) ---
    let neg_inf_y_neg = vandq_u64(x_is_neg_inf, y_is_neg);
    let neg_inf_y_neg_odd = vandq_u64(neg_inf_y_neg, y_is_odd_int);
    let neg_zero = vdupq_n_f64(-0.0);
    result = vbslq_f64(neg_inf_y_neg_odd, neg_zero, result);

    let neg_inf_y_neg_not_odd = vbicq_u64(neg_inf_y_neg, y_is_odd_int);
    result = vbslq_f64(neg_inf_y_neg_not_odd, zero, result);

    let neg_inf_y_pos = vandq_u64(x_is_neg_inf, y_is_pos);
    let neg_inf_y_pos_odd = vandq_u64(neg_inf_y_pos, y_is_odd_int);
    result = vbslq_f64(neg_inf_y_pos_odd, neg_inf, result);

    let neg_inf_y_pos_not_odd = vbicq_u64(neg_inf_y_pos, y_is_odd_int);
    result = vbslq_f64(neg_inf_y_pos_not_odd, inf, result);

    // --- pow(x, ±∞) ---
    let y_pos_inf_gt = vandq_u64(y_is_pos_inf, x_abs_gt_one);
    result = vbslq_f64(y_pos_inf_gt, inf, result);
    let y_pos_inf_lt = vandq_u64(y_is_pos_inf, x_abs_lt_one);
    result = vbslq_f64(y_pos_inf_lt, zero, result);

    let y_neg_inf_gt = vandq_u64(y_is_neg_inf, x_abs_gt_one);
    result = vbslq_f64(y_neg_inf_gt, zero, result);
    let y_neg_inf_lt = vandq_u64(y_is_neg_inf, x_abs_lt_one);
    result = vbslq_f64(y_neg_inf_lt, inf, result);

    // pow(-1, ±∞) = 1
    let neg_one_inf = vandq_u64(x_is_neg_one, y_is_inf);
    result = vbslq_f64(neg_one_inf, one, result);

    // --- Highest-priority rules (applied last so they win) ---

    // pow(x, ±0) = 1 for any x, including NaN
    result = vbslq_f64(y_is_zero, one, result);

    // pow(1, y) = 1 for any y, including NaN
    result = vbslq_f64(x_is_one, one, result);

    // NaN propagation (only when not overridden above)
    let x_nan_y_nonzero = vbicq_u64(x_is_nan, y_is_zero);
    result = vbslq_f64(x_nan_y_nonzero, nan, result);

    let y_nan_x_nonone = vbicq_u64(y_is_nan, x_is_one);
    result = vbslq_f64(y_nan_x_nonone, nan, result);

    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // f32 tests
    // =========================================================================

    const TOL_32: f32 = 5e-7;

    /// Scalar helper: broadcast val into all 4 lanes, compute pow, return lane 0.
    unsafe fn pow_scalar_32(x: f32, y: f32) -> f32 {
        unsafe {
            let vx = vdupq_n_f32(x);
            let vy = vdupq_n_f32(y);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vpow_f32(vx, vy));
            out[0]
        }
    }

    // ---- pow(x, 0) = 1 for all x ------------------------------------------

    #[test]
    fn pow_f32_x_to_zero_is_one() {
        for x in [0.0f32, -0.0, 1.0, -1.0, 42.0, f32::INFINITY, f32::NAN] {
            let r = unsafe { pow_scalar_32(x, 0.0) };
            assert_eq!(r, 1.0, "pow({x}, 0) = {r}, expected 1.0");
        }
    }

    #[test]
    fn pow_f32_x_to_neg_zero_is_one() {
        for x in [0.0f32, -1.0, 42.0, f32::NAN] {
            let r = unsafe { pow_scalar_32(x, -0.0) };
            assert_eq!(r, 1.0, "pow({x}, -0) = {r}, expected 1.0");
        }
    }

    // ---- pow(1, y) = 1 for all y -------------------------------------------

    #[test]
    fn pow_f32_one_to_any_is_one() {
        for y in [0.0f32, -0.0, 1.0, -1.0, 42.5, f32::INFINITY, f32::NAN] {
            let r = unsafe { pow_scalar_32(1.0, y) };
            assert_eq!(r, 1.0, "pow(1, {y}) = {r}, expected 1.0");
        }
    }

    // ---- Basic power computations ------------------------------------------

    #[test]
    fn pow_f32_basic_squares() {
        for x in [2.0f32, 3.0, 0.5, 10.0] {
            let r = unsafe { pow_scalar_32(x, 2.0) };
            let expected = x * x;
            assert!(
                (r - expected).abs() < TOL_32,
                "pow({x}, 2) = {r}, expected {expected}"
            );
        }
    }

    #[test]
    fn pow_f32_basic_cubes() {
        for x in [2.0f32, 3.0, 0.5] {
            let r = unsafe { pow_scalar_32(x, 3.0) };
            let expected = x * x * x;
            assert!(
                (r - expected).abs() < TOL_32 * expected.abs(),
                "pow({x}, 3) = {r}, expected {expected}"
            );
        }
    }

    #[test]
    fn pow_f32_fractional_exponent() {
        let r = unsafe { pow_scalar_32(4.0, 0.5) };
        assert!(
            (r - 2.0).abs() < TOL_32,
            "pow(4, 0.5) = {r}, expected 2.0"
        );
    }

    #[test]
    fn pow_f32_negative_exponent() {
        let r = unsafe { pow_scalar_32(2.0, -1.0) };
        assert!(
            (r - 0.5).abs() < TOL_32,
            "pow(2, -1) = {r}, expected 0.5"
        );
    }

    // ---- Negative bases with integer exponents -----------------------------

    #[test]
    fn pow_f32_neg_base_even_int() {
        let r = unsafe { pow_scalar_32(-2.0, 2.0) };
        assert!(
            (r - 4.0).abs() < TOL_32,
            "pow(-2, 2) = {r}, expected 4.0"
        );
    }

    #[test]
    fn pow_f32_neg_base_odd_int() {
        let r = unsafe { pow_scalar_32(-2.0, 3.0) };
        assert!(
            (r - (-8.0)).abs() < TOL_32,
            "pow(-2, 3) = {r}, expected -8.0"
        );
    }

    #[test]
    fn pow_f32_neg_base_non_int_is_nan() {
        let r = unsafe { pow_scalar_32(-2.0, 0.5) };
        assert!(r.is_nan(), "pow(-2, 0.5) = {r}, expected NaN");
    }

    // ---- Zero base ---------------------------------------------------------

    #[test]
    fn pow_f32_zero_to_positive() {
        let r = unsafe { pow_scalar_32(0.0, 3.0) };
        assert_eq!(r, 0.0, "pow(0, 3) = {r}, expected 0");
        assert!(r.is_sign_positive());
    }

    #[test]
    fn pow_f32_neg_zero_to_odd_positive() {
        let r = unsafe { pow_scalar_32(-0.0, 3.0) };
        assert_eq!(r, 0.0, "pow(-0, 3) = {r}, expected -0");
        assert!(r.is_sign_negative(), "pow(-0, 3) should be -0");
    }

    #[test]
    fn pow_f32_zero_to_negative_is_inf() {
        let r = unsafe { pow_scalar_32(0.0, -2.0) };
        assert!(r.is_infinite() && r.is_sign_positive(), "pow(0, -2) = {r}, expected +∞");
    }

    #[test]
    fn pow_f32_neg_zero_to_neg_odd_is_neg_inf() {
        let r = unsafe { pow_scalar_32(-0.0, -3.0) };
        assert!(
            r.is_infinite() && r.is_sign_negative(),
            "pow(-0, -3) = {r}, expected -∞"
        );
    }

    // ---- Infinity base -----------------------------------------------------

    #[test]
    fn pow_f32_pos_inf_to_positive() {
        let r = unsafe { pow_scalar_32(f32::INFINITY, 2.0) };
        assert_eq!(r, f32::INFINITY, "pow(+∞, 2) = {r}");
    }

    #[test]
    fn pow_f32_pos_inf_to_negative() {
        let r = unsafe { pow_scalar_32(f32::INFINITY, -2.0) };
        assert_eq!(r, 0.0, "pow(+∞, -2) = {r}");
    }

    #[test]
    fn pow_f32_neg_inf_to_pos_odd() {
        let r = unsafe { pow_scalar_32(f32::NEG_INFINITY, 3.0) };
        assert_eq!(r, f32::NEG_INFINITY, "pow(-∞, 3) = {r}");
    }

    #[test]
    fn pow_f32_neg_inf_to_pos_even() {
        let r = unsafe { pow_scalar_32(f32::NEG_INFINITY, 2.0) };
        assert_eq!(r, f32::INFINITY, "pow(-∞, 2) = {r}");
    }

    #[test]
    fn pow_f32_neg_inf_to_neg_odd() {
        let r = unsafe { pow_scalar_32(f32::NEG_INFINITY, -3.0) };
        assert_eq!(r, -0.0, "pow(-∞, -3) = {r}");
        assert!(r.is_sign_negative());
    }

    #[test]
    fn pow_f32_neg_inf_to_neg_even() {
        let r = unsafe { pow_scalar_32(f32::NEG_INFINITY, -2.0) };
        assert_eq!(r, 0.0, "pow(-∞, -2) = {r}");
        assert!(r.is_sign_positive());
    }

    // ---- Infinity exponent -------------------------------------------------

    #[test]
    fn pow_f32_large_base_to_pos_inf() {
        let r = unsafe { pow_scalar_32(2.0, f32::INFINITY) };
        assert_eq!(r, f32::INFINITY, "pow(2, +∞) = {r}");
    }

    #[test]
    fn pow_f32_frac_base_to_pos_inf() {
        let r = unsafe { pow_scalar_32(0.5, f32::INFINITY) };
        assert_eq!(r, 0.0, "pow(0.5, +∞) = {r}");
    }

    #[test]
    fn pow_f32_large_base_to_neg_inf() {
        let r = unsafe { pow_scalar_32(2.0, f32::NEG_INFINITY) };
        assert_eq!(r, 0.0, "pow(2, -∞) = {r}");
    }

    #[test]
    fn pow_f32_frac_base_to_neg_inf() {
        let r = unsafe { pow_scalar_32(0.5, f32::NEG_INFINITY) };
        assert_eq!(r, f32::INFINITY, "pow(0.5, -∞) = {r}");
    }

    #[test]
    fn pow_f32_neg_one_to_inf_is_one() {
        let r = unsafe { pow_scalar_32(-1.0, f32::INFINITY) };
        assert_eq!(r, 1.0, "pow(-1, +∞) = {r}");
    }

    #[test]
    fn pow_f32_neg_one_to_neg_inf_is_one() {
        let r = unsafe { pow_scalar_32(-1.0, f32::NEG_INFINITY) };
        assert_eq!(r, 1.0, "pow(-1, -∞) = {r}");
    }

    // ---- NaN propagation ---------------------------------------------------

    #[test]
    fn pow_f32_nan_base_nonzero_exp_is_nan() {
        let r = unsafe { pow_scalar_32(f32::NAN, 2.0) };
        assert!(r.is_nan(), "pow(NaN, 2) should be NaN");
    }

    #[test]
    fn pow_f32_nonone_base_nan_exp_is_nan() {
        let r = unsafe { pow_scalar_32(2.0, f32::NAN) };
        assert!(r.is_nan(), "pow(2, NaN) should be NaN");
    }

    #[test]
    fn pow_f32_nan_base_zero_exp_is_one() {
        let r = unsafe { pow_scalar_32(f32::NAN, 0.0) };
        assert_eq!(r, 1.0, "pow(NaN, 0) should be 1");
    }

    #[test]
    fn pow_f32_one_base_nan_exp_is_one() {
        let r = unsafe { pow_scalar_32(1.0, f32::NAN) };
        assert_eq!(r, 1.0, "pow(1, NaN) should be 1");
    }

    // ---- All lanes test ----------------------------------------------------

    #[test]
    fn pow_f32_processes_all_4_lanes() {
        let bases: [f32; 4] = [2.0, 3.0, 0.5, 10.0];
        let exps: [f32; 4] = [3.0, 2.0, -1.0, 0.5];
        unsafe {
            let vx = vld1q_f32(bases.as_ptr());
            let vy = vld1q_f32(exps.as_ptr());
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vpow_f32(vx, vy));

            for i in 0..4 {
                let expected = bases[i].powf(exps[i]);
                assert!(
                    (out[i] - expected).abs() < TOL_32 * expected.abs().max(1.0),
                    "lane {i}: pow({}, {}) = {}, expected {expected}",
                    bases[i],
                    exps[i],
                    out[i]
                );
            }
        }
    }

    // ---- ULP sweep test ----------------------------------------------------

    #[test]
    fn pow_f32_ulp_sweep_positive_bases() {
        let mut max_ulp: u32 = 0;
        let mut worst_x: f32 = 0.0;
        let mut worst_y: f32 = 0.0;

        // Sweep positive bases and various exponents
        let exponents = [0.5f32, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -2.0, 0.333];
        let mut bits: u32 = 0x3A800000; // start at ~0.001
        let end: u32 = 0x42C80000; // stop at ~100

        while bits < end {
            let x = f32::from_bits(bits);
            for &y in &exponents {
                let true_val = (x as f64).powf(y as f64) as f32;
                if true_val.is_finite() && true_val != 0.0 {
                    let our_val = unsafe { pow_scalar_32(x, y) };
                    if our_val.is_finite() {
                        let d =
                            (our_val.to_bits() as i32 - true_val.to_bits() as i32).unsigned_abs();
                        if d > max_ulp {
                            max_ulp = d;
                            worst_x = x;
                            worst_y = y;
                        }
                    }
                }
            }
            bits = bits.wrapping_add(1024);
        }
        assert!(
            max_ulp <= 2,
            "max ULP {max_ulp} at x={worst_x:.8}, y={worst_y:.8} — expected ≤ 2"
        );
    }

    // =========================================================================
    // f64 tests
    // =========================================================================

    const TOL_64: f64 = 1e-15;

    /// Scalar helper: broadcast val into all 2 lanes, compute pow, return lane 0.
    unsafe fn pow_scalar_64(x: f64, y: f64) -> f64 {
        unsafe {
            let vx = vdupq_n_f64(x);
            let vy = vdupq_n_f64(y);
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), vpow_f64(vx, vy));
            out[0]
        }
    }

    // ---- pow(x, 0) = 1 for all x ------------------------------------------

    #[test]
    fn pow_f64_x_to_zero_is_one() {
        for x in [0.0f64, -0.0, 1.0, -1.0, 42.0, f64::INFINITY, f64::NAN] {
            let r = unsafe { pow_scalar_64(x, 0.0) };
            assert_eq!(r, 1.0, "pow({x}, 0) = {r}, expected 1.0");
        }
    }

    // ---- pow(1, y) = 1 for all y -------------------------------------------

    #[test]
    fn pow_f64_one_to_any_is_one() {
        for y in [0.0f64, -0.0, 1.0, -1.0, 42.5, f64::INFINITY, f64::NAN] {
            let r = unsafe { pow_scalar_64(1.0, y) };
            assert_eq!(r, 1.0, "pow(1, {y}) = {r}, expected 1.0");
        }
    }

    // ---- Basic power computations ------------------------------------------

    #[test]
    fn pow_f64_basic_squares() {
        for x in [2.0f64, 3.0, 0.5, 10.0] {
            let r = unsafe { pow_scalar_64(x, 2.0) };
            let expected = x * x;
            assert!(
                (r - expected).abs() < TOL_64 * expected.abs(),
                "pow({x}, 2) = {r}, expected {expected}"
            );
        }
    }

    #[test]
    fn pow_f64_fractional_exponent() {
        let r = unsafe { pow_scalar_64(4.0, 0.5) };
        assert!(
            (r - 2.0).abs() < TOL_64,
            "pow(4, 0.5) = {r}, expected 2.0"
        );
    }

    // ---- Negative bases ----------------------------------------------------

    #[test]
    fn pow_f64_neg_base_even_int() {
        let r = unsafe { pow_scalar_64(-2.0, 2.0) };
        assert!(
            (r - 4.0).abs() < TOL_64,
            "pow(-2, 2) = {r}, expected 4.0"
        );
    }

    #[test]
    fn pow_f64_neg_base_odd_int() {
        let r = unsafe { pow_scalar_64(-2.0, 3.0) };
        assert!(
            (r - (-8.0)).abs() < TOL_64 * 8.0,
            "pow(-2, 3) = {r}, expected -8.0"
        );
    }

    #[test]
    fn pow_f64_neg_base_non_int_is_nan() {
        let r = unsafe { pow_scalar_64(-2.0, 0.5) };
        assert!(r.is_nan(), "pow(-2, 0.5) should be NaN");
    }

    // ---- Zero base ---------------------------------------------------------

    #[test]
    fn pow_f64_zero_to_positive() {
        let r = unsafe { pow_scalar_64(0.0, 3.0) };
        assert_eq!(r, 0.0);
        assert!(r.is_sign_positive());
    }

    #[test]
    fn pow_f64_neg_zero_to_odd_positive() {
        let r = unsafe { pow_scalar_64(-0.0, 3.0) };
        assert_eq!(r, 0.0);
        assert!(r.is_sign_negative(), "pow(-0, 3) should be -0");
    }

    #[test]
    fn pow_f64_zero_to_negative_is_inf() {
        let r = unsafe { pow_scalar_64(0.0, -2.0) };
        assert!(r.is_infinite() && r.is_sign_positive());
    }

    #[test]
    fn pow_f64_neg_zero_to_neg_odd_is_neg_inf() {
        let r = unsafe { pow_scalar_64(-0.0, -3.0) };
        assert!(r.is_infinite() && r.is_sign_negative());
    }

    // ---- Infinity base/exponent --------------------------------------------

    #[test]
    fn pow_f64_pos_inf_to_positive() {
        let r = unsafe { pow_scalar_64(f64::INFINITY, 2.0) };
        assert_eq!(r, f64::INFINITY);
    }

    #[test]
    fn pow_f64_pos_inf_to_negative() {
        let r = unsafe { pow_scalar_64(f64::INFINITY, -2.0) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn pow_f64_neg_inf_to_pos_odd() {
        let r = unsafe { pow_scalar_64(f64::NEG_INFINITY, 3.0) };
        assert_eq!(r, f64::NEG_INFINITY);
    }

    #[test]
    fn pow_f64_neg_inf_to_pos_even() {
        let r = unsafe { pow_scalar_64(f64::NEG_INFINITY, 2.0) };
        assert_eq!(r, f64::INFINITY);
    }

    #[test]
    fn pow_f64_large_base_to_pos_inf() {
        let r = unsafe { pow_scalar_64(2.0, f64::INFINITY) };
        assert_eq!(r, f64::INFINITY);
    }

    #[test]
    fn pow_f64_frac_base_to_pos_inf() {
        let r = unsafe { pow_scalar_64(0.5, f64::INFINITY) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn pow_f64_neg_one_to_inf_is_one() {
        let r = unsafe { pow_scalar_64(-1.0, f64::INFINITY) };
        assert_eq!(r, 1.0);
    }

    // ---- NaN ---------------------------------------------------------------

    #[test]
    fn pow_f64_nan_propagation() {
        let r = unsafe { pow_scalar_64(f64::NAN, 2.0) };
        assert!(r.is_nan());

        let r = unsafe { pow_scalar_64(2.0, f64::NAN) };
        assert!(r.is_nan());

        let r = unsafe { pow_scalar_64(f64::NAN, 0.0) };
        assert_eq!(r, 1.0);

        let r = unsafe { pow_scalar_64(1.0, f64::NAN) };
        assert_eq!(r, 1.0);
    }

    // ---- All lanes test ----------------------------------------------------

    #[test]
    fn pow_f64_processes_all_2_lanes() {
        let bases: [f64; 2] = [2.0, 0.5];
        let exps: [f64; 2] = [3.0, -1.0];
        unsafe {
            let vx = vld1q_f64(bases.as_ptr());
            let vy = vld1q_f64(exps.as_ptr());
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), vpow_f64(vx, vy));

            for i in 0..2 {
                let expected = bases[i].powf(exps[i]);
                assert!(
                    (out[i] - expected).abs() < TOL_64 * expected.abs().max(1.0),
                    "lane {i}: pow({}, {}) = {}, expected {expected}",
                    bases[i],
                    exps[i],
                    out[i]
                );
            }
        }
    }

    // ---- ULP sweep test ----------------------------------------------------

    #[test]
    fn pow_f64_ulp_sweep_positive_bases() {
        let mut max_ulp: u64 = 0;
        let mut worst_x: f64 = 0.0;
        let mut worst_y: f64 = 0.0;

        let exponents = [0.5f64, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -2.0, 0.333];
        let step: u64 = 1 << 42;
        let mut bits: u64 = 0x3F50000000000000; // ~0.001
        let end: u64 = 0x4059000000000000; // ~100

        while bits < end {
            let x = f64::from_bits(bits);
            for &y in &exponents {
                let true_val = x.powf(y);
                if true_val.is_finite() && true_val != 0.0 {
                    let our_val = unsafe { pow_scalar_64(x, y) };
                    if our_val.is_finite() {
                        let d =
                            (our_val.to_bits() as i64 - true_val.to_bits() as i64).unsigned_abs();
                        if d > max_ulp {
                            max_ulp = d;
                            worst_x = x;
                            worst_y = y;
                        }
                    }
                }
            }
            let (new_bits, overflow) = bits.overflowing_add(step);
            bits = new_bits;
            if overflow || bits >= end {
                break;
            }
        }
        assert!(
            max_ulp <= 2,
            "max ULP {max_ulp} at x={worst_x:.16}, y={worst_y:.16} — expected ≤ 2"
        );
    }
}
