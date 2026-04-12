//! AVX-512 SIMD implementation of `pow(x, y)` for `f32` and `f64` vectors.
//!
//! This module provides 16-lane f32 and 8-lane f64 power function implementations.
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
//! | Variant           | Max Error |
//! |-------------------|-----------|
//! | `_mm512_pow_ps`   | ≤ 2 ULP  |
//! | `_mm512_pow_pd`   | ≤ 2 ULP  |
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
//!
//! # Implementation Notes
//!
//! - **f32 uses f64 intermediates**: Each 16-lane f32 vector is split into two
//!   8-lane f64 vectors. The shared `pow_core_f64` kernel computes in f64 precision,
//!   then results are converted back and recombined.
//! - **AVX-512 masking**: Uses `__mmask8` integer masks and `_mm512_mask_blend_pd`
//!   instead of vector blends for predicated operations.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::exp::{
    LN2_HI_64 as EXP_LN2_HI, LN2_INV_64, LN2_LO_64 as EXP_LN2_LO, OVERFLOW_THRESH_64,
    P1_64, P2_64, P3_64, P4_64, P5_64, UNDERFLOW_THRESH_64,
};
use crate::arch::consts::ln::{
    LG1_64, LG2_64, LG3_64, LG4_64, LG5_64, LG6_64, LG7_64, LN2_HI_64, LN2_LO_64, SQRT2_64,
    TWO52_64,
};

// =============================================================================
// f32 Implementation (16 lanes, via f64 promotion)
// =============================================================================

/// Computes `x^y` (power) for each lane of an AVX-512 `__m512` register.
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
/// Requires AVX-512F support. The caller must ensure this feature is
/// available at runtime.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_pow_ps(x: __m512, y: __m512) -> __m512 {
    unsafe {
        // Split 16-lane f32 into two 8-lane f64 halves
        let x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(x));
        let x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(x, 1));
        let y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(y));
        let y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(y, 1));

        // Compute pow in f64 precision
        let pow_lo = pow_core_f64(x_lo, y_lo);
        let pow_hi = pow_core_f64(x_hi, y_hi);

        // Convert back to f32 and combine
        let result_lo = _mm512_cvtpd_ps(pow_lo);
        let result_hi = _mm512_cvtpd_ps(pow_hi);

        _mm512_insertf32x8(_mm512_castps256_ps512(result_lo), result_hi, 1)
    }
}

// =============================================================================
// f64 Implementation (8 lanes)
// =============================================================================

/// Computes `x^y` (power) for each lane of an AVX-512 `__m512d` register.
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
/// Requires AVX-512F support. The caller must ensure this feature is
/// available at runtime.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_pow_pd(x: __m512d, y: __m512d) -> __m512d {
    unsafe { pow_core_f64(x, y) }
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
/// Uses the same fdlibm polynomial as the ln kernel, but splits
/// the final reconstruction using Knuth's 2Sum to capture the rounding error.
///
/// # Safety
///
/// Only valid for positive inputs. Zero, negative, infinity, and NaN inputs
/// produce unspecified results (the caller must handle those cases).
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn ln_hilo(x: __m512d) -> (__m512d, __m512d) {
    let one = _mm512_set1_pd(1.0);
        let half = _mm512_set1_pd(0.5);
        let zero = _mm512_setzero_pd();

        // =================================================================
        // Step 1: Handle subnormals by scaling up
        // =================================================================

        let min_normal = _mm512_set1_pd(f64::MIN_POSITIVE);
        let is_subnormal = _mm512_cmp_pd_mask(x, zero, _CMP_GT_OQ)
            & _mm512_cmp_pd_mask(x, min_normal, _CMP_LT_OQ);

        let two52 = _mm512_set1_pd(TWO52_64);
        let x_scaled = _mm512_mul_pd(x, two52);
        let x_work = _mm512_mask_blend_pd(is_subnormal, x, x_scaled);

        let neg_52 = _mm512_set1_pd(-52.0);
        let k_adjust = _mm512_mask_blend_pd(is_subnormal, zero, neg_52);

        // =================================================================
        // Step 2: Extract exponent k and normalize mantissa to [1, 2)
        // =================================================================

        let ix = _mm512_castpd_si512(x_work);

        // Extract biased exponent: bits [52:62] → shift right 52
        let exp_bits = _mm512_srli_epi64(ix, 52);

        // Pack 8×i64 exponents to 8×i32 for conversion to f64
        let exp_i32 = _mm512_cvtepi64_epi32(exp_bits);

        // Convert to f64: k = biased_exponent - 1023
        let bias = _mm512_set1_pd(1023.0);
        let k = _mm512_sub_pd(_mm512_cvtepi32_pd(exp_i32), bias);
        let k = _mm512_add_pd(k, k_adjust);

        // Normalize mantissa: clear exponent, set to biased 1023 → [1, 2)
        let mantissa_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFF_u64 as i64);
        let exp_1023 = _mm512_set1_epi64(0x3FF0000000000000_u64 as i64);
        let m_bits = _mm512_or_si512(_mm512_and_si512(ix, mantissa_mask), exp_1023);
        let m = _mm512_castsi512_pd(m_bits);

        // If m > √2, halve it and increment k
        let sqrt2 = _mm512_set1_pd(SQRT2_64);
        let is_big = _mm512_cmp_pd_mask(m, sqrt2, _CMP_GT_OQ);

        let exp_1022 = _mm512_set1_epi64(0x3FE0000000000000_u64 as i64);
        let m_halved_bits = _mm512_or_si512(_mm512_and_si512(ix, mantissa_mask), exp_1022);
        let m_halved = _mm512_castsi512_pd(m_halved_bits);

        let m = _mm512_mask_blend_pd(is_big, m, m_halved);
        let k = _mm512_mask_blend_pd(is_big, k, _mm512_add_pd(k, one));

        // =================================================================
        // Step 3: f = m - 1, s = f / (2 + f)
        // =================================================================

        let f = _mm512_sub_pd(m, one);
        let two_plus_f = _mm512_add_pd(_mm512_set1_pd(2.0), f);
        let s = _mm512_div_pd(f, two_plus_f);
        let hfsq = _mm512_mul_pd(half, _mm512_mul_pd(f, f));

        // =================================================================
        // Step 4: Minimax polynomial R(z) where z = s²
        // =================================================================

        let z = _mm512_mul_pd(s, s);
        let w = _mm512_mul_pd(z, z);

        // Odd powers:  t1 = Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7))
        let t1 = _mm512_fmadd_pd(w, _mm512_set1_pd(LG7_64), _mm512_set1_pd(LG5_64));
        let t1 = _mm512_fmadd_pd(w, t1, _mm512_set1_pd(LG3_64));
        let t1 = _mm512_fmadd_pd(w, t1, _mm512_set1_pd(LG1_64));

        // Even powers: t2 = Lg2 + w*(Lg4 + w*Lg6)
        let t2 = _mm512_fmadd_pd(w, _mm512_set1_pd(LG6_64), _mm512_set1_pd(LG4_64));
        let t2 = _mm512_fmadd_pd(w, t2, _mm512_set1_pd(LG2_64));

        // R = z * (t1 + z*t2)
        let r = _mm512_fmadd_pd(z, t2, t1);
        let r = _mm512_mul_pd(z, r);

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

        let ln2_hi = _mm512_set1_pd(LN2_HI_64);
        let ln2_lo = _mm512_set1_pd(LN2_LO_64);

        // val_hi = f - hfsq (may round)
        let val_hi = _mm512_sub_pd(f, hfsq);

        // val_lo recovers the rounding error of (f - hfsq) and adds the remaining terms
        let val_lo = _mm512_sub_pd(f, val_hi);
        let val_lo = _mm512_sub_pd(val_lo, hfsq);
        let s_term = _mm512_mul_pd(s, _mm512_add_pd(hfsq, r));
        let val_lo = _mm512_add_pd(val_lo, s_term);
        let val_lo = _mm512_fmadd_pd(k, ln2_lo, val_lo);

        // Knuth 2Sum: hi = val_hi + k*ln2_hi, capturing the rounding error
        let k_ln2_hi = _mm512_mul_pd(k, ln2_hi);
        let hi = _mm512_add_pd(val_hi, k_ln2_hi);
        let b_virt = _mm512_sub_pd(hi, val_hi);
        let a_virt = _mm512_sub_pd(hi, b_virt);
        let b_err = _mm512_sub_pd(k_ln2_hi, b_virt);
        let a_err = _mm512_sub_pd(val_hi, a_virt);
        let lo = _mm512_add_pd(_mm512_add_pd(a_err, b_err), val_lo);

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
/// Requires AVX-512F support. The caller must ensure this feature is
/// available at runtime.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn exp_compensated(ehi: __m512d, elo: __m512d) -> __m512d {
    unsafe {
        let zero = _mm512_setzero_pd();
        let one = _mm512_set1_pd(1.0);
        let two = _mm512_set1_pd(2.0);
        let half = _mm512_set1_pd(0.5);
        let sign_bit = _mm512_set1_pd(-0.0);

        // =================================================================
        // Overflow / underflow detection (on ehi, the dominant term)
        // =================================================================

        let inf = _mm512_set1_pd(f64::INFINITY);
        let overflow_thresh = _mm512_set1_pd(OVERFLOW_THRESH_64);
        let underflow_thresh = _mm512_set1_pd(UNDERFLOW_THRESH_64);

        let is_overflow = _mm512_cmp_pd_mask(ehi, overflow_thresh, _CMP_GT_OQ);
        let is_underflow = _mm512_cmp_pd_mask(ehi, underflow_thresh, _CMP_LT_OQ);
        let is_nan = _mm512_cmp_pd_mask(ehi, ehi, _CMP_UNORD_Q);

        // =================================================================
        // Argument reduction: ehi = k·ln2 + r, then r += elo
        // =================================================================

        let ln2_inv = _mm512_set1_pd(LN2_INV_64);
        let ln2_hi = _mm512_set1_pd(EXP_LN2_HI);
        let ln2_lo = _mm512_set1_pd(EXP_LN2_LO);

        // k = round(ehi / ln2) — copysign(0.5, ehi) trick
        let ehi_sign = _mm512_castpd_si512(_mm512_and_pd(ehi, sign_bit));
        let half_bits = _mm512_castpd_si512(half);
        let sign_half = _mm512_castsi512_pd(_mm512_or_si512(half_bits, ehi_sign));
        let k_f64 = _mm512_roundscale_pd(
            _mm512_fmadd_pd(ehi, ln2_inv, sign_half),
            _MM_FROUND_TO_ZERO,
        );
        let k_i32 = _mm512_cvtpd_epi32(k_f64);

        // r = ehi - k·ln2_hi - k·ln2_lo + elo (extended-precision reduction with tail)
        let r = _mm512_fnmadd_pd(k_f64, ln2_lo, _mm512_fnmadd_pd(k_f64, ln2_hi, ehi));
        let r = _mm512_add_pd(r, elo);

        // =================================================================
        // Polynomial: exp(r) ≈ 1 + r + r·c/(2-c)
        // where c = r - r²·P(r²)
        // =================================================================

        let r2 = _mm512_mul_pd(r, r);

        let p = _mm512_fmadd_pd(r2, _mm512_set1_pd(P5_64), _mm512_set1_pd(P4_64));
        let p = _mm512_fmadd_pd(r2, p, _mm512_set1_pd(P3_64));
        let p = _mm512_fmadd_pd(r2, p, _mm512_set1_pd(P2_64));
        let p = _mm512_fmadd_pd(r2, p, _mm512_set1_pd(P1_64));

        let c = _mm512_sub_pd(r, _mm512_mul_pd(r2, p));

        let rc = _mm512_mul_pd(r, c);
        let exp_r = _mm512_sub_pd(
            one,
            _mm512_sub_pd(_mm512_div_pd(rc, _mm512_sub_pd(c, two)), r),
        );

        // =================================================================
        // Reconstruct: exp(ehi+elo) = 2^k · exp(r+elo)
        // =================================================================

        let k_i64 = _mm512_cvtepi32_epi64(k_i32);
        let k_shifted = _mm512_slli_epi64(k_i64, 52);
        let one_bits = _mm512_set1_epi64(0x3FF0000000000000_u64 as i64);
        let scale = _mm512_castsi512_pd(_mm512_add_epi64(k_shifted, one_bits));

        let result = _mm512_mul_pd(exp_r, scale);

        // =================================================================
        // Apply overflow / underflow / NaN
        // =================================================================

        let nan = _mm512_set1_pd(f64::NAN);
        let result = _mm512_mask_blend_pd(is_overflow, result, inf);
        let result = _mm512_mask_blend_pd(is_underflow, result, zero);
        _mm512_mask_blend_pd(is_nan, result, nan)
    }
}

// =============================================================================
// Core f64 pow kernel (shared by both f32 and f64 paths)
// =============================================================================

/// Core power kernel operating on 8-lane f64 vectors (AVX-512).
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
#[target_feature(enable = "avx512f")]
unsafe fn pow_core_f64(x: __m512d, y: __m512d) -> __m512d {
    unsafe {
        let zero = _mm512_setzero_pd();
        let one = _mm512_set1_pd(1.0);
        let neg_one = _mm512_set1_pd(-1.0);
        let inf = _mm512_set1_pd(f64::INFINITY);
        let neg_inf = _mm512_set1_pd(f64::NEG_INFINITY);
        let nan = _mm512_set1_pd(f64::NAN);
        let half = _mm512_set1_pd(0.5);
        let sign_bit = _mm512_set1_pd(-0.0);

        // =====================================================================
        // Phase 1: Classify inputs
        // =====================================================================

        let x_abs = _mm512_abs_pd(x);

        let x_is_nan = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
        let x_is_zero = _mm512_cmp_pd_mask(x, zero, _CMP_EQ_OQ);
        let x_is_one = _mm512_cmp_pd_mask(x, one, _CMP_EQ_OQ);
        let x_is_neg = _mm512_cmp_pd_mask(x, zero, _CMP_LT_OQ);
        let x_is_neg_one = _mm512_cmp_pd_mask(x, neg_one, _CMP_EQ_OQ);
        let x_is_pos_inf = _mm512_cmp_pd_mask(x, inf, _CMP_EQ_OQ);
        let x_is_neg_inf = _mm512_cmp_pd_mask(x, neg_inf, _CMP_EQ_OQ);
        let x_sign = _mm512_and_pd(x, sign_bit);

        let x_abs_lt_one = _mm512_cmp_pd_mask(x_abs, one, _CMP_LT_OQ);
        let x_abs_gt_one = _mm512_cmp_pd_mask(x_abs, one, _CMP_GT_OQ);

        let y_is_nan = _mm512_cmp_pd_mask(y, y, _CMP_UNORD_Q);
        let y_is_zero = _mm512_cmp_pd_mask(y, zero, _CMP_EQ_OQ);
        let y_is_pos = _mm512_cmp_pd_mask(y, zero, _CMP_GT_OQ);
        let y_is_neg = _mm512_cmp_pd_mask(y, zero, _CMP_LT_OQ);
        let y_is_pos_inf = _mm512_cmp_pd_mask(y, inf, _CMP_EQ_OQ);
        let y_is_neg_inf = _mm512_cmp_pd_mask(y, neg_inf, _CMP_EQ_OQ);
        let y_is_inf = y_is_pos_inf | y_is_neg_inf;

        let y_trunc = _mm512_roundscale_pd(y, _MM_FROUND_TO_ZERO);
        let y_is_integer = _mm512_cmp_pd_mask(y, y_trunc, _CMP_EQ_OQ);

        let y_half = _mm512_mul_pd(y, half);
        let y_half_trunc = _mm512_roundscale_pd(y_half, _MM_FROUND_TO_ZERO);
        let y_half_is_int = _mm512_cmp_pd_mask(y_half, y_half_trunc, _CMP_EQ_OQ);
        let y_is_odd_int = !y_half_is_int & y_is_integer;

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
        let ehi = _mm512_mul_pd(y, ln_hi);
        let elo = _mm512_fmadd_pd(y, ln_lo, _mm512_fmsub_pd(y, ln_hi, ehi));

        // Step 2c: exp(ehi + elo) with compensation
        let mut result = exp_compensated(ehi, elo);

        // =====================================================================
        // Phase 3: Apply IEEE 754 special cases (order matters!)
        // =====================================================================

        // --- Negative-base sign correction ---
        let should_negate = x_is_neg & y_is_odd_int;
        let neg_result = _mm512_castsi512_pd(_mm512_xor_si512(
            _mm512_castpd_si512(result),
            _mm512_castpd_si512(sign_bit),
        ));
        result = _mm512_mask_blend_pd(should_negate, result, neg_result);

        // Negative base with non-integer exponent → NaN (but not -∞)
        let neg_base_non_int = !y_is_integer & x_is_neg;
        let neg_base_non_int = !x_is_neg_inf & neg_base_non_int;
        result = _mm512_mask_blend_pd(neg_base_non_int, result, nan);

        // --- pow(±0, y) ---
        let x_zero_y_neg = x_is_zero & y_is_neg;
        let x_zero_y_neg_odd = x_zero_y_neg & y_is_odd_int;
        let signed_inf = _mm512_castsi512_pd(_mm512_or_si512(
            _mm512_castpd_si512(inf),
            _mm512_castpd_si512(x_sign),
        ));
        result = _mm512_mask_blend_pd(x_zero_y_neg_odd, result, signed_inf);

        let x_zero_y_neg_not_odd = !y_is_odd_int & x_zero_y_neg;
        result = _mm512_mask_blend_pd(x_zero_y_neg_not_odd, result, inf);

        let x_zero_y_pos = x_is_zero & y_is_pos;
        let x_zero_y_pos_odd = x_zero_y_pos & y_is_odd_int;
        let signed_zero = _mm512_castsi512_pd(_mm512_or_si512(
            _mm512_castpd_si512(zero),
            _mm512_castpd_si512(x_sign),
        ));
        result = _mm512_mask_blend_pd(x_zero_y_pos_odd, result, signed_zero);

        let x_zero_y_pos_not_odd = !y_is_odd_int & x_zero_y_pos;
        result = _mm512_mask_blend_pd(x_zero_y_pos_not_odd, result, zero);

        // --- pow(+∞, y) ---
        let pos_inf_y_neg = x_is_pos_inf & y_is_neg;
        result = _mm512_mask_blend_pd(pos_inf_y_neg, result, zero);
        let pos_inf_y_pos = x_is_pos_inf & y_is_pos;
        result = _mm512_mask_blend_pd(pos_inf_y_pos, result, inf);

        // --- pow(-∞, y) ---
        let neg_inf_y_neg = x_is_neg_inf & y_is_neg;
        let neg_inf_y_neg_odd = neg_inf_y_neg & y_is_odd_int;
        let neg_zero = _mm512_set1_pd(-0.0);
        result = _mm512_mask_blend_pd(neg_inf_y_neg_odd, result, neg_zero);

        let neg_inf_y_neg_not_odd = !y_is_odd_int & neg_inf_y_neg;
        result = _mm512_mask_blend_pd(neg_inf_y_neg_not_odd, result, zero);

        let neg_inf_y_pos = x_is_neg_inf & y_is_pos;
        let neg_inf_y_pos_odd = neg_inf_y_pos & y_is_odd_int;
        result = _mm512_mask_blend_pd(neg_inf_y_pos_odd, result, neg_inf);

        let neg_inf_y_pos_not_odd = !y_is_odd_int & neg_inf_y_pos;
        result = _mm512_mask_blend_pd(neg_inf_y_pos_not_odd, result, inf);

        // --- pow(x, ±∞) ---
        let y_pos_inf_gt = y_is_pos_inf & x_abs_gt_one;
        result = _mm512_mask_blend_pd(y_pos_inf_gt, result, inf);
        let y_pos_inf_lt = y_is_pos_inf & x_abs_lt_one;
        result = _mm512_mask_blend_pd(y_pos_inf_lt, result, zero);

        let y_neg_inf_gt = y_is_neg_inf & x_abs_gt_one;
        result = _mm512_mask_blend_pd(y_neg_inf_gt, result, zero);
        let y_neg_inf_lt = y_is_neg_inf & x_abs_lt_one;
        result = _mm512_mask_blend_pd(y_neg_inf_lt, result, inf);

        // pow(-1, ±∞) = 1
        let neg_one_inf = x_is_neg_one & y_is_inf;
        result = _mm512_mask_blend_pd(neg_one_inf, result, one);

        // --- Highest-priority rules (applied last so they win) ---

        // pow(x, ±0) = 1 for any x, including NaN
        result = _mm512_mask_blend_pd(y_is_zero, result, one);

        // pow(1, y) = 1 for any y, including NaN
        result = _mm512_mask_blend_pd(x_is_one, result, one);

        // NaN propagation (only when not overridden above)
        let x_nan_y_nonzero = !y_is_zero & x_is_nan;
        result = _mm512_mask_blend_pd(x_nan_y_nonzero, result, nan);

        let y_nan_x_nonone = !x_is_one & y_is_nan;
        result = _mm512_mask_blend_pd(y_nan_x_nonone, result, nan);

        result
    }
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

    /// Scalar helper: broadcast val into all 16 lanes, compute pow, return lane 0.
    unsafe fn pow_scalar_32(x: f32, y: f32) -> f32 {
        unsafe {
            let vx = _mm512_set1_ps(x);
            let vy = _mm512_set1_ps(y);
            let mut out = [0.0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), _mm512_pow_ps(vx, vy));
            out[0]
        }
    }

    // ---- pow(x, 0) = 1 for all x ------------------------------------------

    #[test]
    fn pow_ps_x_to_zero_is_one() {
        for x in [0.0f32, -0.0, 1.0, -1.0, 42.0, f32::INFINITY, f32::NAN] {
            let r = unsafe { pow_scalar_32(x, 0.0) };
            assert_eq!(r, 1.0, "pow({x}, 0) = {r}, expected 1.0");
        }
    }

    #[test]
    fn pow_ps_x_to_neg_zero_is_one() {
        for x in [0.0f32, -1.0, 42.0, f32::NAN] {
            let r = unsafe { pow_scalar_32(x, -0.0) };
            assert_eq!(r, 1.0, "pow({x}, -0) = {r}, expected 1.0");
        }
    }

    // ---- pow(1, y) = 1 for all y -------------------------------------------

    #[test]
    fn pow_ps_one_to_any_is_one() {
        for y in [0.0f32, -0.0, 1.0, -1.0, 42.5, f32::INFINITY, f32::NAN] {
            let r = unsafe { pow_scalar_32(1.0, y) };
            assert_eq!(r, 1.0, "pow(1, {y}) = {r}, expected 1.0");
        }
    }

    // ---- Basic power computations ------------------------------------------

    #[test]
    fn pow_ps_basic_squares() {
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
    fn pow_ps_basic_cubes() {
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
    fn pow_ps_fractional_exponent() {
        let r = unsafe { pow_scalar_32(4.0, 0.5) };
        assert!(
            (r - 2.0).abs() < TOL_32,
            "pow(4, 0.5) = {r}, expected 2.0"
        );
    }

    #[test]
    fn pow_ps_negative_exponent() {
        let r = unsafe { pow_scalar_32(2.0, -1.0) };
        assert!(
            (r - 0.5).abs() < TOL_32,
            "pow(2, -1) = {r}, expected 0.5"
        );
    }

    // ---- Negative bases with integer exponents -----------------------------

    #[test]
    fn pow_ps_neg_base_even_int() {
        let r = unsafe { pow_scalar_32(-2.0, 2.0) };
        assert!(
            (r - 4.0).abs() < TOL_32,
            "pow(-2, 2) = {r}, expected 4.0"
        );
    }

    #[test]
    fn pow_ps_neg_base_odd_int() {
        let r = unsafe { pow_scalar_32(-2.0, 3.0) };
        assert!(
            (r - (-8.0)).abs() < TOL_32,
            "pow(-2, 3) = {r}, expected -8.0"
        );
    }

    #[test]
    fn pow_ps_neg_base_non_int_is_nan() {
        let r = unsafe { pow_scalar_32(-2.0, 0.5) };
        assert!(r.is_nan(), "pow(-2, 0.5) = {r}, expected NaN");
    }

    // ---- Zero base ---------------------------------------------------------

    #[test]
    fn pow_ps_zero_to_positive() {
        let r = unsafe { pow_scalar_32(0.0, 3.0) };
        assert_eq!(r, 0.0, "pow(0, 3) = {r}, expected 0");
        assert!(r.is_sign_positive());
    }

    #[test]
    fn pow_ps_neg_zero_to_odd_positive() {
        let r = unsafe { pow_scalar_32(-0.0, 3.0) };
        assert_eq!(r, 0.0, "pow(-0, 3) = {r}, expected -0");
        assert!(r.is_sign_negative(), "pow(-0, 3) should be -0");
    }

    #[test]
    fn pow_ps_zero_to_negative_is_inf() {
        let r = unsafe { pow_scalar_32(0.0, -2.0) };
        assert!(r.is_infinite() && r.is_sign_positive(), "pow(0, -2) = {r}, expected +∞");
    }

    #[test]
    fn pow_ps_neg_zero_to_neg_odd_is_neg_inf() {
        let r = unsafe { pow_scalar_32(-0.0, -3.0) };
        assert!(
            r.is_infinite() && r.is_sign_negative(),
            "pow(-0, -3) = {r}, expected -∞"
        );
    }

    // ---- Infinity base -----------------------------------------------------

    #[test]
    fn pow_ps_pos_inf_to_positive() {
        let r = unsafe { pow_scalar_32(f32::INFINITY, 2.0) };
        assert_eq!(r, f32::INFINITY, "pow(+∞, 2) = {r}");
    }

    #[test]
    fn pow_ps_pos_inf_to_negative() {
        let r = unsafe { pow_scalar_32(f32::INFINITY, -2.0) };
        assert_eq!(r, 0.0, "pow(+∞, -2) = {r}");
    }

    #[test]
    fn pow_ps_neg_inf_to_pos_odd() {
        let r = unsafe { pow_scalar_32(f32::NEG_INFINITY, 3.0) };
        assert_eq!(r, f32::NEG_INFINITY, "pow(-∞, 3) = {r}");
    }

    #[test]
    fn pow_ps_neg_inf_to_pos_even() {
        let r = unsafe { pow_scalar_32(f32::NEG_INFINITY, 2.0) };
        assert_eq!(r, f32::INFINITY, "pow(-∞, 2) = {r}");
    }

    #[test]
    fn pow_ps_neg_inf_to_neg_odd() {
        let r = unsafe { pow_scalar_32(f32::NEG_INFINITY, -3.0) };
        assert_eq!(r, -0.0, "pow(-∞, -3) = {r}");
        assert!(r.is_sign_negative());
    }

    #[test]
    fn pow_ps_neg_inf_to_neg_even() {
        let r = unsafe { pow_scalar_32(f32::NEG_INFINITY, -2.0) };
        assert_eq!(r, 0.0, "pow(-∞, -2) = {r}");
        assert!(r.is_sign_positive());
    }

    // ---- Infinity exponent -------------------------------------------------

    #[test]
    fn pow_ps_large_base_to_pos_inf() {
        let r = unsafe { pow_scalar_32(2.0, f32::INFINITY) };
        assert_eq!(r, f32::INFINITY, "pow(2, +∞) = {r}");
    }

    #[test]
    fn pow_ps_frac_base_to_pos_inf() {
        let r = unsafe { pow_scalar_32(0.5, f32::INFINITY) };
        assert_eq!(r, 0.0, "pow(0.5, +∞) = {r}");
    }

    #[test]
    fn pow_ps_large_base_to_neg_inf() {
        let r = unsafe { pow_scalar_32(2.0, f32::NEG_INFINITY) };
        assert_eq!(r, 0.0, "pow(2, -∞) = {r}");
    }

    #[test]
    fn pow_ps_frac_base_to_neg_inf() {
        let r = unsafe { pow_scalar_32(0.5, f32::NEG_INFINITY) };
        assert_eq!(r, f32::INFINITY, "pow(0.5, -∞) = {r}");
    }

    #[test]
    fn pow_ps_neg_one_to_inf_is_one() {
        let r = unsafe { pow_scalar_32(-1.0, f32::INFINITY) };
        assert_eq!(r, 1.0, "pow(-1, +∞) = {r}");
    }

    #[test]
    fn pow_ps_neg_one_to_neg_inf_is_one() {
        let r = unsafe { pow_scalar_32(-1.0, f32::NEG_INFINITY) };
        assert_eq!(r, 1.0, "pow(-1, -∞) = {r}");
    }

    // ---- NaN propagation ---------------------------------------------------

    #[test]
    fn pow_ps_nan_base_nonzero_exp_is_nan() {
        let r = unsafe { pow_scalar_32(f32::NAN, 2.0) };
        assert!(r.is_nan(), "pow(NaN, 2) should be NaN");
    }

    #[test]
    fn pow_ps_nonone_base_nan_exp_is_nan() {
        let r = unsafe { pow_scalar_32(2.0, f32::NAN) };
        assert!(r.is_nan(), "pow(2, NaN) should be NaN");
    }

    #[test]
    fn pow_ps_nan_base_zero_exp_is_one() {
        let r = unsafe { pow_scalar_32(f32::NAN, 0.0) };
        assert_eq!(r, 1.0, "pow(NaN, 0) should be 1");
    }

    #[test]
    fn pow_ps_one_base_nan_exp_is_one() {
        let r = unsafe { pow_scalar_32(1.0, f32::NAN) };
        assert_eq!(r, 1.0, "pow(1, NaN) should be 1");
    }

    // ---- All lanes test ----------------------------------------------------

    #[test]
    fn pow_ps_processes_all_16_lanes() {
        let bases: [f32; 16] = [
            1.0, 2.0, 3.0, 4.0, 0.5, 10.0, 100.0, 0.1,
            5.0, 7.0, 0.25, 8.0, 1.5, 9.0, 0.01, 6.0,
        ];
        let exps: [f32; 16] = [
            5.0, 3.0, 2.0, 0.5, 2.0, -1.0, 0.5, 3.0,
            2.0, 0.5, -1.0, 1.0, 3.0, -0.5, 2.0, 1.5,
        ];
        unsafe {
            let vx = _mm512_loadu_ps(bases.as_ptr());
            let vy = _mm512_loadu_ps(exps.as_ptr());
            let mut out = [0.0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), _mm512_pow_ps(vx, vy));

            for i in 0..16 {
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
    fn pow_ps_ulp_sweep_positive_bases() {
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

    /// Scalar helper: broadcast val into all 8 lanes, compute pow, return lane 0.
    unsafe fn pow_scalar_64(x: f64, y: f64) -> f64 {
        unsafe {
            let vx = _mm512_set1_pd(x);
            let vy = _mm512_set1_pd(y);
            let mut out = [0.0f64; 8];
            _mm512_storeu_pd(out.as_mut_ptr(), _mm512_pow_pd(vx, vy));
            out[0]
        }
    }

    // ---- pow(x, 0) = 1 for all x ------------------------------------------

    #[test]
    fn pow_pd_x_to_zero_is_one() {
        for x in [0.0f64, -0.0, 1.0, -1.0, 42.0, f64::INFINITY, f64::NAN] {
            let r = unsafe { pow_scalar_64(x, 0.0) };
            assert_eq!(r, 1.0, "pow({x}, 0) = {r}, expected 1.0");
        }
    }

    // ---- pow(1, y) = 1 for all y -------------------------------------------

    #[test]
    fn pow_pd_one_to_any_is_one() {
        for y in [0.0f64, -0.0, 1.0, -1.0, 42.5, f64::INFINITY, f64::NAN] {
            let r = unsafe { pow_scalar_64(1.0, y) };
            assert_eq!(r, 1.0, "pow(1, {y}) = {r}, expected 1.0");
        }
    }

    // ---- Basic power computations ------------------------------------------

    #[test]
    fn pow_pd_basic_squares() {
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
    fn pow_pd_fractional_exponent() {
        let r = unsafe { pow_scalar_64(4.0, 0.5) };
        assert!(
            (r - 2.0).abs() < TOL_64,
            "pow(4, 0.5) = {r}, expected 2.0"
        );
    }

    // ---- Negative bases ----------------------------------------------------

    #[test]
    fn pow_pd_neg_base_even_int() {
        let r = unsafe { pow_scalar_64(-2.0, 2.0) };
        assert!(
            (r - 4.0).abs() < TOL_64,
            "pow(-2, 2) = {r}, expected 4.0"
        );
    }

    #[test]
    fn pow_pd_neg_base_odd_int() {
        let r = unsafe { pow_scalar_64(-2.0, 3.0) };
        assert!(
            (r - (-8.0)).abs() < TOL_64 * 8.0,
            "pow(-2, 3) = {r}, expected -8.0"
        );
    }

    #[test]
    fn pow_pd_neg_base_non_int_is_nan() {
        let r = unsafe { pow_scalar_64(-2.0, 0.5) };
        assert!(r.is_nan(), "pow(-2, 0.5) should be NaN");
    }

    // ---- Zero base ---------------------------------------------------------

    #[test]
    fn pow_pd_zero_to_positive() {
        let r = unsafe { pow_scalar_64(0.0, 3.0) };
        assert_eq!(r, 0.0);
        assert!(r.is_sign_positive());
    }

    #[test]
    fn pow_pd_neg_zero_to_odd_positive() {
        let r = unsafe { pow_scalar_64(-0.0, 3.0) };
        assert_eq!(r, 0.0);
        assert!(r.is_sign_negative(), "pow(-0, 3) should be -0");
    }

    #[test]
    fn pow_pd_zero_to_negative_is_inf() {
        let r = unsafe { pow_scalar_64(0.0, -2.0) };
        assert!(r.is_infinite() && r.is_sign_positive());
    }

    #[test]
    fn pow_pd_neg_zero_to_neg_odd_is_neg_inf() {
        let r = unsafe { pow_scalar_64(-0.0, -3.0) };
        assert!(r.is_infinite() && r.is_sign_negative());
    }

    // ---- Infinity base/exponent --------------------------------------------

    #[test]
    fn pow_pd_pos_inf_to_positive() {
        let r = unsafe { pow_scalar_64(f64::INFINITY, 2.0) };
        assert_eq!(r, f64::INFINITY);
    }

    #[test]
    fn pow_pd_pos_inf_to_negative() {
        let r = unsafe { pow_scalar_64(f64::INFINITY, -2.0) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn pow_pd_neg_inf_to_pos_odd() {
        let r = unsafe { pow_scalar_64(f64::NEG_INFINITY, 3.0) };
        assert_eq!(r, f64::NEG_INFINITY);
    }

    #[test]
    fn pow_pd_neg_inf_to_pos_even() {
        let r = unsafe { pow_scalar_64(f64::NEG_INFINITY, 2.0) };
        assert_eq!(r, f64::INFINITY);
    }

    #[test]
    fn pow_pd_large_base_to_pos_inf() {
        let r = unsafe { pow_scalar_64(2.0, f64::INFINITY) };
        assert_eq!(r, f64::INFINITY);
    }

    #[test]
    fn pow_pd_frac_base_to_pos_inf() {
        let r = unsafe { pow_scalar_64(0.5, f64::INFINITY) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn pow_pd_neg_one_to_inf_is_one() {
        let r = unsafe { pow_scalar_64(-1.0, f64::INFINITY) };
        assert_eq!(r, 1.0);
    }

    // ---- NaN ---------------------------------------------------------------

    #[test]
    fn pow_pd_nan_propagation() {
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
    fn pow_pd_processes_all_8_lanes() {
        let bases: [f64; 8] = [2.0, 3.0, 0.5, 10.0, 4.0, 7.0, 0.25, 5.0];
        let exps: [f64; 8] = [3.0, 2.0, -1.0, 0.5, 0.5, 1.5, -2.0, 3.0];
        unsafe {
            let vx = _mm512_loadu_pd(bases.as_ptr());
            let vy = _mm512_loadu_pd(exps.as_ptr());
            let mut out = [0.0f64; 8];
            _mm512_storeu_pd(out.as_mut_ptr(), _mm512_pow_pd(vx, vy));

            for i in 0..8 {
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
    fn pow_pd_ulp_sweep_positive_bases() {
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
