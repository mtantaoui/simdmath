//! AVX2 SIMD implementation of `pow(x, y)` for `f32` and `f64` vectors.
//!
//! This module provides 8-lane f32 and 4-lane f64 power function implementations.
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
//! | `_mm256_pow_ps`   | ≤ 2 ULP  |
//! | `_mm256_pow_pd`   | ≤ 2 ULP  |
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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::exp::{
    LN2_HI_64 as EXP_LN2_HI, LN2_INV_64, LN2_LO_64 as EXP_LN2_LO, OVERFLOW_THRESH_64, P1_64, P2_64,
    P3_64, P4_64, P5_64, UNDERFLOW_THRESH_64,
};
use crate::arch::consts::ln::{
    LG1_64, LG2_64, LG3_64, LG4_64, LG5_64, LG6_64, LG7_64, LN2_HI_64, LN2_LO_64, SQRT2_64,
    TWO52_64,
};

// =============================================================================
// f32 Implementation (8 lanes, via f64 promotion)
// =============================================================================

/// Computes `x^y` (power) for each lane of an AVX2 `__m256` register.
///
/// Promotes `f32` inputs to `f64` and runs the plain (non-compensated) f64
/// `ln → mul → exp` chain per half: the absolute error of `y·ln|x|` computed
/// in f64 is ~|y·ln|x||·2⁻⁵² ≤ 89·2⁻⁵², orders of magnitude below half an
/// f32 ULP of the result, so the double-double machinery the f64 path needs
/// is skipped. The IEEE special-case cascade then runs once on the 8 f32
/// lanes instead of twice on the f64 halves.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_pow_ps(x: __m256, y: __m256) -> __m256 {
    unsafe {
        let x_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

        // Split 8-lane f32 into two 4-lane f64 halves
        let x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x_abs));
        let x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x_abs, 1));
        let y_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(y));
        let y_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(y, 1));

        // Raw exp(y·ln|x|) in f64 precision (no special-case handling)
        let raw_lo = pow_raw_f64::<false>(x_lo, y_lo);
        let raw_hi = pow_raw_f64::<false>(x_hi, y_hi);

        // Narrow back to f32 and combine
        let raw = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm256_cvtpd_ps(raw_lo)),
            _mm256_cvtpd_ps(raw_hi),
            1,
        );

        // IEEE 754 special cases, applied once across all 8 lanes
        pow_special_cases_ps(x, y, raw)
    }
}

// =============================================================================
// f64 Implementation (4 lanes)
// =============================================================================

/// Computes `x^y` (power) for each lane of an AVX2 `__m256d` register.
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
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_pow_pd(x: __m256d, y: __m256d) -> __m256d {
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
/// Uses the same fdlibm polynomial as [`ln_core_f64`](super::ln), but splits
/// the final reconstruction using Knuth's 2Sum to capture the rounding error.
///
/// When `PRECISE` is `false` the subnormal pre-scaling and the 2Sum split are
/// skipped: the returned `lo` is zero and `hi` carries the plain (~1 ULP f64)
/// logarithm. This is only valid when the caller guarantees `x` was promoted
/// from `f32` (an f32 subnormal is still a normal f64) and only needs f32
/// output accuracy, where an f64 ULP of slack is invisible.
///
/// # Safety
///
/// Only valid for positive inputs. Zero, negative, infinity, and NaN inputs
/// produce unspecified results (the caller must handle those cases).
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn ln_hilo<const PRECISE: bool>(x: __m256d) -> (__m256d, __m256d) {
    {
        let one = _mm256_set1_pd(1.0);
        let half = _mm256_set1_pd(0.5);
        let zero = _mm256_setzero_pd();

        // =================================================================
        // Step 1: Handle subnormals by scaling up
        // =================================================================

        let (x_work, k_adjust) = if PRECISE {
            let min_normal = _mm256_set1_pd(f64::MIN_POSITIVE);
            let is_subnormal = _mm256_and_pd(
                _mm256_cmp_pd(x, zero, _CMP_GT_OQ),
                _mm256_cmp_pd(x, min_normal, _CMP_LT_OQ),
            );

            let two52 = _mm256_set1_pd(TWO52_64);
            let x_scaled = _mm256_mul_pd(x, two52);
            let x_work = _mm256_blendv_pd(x, x_scaled, is_subnormal);

            let k_adjust = _mm256_and_pd(
                _mm256_set1_pd(f64::from_bits((-52.0_f64).to_bits())),
                is_subnormal,
            );
            (x_work, k_adjust)
        } else {
            // Inputs promoted from f32 are never subnormal in f64.
            (x, zero)
        };

        // =================================================================
        // Step 2: Extract exponent k and normalize mantissa to [1, 2)
        // =================================================================

        let ix = _mm256_castpd_si256(x_work);

        let exp_bits = _mm256_srli_epi64(ix, 52);
        let pack_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
        let exp_i32 = _mm256_permutevar8x32_epi32(exp_bits, pack_idx);

        let bias = _mm256_set1_pd(1023.0);
        let k = _mm256_sub_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(exp_i32)), bias);
        let k = if PRECISE {
            _mm256_add_pd(k, k_adjust)
        } else {
            k
        };

        // Normalize mantissa: clear exponent, set to biased 1023 → [1, 2)
        let mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFF_u64 as i64);
        let exp_1023 = _mm256_set1_epi64x(0x3FF0000000000000_u64 as i64);
        let m_bits = _mm256_or_si256(_mm256_and_si256(ix, mantissa_mask), exp_1023);
        let m = _mm256_castsi256_pd(m_bits);

        // If m > √2, halve it and increment k
        let sqrt2 = _mm256_set1_pd(SQRT2_64);
        let is_big = _mm256_cmp_pd(m, sqrt2, _CMP_GT_OQ);

        let exp_1022 = _mm256_set1_epi64x(0x3FE0000000000000_u64 as i64);
        let m_halved_bits = _mm256_or_si256(_mm256_and_si256(ix, mantissa_mask), exp_1022);
        let m_halved = _mm256_castsi256_pd(m_halved_bits);

        let m = _mm256_blendv_pd(m, m_halved, is_big);
        let k = _mm256_add_pd(k, _mm256_and_pd(one, is_big));

        // =================================================================
        // Step 3: f = m - 1, s = f / (2 + f)
        // =================================================================

        let f = _mm256_sub_pd(m, one);
        let two_plus_f = _mm256_add_pd(_mm256_set1_pd(2.0), f);
        let s = _mm256_div_pd(f, two_plus_f);
        let hfsq = _mm256_mul_pd(half, _mm256_mul_pd(f, f));

        // =================================================================
        // Step 4: Minimax polynomial R(z) where z = s²
        // =================================================================

        let z = _mm256_mul_pd(s, s);
        let w = _mm256_mul_pd(z, z);

        // Odd powers:  t1 = Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7))
        let t1 = _mm256_fmadd_pd(w, _mm256_set1_pd(LG7_64), _mm256_set1_pd(LG5_64));
        let t1 = _mm256_fmadd_pd(w, t1, _mm256_set1_pd(LG3_64));
        let t1 = _mm256_fmadd_pd(w, t1, _mm256_set1_pd(LG1_64));

        // Even powers: t2 = Lg2 + w*(Lg4 + w*Lg6)
        let t2 = _mm256_fmadd_pd(w, _mm256_set1_pd(LG6_64), _mm256_set1_pd(LG4_64));
        let t2 = _mm256_fmadd_pd(w, t2, _mm256_set1_pd(LG2_64));

        // R = z * (t1 + z*t2)
        let r = _mm256_fmadd_pd(z, t2, t1);
        let r = _mm256_mul_pd(z, r);

        // =================================================================
        // Step 5: Split result into (hi, lo) using 2Sum
        //
        // ln(x) = f - hfsq + s*(hfsq+R) + k*ln2_hi + k*ln2_lo
        //
        // We split into:
        //   val_hi = f - hfsq
        //   val_lo = (f - val_hi - hfsq) + s*(hfsq+R) + k*ln2_lo
        //          ≈ rounding error of (f-hfsq)  +  polynomial  +  k*ln2_lo
        //
        // Then combine with k*ln2_hi using Knuth's 2Sum:
        //   hi = val_hi + k*ln2_hi   (rounded)
        //   lo = (rounding error of that addition) + val_lo
        // =================================================================

        let ln2_hi = _mm256_set1_pd(LN2_HI_64);
        let ln2_lo = _mm256_set1_pd(LN2_LO_64);

        // val_hi = f - hfsq (may round)
        let val_hi = _mm256_sub_pd(f, hfsq);

        // val_lo recovers the rounding error of (f - hfsq) and adds the remaining terms
        // val_lo = (f - val_hi) - hfsq + s*(hfsq+R) + k*ln2_lo
        let val_lo = _mm256_sub_pd(f, val_hi);
        let val_lo = _mm256_sub_pd(val_lo, hfsq);
        let s_term = _mm256_mul_pd(s, _mm256_add_pd(hfsq, r));
        let val_lo = _mm256_add_pd(val_lo, s_term);
        let val_lo = _mm256_fmadd_pd(k, ln2_lo, val_lo);

        let k_ln2_hi = _mm256_mul_pd(k, ln2_hi);

        if PRECISE {
            // Knuth 2Sum: hi = val_hi + k*ln2_hi, capturing the rounding error
            let hi = _mm256_add_pd(val_hi, k_ln2_hi);
            let b_virt = _mm256_sub_pd(hi, val_hi);
            let a_virt = _mm256_sub_pd(hi, b_virt);
            let b_err = _mm256_sub_pd(k_ln2_hi, b_virt);
            let a_err = _mm256_sub_pd(val_hi, a_virt);
            let lo = _mm256_add_pd(_mm256_add_pd(a_err, b_err), val_lo);

            (hi, lo)
        } else {
            // Plain reconstruction (~1 f64 ULP), ample for f32 output accuracy.
            let hi = _mm256_add_pd(_mm256_add_pd(val_hi, val_lo), k_ln2_hi);
            (hi, zero)
        }
    }
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
/// When `PRECISE` is `false`, `elo` is ignored (the caller passes zero) and
/// the fold-in is skipped — used by the f32 path, which doesn't need the
/// compensation tail.
///
/// # Safety
///
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_compensated<const PRECISE: bool>(ehi: __m256d, elo: __m256d) -> __m256d {
    {
        let zero = _mm256_setzero_pd();
        let one = _mm256_set1_pd(1.0);
        let two = _mm256_set1_pd(2.0);
        let half = _mm256_set1_pd(0.5);
        let sign_bit = _mm256_set1_pd(-0.0);

        // =================================================================
        // Overflow / underflow detection (on ehi, the dominant term)
        // =================================================================

        let inf = _mm256_set1_pd(f64::INFINITY);
        let overflow_thresh = _mm256_set1_pd(OVERFLOW_THRESH_64);
        let underflow_thresh = _mm256_set1_pd(UNDERFLOW_THRESH_64);

        let is_overflow = _mm256_cmp_pd(ehi, overflow_thresh, _CMP_GT_OQ);
        let is_underflow = _mm256_cmp_pd(ehi, underflow_thresh, _CMP_LT_OQ);
        let is_nan = _mm256_cmp_pd(ehi, ehi, _CMP_UNORD_Q);

        // =================================================================
        // Argument reduction: ehi = k·ln2 + r, then r += elo
        // =================================================================

        let ln2_inv = _mm256_set1_pd(LN2_INV_64);
        let ln2_hi = _mm256_set1_pd(EXP_LN2_HI);
        let ln2_lo = _mm256_set1_pd(EXP_LN2_LO);

        // k = round(ehi / ln2)
        let sign_half = _mm256_or_pd(half, _mm256_and_pd(ehi, sign_bit));
        let k_f64 = _mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(_mm256_fmadd_pd(
            ehi, ln2_inv, sign_half,
        ));
        let k_i32 = _mm256_cvtpd_epi32(k_f64);

        // r = ehi - k·ln2_hi - k·ln2_lo + elo (extended-precision reduction with tail)
        let r = _mm256_fnmadd_pd(k_f64, ln2_lo, _mm256_fnmadd_pd(k_f64, ln2_hi, ehi));
        let r = if PRECISE { _mm256_add_pd(r, elo) } else { r };

        // =================================================================
        // Polynomial: exp(r) ≈ 1 + r + r·c/(2-c)
        // where c = r - r²·P(r²)
        // =================================================================

        let exp_r = if PRECISE {
            let r2 = _mm256_mul_pd(r, r);

            let p = _mm256_fmadd_pd(r2, _mm256_set1_pd(P5_64), _mm256_set1_pd(P4_64));
            let p = _mm256_fmadd_pd(r2, p, _mm256_set1_pd(P3_64));
            let p = _mm256_fmadd_pd(r2, p, _mm256_set1_pd(P2_64));
            let p = _mm256_fmadd_pd(r2, p, _mm256_set1_pd(P1_64));

            let c = _mm256_sub_pd(r, _mm256_mul_pd(r2, p));

            let rc = _mm256_mul_pd(r, c);
            _mm256_sub_pd(
                one,
                _mm256_sub_pd(_mm256_div_pd(rc, _mm256_sub_pd(c, two)), r),
            )
        } else {
            // Division-free degree-10 Taylor polynomial (coefficients 1/k!
            // are exact compile-time constants). On |r| ≤ ln2/2 truncation is
            // ≤ 2⁻⁴¹ relative — invisible at the f32 output precision this
            // path serves — and it avoids the Padé form's vdivpd.
            //
            // Split into even/odd halves in r² so the two Horner chains run
            // in parallel: exp(r) = E(r²) + r·O(r²).
            let r2 = _mm256_mul_pd(r, r);

            // E(z) = 1 + z/2 + z²/24 + z³/720 + z⁴/40320 + z⁵/3628800
            let e = _mm256_set1_pd(1.0 / 3_628_800.0); // 1/10!
            let e = _mm256_fmadd_pd(e, r2, _mm256_set1_pd(1.0 / 40_320.0)); // 1/8!
            let e = _mm256_fmadd_pd(e, r2, _mm256_set1_pd(1.0 / 720.0)); // 1/6!
            let e = _mm256_fmadd_pd(e, r2, _mm256_set1_pd(1.0 / 24.0)); // 1/4!
            let e = _mm256_fmadd_pd(e, r2, half); // 1/2!
            let e = _mm256_fmadd_pd(e, r2, one);

            // O(z) = 1 + z/6 + z²/120 + z³/5040 + z⁴/362880
            let o = _mm256_set1_pd(1.0 / 362_880.0); // 1/9!
            let o = _mm256_fmadd_pd(o, r2, _mm256_set1_pd(1.0 / 5_040.0)); // 1/7!
            let o = _mm256_fmadd_pd(o, r2, _mm256_set1_pd(1.0 / 120.0)); // 1/5!
            let o = _mm256_fmadd_pd(o, r2, _mm256_set1_pd(1.0 / 6.0)); // 1/3!
            let o = _mm256_fmadd_pd(o, r2, one);

            // exp(r) = E + r·O
            _mm256_fmadd_pd(r, o, e)
        };

        // =================================================================
        // Reconstruct: exp(ehi+elo) = 2^k · exp(r+elo)
        // =================================================================

        let k_i64 = _mm256_cvtepi32_epi64(k_i32);
        let k_shifted = _mm256_slli_epi64(k_i64, 52);
        let one_bits = _mm256_set1_epi64x(0x3FF0000000000000_u64 as i64);
        let scale = _mm256_castsi256_pd(_mm256_add_epi64(k_shifted, one_bits));

        let result = _mm256_mul_pd(exp_r, scale);

        // =================================================================
        // Apply overflow / underflow / NaN
        // =================================================================

        let nan = _mm256_set1_pd(f64::NAN);
        let result = _mm256_blendv_pd(result, inf, is_overflow);
        let result = _mm256_blendv_pd(result, zero, is_underflow);
        _mm256_blendv_pd(result, nan, is_nan)
    }
}

// =============================================================================
// Raw pow: exp(y · ln(x_abs)) without special-case handling
// =============================================================================

/// Computes the raw `exp(y · ln(x_abs))` on 4 f64 lanes; the caller applies
/// the IEEE 754 special-case rules.
///
/// `PRECISE` selects the compensated path (hi/lo log, Dekker product, tail
/// folded into exp) required for ≤2 ULP at f64 output. The plain path
/// (`PRECISE = false`) serves f32 output computed in f64, where a couple of
/// f64 ULPs of argument error are invisible.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn pow_raw_f64<const PRECISE: bool>(x_abs: __m256d, y: __m256d) -> __m256d {
    unsafe {
        // For non-positive x, ln_hilo produces garbage — those lanes are
        // overwritten by the caller's special-case handling.
        let (ln_hi, ln_lo) = ln_hilo::<PRECISE>(x_abs);

        if PRECISE {
            // Dekker multiplication using FMA
            // ehi = y * ln_hi                       (rounded product)
            // elo = fma(y, ln_hi, -ehi) + y * ln_lo (exact error + low-order term)
            let ehi = _mm256_mul_pd(y, ln_hi);
            let elo = _mm256_fmadd_pd(y, ln_lo, _mm256_fmsub_pd(y, ln_hi, ehi));
            exp_compensated::<true>(ehi, elo)
        } else {
            // ln_lo is zero here; passed through only to satisfy the signature.
            exp_compensated::<false>(_mm256_mul_pd(y, ln_hi), ln_lo)
        }
    }
}

// =============================================================================
// Core f64 pow kernel (f64 path)
// =============================================================================

/// Core power kernel operating on 4-lane f64 vectors.
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
#[target_feature(enable = "avx2,fma")]
unsafe fn pow_core_f64(x: __m256d, y: __m256d) -> __m256d {
    unsafe {
        let zero = _mm256_setzero_pd();
        let one = _mm256_set1_pd(1.0);
        let neg_one = _mm256_set1_pd(-1.0);
        let inf = _mm256_set1_pd(f64::INFINITY);
        let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
        let nan = _mm256_set1_pd(f64::NAN);
        let half = _mm256_set1_pd(0.5);
        let sign_bit = _mm256_set1_pd(-0.0);

        // =====================================================================
        // Phase 1: Classify inputs
        // =====================================================================

        let x_abs = _mm256_andnot_pd(sign_bit, x);

        let x_is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
        let x_is_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
        let x_is_one = _mm256_cmp_pd(x, one, _CMP_EQ_OQ);
        let x_is_neg = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
        let x_is_neg_one = _mm256_cmp_pd(x, neg_one, _CMP_EQ_OQ);
        let x_is_pos_inf = _mm256_cmp_pd(x, inf, _CMP_EQ_OQ);
        let x_is_neg_inf = _mm256_cmp_pd(x, neg_inf, _CMP_EQ_OQ);
        let x_sign = _mm256_and_pd(x, sign_bit);

        let x_abs_lt_one = _mm256_cmp_pd(x_abs, one, _CMP_LT_OQ);
        let x_abs_gt_one = _mm256_cmp_pd(x_abs, one, _CMP_GT_OQ);

        let y_is_nan = _mm256_cmp_pd(y, y, _CMP_UNORD_Q);
        let y_is_zero = _mm256_cmp_pd(y, zero, _CMP_EQ_OQ);
        let y_is_pos = _mm256_cmp_pd(y, zero, _CMP_GT_OQ);
        let y_is_neg = _mm256_cmp_pd(y, zero, _CMP_LT_OQ);
        let y_is_pos_inf = _mm256_cmp_pd(y, inf, _CMP_EQ_OQ);
        let y_is_neg_inf = _mm256_cmp_pd(y, neg_inf, _CMP_EQ_OQ);
        let y_is_inf = _mm256_or_pd(y_is_pos_inf, y_is_neg_inf);

        let y_trunc = _mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(y);
        let y_is_integer = _mm256_cmp_pd(y, y_trunc, _CMP_EQ_OQ);

        let y_half = _mm256_mul_pd(y, half);
        let y_half_trunc = _mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(y_half);
        let y_half_is_int = _mm256_cmp_pd(y_half, y_half_trunc, _CMP_EQ_OQ);
        let y_is_odd_int = _mm256_andnot_pd(y_half_is_int, y_is_integer);

        // =====================================================================
        // Phase 2: Compensated exp(y · ln(|x|))
        // =====================================================================

        let mut result = pow_raw_f64::<true>(x_abs, y);

        // =====================================================================
        // Phase 3: Apply IEEE 754 special cases (compressed cascade)
        //
        // Every special-case rule produces one of four output categories:
        //   • +inf / +0 (sign-agnostic)
        //   • signed inf / signed zero (sign copied from x)
        //   • NaN
        //   • 1.0
        //
        // Category masks are built in parallel (no cross-dependency), then
        // applied to `result` in priority order with only 4 serial blends
        // (sign-correction, ±inf/±0, NaN, 1.0) — vs. ~20 in the original
        // serial cascade. Independent value-side blends build `special_val`
        // and run in parallel with the mask computation.
        // =====================================================================

        // --- 1. Sign-correct the raw result for x<0 & y odd-integer ---
        let should_negate = _mm256_and_pd(x_is_neg, y_is_odd_int);
        let neg_result = _mm256_xor_pd(result, sign_bit);
        result = _mm256_blendv_pd(result, neg_result, should_negate);

        // --- 2. Build category masks (all independent / parallel) ---

        // mask_inf_pos: output = +∞
        //   pow(0, neg-even), pow(+∞, pos), pow(-∞, pos-even),
        //   pow(|x|>1, +∞), pow(|x|<1, -∞)
        let mask_inf_pos = _mm256_or_pd(
            _mm256_or_pd(
                _mm256_andnot_pd(y_is_odd_int, _mm256_and_pd(x_is_zero, y_is_neg)),
                _mm256_and_pd(x_is_pos_inf, y_is_pos),
            ),
            _mm256_or_pd(
                _mm256_andnot_pd(y_is_odd_int, _mm256_and_pd(x_is_neg_inf, y_is_pos)),
                _mm256_or_pd(
                    _mm256_and_pd(y_is_pos_inf, x_abs_gt_one),
                    _mm256_and_pd(y_is_neg_inf, x_abs_lt_one),
                ),
            ),
        );

        // mask_zero_pos: output = +0
        //   pow(0, pos-even), pow(+∞, neg), pow(-∞, neg-even),
        //   pow(|x|<1, +∞), pow(|x|>1, -∞)
        let mask_zero_pos = _mm256_or_pd(
            _mm256_or_pd(
                _mm256_andnot_pd(y_is_odd_int, _mm256_and_pd(x_is_zero, y_is_pos)),
                _mm256_and_pd(x_is_pos_inf, y_is_neg),
            ),
            _mm256_or_pd(
                _mm256_andnot_pd(y_is_odd_int, _mm256_and_pd(x_is_neg_inf, y_is_neg)),
                _mm256_or_pd(
                    _mm256_and_pd(y_is_pos_inf, x_abs_lt_one),
                    _mm256_and_pd(y_is_neg_inf, x_abs_gt_one),
                ),
            ),
        );

        // mask_inf_signed: output = inf | sign(x)
        //   pow(±0, neg-odd) = ±∞
        //   pow(-∞, pos-odd) = -∞   (x_sign is negative for x=-∞)
        let mask_inf_signed = _mm256_or_pd(
            _mm256_and_pd(_mm256_and_pd(x_is_zero, y_is_neg), y_is_odd_int),
            _mm256_and_pd(_mm256_and_pd(x_is_neg_inf, y_is_pos), y_is_odd_int),
        );

        // mask_zero_signed: output = zero | sign(x)
        //   pow(±0, pos-odd) = ±0
        //   pow(-∞, neg-odd) = -0   (x_sign is negative for x=-∞)
        let mask_zero_signed = _mm256_or_pd(
            _mm256_and_pd(_mm256_and_pd(x_is_zero, y_is_pos), y_is_odd_int),
            _mm256_and_pd(_mm256_and_pd(x_is_neg_inf, y_is_neg), y_is_odd_int),
        );

        // mask_nan: output = NaN
        //   x<0 with non-integer y (excluding x=-∞, which has IEEE rules)
        //   pow(NaN, y) with y!=0
        //   pow(x, NaN) with x!=1
        let neg_base_non_int =
            _mm256_andnot_pd(x_is_neg_inf, _mm256_andnot_pd(y_is_integer, x_is_neg));
        let x_nan_y_nonzero = _mm256_andnot_pd(y_is_zero, x_is_nan);
        let y_nan_x_nonone = _mm256_andnot_pd(x_is_one, y_is_nan);
        let mask_nan = _mm256_or_pd(
            neg_base_non_int,
            _mm256_or_pd(x_nan_y_nonzero, y_nan_x_nonone),
        );

        // mask_one: output = 1.0   (highest priority — applied last)
        //   pow(-1, ±∞), pow(x, ±0), pow(1, y)
        let neg_one_inf = _mm256_and_pd(x_is_neg_one, y_is_inf);
        let mask_one = _mm256_or_pd(neg_one_inf, _mm256_or_pd(y_is_zero, x_is_one));

        // --- 3. Build the signed value vectors (independent of cascade) ---
        let signed_inf = _mm256_or_pd(inf, x_sign);
        let signed_zero = _mm256_or_pd(zero, x_sign);

        // --- 4. Merge the four magnitude categories into one `special_val`
        //        via independent blends (parallel with the mask construction) ---
        let inf_val = _mm256_blendv_pd(inf, signed_inf, mask_inf_signed);
        let zero_val = _mm256_blendv_pd(zero, signed_zero, mask_zero_signed);
        let mask_inf_any = _mm256_or_pd(mask_inf_pos, mask_inf_signed);
        let mask_zero_any = _mm256_or_pd(mask_zero_pos, mask_zero_signed);
        let mask_special = _mm256_or_pd(mask_inf_any, mask_zero_any);
        let special_val = _mm256_blendv_pd(zero_val, inf_val, mask_inf_any);

        // --- 5. Apply in priority order — only 3 more serial blends ---
        result = _mm256_blendv_pd(result, special_val, mask_special);
        result = _mm256_blendv_pd(result, nan, mask_nan);
        result = _mm256_blendv_pd(result, one, mask_one);

        result
    }
}

// =============================================================================
// IEEE 754 special-case cascade in the f32 domain (8 lanes at once)
// =============================================================================

/// Applies the IEEE 754 / C99 `pow` special-case rules to `raw` in the f32
/// domain. `raw` holds `exp(y·ln|x|)` narrowed to f32 — correct for ordinary
/// positive bases, garbage in lanes a special rule covers.
///
/// This is the f32 twin of Phase 1 + Phase 3 of [`pow_core_f64`]; running it
/// once on the 8-wide f32 vectors costs half of running it per f64 half.
/// See [`pow_core_f64`] for the rule-by-rule commentary.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn pow_special_cases_ps(x: __m256, y: __m256, raw: __m256) -> __m256 {
    {
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let neg_one = _mm256_set1_ps(-1.0);
        let inf = _mm256_set1_ps(f32::INFINITY);
        let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
        let nan = _mm256_set1_ps(f32::NAN);
        let half = _mm256_set1_ps(0.5);
        let sign_bit = _mm256_set1_ps(-0.0);

        // --- Classify inputs (Phase 1 twin) ---
        let x_abs = _mm256_andnot_ps(sign_bit, x);

        let x_is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
        let x_is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
        let x_is_one = _mm256_cmp_ps(x, one, _CMP_EQ_OQ);
        let x_is_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
        let x_is_neg_one = _mm256_cmp_ps(x, neg_one, _CMP_EQ_OQ);
        let x_is_pos_inf = _mm256_cmp_ps(x, inf, _CMP_EQ_OQ);
        let x_is_neg_inf = _mm256_cmp_ps(x, neg_inf, _CMP_EQ_OQ);
        let x_sign = _mm256_and_ps(x, sign_bit);

        let x_abs_lt_one = _mm256_cmp_ps(x_abs, one, _CMP_LT_OQ);
        let x_abs_gt_one = _mm256_cmp_ps(x_abs, one, _CMP_GT_OQ);

        let y_is_nan = _mm256_cmp_ps(y, y, _CMP_UNORD_Q);
        let y_is_zero = _mm256_cmp_ps(y, zero, _CMP_EQ_OQ);
        let y_is_pos = _mm256_cmp_ps(y, zero, _CMP_GT_OQ);
        let y_is_neg = _mm256_cmp_ps(y, zero, _CMP_LT_OQ);
        let y_is_pos_inf = _mm256_cmp_ps(y, inf, _CMP_EQ_OQ);
        let y_is_neg_inf = _mm256_cmp_ps(y, neg_inf, _CMP_EQ_OQ);
        let y_is_inf = _mm256_or_ps(y_is_pos_inf, y_is_neg_inf);

        let y_trunc = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(y);
        let y_is_integer = _mm256_cmp_ps(y, y_trunc, _CMP_EQ_OQ);

        let y_half = _mm256_mul_ps(y, half);
        let y_half_trunc = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(y_half);
        let y_half_is_int = _mm256_cmp_ps(y_half, y_half_trunc, _CMP_EQ_OQ);
        let y_is_odd_int = _mm256_andnot_ps(y_half_is_int, y_is_integer);

        // --- 1. Sign-correct the raw result for x<0 & y odd-integer ---
        let should_negate = _mm256_and_ps(x_is_neg, y_is_odd_int);
        let mut result = _mm256_xor_ps(raw, _mm256_and_ps(should_negate, sign_bit));

        // --- 2. Build category masks (all independent / parallel) ---

        // mask_inf_pos: output = +∞
        let mask_inf_pos = _mm256_or_ps(
            _mm256_or_ps(
                _mm256_andnot_ps(y_is_odd_int, _mm256_and_ps(x_is_zero, y_is_neg)),
                _mm256_and_ps(x_is_pos_inf, y_is_pos),
            ),
            _mm256_or_ps(
                _mm256_andnot_ps(y_is_odd_int, _mm256_and_ps(x_is_neg_inf, y_is_pos)),
                _mm256_or_ps(
                    _mm256_and_ps(y_is_pos_inf, x_abs_gt_one),
                    _mm256_and_ps(y_is_neg_inf, x_abs_lt_one),
                ),
            ),
        );

        // mask_zero_pos: output = +0
        let mask_zero_pos = _mm256_or_ps(
            _mm256_or_ps(
                _mm256_andnot_ps(y_is_odd_int, _mm256_and_ps(x_is_zero, y_is_pos)),
                _mm256_and_ps(x_is_pos_inf, y_is_neg),
            ),
            _mm256_or_ps(
                _mm256_andnot_ps(y_is_odd_int, _mm256_and_ps(x_is_neg_inf, y_is_neg)),
                _mm256_or_ps(
                    _mm256_and_ps(y_is_pos_inf, x_abs_lt_one),
                    _mm256_and_ps(y_is_neg_inf, x_abs_gt_one),
                ),
            ),
        );

        // mask_inf_signed: output = inf | sign(x)
        let mask_inf_signed = _mm256_or_ps(
            _mm256_and_ps(_mm256_and_ps(x_is_zero, y_is_neg), y_is_odd_int),
            _mm256_and_ps(_mm256_and_ps(x_is_neg_inf, y_is_pos), y_is_odd_int),
        );

        // mask_zero_signed: output = zero | sign(x)
        let mask_zero_signed = _mm256_or_ps(
            _mm256_and_ps(_mm256_and_ps(x_is_zero, y_is_pos), y_is_odd_int),
            _mm256_and_ps(_mm256_and_ps(x_is_neg_inf, y_is_neg), y_is_odd_int),
        );

        // mask_nan: output = NaN
        let neg_base_non_int =
            _mm256_andnot_ps(x_is_neg_inf, _mm256_andnot_ps(y_is_integer, x_is_neg));
        let x_nan_y_nonzero = _mm256_andnot_ps(y_is_zero, x_is_nan);
        let y_nan_x_nonone = _mm256_andnot_ps(x_is_one, y_is_nan);
        let mask_nan = _mm256_or_ps(
            neg_base_non_int,
            _mm256_or_ps(x_nan_y_nonzero, y_nan_x_nonone),
        );

        // mask_one: output = 1.0   (highest priority — applied last)
        let neg_one_inf = _mm256_and_ps(x_is_neg_one, y_is_inf);
        let mask_one = _mm256_or_ps(neg_one_inf, _mm256_or_ps(y_is_zero, x_is_one));

        // --- 3. Build the signed value vectors (independent of cascade) ---
        let signed_inf = _mm256_or_ps(inf, x_sign);
        let signed_zero = _mm256_or_ps(zero, x_sign);

        // --- 4. Merge the four magnitude categories into one `special_val` ---
        let inf_val = _mm256_blendv_ps(inf, signed_inf, mask_inf_signed);
        let zero_val = _mm256_blendv_ps(zero, signed_zero, mask_zero_signed);
        let mask_inf_any = _mm256_or_ps(mask_inf_pos, mask_inf_signed);
        let mask_zero_any = _mm256_or_ps(mask_zero_pos, mask_zero_signed);
        let mask_special = _mm256_or_ps(mask_inf_any, mask_zero_any);
        let special_val = _mm256_blendv_ps(zero_val, inf_val, mask_inf_any);

        // --- 5. Apply in priority order — only 3 more serial blends ---
        result = _mm256_blendv_ps(result, special_val, mask_special);
        result = _mm256_blendv_ps(result, nan, mask_nan);
        result = _mm256_blendv_ps(result, one, mask_one);

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

    /// Scalar helper: broadcast val into all 8 lanes, compute pow, return lane 0.
    unsafe fn pow_scalar_32(x: f32, y: f32) -> f32 {
        unsafe {
            let vx = _mm256_set1_ps(x);
            let vy = _mm256_set1_ps(y);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), _mm256_pow_ps(vx, vy));
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
        assert!((r - 2.0).abs() < TOL_32, "pow(4, 0.5) = {r}, expected 2.0");
    }

    #[test]
    fn pow_ps_negative_exponent() {
        let r = unsafe { pow_scalar_32(2.0, -1.0) };
        assert!((r - 0.5).abs() < TOL_32, "pow(2, -1) = {r}, expected 0.5");
    }

    // ---- Negative bases with integer exponents -----------------------------

    #[test]
    fn pow_ps_neg_base_even_int() {
        let r = unsafe { pow_scalar_32(-2.0, 2.0) };
        assert!((r - 4.0).abs() < TOL_32, "pow(-2, 2) = {r}, expected 4.0");
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
        assert!(
            r.is_infinite() && r.is_sign_positive(),
            "pow(0, -2) = {r}, expected +∞"
        );
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
    fn pow_ps_processes_all_8_lanes() {
        let bases: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 0.5, 10.0, 100.0, 0.1];
        let exps: [f32; 8] = [5.0, 3.0, 2.0, 0.5, 2.0, -1.0, 0.5, 3.0];
        unsafe {
            let vx = _mm256_loadu_ps(bases.as_ptr());
            let vy = _mm256_loadu_ps(exps.as_ptr());
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), _mm256_pow_ps(vx, vy));

            for i in 0..8 {
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

    /// Scalar helper: broadcast val into all 4 lanes, compute pow, return lane 0.
    unsafe fn pow_scalar_64(x: f64, y: f64) -> f64 {
        unsafe {
            let vx = _mm256_set1_pd(x);
            let vy = _mm256_set1_pd(y);
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), _mm256_pow_pd(vx, vy));
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
        assert!((r - 2.0).abs() < TOL_64, "pow(4, 0.5) = {r}, expected 2.0");
    }

    // ---- Negative bases ----------------------------------------------------

    #[test]
    fn pow_pd_neg_base_even_int() {
        let r = unsafe { pow_scalar_64(-2.0, 2.0) };
        assert!((r - 4.0).abs() < TOL_64, "pow(-2, 2) = {r}, expected 4.0");
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
    fn pow_pd_processes_all_4_lanes() {
        let bases: [f64; 4] = [2.0, 3.0, 0.5, 10.0];
        let exps: [f64; 4] = [3.0, 2.0, -1.0, 0.5];
        unsafe {
            let vx = _mm256_loadu_pd(bases.as_ptr());
            let vy = _mm256_loadu_pd(exps.as_ptr());
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), _mm256_pow_pd(vx, vy));

            for i in 0..4 {
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
