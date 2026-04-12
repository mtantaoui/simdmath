//! AVX-512 SIMD implementation of `atan(x)` for `f32` and `f64` vectors.
//!
//! # Algorithm
//!
//! Both implementations use a branchless argument reduction followed by a
//! minimax polynomial approximation, identical to the AVX2 versions but
//! operating on wider registers (16 lanes for f32, 8 lanes for f64).
//!
//! ## f32 — Single-range reduction
//!
//! For `|x| > 1`: `atan(x) = π/2 - atan(1/x)`, reducing to `[-1, 1]`.
//! A 9-term odd minimax polynomial approximates `atan` on that range.
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
//! | Implementation   | Accuracy    |
//! |------------------|-------------|
//! | `_mm512_atan_ps` | ≤ 3 ULP     |
//! | `_mm512_atan_pd` | ≤ 1 ULP     |
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
//! AVX-512 mask blend operations. This keeps all SIMD lanes active
//! simultaneously and avoids scalar fall-backs.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::avx512::abs::{_mm512_abs_pd, _mm512_abs_ps};
use crate::arch::consts::atan::{
    AT0, AT1, AT2, AT3, AT4, AT5, AT6, AT7, AT8, AT9, AT10, ATAN_P0_32, ATAN_P1_32, ATAN_P2_32,
    ATAN_P3_32, ATAN_P4_32, ATAN_P5_32, ATAN_P6_32, ATAN_P7_32, ATAN_P8_32, ATAN_THRESH_0,
    ATAN_THRESH_1, ATAN_THRESH_2, ATAN_THRESH_3, ATANHI_0, ATANHI_1, ATANHI_2, ATANHI_3, ATANLO_0,
    ATANLO_1, ATANLO_2, ATANLO_3, FRAC_PI_2_32,
};

// ---------------------------------------------------------------------------
// f32 Implementation (16 lanes)
// ---------------------------------------------------------------------------

/// Computes `atan(x)` for each lane of an AVX-512 `__m512` register.
///
/// # Precision
///
/// **≤ 3 ULP** error across the entire domain.
///
/// # Description
///
/// All 16 lanes are processed simultaneously without branches. The algorithm
/// uses argument reduction for `|x| > 1` and a 9-term minimax polynomial
/// for the core approximation.
///
/// # Safety
///
/// Requires AVX-512F. `x` must be a valid `__m512` register.
///
/// # Example
///
/// ```ignore
/// let input = _mm512_set1_ps(1.0);
/// let result = _mm512_atan_ps(input);
/// // All 16 lanes ≈ π/4 ≈ 0.7854
/// ```
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_atan_ps(x: __m512) -> __m512 {
    unsafe {
        // ---------------------------------------------------------------------
        // Broadcast constants to SIMD registers
        // ---------------------------------------------------------------------
        let one = _mm512_set1_ps(1.0);
        let frac_pi_2 = _mm512_set1_ps(FRAC_PI_2_32);

        // Polynomial coefficients
        let p0 = _mm512_set1_ps(ATAN_P0_32);
        let p1 = _mm512_set1_ps(ATAN_P1_32);
        let p2 = _mm512_set1_ps(ATAN_P2_32);
        let p3 = _mm512_set1_ps(ATAN_P3_32);
        let p4 = _mm512_set1_ps(ATAN_P4_32);
        let p5 = _mm512_set1_ps(ATAN_P5_32);
        let p6 = _mm512_set1_ps(ATAN_P6_32);
        let p7 = _mm512_set1_ps(ATAN_P7_32);
        let p8 = _mm512_set1_ps(ATAN_P8_32);

        // ---------------------------------------------------------------------
        // Extract sign and compute |x|
        //
        // We extract the sign bit directly to preserve -0.0 correctly.
        // atan(-x) = -atan(x), so we work with |x| and restore sign at end.
        //
        // Note: _mm512_and_ps requires avx512dq, so we use integer AND instead.
        // ---------------------------------------------------------------------
        let sign_mask_i = _mm512_set1_epi32(0x8000_0000_u32 as i32);
        let x_i = _mm512_castps_si512(x);
        let sign_bits_i = _mm512_and_epi32(x_i, sign_mask_i);
        let _sign_bits = _mm512_castsi512_ps(sign_bits_i);
        let abs_x = _mm512_abs_ps(x);

        // ---------------------------------------------------------------------
        // Argument reduction
        //
        // For |x| > 1: atan(|x|) = π/2 - atan(1/|x|)
        // ---------------------------------------------------------------------
        let needs_reduction = _mm512_cmp_ps_mask(abs_x, one, _CMP_GT_OQ);

        // Reduced argument: use 1/|x| if |x| > 1, otherwise |x|
        let recip = _mm512_div_ps(one, abs_x);
        let t = _mm512_mask_blend_ps(needs_reduction, abs_x, recip);

        // ---------------------------------------------------------------------
        // Polynomial evaluation using Horner's method
        //
        // atan(t) ≈ t · (P0 + t² · (P1 + t² · (P2 + t² · (P3 + ...))))
        //
        // We evaluate the polynomial in t² to exploit the odd symmetry of atan.
        // ---------------------------------------------------------------------
        let t2 = _mm512_mul_ps(t, t);

        // Horner's method from highest to lowest coefficient
        let mut u = p8;
        u = _mm512_fmadd_ps(u, t2, p7); // P7 + t² · P8
        u = _mm512_fmadd_ps(u, t2, p6); // P6 + t² · (P7 + ...)
        u = _mm512_fmadd_ps(u, t2, p5); // P5 + ...
        u = _mm512_fmadd_ps(u, t2, p4); // P4 + ...
        u = _mm512_fmadd_ps(u, t2, p3); // P3 + ...
        u = _mm512_fmadd_ps(u, t2, p2); // P2 + ...
        u = _mm512_fmadd_ps(u, t2, p1); // P1 + ...
        u = _mm512_fmadd_ps(u, t2, p0); // P0 + ...

        // Multiply by t to get the final polynomial value
        let poly_result = _mm512_mul_ps(u, t);

        // ---------------------------------------------------------------------
        // Apply argument reduction correction
        //
        // If |x| > 1: atan(|x|) = π/2 - atan(1/|x|) = π/2 - poly_result
        // Otherwise: atan(|x|) = poly_result
        // ---------------------------------------------------------------------
        let reduced_result = _mm512_sub_ps(frac_pi_2, poly_result);
        let abs_result = _mm512_mask_blend_ps(needs_reduction, poly_result, reduced_result);

        // ---------------------------------------------------------------------
        // Restore sign: atan(-x) = -atan(x)
        //
        // XOR the result with the original sign bits to restore the sign.
        // This correctly handles -0.0 → -0.0.
        //
        // Note: _mm512_xor_ps requires avx512dq, so we use integer XOR instead.
        // ---------------------------------------------------------------------
        let abs_result_i = _mm512_castps_si512(abs_result);
        let result_i = _mm512_xor_epi32(abs_result_i, sign_bits_i);
        _mm512_castsi512_ps(result_i)
    }
}

// ===========================================================================
// f64 Implementation (8 lanes)
// ===========================================================================

/// Computes `atan(x)` for each lane of an AVX-512 `__m512d` register (8 × f64).
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
/// All 8 lanes are processed simultaneously without branches.
///
/// # Safety
///
/// Requires AVX-512F. `x` must be a valid `__m512d` register.
///
/// # Example
///
/// ```ignore
/// let input = _mm512_set1_pd(1.0);
/// let result = _mm512_atan_pd(input);
/// // All 8 lanes ≈ π/4 ≈ 0.7854
/// ```
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_atan_pd(x: __m512d) -> __m512d {
    unsafe {
        // ---------------------------------------------------------------------
        // Broadcast constants
        // ---------------------------------------------------------------------
        let one = _mm512_set1_pd(1.0);
        let two = _mm512_set1_pd(2.0);
        let three = _mm512_set1_pd(3.0);

        // Polynomial coefficients (musl aT[])
        let at0 = _mm512_set1_pd(AT0);
        let at1 = _mm512_set1_pd(AT1);
        let at2 = _mm512_set1_pd(AT2);
        let at3 = _mm512_set1_pd(AT3);
        let at4 = _mm512_set1_pd(AT4);
        let at5 = _mm512_set1_pd(AT5);
        let at6 = _mm512_set1_pd(AT6);
        let at7 = _mm512_set1_pd(AT7);
        let at8 = _mm512_set1_pd(AT8);
        let at9 = _mm512_set1_pd(AT9);
        let at10 = _mm512_set1_pd(AT10);

        // Two-sum offsets for each reduction range
        let hi0 = _mm512_set1_pd(ATANHI_0);
        let lo0 = _mm512_set1_pd(ATANLO_0);
        let hi1 = _mm512_set1_pd(ATANHI_1);
        let lo1 = _mm512_set1_pd(ATANLO_1);
        let hi2 = _mm512_set1_pd(ATANHI_2);
        let lo2 = _mm512_set1_pd(ATANLO_2);
        let hi3 = _mm512_set1_pd(ATANHI_3);
        let lo3 = _mm512_set1_pd(ATANLO_3);

        // Range thresholds
        let thr0 = _mm512_set1_pd(ATAN_THRESH_0); // 7/16 = 0.4375
        let thr1 = _mm512_set1_pd(ATAN_THRESH_1); // 11/16 = 0.6875
        let thr2 = _mm512_set1_pd(ATAN_THRESH_2); // 19/16 = 1.1875
        let thr3 = _mm512_set1_pd(ATAN_THRESH_3); // 39/16 = 2.4375

        // ---------------------------------------------------------------------
        // Extract sign and compute |x|
        //
        // Note: _mm512_and_pd requires avx512dq, so we use integer AND instead.
        // ---------------------------------------------------------------------
        let sign_mask_i = _mm512_set1_epi64(0x8000_0000_0000_0000_u64 as i64);
        let x_i = _mm512_castpd_si512(x);
        let sign_bits_i = _mm512_and_epi64(x_i, sign_mask_i);
        let _sign_bits = _mm512_castsi512_pd(sign_bits_i);
        let abs_x = _mm512_abs_pd(x);

        // ---------------------------------------------------------------------
        // Compute range masks (from smallest to largest)
        // AVX-512 uses __mmask8 for f64 comparisons
        // ---------------------------------------------------------------------
        let is_lt_thr0 = _mm512_cmp_pd_mask(abs_x, thr0, _CMP_LT_OQ); // |x| < 7/16
        let is_lt_thr1 = _mm512_cmp_pd_mask(abs_x, thr1, _CMP_LT_OQ); // |x| < 11/16
        let is_lt_thr2 = _mm512_cmp_pd_mask(abs_x, thr2, _CMP_LT_OQ); // |x| < 19/16
        let is_lt_thr3 = _mm512_cmp_pd_mask(abs_x, thr3, _CMP_LT_OQ); // |x| < 39/16

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
        let t0 = _mm512_div_pd(
            _mm512_sub_pd(_mm512_mul_pd(two, abs_x), one),
            _mm512_add_pd(two, abs_x),
        );

        // id=1: (|x| − 1) / (|x| + 1)
        let t1 = _mm512_div_pd(_mm512_sub_pd(abs_x, one), _mm512_add_pd(abs_x, one));

        // id=2: (2·|x| − 3) / (2 + 3·|x|)
        let t2 = _mm512_div_pd(
            _mm512_sub_pd(_mm512_mul_pd(two, abs_x), three),
            _mm512_add_pd(two, _mm512_mul_pd(three, abs_x)),
        );

        // id=3: −1 / |x|  (atan(|x|) = π/2 + atan(−1/|x|))
        let neg_one = _mm512_sub_pd(_mm512_setzero_pd(), one);
        let t3 = _mm512_div_pd(neg_one, abs_x);

        // ---------------------------------------------------------------------
        // Select the reduced argument by cascaded blending.
        //
        // Priority (highest first): id=3, id=2, id=1, id=0, id=-1
        // AVX-512 mask_blend: mask selects from second operand when bit is 1
        // ---------------------------------------------------------------------
        // Start from id=3, blend down toward id=-1
        let t = {
            let t = _mm512_mask_blend_pd(is_lt_thr3, t3, t2); // id=3 or id=2
            let t = _mm512_mask_blend_pd(is_lt_thr2, t, t1); // …or id=1
            let t = _mm512_mask_blend_pd(is_lt_thr1, t, t0); // …or id=0
            _mm512_mask_blend_pd(is_lt_thr0, t, abs_x) // …or no reduction
        };

        // ---------------------------------------------------------------------
        // Select the matching hi/lo offsets by cascaded blending
        //
        // id=-1 → hi=0, lo=0
        // ---------------------------------------------------------------------
        let zero_pd = _mm512_setzero_pd();
        let hi = {
            let hi = _mm512_mask_blend_pd(is_lt_thr3, hi3, hi2);
            let hi = _mm512_mask_blend_pd(is_lt_thr2, hi, hi1);
            let hi = _mm512_mask_blend_pd(is_lt_thr1, hi, hi0);
            _mm512_mask_blend_pd(is_lt_thr0, hi, zero_pd)
        };
        let lo = {
            let lo = _mm512_mask_blend_pd(is_lt_thr3, lo3, lo2);
            let lo = _mm512_mask_blend_pd(is_lt_thr2, lo, lo1);
            let lo = _mm512_mask_blend_pd(is_lt_thr1, lo, lo0);
            _mm512_mask_blend_pd(is_lt_thr0, lo, zero_pd)
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
        // ---------------------------------------------------------------------
        let z = _mm512_mul_pd(t, t); // t²
        let w = _mm512_mul_pd(z, z); // t⁴

        // s1: odd-indexed coefficients (aT[0], aT[2], aT[4], aT[6], aT[8], aT[10])
        let s1 = _mm512_mul_pd(
            z,
            _mm512_fmadd_pd(
                w,
                _mm512_fmadd_pd(
                    w,
                    _mm512_fmadd_pd(
                        w,
                        _mm512_fmadd_pd(w, _mm512_fmadd_pd(w, at10, at8), at6),
                        at4,
                    ),
                    at2,
                ),
                at0,
            ),
        );

        // s2: even-indexed coefficients (aT[1], aT[3], aT[5], aT[7], aT[9])
        let s2 = _mm512_mul_pd(
            w,
            _mm512_fmadd_pd(
                w,
                _mm512_fmadd_pd(
                    w,
                    _mm512_fmadd_pd(w, _mm512_fmadd_pd(w, at9, at7), at5),
                    at3,
                ),
                at1,
            ),
        );

        // ---------------------------------------------------------------------
        // Combine: result = hi + lo + t − t·(s1 + s2)
        //
        // Expanding the musl formula: atanhi + atanlo + t*(1 − (s1+s2))
        // Written as two FMAs to keep accuracy:
        //   correction = t * (s1 + s2)
        //   result     = hi + lo + t − correction
        // ---------------------------------------------------------------------
        let sum_s = _mm512_add_pd(s1, s2);
        let correction = _mm512_mul_pd(t, sum_s); // t·(s1+s2)

        // hi + lo + t − correction  (order matters for cancellation)
        let result_abs = _mm512_add_pd(_mm512_add_pd(hi, lo), _mm512_sub_pd(t, correction));

        // ---------------------------------------------------------------------
        // Restore sign: atan(−x) = −atan(x)
        //
        // XOR with original sign bits correctly propagates −0.0 → −0.0.
        //
        // Note: _mm512_xor_pd requires avx512dq, so we use integer XOR instead.
        // ---------------------------------------------------------------------
        let result_abs_i = _mm512_castpd_si512(result_abs);
        let result_i = _mm512_xor_epi64(result_abs_i, sign_bits_i);
        _mm512_castsi512_pd(result_i)
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

    // Helper to extract f32 lanes from __m512
        unsafe fn extract_f32(v: __m512) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), v) };
        out
    }

    // Helper to extract f64 lanes from __m512d
        unsafe fn extract_f64(v: __m512d) -> [f64; 8] {
        let mut out = [0.0f64; 8];
        unsafe { _mm512_storeu_pd(out.as_mut_ptr(), v) };
        out
    }

    // =========================================================================
    // f32 tests
    // =========================================================================

    #[test]
    fn atan_ps_zero_returns_zero() {
        unsafe {
            let input = _mm512_set1_ps(0.0);
            let result = extract_f32(_mm512_atan_ps(input));
            assert!(result.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn atan_ps_negative_zero_returns_negative_zero() {
        unsafe {
            let input = _mm512_set1_ps(-0.0);
            let result = extract_f32(_mm512_atan_ps(input));
            for &x in &result {
                assert!(x == 0.0 && x.is_sign_negative());
            }
        }
    }

    #[test]
    fn atan_ps_one_returns_pi_over_4() {
        unsafe {
            let input = _mm512_set1_ps(1.0);
            let result = extract_f32(_mm512_atan_ps(input));
            for &x in &result {
                assert!(
                    (x - FRAC_PI_4_F32).abs() < TOL_F32,
                    "got {x}, expected {FRAC_PI_4_F32}"
                );
            }
        }
    }

    #[test]
    fn atan_ps_neg_one_returns_neg_pi_over_4() {
        unsafe {
            let input = _mm512_set1_ps(-1.0);
            let result = extract_f32(_mm512_atan_ps(input));
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
    fn atan_ps_infinity_returns_pi_over_2() {
        unsafe {
            let input = _mm512_set1_ps(f32::INFINITY);
            let result = extract_f32(_mm512_atan_ps(input));
            for &x in &result {
                assert!(
                    (x - FRAC_PI_2_F32).abs() < TOL_F32,
                    "got {x}, expected {FRAC_PI_2_F32}"
                );
            }
        }
    }

    #[test]
    fn atan_ps_neg_infinity_returns_neg_pi_over_2() {
        unsafe {
            let input = _mm512_set1_ps(f32::NEG_INFINITY);
            let result = extract_f32(_mm512_atan_ps(input));
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
    fn atan_ps_nan_returns_nan() {
        unsafe {
            let input = _mm512_set1_ps(f32::NAN);
            let result = extract_f32(_mm512_atan_ps(input));
            assert!(result.iter().all(|x| x.is_nan()));
        }
    }

    #[test]
    fn atan_ps_small_values() {
        unsafe {
            let input = _mm512_setr_ps(
                0.1, 0.01, 0.001, 0.0001, -0.1, -0.01, -0.001, -0.0001, 0.05, -0.05, 0.2, -0.2,
                0.3, -0.3, 0.4, -0.4,
            );
            let result = extract_f32(_mm512_atan_ps(input));
            let expected: [f32; 16] = [
                0.1f32.atan(),
                0.01f32.atan(),
                0.001f32.atan(),
                0.0001f32.atan(),
                (-0.1f32).atan(),
                (-0.01f32).atan(),
                (-0.001f32).atan(),
                (-0.0001f32).atan(),
                0.05f32.atan(),
                (-0.05f32).atan(),
                0.2f32.atan(),
                (-0.2f32).atan(),
                0.3f32.atan(),
                (-0.3f32).atan(),
                0.4f32.atan(),
                (-0.4f32).atan(),
            ];
            for (i, (&r, &e)) in result.iter().zip(&expected).enumerate() {
                assert!((r - e).abs() < TOL_F32, "lane {i}: got {r}, expected {e}");
            }
        }
    }

    #[test]
    fn atan_ps_large_values() {
        unsafe {
            let input = _mm512_setr_ps(
                10.0, 100.0, 1000.0, 10000.0, -10.0, -100.0, -1000.0, -10000.0, 2.0, -2.0, 5.0,
                -5.0, 20.0, -20.0, 50.0, -50.0,
            );
            let result = extract_f32(_mm512_atan_ps(input));
            let expected: [f32; 16] = [
                10.0f32.atan(),
                100.0f32.atan(),
                1000.0f32.atan(),
                10000.0f32.atan(),
                (-10.0f32).atan(),
                (-100.0f32).atan(),
                (-1000.0f32).atan(),
                (-10000.0f32).atan(),
                2.0f32.atan(),
                (-2.0f32).atan(),
                5.0f32.atan(),
                (-5.0f32).atan(),
                20.0f32.atan(),
                (-20.0f32).atan(),
                50.0f32.atan(),
                (-50.0f32).atan(),
            ];
            for (i, (&r, &e)) in result.iter().zip(&expected).enumerate() {
                assert!((r - e).abs() < TOL_F32, "lane {i}: got {r}, expected {e}");
            }
        }
    }

    #[test]
    fn atan_ps_all_lanes_independent() {
        unsafe {
            let input = _mm512_setr_ps(
                -3.0, -2.0, -1.5, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0,
                4.0,
            );
            let result = extract_f32(_mm512_atan_ps(input));
            let expected: [f32; 16] = [
                (-3.0f32).atan(),
                (-2.0f32).atan(),
                (-1.5f32).atan(),
                (-1.0f32).atan(),
                (-0.5f32).atan(),
                (-0.25f32).atan(),
                (-0.1f32).atan(),
                0.0f32.atan(),
                0.1f32.atan(),
                0.25f32.atan(),
                0.5f32.atan(),
                1.0f32.atan(),
                1.5f32.atan(),
                2.0f32.atan(),
                3.0f32.atan(),
                4.0f32.atan(),
            ];
            for (i, (&r, &e)) in result.iter().zip(&expected).enumerate() {
                assert!((r - e).abs() < TOL_F32, "lane {i}: got {r}, expected {e}");
            }
        }
    }

    #[test]
    fn atan_ps_ulp_sweep() {
        unsafe {
            let mut max_ulp: u32 = 0;

            // Sweep across domain
            for i in -500_000i32..=500_000 {
                let x = (i as f32) * 0.00002; // [-10, 10]
                let expected = x.atan();
                if !expected.is_finite() {
                    continue;
                }

                let input = _mm512_set1_ps(x);
                let result = extract_f32(_mm512_atan_ps(input))[0];

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
    fn atan_pd_zero_returns_zero() {
        unsafe {
            let result = extract_f64(_mm512_atan_pd(_mm512_set1_pd(0.0)));
            assert!(result.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn atan_pd_negative_zero_returns_negative_zero() {
        unsafe {
            let result = extract_f64(_mm512_atan_pd(_mm512_set1_pd(-0.0)));
            for &x in &result {
                assert!(x == 0.0 && x.is_sign_negative());
            }
        }
    }

    #[test]
    fn atan_pd_one_returns_pi_over_4() {
        unsafe {
            let result = extract_f64(_mm512_atan_pd(_mm512_set1_pd(1.0)));
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
    fn atan_pd_neg_one_returns_neg_pi_over_4() {
        unsafe {
            let result = extract_f64(_mm512_atan_pd(_mm512_set1_pd(-1.0)));
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
    fn atan_pd_infinity_returns_pi_over_2() {
        unsafe {
            let result = extract_f64(_mm512_atan_pd(_mm512_set1_pd(f64::INFINITY)));
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
    fn atan_pd_neg_infinity_returns_neg_pi_over_2() {
        unsafe {
            let result = extract_f64(_mm512_atan_pd(_mm512_set1_pd(f64::NEG_INFINITY)));
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
    fn atan_pd_nan_returns_nan() {
        unsafe {
            let result = extract_f64(_mm512_atan_pd(_mm512_set1_pd(f64::NAN)));
            assert!(result.iter().all(|x| x.is_nan()));
        }
    }

    #[test]
    fn atan_pd_all_lanes_independent() {
        unsafe {
            let input = _mm512_setr_pd(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0);
            let result = extract_f64(_mm512_atan_pd(input));
            let expected = [
                (-2.0f64).atan(),
                (-1.0f64).atan(),
                (-0.5f64).atan(),
                0.0f64.atan(),
                0.5f64.atan(),
                1.0f64.atan(),
                2.0f64.atan(),
                3.0f64.atan(),
            ];
            for (i, (&r, &e)) in result.iter().zip(&expected).enumerate() {
                assert!((r - e).abs() < TOL_F64, "lane {i}: got {r}, expected {e}");
            }
        }
    }

    #[test]
    fn atan_pd_ulp_sweep() {
        unsafe {
            let mut max_ulp: u64 = 0;

            // Sweep across a wide domain including all 5 reduction ranges
            for i in 0..2000 {
                let t = (i as f64 / 1999.0) * 20.0 - 10.0; // [-10.0, 10.0]
                let result = extract_f64(_mm512_atan_pd(_mm512_set1_pd(t)));
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
