//! AVX-512 SIMD implementation of `cbrt(x)` (cube root) for `f32` and `f64` vectors.
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
//! | Implementation   | Accuracy     |
//! |------------------|--------------|
//! | `_mm512_cbrt_ps` | ≤ 1 ULP      |
//! | `_mm512_cbrt_pd` | ≤ 1 ULP      |
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
//! # Implementation Notes
//!
//! - **f32 uses f64 intermediates**: Each 16-lane f32 vector is split into two
//!   8-lane f64 vectors for Newton iterations to maintain precision.
//! - **Subnormal handling**: Detected and scaled before processing.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::cbrt::{
    B1_32, B1_64, B2_32, B2_64, P0, P1, P2, P3, P4, ROUND_BIAS_64, ROUND_MASK_64, X1P24_32,
    X1P54_64,
};

// ===========================================================================
// Integer division by 3 for 32-bit lanes in a 512-bit register
// ===========================================================================

/// Divides each unsigned 32-bit integer lane by 3 using the multiply-by-magic trick.
///
/// AVX-512 has no integer division. We use: `x / 3 = mulhi(x, 0xAAAAAAAB) >> 1`.
/// Since `_mm512_mul_epu32` only multiplies the low 32 bits of each 64-bit lane,
/// we process even and odd 32-bit lanes separately.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn div_by_3_epi32_512(x: __m512i) -> __m512i {
    let magic = _mm512_set1_epi32(0xAAAAAAABu32 as i32);

    // Even lanes (0, 2, 4, …): low 32 bits of each 64-bit lane
    let even_prod = _mm512_mul_epu32(x, magic); // 64-bit products
    let even_hi = _mm512_srli_epi64(even_prod, 32); // high 32 bits
    let even_result = _mm512_srli_epi64(even_hi, 1); // >> 1

    // Odd lanes (1, 3, 5, …): shift into even positions, multiply, shift back
    let x_odd = _mm512_srli_epi64(x, 32);
    let odd_prod = _mm512_mul_epu32(x_odd, magic);
    let odd_hi = _mm512_srli_epi64(odd_prod, 32);
    let odd_result = _mm512_slli_epi64(_mm512_srli_epi64(odd_hi, 1), 32);

    _mm512_or_si512(even_result, odd_result)
}

// ===========================================================================
// f32 Implementation (16 lanes)
// ===========================================================================

/// Computes `cbrt(x)` (cube root) for each lane of an AVX-512 `__m512` register.
///
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain including subnormals.
///
/// # Description
///
/// Uses a bit-manipulation trick for initial approximation followed by two
/// Newton–Raphson iterations in double precision. All 16 lanes are processed
/// simultaneously using AVX-512 mask operations for special cases.
///
/// # Safety
///
/// Requires AVX-512F. `x` must be a valid `__m512` register.
///
/// # Example
///
/// ```ignore
/// let x = _mm512_set1_ps(27.0);
/// let result = _mm512_cbrt_ps(x);
/// // All 16 lanes ≈ 3.0
/// ```
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_cbrt_ps(x: __m512) -> __m512 {
    unsafe {
        // -----------------------------------------------------------------------
        // Constants
        // -----------------------------------------------------------------------
        let x1p24 = _mm512_set1_ps(X1P24_32);
        let abs_mask = _mm512_set1_epi32(0x7FFFFFFFu32 as i32);
        let sign_mask = _mm512_set1_epi32(0x80000000u32 as i32);
        let inf_threshold = _mm512_set1_epi32(0x7F800000u32 as i32);
        let subnormal_threshold = _mm512_set1_epi32(0x00800000u32 as i32);
        let b1 = _mm512_set1_epi32(B1_32 as i32);
        let b2 = _mm512_set1_epi32(B2_32 as i32);
        let zero = _mm512_setzero_si512();
        let one_i32 = _mm512_set1_epi32(1);

        // -----------------------------------------------------------------------
        // Extract bit representation and sign
        // -----------------------------------------------------------------------
        let ui = _mm512_castps_si512(x);
        let hx = _mm512_and_si512(ui, abs_mask);
        let sign_bits = _mm512_and_si512(ui, sign_mask);

        // -----------------------------------------------------------------------
        // Special case detection (AVX-512 mask registers)
        // -----------------------------------------------------------------------

        // Inf or NaN: hx >= 0x7f800000 (hx is always non-negative, signed cmp works)
        let is_inf_nan: __mmask16 =
            _mm512_cmpgt_epi32_mask(hx, _mm512_sub_epi32(inf_threshold, one_i32));

        // Zero: hx == 0
        let is_zero: __mmask16 = _mm512_cmpeq_epi32_mask(hx, zero);

        // Subnormal: 0 < hx < 0x00800000
        let hx_lt_subnormal: __mmask16 = _mm512_cmpgt_epi32_mask(subnormal_threshold, hx);
        let is_subnormal: __mmask16 = hx_lt_subnormal & !is_zero;

        // -----------------------------------------------------------------------
        // Initial approximation via bit manipulation
        // -----------------------------------------------------------------------

        // Normal case: hx/3 + B1
        let hx_normal = _mm512_add_epi32(div_by_3_epi32_512(hx), b1);

        // Subnormal case: reconstruct |x| with sign, scale by 2^24, then hx_scaled/3 + B2
        let x_with_sign = _mm512_castsi512_ps(_mm512_or_si512(sign_bits, hx));
        let x_scaled = _mm512_mul_ps(x_with_sign, x1p24);
        let hx_scaled = _mm512_and_si512(_mm512_castps_si512(x_scaled), abs_mask);
        let hx_subnormal = _mm512_add_epi32(div_by_3_epi32_512(hx_scaled), b2);

        // Select between normal and subnormal paths
        let hx_approx = _mm512_mask_blend_epi32(is_subnormal, hx_normal, hx_subnormal);

        // Restore sign and convert to float
        let t_f32 = _mm512_castsi512_ps(_mm512_or_si512(sign_bits, hx_approx));

        // -----------------------------------------------------------------------
        // Newton–Raphson iterations in f64 (lower 8 lanes)
        // -----------------------------------------------------------------------
        let x_si = _mm512_castps_si512(x);
        let t_si = _mm512_castps_si512(t_f32);

        let x_low_f32 = _mm256_castsi256_ps(_mm512_castsi512_si256(x_si));
        let x_low = _mm512_cvtps_pd(x_low_f32);

        let t_low_f32 = _mm256_castsi256_ps(_mm512_castsi512_si256(t_si));
        let t_low = _mm512_cvtps_pd(t_low_f32);

        // First iteration: t = t * (2x + t³) / (x + 2t³)
        let r_low = _mm512_mul_pd(t_low, _mm512_mul_pd(t_low, t_low));
        let two_x_low = _mm512_add_pd(x_low, x_low);
        let t_low = _mm512_mul_pd(
            t_low,
            _mm512_div_pd(
                _mm512_add_pd(two_x_low, r_low),
                _mm512_add_pd(x_low, _mm512_add_pd(r_low, r_low)),
            ),
        );

        // Second iteration
        let r_low = _mm512_mul_pd(t_low, _mm512_mul_pd(t_low, t_low));
        let t_low_final = _mm512_mul_pd(
            t_low,
            _mm512_div_pd(
                _mm512_add_pd(two_x_low, r_low),
                _mm512_add_pd(x_low, _mm512_add_pd(r_low, r_low)),
            ),
        );

        // -----------------------------------------------------------------------
        // Newton–Raphson iterations in f64 (upper 8 lanes)
        // -----------------------------------------------------------------------
        let x_high_f32 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(x_si, 1));
        let x_high = _mm512_cvtps_pd(x_high_f32);

        let t_high_f32 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(t_si, 1));
        let t_high = _mm512_cvtps_pd(t_high_f32);

        // First iteration
        let r_high = _mm512_mul_pd(t_high, _mm512_mul_pd(t_high, t_high));
        let two_x_high = _mm512_add_pd(x_high, x_high);
        let t_high = _mm512_mul_pd(
            t_high,
            _mm512_div_pd(
                _mm512_add_pd(two_x_high, r_high),
                _mm512_add_pd(x_high, _mm512_add_pd(r_high, r_high)),
            ),
        );

        // Second iteration
        let r_high = _mm512_mul_pd(t_high, _mm512_mul_pd(t_high, t_high));
        let t_high_final = _mm512_mul_pd(
            t_high,
            _mm512_div_pd(
                _mm512_add_pd(two_x_high, r_high),
                _mm512_add_pd(x_high, _mm512_add_pd(r_high, r_high)),
            ),
        );

        // -----------------------------------------------------------------------
        // Combine halves back to 16×f32 and handle special cases
        // -----------------------------------------------------------------------
        let result_low = _mm512_cvtpd_ps(t_low_final); // __m256
        let result_high = _mm512_cvtpd_ps(t_high_final); // __m256
        let mut result = _mm512_castsi512_ps(_mm512_inserti64x4(
            _mm512_castps_si512(_mm512_castps256_ps512(result_low)),
            _mm256_castps_si256(result_high),
            1,
        ));

        // Zero: return x (preserves sign of ±0)
        result = _mm512_mask_blend_ps(is_zero, result, x);

        // Inf/NaN: return x + x (propagates NaN, returns ±∞ for ±∞)
        result = _mm512_mask_blend_ps(is_inf_nan, result, _mm512_add_ps(x, x));

        result
    }
}

// ===========================================================================
// f64 Implementation (8 lanes)
// ===========================================================================

/// Computes `cbrt(x)` (cube root) for each lane of an AVX-512 `__m512d` register.
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
/// Requires AVX-512F. `x` must be a valid `__m512d` register.
///
/// # Example
///
/// ```ignore
/// let x = _mm512_set1_pd(27.0);
/// let result = _mm512_cbrt_pd(x);
/// // All 8 lanes ≈ 3.0
/// ```
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn _mm512_cbrt_pd(x: __m512d) -> __m512d {
    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------
    let abs_mask_64 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFu64 as i64);
    let abs_mask_32 = _mm512_set1_epi64(0x7FFFFFFF_i64);
    let sign_mask = _mm512_set1_epi64(1_i64 << 63);
    let inf_threshold = _mm512_set1_epi64(0x7FF00000_i64);
    let subnormal_threshold = _mm512_set1_epi64(0x00100000_i64);
    let zero_si = _mm512_setzero_si512();

    // -----------------------------------------------------------------------
    // Extract bit representation, upper 32 bits (exponent+mantissa), sign
    // -----------------------------------------------------------------------
    let bits = _mm512_castpd_si512(x);
    let hx_full = _mm512_srli_epi64(bits, 32); // upper 32 bits → low half of each 64-bit lane
    let hx = _mm512_and_si512(hx_full, abs_mask_32); // strip sign
    let sign_bits = _mm512_and_si512(bits, sign_mask);

    // -----------------------------------------------------------------------
    // Special case detection
    // -----------------------------------------------------------------------
    let abs_bits = _mm512_and_si512(bits, abs_mask_64);

    // Inf/NaN: hx >= 0x7ff00000 (signed cmp works since hx ∈ [0, 0x7fffffff])
    let is_inf_nan: __mmask8 =
        _mm512_cmpgt_epi64_mask(hx, _mm512_sub_epi64(inf_threshold, _mm512_set1_epi64(1)));

    // Zero: all 64 bits of |x| == 0
    let is_zero: __mmask8 = _mm512_cmpeq_epi64_mask(abs_bits, zero_si);

    // Subnormal: 0 < |x| < 2^-1022 → hx < 0x00100000 and not zero
    let hx_lt_sub: __mmask8 = _mm512_cmpgt_epi64_mask(subnormal_threshold, hx);
    let is_subnormal: __mmask8 = hx_lt_sub & !is_zero;

    // -----------------------------------------------------------------------
    // Subnormal path: scale by 2^54, recompute bits and hx
    // -----------------------------------------------------------------------
    let x_scaled = _mm512_mul_pd(x, _mm512_set1_pd(X1P54_64));
    let bits_scaled = _mm512_castpd_si512(x_scaled);
    let hx_scaled = _mm512_and_si512(_mm512_srli_epi64(bits_scaled, 32), abs_mask_32);

    // -----------------------------------------------------------------------
    // Integer division by 3 on upper-32-bit values (in low 32 bits of 64-bit lanes)
    // _mm512_mul_epu32 multiplies the low 32 bits of each 64-bit lane
    // -----------------------------------------------------------------------
    let magic = _mm512_set1_epi64(0xAAAAAAABu64 as i64);
    let b1_vec = _mm512_set1_epi64(B1_64 as i64);
    let b2_vec = _mm512_set1_epi64(B2_64 as i64);

    // Normal: hx/3 + B1_64
    let prod_n = _mm512_mul_epu32(hx, magic);
    let hx_normal = _mm512_add_epi64(_mm512_srli_epi64(_mm512_srli_epi64(prod_n, 32), 1), b1_vec);

    // Subnormal: hx_scaled/3 + B2_64
    let prod_s = _mm512_mul_epu32(hx_scaled, magic);
    let hx_subnormal =
        _mm512_add_epi64(_mm512_srli_epi64(_mm512_srli_epi64(prod_s, 32), 1), b2_vec);

    // Select normal / subnormal
    let hx_approx = _mm512_mask_blend_epi64(is_subnormal, hx_normal, hx_subnormal);

    // -----------------------------------------------------------------------
    // Construct initial approximation: sign | (hx_approx << 32)
    // -----------------------------------------------------------------------
    let t_bits = _mm512_or_si512(sign_bits, _mm512_slli_epi64(hx_approx, 32));
    let mut t = _mm512_castsi512_pd(t_bits);

    // -----------------------------------------------------------------------
    // Polynomial refinement to ~23 bits: t = t * P(r), r = t²*(t/x)
    // -----------------------------------------------------------------------
    let r = _mm512_mul_pd(_mm512_mul_pd(t, t), _mm512_div_pd(t, x));

    let p0 = _mm512_set1_pd(P0);
    let p1 = _mm512_set1_pd(P1);
    let p2 = _mm512_set1_pd(P2);
    let p3 = _mm512_set1_pd(P3);
    let p4 = _mm512_set1_pd(P4);

    // (P0 + r*(P1 + r*P2)) + r³*(P3 + r*P4)
    let poly_lo = _mm512_fmadd_pd(r, _mm512_fmadd_pd(r, p2, p1), p0);
    let r2 = _mm512_mul_pd(r, r);
    let r3 = _mm512_mul_pd(r2, r);
    let poly_hi = _mm512_mul_pd(r3, _mm512_fmadd_pd(r, p4, p3));
    t = _mm512_mul_pd(t, _mm512_add_pd(poly_lo, poly_hi));

    // -----------------------------------------------------------------------
    // Round t to 23 significant bits (away from zero) so t*t is exact
    // -----------------------------------------------------------------------
    let t_bits_r = _mm512_castpd_si512(t);
    let rounded = _mm512_and_si512(
        _mm512_add_epi64(t_bits_r, _mm512_set1_epi64(ROUND_BIAS_64 as i64)),
        _mm512_set1_epi64(ROUND_MASK_64 as i64),
    );
    t = _mm512_castsi512_pd(rounded);

    // -----------------------------------------------------------------------
    // One Newton iteration to 53 bits: t += t * (x/t² - t) / (2t + x/t²)
    // -----------------------------------------------------------------------
    let s = _mm512_mul_pd(t, t);
    let r = _mm512_div_pd(x, s);
    let w = _mm512_add_pd(t, t);
    t = _mm512_fmadd_pd(
        t,
        _mm512_div_pd(_mm512_sub_pd(r, t), _mm512_add_pd(w, r)),
        t,
    );

    // -----------------------------------------------------------------------
    // Blend in special cases
    // -----------------------------------------------------------------------
    // Zero: return x (preserves ±0)
    t = _mm512_mask_blend_pd(is_zero, t, x);

    // Inf/NaN: return x + x (propagates NaN, returns ±∞ for ±∞)
    _mm512_mask_blend_pd(is_inf_nan, t, _mm512_add_pd(x, x))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to extract f32 lanes
    unsafe fn to_array_ps(v: __m512) -> [f32; 16] {
        let mut arr = [0.0_f32; 16];
        unsafe { _mm512_storeu_ps(arr.as_mut_ptr(), v) };
        arr
    }

    // Helper to extract f64 lanes
    unsafe fn to_array_pd(v: __m512d) -> [f64; 8] {
        let mut arr = [0.0_f64; 8];
        unsafe { _mm512_storeu_pd(arr.as_mut_ptr(), v) };
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
    fn test_cbrt_ps_perfect_cubes() {
        unsafe {
            let inputs: [f32; 16] = [
                1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0, 729.0, 1000.0, 1.0, 8.0, 27.0,
                64.0, 125.0, 216.0,
            ];
            let expected: [f32; 16] = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ];
            let x = _mm512_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm512_cbrt_ps(x));

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
    fn test_cbrt_ps_negative_values() {
        unsafe {
            let inputs: [f32; 16] = [
                -1.0, -8.0, -27.0, -64.0, -125.0, -216.0, -343.0, -512.0, -729.0, -1000.0, -1.0,
                -8.0, -27.0, -64.0, -125.0, -216.0,
            ];
            let x = _mm512_loadu_ps(inputs.as_ptr());
            let result = to_array_ps(_mm512_cbrt_ps(x));

            for (i, &r) in result.iter().enumerate() {
                let expected = inputs[i].cbrt();
                let ulp = ulp_distance_f32(r, expected);
                assert!(
                    ulp <= 1,
                    "Lane {}: cbrt({}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_cbrt_ps_zero() {
        unsafe {
            let pos_zero = _mm512_set1_ps(0.0);
            let neg_zero = _mm512_set1_ps(-0.0);

            let result_pos = to_array_ps(_mm512_cbrt_ps(pos_zero));
            let result_neg = to_array_ps(_mm512_cbrt_ps(neg_zero));

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
    fn test_cbrt_ps_infinity() {
        unsafe {
            let pos_inf = _mm512_set1_ps(f32::INFINITY);
            let neg_inf = _mm512_set1_ps(f32::NEG_INFINITY);

            let result_pos = to_array_ps(_mm512_cbrt_ps(pos_inf));
            let result_neg = to_array_ps(_mm512_cbrt_ps(neg_inf));

            for &r in &result_pos {
                assert!(r.is_infinite() && r > 0.0, "Should be +∞");
            }
            for &r in &result_neg {
                assert!(r.is_infinite() && r < 0.0, "Should be -∞");
            }
        }
    }

    #[test]
    fn test_cbrt_ps_nan() {
        unsafe {
            let nan = _mm512_set1_ps(f32::NAN);
            let result = to_array_ps(_mm512_cbrt_ps(nan));

            for &r in &result {
                assert!(r.is_nan(), "cbrt(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_cbrt_ps_ulp_sweep() {
        unsafe {
            let test_values = [
                0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1000.0, 1e10, 1e20, 1e-10, 1e-20,
            ];

            for &val in &test_values {
                for sign in [1.0_f32, -1.0] {
                    let x_val = sign * val;
                    let x = _mm512_set1_ps(x_val);
                    let result = to_array_ps(_mm512_cbrt_ps(x));
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
    fn test_cbrt_pd_perfect_cubes() {
        unsafe {
            let inputs: [f64; 8] = [1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0];
            let expected: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let x = _mm512_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm512_cbrt_pd(x));

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
    fn test_cbrt_pd_negative_values() {
        unsafe {
            let inputs: [f64; 8] = [-1.0, -8.0, -27.0, -64.0, -125.0, -216.0, -343.0, -512.0];
            let x = _mm512_loadu_pd(inputs.as_ptr());
            let result = to_array_pd(_mm512_cbrt_pd(x));

            for (i, &r) in result.iter().enumerate() {
                let expected = inputs[i].cbrt();
                let ulp = ulp_distance_f64(r, expected);
                assert!(
                    ulp <= 1,
                    "Lane {}: cbrt({}) = {}, expected {}, ULP = {}",
                    i,
                    inputs[i],
                    r,
                    expected,
                    ulp
                );
            }
        }
    }

    #[test]
    fn test_cbrt_pd_zero() {
        unsafe {
            let pos_zero = _mm512_set1_pd(0.0);
            let neg_zero = _mm512_set1_pd(-0.0);

            let result_pos = to_array_pd(_mm512_cbrt_pd(pos_zero));
            let result_neg = to_array_pd(_mm512_cbrt_pd(neg_zero));

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
    fn test_cbrt_pd_infinity() {
        unsafe {
            let pos_inf = _mm512_set1_pd(f64::INFINITY);
            let neg_inf = _mm512_set1_pd(f64::NEG_INFINITY);

            let result_pos = to_array_pd(_mm512_cbrt_pd(pos_inf));
            let result_neg = to_array_pd(_mm512_cbrt_pd(neg_inf));

            for &r in &result_pos {
                assert!(r.is_infinite() && r > 0.0, "Should be +∞");
            }
            for &r in &result_neg {
                assert!(r.is_infinite() && r < 0.0, "Should be -∞");
            }
        }
    }

    #[test]
    fn test_cbrt_pd_nan() {
        unsafe {
            let nan = _mm512_set1_pd(f64::NAN);
            let result = to_array_pd(_mm512_cbrt_pd(nan));

            for &r in &result {
                assert!(r.is_nan(), "cbrt(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_cbrt_pd_ulp_sweep() {
        unsafe {
            let test_values = [
                0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1000.0, 1e10, 1e50, 1e100,
                1e-10, 1e-50, 1e-100,
            ];

            for &val in &test_values {
                for sign in [1.0_f64, -1.0] {
                    let x_val = sign * val;
                    let x = _mm512_set1_pd(x_val);
                    let result = to_array_pd(_mm512_cbrt_pd(x));
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
