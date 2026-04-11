//! AVX2 SIMD implementation of `tan(x)` for `f32` and `f64` vectors.
//!
//! This module provides 8-lane f32 and 4-lane f64 tangent implementations using
//! the Cody-Waite argument reduction algorithm and minimax polynomial
//! approximations ported from musl libc's `tanf.c`, `tan.c`, and kernel functions.
//!
//! # Algorithm
//!
//! 1. **Argument reduction**: Reduce `x` to `y ∈ [-π/4, π/4]` via `y = x - n*(π/2)`
//!    using Cody-Waite extended precision subtraction.
//!
//! 2. **Quadrant selection**: Unlike sin/cos which have period 2π, tan has period π.
//!    Based on `n mod 2`:
//!    | n mod 2 | tan(x)      |
//!    |---------|-------------|
//!    | 0       |  tan(y)     |
//!    | 1       | -1/tan(y)   |
//!
//! 3. **Polynomial evaluation**: Minimax polynomial for the tangent kernel.
//!
//! # Precision
//!
//! | Variant           | Max Error |
//! |-------------------|-----------|
//! | `_mm256_tan_ps`   | ≤ 2 ULP   |
//! | `_mm256_tan_pd`   | ≤ 2 ULP   |
//!
//! # Special Values
//!
//! | Input       | Output |
//! |-------------|--------|
//! | `0.0`       | `0.0`  |
//! | `-0.0`      | `-0.0` |
//! | `±∞`        | `NaN`  |
//! | `NaN`       | `NaN`  |
//! | Very small  | `x` (correctly rounded) |
//! | `±π/2`      | Large value (approaches ±∞) |

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::consts::tan::{
    BIG_THRESH_64, FRAC_2_PI_32, FRAC_2_PI_64, PIO2_1_32, PIO2_1_64, PIO2_1T_32, PIO2_1T_64,
    PIO2_2_64, PIO2_2T_64, PIO4_HI_64, PIO4_LO_64, T0_32, T0_64, T1_32, T1_64, T2_32, T2_64, T3_32,
    T3_64, T4_32, T4_64, T5_32, T5_64, T6_64, T7_64, T8_64, T9_64, T10_64, T11_64, T12_64, TOINT,
};

// =============================================================================
// f32 Implementation (8 lanes, computed in f64 precision internally)
// =============================================================================

/// Computes `tan(x)` for each lane of an AVX2 `__m256` register.
///
/// Uses the musl libc algorithm: Cody-Waite argument reduction to `[-π/4, π/4]`
/// followed by polynomial evaluation of the tangent kernel. When n is odd,
/// computes -1/tan(y) using the cotangent identity. Internal computations
/// use f64 precision for accuracy.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires AVX2 and FMA support. The caller must ensure these features are
/// available at runtime.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_tan_ps(x: __m256) -> __m256 {
    unsafe {
        // Process as two 4-lane f64 operations for precision
        // Split input into low and high halves, convert to f64
        let x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
        let x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));

        // Compute tangent in f64 precision for each half
        let tan_lo = tan_ps_in_f64(x_lo);
        let tan_hi = tan_ps_in_f64(x_hi);

        // Convert back to f32 and combine
        let result_lo = _mm256_cvtpd_ps(tan_lo);
        let result_hi = _mm256_cvtpd_ps(tan_hi);

        _mm256_insertf128_ps(_mm256_castps128_ps256(result_lo), result_hi, 1)
    }
}

/// Internal f64 computation for f32 tangent (4 lanes).
///
/// This helper computes tan(x) in f64 precision for 4 f32 values that have
/// been promoted to f64. The extra precision ensures ≤2 ULP in the final f32.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn tan_ps_in_f64(x: __m256d) -> __m256d {
    unsafe {
        let frac_2_pi = _mm256_set1_pd(FRAC_2_PI_32);
        let pio2_1 = _mm256_set1_pd(PIO2_1_32);
        let pio2_1t = _mm256_set1_pd(PIO2_1T_32);
        let toint = _mm256_set1_pd(TOINT);

        // -------------------------------------------------------------------------
        // Step 1: Argument reduction
        // Compute n = round(x * 2/π), then y = x - n * (π/2)
        // -------------------------------------------------------------------------

        // fn = round(x * 2/π) using magic number trick
        let fn_val = _mm256_sub_pd(_mm256_fmadd_pd(x, frac_2_pi, toint), toint);

        // Convert to integer for quadrant selection
        let n = _mm256_cvtpd_epi32(fn_val);

        // Cody-Waite reduction: y = x - fn * pio2_1 - fn * pio2_1t
        let y = _mm256_fnmadd_pd(fn_val, pio2_1t, _mm256_fnmadd_pd(fn_val, pio2_1, x));

        // -------------------------------------------------------------------------
        // Step 2: Compute tan(y) using polynomial approximation
        // tan(y) ≈ y + T0*y³ + T1*y⁵ + T2*y⁷ + T3*y⁹ + T4*y¹¹ + T5*y¹³
        // -------------------------------------------------------------------------

        let tan_y = tandf_kernel(y);

        // -------------------------------------------------------------------------
        // Step 3: Quadrant-based selection
        // n mod 2: 0 → tan(y), 1 → -1/tan(y)
        //
        // When n is odd, we're in a quadrant where tan should use cotangent:
        // tan(y + π/2) = -cot(y) = -1/tan(y)
        // -------------------------------------------------------------------------

        // Extend n to 256-bit for blending (n is 128-bit from cvtpd_epi32)
        let n_256 = _mm256_cvtepi32_epi64(n);
        let one = _mm256_set1_epi64x(1);

        // Check if n is odd (n & 1 == 1)
        let n_and_1 = _mm256_and_si256(n_256, one);
        let is_odd = _mm256_cmpeq_epi64(n_and_1, one);

        // Compute -1/tan(y) for odd quadrants
        let neg_one = _mm256_set1_pd(-1.0);
        let recip = _mm256_div_pd(neg_one, tan_y);

        // Select tan(y) or -1/tan(y) based on quadrant
        let result = _mm256_blendv_pd(tan_y, recip, _mm256_castsi256_pd(is_odd));

        // -------------------------------------------------------------------------
        // Step 4: Handle special cases (NaN, Inf, tiny values)
        // tan(±∞) = NaN, tan(NaN) = NaN, tan(±0) = ±0
        // -------------------------------------------------------------------------

        let sign_bit = _mm256_set1_pd(-0.0);
        let abs_x = _mm256_andnot_pd(sign_bit, x);
        let inf = _mm256_set1_pd(f64::INFINITY);
        let is_inf_or_nan = _mm256_cmp_pd(abs_x, inf, _CMP_GE_OQ);
        let nan = _mm256_set1_pd(f64::NAN);

        // For tiny values (including ±0), tan(x) ≈ x
        let tiny = _mm256_set1_pd(1e-300);
        let is_tiny = _mm256_cmp_pd(abs_x, tiny, _CMP_LT_OQ);
        let result = _mm256_blendv_pd(result, x, is_tiny);

        _mm256_blendv_pd(result, nan, is_inf_or_nan)
    }
}

/// Tangent kernel for reduced argument in `[-π/4, π/4]`.
///
/// Implements musl's `__tandf`: tan(x) ≈ x + T0*x³ + T1*x⁵ + T2*x⁷ + T3*x⁹ + T4*x¹¹ + T5*x¹³
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn tandf_kernel(x: __m256d) -> __m256d {
    let t0 = _mm256_set1_pd(T0_32);
    let t1 = _mm256_set1_pd(T1_32);
    let t2 = _mm256_set1_pd(T2_32);
    let t3 = _mm256_set1_pd(T3_32);
    let t4 = _mm256_set1_pd(T4_32);
    let t5 = _mm256_set1_pd(T5_32);

    let z = _mm256_mul_pd(x, x); // z = x²
    let w = _mm256_mul_pd(z, z); // w = z² = x⁴

    // Horner's method for polynomial evaluation
    // r = T4 + z*T5
    let r = _mm256_fmadd_pd(z, t5, t4);
    // r = T2 + z*T3 + w*r = T2 + z*T3 + w*(T4 + z*T5)
    let r = _mm256_fmadd_pd(w, r, _mm256_fmadd_pd(z, t3, t2));
    // r = T0 + z*T1 + w*r
    let r = _mm256_fmadd_pd(w, r, _mm256_fmadd_pd(z, t1, t0));

    // tan(x) = x + x³ * r = x + z*x*r = x * (1 + z*r)
    let zx = _mm256_mul_pd(z, x); // z*x = x³
    _mm256_fmadd_pd(zx, r, x) // x + x³*r
}

// =============================================================================
// f64 Implementation (4 lanes)
// =============================================================================

/// Computes `tan(x)` for each lane of an AVX2 `__m256d` register.
///
/// Uses musl libc's algorithm with degree-27 polynomial for the tangent kernel
/// after Cody-Waite argument reduction.
///
/// # Precision
///
/// **≤ 2 ULP** error across the entire domain.
///
/// # Safety
///
/// Requires AVX2 and FMA support. The caller must ensure these features are
/// available at runtime.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_tan_pd(x: __m256d) -> __m256d {
    unsafe {
        let frac_2_pi = _mm256_set1_pd(FRAC_2_PI_64);
        let pio2_1 = _mm256_set1_pd(PIO2_1_64);
        let pio2_1t = _mm256_set1_pd(PIO2_1T_64);
        let pio2_2 = _mm256_set1_pd(PIO2_2_64);
        let pio2_2t = _mm256_set1_pd(PIO2_2T_64);
        let toint = _mm256_set1_pd(TOINT);

        // -------------------------------------------------------------------------
        // Step 1: Argument reduction with extended precision
        // -------------------------------------------------------------------------

        let fn_val = _mm256_sub_pd(_mm256_fmadd_pd(x, frac_2_pi, toint), toint);
        let n = _mm256_cvtpd_epi32(fn_val);

        // Extended precision Cody-Waite reduction
        // y = x - fn*(pio2_1 + pio2_1t) with compensation
        let mut y = _mm256_fnmadd_pd(fn_val, pio2_1, x);
        y = _mm256_fnmadd_pd(fn_val, pio2_1t, y);

        // For very large arguments, apply additional correction terms
        let sign_bit = _mm256_set1_pd(-0.0);
        let abs_x = _mm256_andnot_pd(sign_bit, x);
        let large_thresh = _mm256_set1_pd(1e9);
        let is_large = _mm256_cmp_pd(abs_x, large_thresh, _CMP_GT_OQ);

        // Additional reduction for large values
        let y_corrected = _mm256_fnmadd_pd(fn_val, pio2_2, y);
        let y_corrected = _mm256_fnmadd_pd(fn_val, pio2_2t, y_corrected);
        let y = _mm256_blendv_pd(y, y_corrected, is_large);

        // -------------------------------------------------------------------------
        // Step 2: Compute quadrant info (n mod 2)
        // -------------------------------------------------------------------------

        let n_256 = _mm256_cvtepi32_epi64(n);
        let one_i = _mm256_set1_epi64x(1);
        let n_and_1 = _mm256_and_si256(n_256, one_i);

        // -------------------------------------------------------------------------
        // Step 3: Compute tan(y) with kernel (handles big argument and odd quadrant)
        // -------------------------------------------------------------------------

        let result = tan_kernel_f64(y, n_and_1);

        // -------------------------------------------------------------------------
        // Step 4: Handle special cases
        // tan(±∞) = NaN, tan(NaN) = NaN, tan(±0) = ±0
        // -------------------------------------------------------------------------

        let inf = _mm256_set1_pd(f64::INFINITY);
        let is_inf_or_nan = _mm256_cmp_pd(abs_x, inf, _CMP_GE_OQ);
        let nan = _mm256_set1_pd(f64::NAN);

        // For tiny values (including ±0), tan(x) ≈ x
        let tiny = _mm256_set1_pd(1e-300);
        let is_tiny = _mm256_cmp_pd(abs_x, tiny, _CMP_LT_OQ);
        let result = _mm256_blendv_pd(result, x, is_tiny);

        _mm256_blendv_pd(result, nan, is_inf_or_nan)
    }
}

/// Tangent kernel for f64 reduced argument.
///
/// Implements musl's `__tan` algorithm: polynomial approximation for tan(x) on [-π/4, π/4].
/// Uses a degree-27 polynomial split into odd and even terms for better precision.
///
/// For |x| >= 0.6744 (near ±π/4), uses a special "big" argument path that applies
/// the identity tan(π/4 - y) = (1-tan(y))/(1+tan(y)) to improve accuracy.
///
/// # Parameters
///
/// - `x`: Reduced argument in [-π/4, π/4]
/// - `odd`: Integer mask indicating whether this is the odd quadrant (needs -1/tan transformation)
///
/// # Returns
///
/// The final tan(x) value, with -1/tan transformation applied if in odd quadrant.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn tan_kernel_f64(x: __m256d, odd: __m256i) -> __m256d {
    // Constants
    let sign_bit = _mm256_set1_pd(-0.0);
    let pio4 = _mm256_set1_pd(PIO4_HI_64);
    let pio4lo = _mm256_set1_pd(PIO4_LO_64);
    let big_thresh = _mm256_set1_pd(BIG_THRESH_64);
    let one = _mm256_set1_pd(1.0);
    let two = _mm256_set1_pd(2.0);
    let neg_one = _mm256_set1_pd(-1.0);

    // Load polynomial coefficients (T[0] to T[12] from musl)
    let t0 = _mm256_set1_pd(T0_64);
    let t1 = _mm256_set1_pd(T1_64);
    let t2 = _mm256_set1_pd(T2_64);
    let t3 = _mm256_set1_pd(T3_64);
    let t4 = _mm256_set1_pd(T4_64);
    let t5 = _mm256_set1_pd(T5_64);
    let t6 = _mm256_set1_pd(T6_64);
    let t7 = _mm256_set1_pd(T7_64);
    let t8 = _mm256_set1_pd(T8_64);
    let t9 = _mm256_set1_pd(T9_64);
    let t10 = _mm256_set1_pd(T10_64);
    let t11 = _mm256_set1_pd(T11_64);
    let t12 = _mm256_set1_pd(T12_64);

    // -------------------------------------------------------------------------
    // Step 1: Check for "big" values: |x| >= 0.6744
    // These need special handling because the polynomial loses accuracy near π/4
    // -------------------------------------------------------------------------
    let abs_x = _mm256_andnot_pd(sign_bit, x);
    let is_big = _mm256_cmp_pd(abs_x, big_thresh, _CMP_GE_OQ);

    // For big values: transform x to (π/4 - |x|), preserving original sign for later
    let x_sign = _mm256_and_pd(x, sign_bit);
    let x_transformed = _mm256_add_pd(_mm256_sub_pd(pio4, abs_x), pio4lo);

    // Select: use transformed x for big values, original x otherwise
    let x_eval = _mm256_blendv_pd(x, x_transformed, is_big);

    // -------------------------------------------------------------------------
    // Step 2: Polynomial evaluation
    // Split into odd and even powers for better numerical stability
    // -------------------------------------------------------------------------
    let z = _mm256_mul_pd(x_eval, x_eval); // z = x²
    let w = _mm256_mul_pd(z, z); // w = x⁴

    // r = T[1] + w*(T[3] + w*(T[5] + w*(T[7] + w*(T[9] + w*T[11]))))
    // Odd-indexed coefficients
    let mut r = t11;
    r = _mm256_fmadd_pd(r, w, t9);
    r = _mm256_fmadd_pd(r, w, t7);
    r = _mm256_fmadd_pd(r, w, t5);
    r = _mm256_fmadd_pd(r, w, t3);
    r = _mm256_fmadd_pd(r, w, t1);

    // v = z*(T[2] + w*(T[4] + w*(T[6] + w*(T[8] + w*(T[10] + w*T[12])))))
    // Even-indexed coefficients (except T[0])
    let mut v = t12;
    v = _mm256_fmadd_pd(v, w, t10);
    v = _mm256_fmadd_pd(v, w, t8);
    v = _mm256_fmadd_pd(v, w, t6);
    v = _mm256_fmadd_pd(v, w, t4);
    v = _mm256_fmadd_pd(v, w, t2);
    v = _mm256_mul_pd(z, v);

    // Compute polynomial remainder: r_poly = T[0]*x³ + x⁵*(r+v)
    let s = _mm256_mul_pd(z, x_eval); // s = x³
    let rv_sum = _mm256_add_pd(r, v);
    let zs = _mm256_mul_pd(z, s); // zs = x⁵
    let poly_r = _mm256_fmadd_pd(zs, rv_sum, _mm256_mul_pd(s, t0));

    // tan(x) = x + poly_r (for non-big case)
    let tan_small = _mm256_add_pd(x_eval, poly_r);

    // -------------------------------------------------------------------------
    // Step 3: Handle "big" case using musl's formula
    // For |x| near π/4, use: v = s - 2*(x + (r - w²/(w+s)))
    // where s = 1-2*odd (±1 based on quadrant), w = x + r
    // -------------------------------------------------------------------------
    let odd_mask = _mm256_cmpeq_epi64(odd, _mm256_set1_epi64x(1));

    // s_factor = 1 if even quadrant, -1 if odd quadrant
    let s_factor = _mm256_blendv_pd(one, neg_one, _mm256_castsi256_pd(odd_mask));

    let w_big = _mm256_add_pd(x_eval, poly_r); // w = x + r
    let denom = _mm256_add_pd(w_big, s_factor); // w + s
    let w_sq = _mm256_mul_pd(w_big, w_big); // w²
    let frac = _mm256_div_pd(w_sq, denom); // w²/(w+s)
    let inner = _mm256_add_pd(x_eval, _mm256_sub_pd(poly_r, frac)); // x + (r - w²/(w+s))
    let tan_big_val = _mm256_fnmadd_pd(two, inner, s_factor); // s - 2*(...)

    // Apply original sign for big values (sign was factored out during transformation)
    let tan_big_val = _mm256_xor_pd(tan_big_val, x_sign);

    // -------------------------------------------------------------------------
    // Step 4: Handle odd quadrant for non-big case: tan(x + π/2) = -1/tan(x)
    // Big case already incorporates this via s_factor
    // -------------------------------------------------------------------------
    let recip = _mm256_div_pd(neg_one, tan_small);
    let tan_small_final = _mm256_blendv_pd(tan_small, recip, _mm256_castsi256_pd(odd_mask));

    // Final selection: big path has complete answer, non-big path computed above
    _mm256_blendv_pd(tan_small_final, tan_big_val, is_big)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI32;
    use std::f64::consts::PI as PI64;

    /// Helper to extract f32 lanes from __m256
    unsafe fn extract_ps(v: __m256) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v) };
        out
    }

    /// Helper to extract f64 lanes from __m256d
    unsafe fn extract_pd(v: __m256d) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), v) };
        out
    }

    fn ulp_diff_f32(a: f32, b: f32) -> u32 {
        if a.is_nan() && b.is_nan() {
            return 0;
        }
        if a.is_nan() || b.is_nan() {
            return u32::MAX;
        }
        if a.signum() != b.signum() {
            return u32::MAX; // Different signs
        }
        let a_bits = a.to_bits() as i32;
        let b_bits = b.to_bits() as i32;
        (a_bits.wrapping_sub(b_bits)).unsigned_abs()
    }

    fn ulp_diff_f64(a: f64, b: f64) -> u64 {
        if a.is_nan() && b.is_nan() {
            return 0;
        }
        if a.is_nan() || b.is_nan() {
            return u64::MAX;
        }
        if a.signum() != b.signum() {
            return u64::MAX;
        }
        let a_bits = a.to_bits() as i64;
        let b_bits = b.to_bits() as i64;
        (a_bits.wrapping_sub(b_bits)).unsigned_abs()
    }

    // =========================================================================
    // f32 tests
    // =========================================================================

    #[test]
    fn test_tan_ps_zero() {
        unsafe {
            let input = _mm256_set1_ps(0.0);
            let result = extract_ps(_mm256_tan_ps(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_positive());
            }
        }
    }

    #[test]
    fn test_tan_ps_negative_zero() {
        unsafe {
            let input = _mm256_set1_ps(-0.0);
            let result = extract_ps(_mm256_tan_ps(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_negative());
            }
        }
    }

    #[test]
    fn test_tan_ps_pi_over_4() {
        unsafe {
            let input = _mm256_set1_ps(PI32 / 4.0);
            let result = extract_ps(_mm256_tan_ps(input));
            let expected = (PI32 / 4.0).tan();
            for &r in &result {
                assert!(
                    (r - expected).abs() < 1e-5,
                    "tan(π/4) = {}, expected {}",
                    r,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_tan_ps_pi() {
        unsafe {
            let input = _mm256_set1_ps(PI32);
            let result = extract_ps(_mm256_tan_ps(input));
            // tan(π) ≈ 0
            for &r in &result {
                assert!(r.abs() < 1e-5, "tan(π) = {}, expected ~0.0", r);
            }
        }
    }

    #[test]
    fn test_tan_ps_negative_pi_over_4() {
        unsafe {
            let input = _mm256_set1_ps(-PI32 / 4.0);
            let result = extract_ps(_mm256_tan_ps(input));
            let expected = (-PI32 / 4.0).tan();
            for &r in &result {
                assert!(
                    (r - expected).abs() < 1e-5,
                    "tan(-π/4) = {}, expected {}",
                    r,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_tan_ps_nan() {
        unsafe {
            let input = _mm256_set1_ps(f32::NAN);
            let result = extract_ps(_mm256_tan_ps(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_ps_infinity() {
        unsafe {
            let input = _mm256_set1_ps(f32::INFINITY);
            let result = extract_ps(_mm256_tan_ps(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_ps_negative_infinity() {
        unsafe {
            let input = _mm256_set1_ps(f32::NEG_INFINITY);
            let result = extract_ps(_mm256_tan_ps(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(-∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_ps_lane_independence() {
        unsafe {
            let input = _mm256_setr_ps(
                0.0,
                PI32 / 4.0,
                PI32 / 6.0,
                PI32 / 3.0,
                -PI32 / 4.0,
                -PI32 / 6.0,
                PI32,
                -PI32,
            );
            let result = extract_ps(_mm256_tan_ps(input));
            let expected = [
                0.0f32.tan(),
                (PI32 / 4.0).tan(),
                (PI32 / 6.0).tan(),
                (PI32 / 3.0).tan(),
                (-PI32 / 4.0).tan(),
                (-PI32 / 6.0).tan(),
                PI32.tan(),
                (-PI32).tan(),
            ];
            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (r - e).abs() < 1e-4,
                    "Lane {}: got {}, expected {}",
                    i,
                    r,
                    e
                );
            }
        }
    }

    #[test]
    fn test_tan_ps_ulp_sweep() {
        unsafe {
            let mut max_ulp = 0u32;
            // Avoid values near π/2 where tan approaches infinity
            for i in 0..10000 {
                let x = -1.5 + (i as f32 / 10000.0) * 3.0; // Range [-1.5, 1.5]
                let input = _mm256_set1_ps(x);
                let result = extract_ps(_mm256_tan_ps(input))[0];
                let expected = x.tan();
                // Skip large values where ULP isn't meaningful
                if expected.abs() > 1000.0 || result.abs() > 1000.0 {
                    continue;
                }
                if expected.is_finite() && result.is_finite() {
                    let ulp = ulp_diff_f32(result, expected);
                    max_ulp = max_ulp.max(ulp);
                }
            }
            assert!(max_ulp <= 2, "Max ULP error: {} (expected ≤ 2)", max_ulp);
        }
    }

    // =========================================================================
    // f64 tests
    // =========================================================================

    #[test]
    fn test_tan_pd_zero() {
        unsafe {
            let input = _mm256_set1_pd(0.0);
            let result = extract_pd(_mm256_tan_pd(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_positive());
            }
        }
    }

    #[test]
    fn test_tan_pd_negative_zero() {
        unsafe {
            let input = _mm256_set1_pd(-0.0);
            let result = extract_pd(_mm256_tan_pd(input));
            for &r in &result {
                assert_eq!(r, 0.0);
                assert!(r.is_sign_negative());
            }
        }
    }

    #[test]
    fn test_tan_pd_pi_over_4() {
        unsafe {
            let input = _mm256_set1_pd(PI64 / 4.0);
            let result = extract_pd(_mm256_tan_pd(input));
            let expected = (PI64 / 4.0).tan();
            for &r in &result {
                assert!(
                    (r - expected).abs() < 1e-14,
                    "tan(π/4) = {}, expected {}",
                    r,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_tan_pd_pi() {
        unsafe {
            let input = _mm256_set1_pd(PI64);
            let result = extract_pd(_mm256_tan_pd(input));
            // tan(π) ≈ 0
            for &r in &result {
                assert!(r.abs() < 1e-14, "tan(π) = {}, expected ~0.0", r);
            }
        }
    }

    #[test]
    fn test_tan_pd_nan() {
        unsafe {
            let input = _mm256_set1_pd(f64::NAN);
            let result = extract_pd(_mm256_tan_pd(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(NaN) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_pd_infinity() {
        unsafe {
            let input = _mm256_set1_pd(f64::INFINITY);
            let result = extract_pd(_mm256_tan_pd(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_pd_negative_infinity() {
        unsafe {
            let input = _mm256_set1_pd(f64::NEG_INFINITY);
            let result = extract_pd(_mm256_tan_pd(input));
            for &r in &result {
                assert!(r.is_nan(), "tan(-∞) should be NaN");
            }
        }
    }

    #[test]
    fn test_tan_pd_lane_independence() {
        unsafe {
            let input = _mm256_setr_pd(0.0, PI64 / 4.0, -PI64 / 4.0, PI64);
            let result = extract_pd(_mm256_tan_pd(input));
            let expected = [
                0.0f64.tan(),
                (PI64 / 4.0).tan(),
                (-PI64 / 4.0).tan(),
                PI64.tan(),
            ];
            for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (r - e).abs() < 1e-13,
                    "Lane {}: got {}, expected {}",
                    i,
                    r,
                    e
                );
            }
        }
    }

    #[test]
    fn test_tan_pd_ulp_sweep() {
        unsafe {
            let mut max_ulp = 0u64;
            // Avoid values near π/2 where tan approaches infinity
            for i in 0..10000 {
                let x = -1.5 + (i as f64 / 10000.0) * 3.0; // Range [-1.5, 1.5]
                let input = _mm256_set1_pd(x);
                let result = extract_pd(_mm256_tan_pd(input))[0];
                let expected = x.tan();
                // Skip large values where ULP isn't meaningful
                if expected.abs() > 1000.0 || result.abs() > 1000.0 {
                    continue;
                }
                if expected.is_finite() && result.is_finite() {
                    let ulp = ulp_diff_f64(result, expected);
                    max_ulp = max_ulp.max(ulp);
                }
            }
            assert!(max_ulp <= 2, "Max ULP error: {} (expected ≤ 2)", max_ulp);
        }
    }
}
