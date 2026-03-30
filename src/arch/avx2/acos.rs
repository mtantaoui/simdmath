//! AVX2 SIMD implementation of `acos(x)` for `f32` vectors.
//!
//! # Algorithm
//!
//! This is a branchless, three-range minimax rational approximation ported
//! directly from **musl libc's `acosf.c`** (which itself descends from Sun's
//! fdlibm). It achieves **≤ 1 ULP** accuracy across the entire domain `[-1, 1]`
//! (verified by sweeping every 1024th `f32` in `[-1, 1]`, ~2 million values).
//!
//! Three FMA instructions fuse the compound multiply-add steps in each range,
//! eliminating one intermediate rounding per case:
//! - Case C: `fnmadd(x, r, pio2_lo)` instead of separate mul + sub
//! - Case D: `fmsub(r, s, pio2_lo)` instead of separate mul + sub
//! - Case E: `fmadd(r, s, c)` instead of separate mul + add
//!
//! The core building block is a Padé rational approximation `r(z)` that
//! approximates `(asin(√z)/√z − 1)/z` on `[0, 0.25]`, yielding:
//!
//! ```text
//! asin(x) ≈ x + x·r(x²)      for |x| ≤ 0.5
//! ```
//!
//! ## Three computational ranges
//!
//! | Range           | Identity used                                        |
//! |-----------------|------------------------------------------------------|
//! | `\|x\| < 0.5`  | `acos(x) = π/2 − asin(x)`                           |
//! | `x > 0.5`       | `acos(x) = 2·asin(√((1−x)/2))` (half-angle formula) |
//! | `x < −0.5`      | `acos(x) = π − 2·asin(√((1+x)/2))`                 |
//!
//! The half-angle formula avoids catastrophic cancellation near `|x| = 1`,
//! where the derivative of `acos(x)` diverges.
//!
//! ## Compensated sqrt (case: x > 0.5)
//!
//! For `x > 0.5` the sqrt argument `z = (1−x)/2` is very small, so a
//! compensated-sqrt is used for extra precision. The low 12 bits of `s = √z`
//! are masked off to produce `df`; then the rounding error is recovered as
//! `c = (z − df²)/(s + df)` using exact arithmetic (since `df` has only
//! 11 significant mantissa bits, `df²` fits exactly in 23 bits). This is the
//! **Dekker split** applied to square roots.
//!
//! ## Special values
//!
//! | Input        | Output          |
//! |--------------|-----------------|
//! | `1.0`        | `0.0`           |
//! | `-1.0`       | `π ≈ 3.1415927` |
//! | `\|x\| > 1` | `NaN`           |
//! | `NaN`        | `NaN`           |
//!
//! ## Blending strategy
//!
//! All result branches are computed unconditionally and blended at the end
//! using `_mm256_blendv_ps`. This eliminates branches and keeps all 8 lanes
//! in flight simultaneously.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::arch::avx2::abs::_mm256_abs_ps;

// ---------------------------------------------------------------------------
// Constants (from musl acosf.c / fdlibm)
// ---------------------------------------------------------------------------

/// High word of π/2 in the Dekker two-sum `PIO2_HI + PIO2_LO = π/2`.
///
/// The split keeps `PIO2_HI` exact to 23 significant bits so that arithmetic
/// involving π/2 avoids catastrophic cancellation.
// 
pub(crate) const PIO2_HI: f32 = 1.570_796_3;

/// Low word of π/2: `PIO2_LO = π/2 − PIO2_HI` (≈ 7.55e-8).
///
/// Adding `PIO2_LO` to a result that already includes `PIO2_HI` restores the
/// full precision of π/2.
pub(crate) const PIO2_LO: f32 = 7.549_789_4e-8;

/// Padé numerator coefficient: z⁰ term of p(z) in r(z) = z·p(z)/q(z).
pub(crate) const P_S0: f32 = 1.666_658_7e-1;

/// Padé numerator coefficient: z¹ term of p(z).
pub(crate) const P_S1: f32 = -4.274_342_2e-2;

/// Padé numerator coefficient: z² term of p(z).
pub(crate) const P_S2: f32 = -8.656_363e-3;

/// Padé denominator coefficient: z¹ term of q(z) = 1 + z·Q_S1.
pub(crate) const Q_S1: f32 = -7.066_296_3e-1;

/// Smallest positive normal `f32` (≈ 2⁻¹²⁶ ≈ 1.175e-38).
///
/// Adding this to `2·PIO2_HI` when `x = -1` forces the FPU to raise the
/// IEEE 754 *inexact* flag, which is mandated because π is not exactly
/// representable in binary floating point. The value is so small (~10⁷ times
/// smaller than 1 ULP of π in f32) that it has no effect on the numerical
/// result.
pub(crate) const X1P_120: f32 = 1.175_494_4e-38;

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/// Computes `acos(x)` for each lane of an AVX2 `__m256` register.
///
/// All 8 lanes are processed simultaneously without branches. The result
/// paths (small |x|, large positive, large negative, |x|=1, out-of-domain)
/// are computed unconditionally and merged with `_mm256_blendv_ps`.
///
/// # Safety
/// `x` must be a valid `__m256` register. No alignment or memory constraints.
#[inline]
pub(crate) unsafe fn _mm256_acos_ps(x: __m256) -> __m256 {
    // Broadcast scalar constants to SIMD registers once.
    let pio2_hi = _mm256_set1_ps(PIO2_HI);
    let pio2_lo = _mm256_set1_ps(PIO2_LO);
    let p_s0 = _mm256_set1_ps(P_S0);
    let p_s1 = _mm256_set1_ps(P_S1);
    let p_s2 = _mm256_set1_ps(P_S2);
    let q_s1 = _mm256_set1_ps(Q_S1);
    let x1p_120 = _mm256_set1_ps(X1P_120);
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let two = _mm256_set1_ps(2.0);
    let zero = _mm256_setzero_ps();

    // |x| — used to determine which range each lane falls into.
    let abs_x = _mm256_abs_ps(x);

    // -------------------------------------------------------------------------
    // Padé rational approximation r(z)
    //
    //   p(z) = P_S0 + z·(P_S1 + z·P_S2)   [degree-2 numerator, Horner form]
    //   q(z) = 1 + z·Q_S1                  [degree-1 denominator]
    //   r(z) = z·p(z) / q(z)
    //
    // r(z) ≈ (asin(√z)/√z − 1)/z on [0, 0.25], so asin(x) ≈ x + x·r(x²).
    // -------------------------------------------------------------------------
    let r = |z: __m256| -> __m256 {
        let p = _mm256_mul_ps(z, _mm256_fmadd_ps(z, _mm256_fmadd_ps(z, p_s2, p_s1), p_s0));
        let q = _mm256_fmadd_ps(z, q_s1, one);
        _mm256_div_ps(p, q)
    };

    // -------------------------------------------------------------------------
    // Lane classification masks (ordered, quiet: NaN → false for all)
    // -------------------------------------------------------------------------
    let is_abs_ge_1 = _mm256_cmp_ps(abs_x, one, _CMP_GE_OQ); // |x| >= 1
    let is_abs_eq_1 = _mm256_cmp_ps(abs_x, one, _CMP_EQ_OQ); // |x| == 1
    let is_abs_lt_half = _mm256_cmp_ps(abs_x, half, _CMP_LT_OQ); // |x| < 0.5
    let is_x_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ); // x < 0

    // Lanes with x in (-1, -0.5]: x is negative AND |x| >= 0.5.
    // _mm256_andnot_ps(a, b) = ~a & b
    let is_x_neg_large = _mm256_andnot_ps(is_abs_lt_half, is_x_neg);

    // -------------------------------------------------------------------------
    // Case A — |x| == 1  (exact boundary values)
    //
    //   acos( 1) = 0
    //   acos(-1) = 2·pio2_hi + x1p_120  ≈ π
    //              (x1p_120 raises the IEEE 754 inexact flag as mandated)
    // -------------------------------------------------------------------------
    let result_eq_1 = _mm256_blendv_ps(
        zero,                                                // x = +1 → 0
        _mm256_add_ps(_mm256_mul_ps(two, pio2_hi), x1p_120), // x = -1 → π
        is_x_neg,
    );

    // -------------------------------------------------------------------------
    // Case B — |x| > 1  (out-of-domain → NaN)
    //
    // x − x = 0 for all finite x; dividing 0/0 produces NaN. The subtraction
    // prevents compilers from constant-folding the expression.
    // -------------------------------------------------------------------------
    let nan = _mm256_div_ps(zero, _mm256_sub_ps(x, x));

    // For |x| >= 1: select exact result where |x| == 1, NaN otherwise.
    let result_ge_1 = _mm256_blendv_ps(nan, result_eq_1, is_abs_eq_1);

    // -------------------------------------------------------------------------
    // Case C — |x| < 0.5  (small argument)
    //
    // acos(x) = π/2 − asin(x) ≈ π/2 − x − x·r(x²)
    //         = pio2_hi − (x − (pio2_lo − x·r(x²)))
    //
    // The nested subtraction keeps pio2_hi and pio2_lo paired correctly so
    // that precision is not lost through the two-part representation of π/2.
    // -------------------------------------------------------------------------
    let z_small = _mm256_mul_ps(x, x);
    // Case C: pio2_lo − x·r(z) as a single FMA (fnmadd = −a·b + c).
    let result_small = _mm256_sub_ps(
        pio2_hi,
        _mm256_sub_ps(x, _mm256_fnmadd_ps(x, r(z_small), pio2_lo)),
    );

    // -------------------------------------------------------------------------
    // Case D — x < -0.5  (large negative argument)
    //
    // Identity: acos(x) = π − 2·asin(√((1+x)/2))
    //         = 2·(π/2 − asin(s))  where s = √z, z = (1+x)/2
    //         = 2·(pio2_hi − (s + w))
    //
    //   w = r(z)·s − pio2_lo   (the pio2_lo term completes the two-sum)
    // -------------------------------------------------------------------------
    let z_neg = _mm256_mul_ps(_mm256_add_ps(one, x), half);
    let s_neg = _mm256_sqrt_ps(z_neg);
    // Case D: r(z)·s − pio2_lo as a single FMA (fmsub = a·b − c).
    let w_neg = _mm256_fmsub_ps(r(z_neg), s_neg, pio2_lo);
    let result_neg = _mm256_mul_ps(two, _mm256_sub_ps(pio2_hi, _mm256_add_ps(s_neg, w_neg)));

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
    let z_pos = _mm256_mul_ps(_mm256_sub_ps(one, x), half);
    let s_pos = _mm256_sqrt_ps(z_pos);

    // Mask off the low 12 bits of the f32 bit representation to form df.
    let df = _mm256_castsi256_ps(_mm256_and_si256(
        _mm256_castps_si256(s_pos),
        _mm256_set1_epi32(0xfffff000_u32 as i32),
    ));
    // Recover the rounding error introduced by truncating s to df.
    let c_pos = _mm256_div_ps(
        _mm256_sub_ps(z_pos, _mm256_mul_ps(df, df)),
        _mm256_add_ps(s_pos, df),
    );
    // Case E: r(z)·s + c as a single FMA.
    let w_pos = _mm256_fmadd_ps(r(z_pos), s_pos, c_pos);
    let result_pos = _mm256_mul_ps(two, _mm256_add_ps(df, w_pos));

    // -------------------------------------------------------------------------
    // Merge: blend all cases. Priority (highest → lowest):
    //   |x| >= 1   → result_ge_1  (exact or NaN)
    //   |x| < 0.5  → result_small (Case C)
    //   x ∈ (-1,-0.5] → result_neg (Case D)
    //   x ∈ [0.5, 1)  → result_pos (Case E)
    // -------------------------------------------------------------------------
    let result_large = _mm256_blendv_ps(result_pos, result_neg, is_x_neg_large);
    let result_valid = _mm256_blendv_ps(result_large, result_small, is_abs_lt_half);
    _mm256_blendv_ps(result_valid, result_ge_1, is_abs_ge_1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance: ~2 ULPs at the scale of π. One ULP of π in f32 ≈ 2.38e-7.
    const TOL: f32 = 5e-7;

    /// Load 8 copies of `val`, call `_mm256_acos_ps`, and return lane 0.
    unsafe fn acos_scalar(val: f32) -> f32 {
        let v = _mm256_set1_ps(val);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), _mm256_acos_ps(v));
        out[0]
    }

    // ---- Special / boundary values -------------------------------------------

    #[test]
    fn acos_of_one_is_zero() {
        unsafe {
            assert_eq!(acos_scalar(1.0), 0.0);
        }
    }

    #[test]
    fn acos_of_neg_one_is_pi() {
        unsafe {
            let result = acos_scalar(-1.0);
            assert!(
                (result - std::f32::consts::PI).abs() < TOL,
                "acos(-1) = {result}, expected π ≈ {}",
                std::f32::consts::PI
            );
        }
    }

    #[test]
    fn acos_of_zero_is_pio2() {
        unsafe {
            let result = acos_scalar(0.0);
            assert!(
                (result - std::f32::consts::FRAC_PI_2).abs() < TOL,
                "acos(0) = {result}"
            );
        }
    }

    #[test]
    fn acos_of_neg_zero_equals_acos_of_pos_zero() {
        // -0.0 and +0.0 are equal in IEEE 754, so acos(-0.0) == acos(0.0) == π/2.
        unsafe {
            let result = acos_scalar(-0.0f32);
            assert!(
                (result - std::f32::consts::FRAC_PI_2).abs() < TOL,
                "acos(-0.0) = {result}"
            );
        }
    }

    // ---- Range-boundary values (|x| = 0.5) -----------------------------------

    #[test]
    fn acos_of_pos_half_is_pi_over_3() {
        // acos(0.5) = π/3
        unsafe {
            let result = acos_scalar(0.5);
            let expected = std::f32::consts::PI / 3.0;
            assert!(
                (result - expected).abs() < TOL,
                "acos(0.5) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn acos_of_neg_half_is_2pi_over_3() {
        // acos(-0.5) = 2π/3
        unsafe {
            let result = acos_scalar(-0.5);
            let expected = 2.0 * std::f32::consts::PI / 3.0;
            assert!(
                (result - expected).abs() < TOL,
                "acos(-0.5) = {result}, expected {expected}"
            );
        }
    }

    // ---- Common angles --------------------------------------------------------

    #[test]
    fn acos_of_sqrt2_over_2_is_pi_over_4() {
        // acos(√2/2) = π/4
        unsafe {
            let result = acos_scalar(std::f32::consts::FRAC_1_SQRT_2);
            let expected = std::f32::consts::FRAC_PI_4;
            assert!(
                (result - expected).abs() < TOL,
                "acos(√2/2) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn acos_of_neg_sqrt2_over_2_is_3pi_over_4() {
        // acos(-√2/2) = 3π/4
        unsafe {
            let result = acos_scalar(-std::f32::consts::FRAC_1_SQRT_2);
            let expected = 3.0 * std::f32::consts::FRAC_PI_4;
            assert!(
                (result - expected).abs() < TOL,
                "acos(-√2/2) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn acos_of_sqrt3_over_2_is_pi_over_6() {
        // acos(√3/2) = π/6
        unsafe {
            let result = acos_scalar(3.0f32.sqrt() / 2.0);
            let expected = std::f32::consts::FRAC_PI_6;
            assert!(
                (result - expected).abs() < TOL,
                "acos(√3/2) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn acos_of_neg_sqrt3_over_2_is_5pi_over_6() {
        // acos(-√3/2) = 5π/6
        unsafe {
            let result = acos_scalar(-3.0f32.sqrt() / 2.0);
            let expected = 5.0 * std::f32::consts::FRAC_PI_6;
            assert!(
                (result - expected).abs() < TOL,
                "acos(-√3/2) = {result}, expected {expected}"
            );
        }
    }

    // ---- Out-of-domain inputs → NaN ------------------------------------------

    #[test]
    fn acos_above_one_is_nan() {
        unsafe {
            assert!(acos_scalar(1.5).is_nan());
        }
    }

    #[test]
    fn acos_below_neg_one_is_nan() {
        unsafe {
            assert!(acos_scalar(-1.5).is_nan());
        }
    }

    #[test]
    fn acos_of_infinity_is_nan() {
        unsafe {
            assert!(acos_scalar(f32::INFINITY).is_nan());
        }
    }

    #[test]
    fn acos_of_neg_infinity_is_nan() {
        unsafe {
            assert!(acos_scalar(f32::NEG_INFINITY).is_nan());
        }
    }

    #[test]
    fn acos_of_nan_is_nan() {
        unsafe {
            assert!(acos_scalar(f32::NAN).is_nan());
        }
    }

    // ---- All 8 lanes processed correctly -------------------------------------

    #[test]
    fn acos_processes_all_8_lanes_independently() {
        // Mix of inputs spanning all three computational ranges.
        let inputs = [0.0f32, 0.5, -0.5, 0.9, -0.9, 1.0, -1.0, 0.25];
        unsafe {
            let v = _mm256_loadu_ps(inputs.as_ptr());
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), _mm256_acos_ps(v));

            let expected: [f32; 8] = inputs.map(|x| x.acos());
            for (i, (&r, &e)) in out.iter().zip(&expected).enumerate() {
                assert!(
                    (r - e).abs() < TOL,
                    "lane {i}: acos({}) = {r}, expected {e}",
                    inputs[i]
                );
            }
        }
    }

    // ---- ULP accuracy sweep --------------------------------------------------

    /// Verify ≤ 1 ULP vs correctly-rounded `(x as f64).acos() as f32` for
    /// every 1024th `f32` in `[-1, 1]` (~2 million values).
    #[test]
    fn max_ulp_error_is_at_most_1() {
        let mut max_ulp: u32 = 0;
        let mut worst_x: f32 = 0.0;
        let mut bits: u32 = 0u32;
        loop {
            let x = f32::from_bits(bits);
            if x.abs() <= 1.0 {
                let true_val = (x as f64).acos() as f32;
                let our_val = unsafe { acos_scalar(x) };
                let d = (our_val.to_bits() as i32 - true_val.to_bits() as i32).unsigned_abs();
                if d > max_ulp { max_ulp = d; worst_x = x; }
            }
            bits = bits.wrapping_add(1024);
            if bits == 0 { break; }
        }
        assert!(
            max_ulp <= 1,
            "max ULP {max_ulp} at x={worst_x:.8} — expected ≤ 1"
        );
    }
}
