//! Constants for `acos` implementations.
//!
//! These constants are taken verbatim from musl libc's `acosf.c` and `acos.c`
//! (which descend from Sun's fdlibm). They have specific bit patterns designed
//! for the Dekker two-sum representation and rational approximation.
//!
//! Do not replace them with Rust's `std::f32::consts` or `std::f64::consts` values.

#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]

// ===========================================================================
// f32 Constants (from musl acosf.c / fdlibm)
// ===========================================================================

/// High word of π/2 in the Dekker two-sum `PIO2_HI + PIO2_LO = π/2`.
///
/// The split keeps `PIO2_HI` exact to 23 significant bits so that arithmetic
/// involving π/2 avoids catastrophic cancellation.
pub(crate) const PIO2_HI_32: f32 = 1.570_796_3;

/// Low word of π/2: `PIO2_LO = π/2 − PIO2_HI` (≈ 7.55e-8).
///
/// Adding `PIO2_LO` to a result that already includes `PIO2_HI` restores the
/// full precision of π/2.
pub(crate) const PIO2_LO_32: f32 = 7.549_789_4e-8;

/// Padé numerator coefficient: z⁰ term of p(z) in r(z) = z·p(z)/q(z).
pub(crate) const P_S0_32: f32 = 1.666_658_7e-1;

/// Padé numerator coefficient: z¹ term of p(z).
pub(crate) const P_S1_32: f32 = -4.274_342_2e-2;

/// Padé numerator coefficient: z² term of p(z).
pub(crate) const P_S2_32: f32 = -8.656_363e-3;

/// Padé denominator coefficient: z¹ term of q(z) = 1 + z·Q_S1.
pub(crate) const Q_S1_32: f32 = -7.066_296_3e-1;

/// Smallest positive normal `f32` (≈ 2⁻¹²⁶ ≈ 1.175e-38).
///
/// Adding this to `2·PIO2_HI` when `x = -1` forces the FPU to raise the
/// IEEE 754 *inexact* flag, which is mandated because π is not exactly
/// representable in binary floating point. The value is so small (~10⁷ times
/// smaller than 1 ULP of π in f32) that it has no effect on the numerical
/// result.
pub(crate) const X1P_120_32: f32 = 1.175_494_4e-38;

// ===========================================================================
// f64 Constants (from musl acos.c / fdlibm)
// ===========================================================================

/// High word of π/2 in the Dekker two-sum `PIO2_HI_64 + PIO2_LO_64 = π/2`.
///
/// The split keeps `PIO2_HI_64` exact so that arithmetic involving π/2
/// avoids catastrophic cancellation.
pub(crate) const PIO2_HI_64: f64 = 1.570_796_326_794_896_558_00e+00;

/// Low word of π/2: `PIO2_LO_64 = π/2 − PIO2_HI_64`.
///
/// Adding `PIO2_LO_64` to a result that already includes `PIO2_HI_64` restores
/// the full precision of π/2.
pub(crate) const PIO2_LO_64: f64 = 6.123_233_995_736_766_035_87e-17;

/// Padé numerator coefficient: z⁰ term of p(z) in r(z) = z·p(z)/q(z).
pub(crate) const P_S0_64: f64 = 1.666_666_666_666_666_574_15e-01;

/// Padé numerator coefficient: z¹ term of p(z).
pub(crate) const P_S1_64: f64 = -3.255_658_186_224_009_154_05e-01;

/// Padé numerator coefficient: z² term of p(z).
pub(crate) const P_S2_64: f64 = 2.012_125_321_348_629_258_81e-01;

/// Padé numerator coefficient: z³ term of p(z).
pub(crate) const P_S3_64: f64 = -4.005_553_450_067_941_140_27e-02;

/// Padé numerator coefficient: z⁴ term of p(z).
pub(crate) const P_S4_64: f64 = 7.915_349_942_898_145_321_76e-04;

/// Padé numerator coefficient: z⁵ term of p(z).
pub(crate) const P_S5_64: f64 = 3.479_331_075_960_211_675_70e-05;

/// Padé denominator coefficient: z¹ term of q(z) = 1 + z·(...).
pub(crate) const Q_S1_64: f64 = -2.403_394_911_734_414_218_78e+00;

/// Padé denominator coefficient: z² term of q(z).
pub(crate) const Q_S2_64: f64 = 2.020_945_760_233_505_694_71e+00;

/// Padé denominator coefficient: z³ term of q(z).
pub(crate) const Q_S3_64: f64 = -6.882_839_716_054_532_930_30e-01;

/// Padé denominator coefficient: z⁴ term of q(z).
pub(crate) const Q_S4_64: f64 = 7.703_815_055_590_193_527_91e-02;
