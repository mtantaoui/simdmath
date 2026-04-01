//! Constants for `atan` implementations.
//!
//! The `atan` function uses argument reduction followed by a minimax polynomial.
//!
//! **f32**: degree-9 odd polynomial on `[-1, 1]`, argument reduced via `atan(x) = π/2 - atan(1/x)`.
//!
//! **f64**: musl libc 4-range reduction with a degree-11 polynomial split into
//! odd/even parts for efficiency. The 4 breakpoints are `|x| ∈ {7/16, 11/16, 19/16, 39/16}`.
//! Each range uses a two-sum `atanhi + atanlo` offset for ≤ 1 ULP accuracy.
//!
//! All constants are taken verbatim from **musl libc `atan.c`** (fdlibm descent).

#![allow(clippy::excessive_precision)]

// ===========================================================================
// f32 Constants
// ===========================================================================

/// π/2 for f32, used as the correction for |x| > 1 reduction.
pub(crate) const FRAC_PI_2_32: f32 = 1.570_796_326_794_896_619_231_3;

/// atan polynomial coefficient 0 (x term, ≈ 1).
pub(crate) const ATAN_P0_32: f32 = 0.999_999_871_164;
/// atan polynomial coefficient 1 (x³ term).
pub(crate) const ATAN_P1_32: f32 = -0.333_325_240_026;
/// atan polynomial coefficient 2 (x⁵ term).
pub(crate) const ATAN_P2_32: f32 = 0.199_848_846_856;
/// atan polynomial coefficient 3 (x⁷ term).
pub(crate) const ATAN_P3_32: f32 = -0.141_548_060_419;
/// atan polynomial coefficient 4 (x⁹ term).
pub(crate) const ATAN_P4_32: f32 = 0.104_775_391_987;
/// atan polynomial coefficient 5 (x¹¹ term).
pub(crate) const ATAN_P5_32: f32 = -0.071_943_845_424_6;
/// atan polynomial coefficient 6 (x¹³ term).
pub(crate) const ATAN_P6_32: f32 = 0.039_345_413_147_9;
/// atan polynomial coefficient 7 (x¹⁵ term).
pub(crate) const ATAN_P7_32: f32 = -0.014_152_348_036_2;
/// atan polynomial coefficient 8 (x¹⁷ term).
pub(crate) const ATAN_P8_32: f32 = 0.002_398_139_012_51;

// ===========================================================================
// f64 Constants — from musl libc atan.c (fdlibm)
// ===========================================================================

// ---------------------------------------------------------------------------
// Two-sum offsets for the 4 argument-reduction ranges
//
// Each atan(breakpoint) is split into hi + lo so that:
//   atan(breakpoint) = atanhi[i] + atanlo[i]
// with atanhi exact to the last bit and atanlo capturing the rounding residual.
// ---------------------------------------------------------------------------

/// High part of atan(0.5) — range id=0 (7/16 ≤ |x| < 11/16).
/// IEEE hex: 0x3FDDAC67_0561BB4F
pub(crate) const ATANHI_0: f64 = 4.636_476_090_008_060_935_15e-01;
/// Low part of atan(0.5).
/// IEEE hex: 0x3C7A2B7F_222F65E2
pub(crate) const ATANLO_0: f64 = 2.269_877_745_296_168_709_24e-17;

/// High part of atan(1.0) = π/4 — range id=1 (11/16 ≤ |x| < 19/16).
/// IEEE hex: 0x3FE921FB_54442D18
pub(crate) const ATANHI_1: f64 = 7.853_981_633_974_482_789_99e-01;
/// Low part of atan(1.0).
/// IEEE hex: 0x3C81A626_33145C07
pub(crate) const ATANLO_1: f64 = 3.061_616_997_868_383_017_93e-17;

/// High part of atan(1.5) — range id=2 (19/16 ≤ |x| < 39/16).
/// IEEE hex: 0x3FEF730B_D281F69B
pub(crate) const ATANHI_2: f64 = 9.827_937_232_473_290_540_82e-01;
/// Low part of atan(1.5).
/// IEEE hex: 0x3C700788_7AF0CBBD
pub(crate) const ATANLO_2: f64 = 1.390_331_103_123_099_845_16e-17;

/// High part of atan(∞) = π/2 — range id=3 (|x| ≥ 39/16).
/// IEEE hex: 0x3FF921FB_54442D18
pub(crate) const ATANHI_3: f64 = 1.570_796_326_794_896_558_00e+00;
/// Low part of atan(∞).
/// IEEE hex: 0x3C91A626_33145C07
pub(crate) const ATANLO_3: f64 = 6.123_233_995_736_766_035_87e-17;

// ---------------------------------------------------------------------------
// Polynomial coefficients aT[0..10] — musl libc atan.c
//
// The polynomial is split into odd-indexed (s1) and even-indexed (s2) parts
// for efficiency. With z = t² and w = z²:
//   s1 = z*(aT[0] + w*(aT[2] + w*(aT[4] + w*(aT[6] + w*(aT[8] + w*aT[10])))))
//   s2 = w*(aT[1] + w*(aT[3] + w*(aT[5] + w*(aT[7] + w*aT[9]))))
//   correction = t * (s1 + s2)
// ---------------------------------------------------------------------------

/// aT[0] ≈ 1/3
pub(crate) const AT0: f64 = 3.333_333_333_333_293_180_27e-01;
/// aT[1] ≈ -1/5
pub(crate) const AT1: f64 = -1.999_999_999_987_648_324_76e-01;
/// aT[2] ≈ 1/7
pub(crate) const AT2: f64 = 1.428_571_427_250_346_637_11e-01;
/// aT[3] ≈ -1/9
pub(crate) const AT3: f64 = -1.111_111_040_546_235_578_80e-01;
/// aT[4] ≈ 1/11
pub(crate) const AT4: f64 = 9.090_887_133_436_506_561_96e-02;
/// aT[5] ≈ -1/13
pub(crate) const AT5: f64 = -7.691_876_205_044_829_994_95e-02;
/// aT[6] ≈ 1/15
pub(crate) const AT6: f64 = 6.661_073_137_387_531_206_69e-02;
/// aT[7] ≈ -1/17
pub(crate) const AT7: f64 = -5.833_570_133_790_573_486_45e-02;
/// aT[8] ≈ 1/19
pub(crate) const AT8: f64 = 4.976_877_994_615_932_360_17e-02;
/// aT[9] ≈ -1/21
pub(crate) const AT9: f64 = -3.653_157_274_421_691_552_70e-02;
/// aT[10] ≈ 1/23
pub(crate) const AT10: f64 = 1.628_582_011_536_578_236_23e-02;

// ---------------------------------------------------------------------------
// Range boundaries for argument reduction
// ---------------------------------------------------------------------------

/// Lower boundary: |x| < 7/16 → no reduction needed.
pub(crate) const ATAN_THRESH_0: f64 = 7.0 / 16.0; // 0.4375
/// Upper boundary of range 0: 7/16 ≤ |x| < 11/16.
pub(crate) const ATAN_THRESH_1: f64 = 11.0 / 16.0; // 0.6875
/// Upper boundary of range 1: 11/16 ≤ |x| < 19/16.
pub(crate) const ATAN_THRESH_2: f64 = 19.0 / 16.0; // 1.1875
/// Upper boundary of range 2: 19/16 ≤ |x| < 39/16.
pub(crate) const ATAN_THRESH_3: f64 = 39.0 / 16.0; // 2.4375
