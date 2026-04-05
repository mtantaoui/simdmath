//! Constants for `atan2` implementations.
//!
//! The `atan2(y, x)` function computes the angle θ of the point `(x, y)` in radians,
//! measured counter-clockwise from the positive x-axis. The result is in `(-π, π]`.
//!
//! These constants are used to handle edge cases and provide the necessary π/2, π,
//! and 3π/4 values with proper two-sum splits for full precision.
//!
//! All constants are taken from **musl libc `atan2f.c`** (fdlibm descent).

// These are intentional musl libc constants with specific bit patterns, not approximations.
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]

// ===========================================================================
// f32 Constants
// ===========================================================================

/// High part of π in two-sum: `PI_HI_32 + PI_LO_32 = π` to full f32 precision.
/// IEEE 754 hex: 0x40490FDB
pub(crate) const PI_HI_32: f32 = 3.141_592_741_01;

/// Low part of π: captures the rounding residual of `π - PI_HI_32`.
/// IEEE 754 hex: 0xB3BBBD2E (≈ -8.74e-8)
pub(crate) const PI_LO_32: f32 = -8.742_278e-8;

/// π/2 for f32.
pub(crate) const FRAC_PI_2_32: f32 = 1.570_796_370_51;

/// π/4 for f32.
pub(crate) const FRAC_PI_4_32: f32 = 0.785_398_185_25;

/// 3π/4 for f32.
pub(crate) const FRAC_3_PI_4_32: f32 = 2.356_194_496_155;

/// Exponent threshold for detecting |y/x| > 2^26 (tiny x relative to y).
/// When `exp(|y|) - exp(|x|) > 26`, the ratio is so large that atan2 ≈ ±π/2.
pub(crate) const HUGE_RATIO_THRESHOLD_32: i32 = 26 << 23;

// ===========================================================================
// f64 Constants
// ===========================================================================

/// High part of π in two-sum for f64.
pub(crate) const PI_HI_64: f64 = 3.141_592_653_589_793_115_998;

/// Low part of π for f64.
pub(crate) const PI_LO_64: f64 = 1.224_646_799_147_353_207_17e-16;

/// π/2 for f64.
pub(crate) const FRAC_PI_2_64: f64 = 1.570_796_326_794_896_558_00;

/// π/4 for f64.
pub(crate) const FRAC_PI_4_64: f64 = 0.785_398_163_397_448_279_00;

/// 3π/4 for f64.
pub(crate) const FRAC_3_PI_4_64: f64 = 2.356_194_490_192_344_836_91;

/// Exponent threshold for detecting |y/x| > 2^60 for f64.
pub(crate) const HUGE_RATIO_THRESHOLD_64: i64 = 60 << 52;
