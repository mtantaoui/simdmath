//! Constants for SIMD `exp(x)` (exponential) implementations.
//!
//! These constants are derived from Sun Microsystems' fdlibm `e_exp.c`
//! (via musl libc). They have specific bit patterns designed for numerical accuracy.
//!
//! # Algorithm Overview
//!
//! The exponential function is computed using:
//! 1. **Argument reduction**: Reduce `x` to `r` via `x = k*ln(2) + r` where
//!    `|r| ≤ ln(2)/2 ≈ 0.3466`. The integer `k` is chosen as `round(x / ln(2))`.
//! 2. **Polynomial approximation**: Compute `exp(r) ≈ 1 + r + r*P(r²)` where
//!    `P` is a degree-5 minimax polynomial in `r²`.
//! 3. **Reconstruction**: `exp(x) = 2^k * exp(r)` by adding `k` to the IEEE 754 exponent.
//!
//! # Note
//!
//! `ln(2)` is split into high and low parts (`LN2_HI + LN2_LO`) so that
//! `k * LN2_HI` is computed exactly.
//!
//! The inverse `1/ln(2) = log₂(e)` is stored as `LN2_INV` for the initial
//! scaling step.

// Allow clippy warnings for fdlibm constants - they have specific bit patterns
#![allow(clippy::excessive_precision)]

// =============================================================================
// f64 polynomial coefficients (P1–P5) from fdlibm e_exp.c
// =============================================================================

// These are the coefficients of the minimax polynomial approximation:
//   P(r²) = P1 + r²*(P2 + r²*(P3 + r²*(P4 + r²*P5)))
// such that exp(r) ≈ 1 + r + r²/2 * (1 - r*P(r²) / (r*P(r²) - 2))

/// P1 coefficient (≈ 1/6 = 1.66...e-1)
#[allow(dead_code)]
pub const P1_64: f64 = f64::from_bits(0x3FC555555555553E); //  1.66666666666666019037e-01

/// P2 coefficient (≈ -1/360 = -2.77...e-3)
#[allow(dead_code)]
pub const P2_64: f64 = f64::from_bits(0xBF66C16C16BEBD93); // -2.77777777770155933842e-03

/// P3 coefficient (≈ 1/15120 = 6.61...e-5)
#[allow(dead_code)]
pub const P3_64: f64 = f64::from_bits(0x3F11566AAF25DE2C); //  6.61375632143793436117e-05

/// P4 coefficient (≈ -1/604800 = -1.65...e-6)
#[allow(dead_code)]
pub const P4_64: f64 = f64::from_bits(0xBEBBBD41C5D26BF1); // -1.65339022054218515266e-06

/// P5 coefficient (≈ 4.13...e-8)
#[allow(dead_code)]
pub const P5_64: f64 = f64::from_bits(0x3E66376972BEA4D0); //  4.13813679705723846039e-08

// =============================================================================
// ln(2) split for extended precision argument reduction
// =============================================================================

/// High part of ln(2) — only 33 significant bits so that `k * LN2_HI` is exact
/// for integer k with |k| < 2^20.
#[allow(dead_code)]
pub const LN2_HI_64: f64 = f64::from_bits(0x3FE62E42FEE00000); //  6.93147180369123816490e-01

/// Low part of ln(2): `ln(2) = LN2_HI + LN2_LO` to ~70 bits of precision.
#[allow(dead_code)]
pub const LN2_LO_64: f64 = f64::from_bits(0x3DEA39EF35793C76); //  1.90821492927058500170e-10

/// 1/ln(2) = log₂(e) — used for initial argument scaling: k = round(x / ln(2)).
#[allow(dead_code)]
pub const LN2_INV_64: f64 = f64::from_bits(0x3FF71547652B82FE); //  1.44269504088896338700e+00

/// Half of ln(2), used in the magic-number rounding trick.
#[allow(dead_code)]
pub const HALF_LN2_64: f64 = f64::from_bits(0x3FE62E42FEFA39EF); //  6.93147180559945286227e-01

// =============================================================================
// Overflow / underflow thresholds
// =============================================================================

/// Overflow threshold for f64: x > this → +∞.
/// This is ln(2^1024) ≈ 709.78.
#[allow(dead_code)]
pub const OVERFLOW_THRESH_64: f64 = f64::from_bits(0x40862E42FEFA39EF); //  7.09782712893383973096e+02

/// Underflow threshold for f64: x < this → 0.
/// This is ln(2^-1075) ≈ -745.13.
#[allow(dead_code)]
pub const UNDERFLOW_THRESH_64: f64 = f64::from_bits(0xC0874910D52D3051); // -7.45133219101941108420e+02

/// Threshold below which exp(x) - 1 ≈ x: |x| < 2^-54.
#[allow(dead_code)]
pub const TINY_THRESH_64: f64 = f64::from_bits(0x3C90000000000000); //  2^-54 ≈ 5.55e-17
