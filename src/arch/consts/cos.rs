//! Constants for SIMD `cos(x)` implementations.
//!
//! These constants are derived from musl libc's `cosf.c`, `cos.c`, `__cosdf.c`,
//! `__sindf.c`, `__cos.c`, `__sin.c`, and `__rem_pio2f.c` (which in turn come
//! from Sun Microsystems' fdlibm).
//!
//! # Algorithm Overview
//!
//! The cosine function is computed using:
//! 1. **Argument reduction**: Reduce `x` to `y` in `[-π/4, π/4]` via `y = x - n*(π/2)`
//! 2. **Quadrant selection**: Based on `n mod 4`, compute `±sin(y)` or `±cos(y)`
//! 3. **Polynomial approximation**: Use minimax polynomials for the kernel functions
//!
//! ## Quadrant Table (n mod 4)
//!
//! | n | cos(x) |
//! |---|--------|
//! | 0 |  cos(y) |
//! | 1 | -sin(y) |
//! | 2 | -cos(y) |
//! | 3 |  sin(y) |

// Allow clippy warnings for musl constants - they have specific bit patterns
#![allow(clippy::excessive_precision, clippy::approx_constant)]

// =============================================================================
// f32 Constants (for computations done in f64 precision internally)
// =============================================================================

/// 2/π for argument reduction: computes n = round(x * (2/π))
#[allow(dead_code)]
pub const FRAC_2_PI_32: f64 = 0.6366197723675814; // 0x3FE45F306DC9C883

/// First 25 bits of π/2 for Cody-Waite reduction
#[allow(dead_code)]
pub const PIO2_1_32: f64 = 1.5707963109016418; // 0x3FF921FB50000000

/// π/2 - PIO2_1 (tail for extended precision)
#[allow(dead_code)]
pub const PIO2_1T_32: f64 = 1.5893254773528197e-08; // 0x3E5110B4611A6263

/// π/4 for range checking after reduction
#[allow(dead_code)]
pub const PIO4_32: f64 = 0.7853981633974483; // 0x3FE921FB54442D18

/// Threshold below which cos(x) ≈ 1: |x| < 2^-12
#[allow(dead_code)]
pub const TINY_COS_32: f32 = 2.44140625e-4; // 0x39800000 as f32

/// Threshold for medium-size argument reduction: |x| < 2^28 * (π/2)
#[allow(dead_code)]
pub const MEDIUM_32: u32 = 0x4DC90FDB;

// -----------------------------------------------------------------------------
// f32 Cosine kernel coefficients (__cosdf)
// Approximates cos(x) on [-π/4, π/4] with |error| < 2^-34.1
// cos(x) ≈ 1 + C0*x² + C1*x⁴ + C2*x⁶ + C3*x⁸
// -----------------------------------------------------------------------------

#[allow(dead_code)]
pub const C0_32: f64 = -0.499999997251031003120; // -0x1ffffffd0c5e81p-54
#[allow(dead_code)]
pub const C1_32: f64 = 0.0416666233237390631894; // 0x155553e1053a42p-57
#[allow(dead_code)]
pub const C2_32: f64 = -0.00138867637746099294692; // -0x16c087e80f1e27p-62
#[allow(dead_code)]
pub const C3_32: f64 = 0.0000243904487962774090654; // 0x199342e0ee5069p-68

// -----------------------------------------------------------------------------
// f32 Sine kernel coefficients (__sindf)
// Approximates sin(x)/x on [-π/4, π/4] with |error| < 2^-37.5
// sin(x) ≈ x + S1*x³ + S2*x⁵ + S3*x⁷ + S4*x⁹
// -----------------------------------------------------------------------------

#[allow(dead_code)]
pub const S1_32: f64 = -0.166666666416265235595; // -0x15555554cbac77p-55
#[allow(dead_code)]
pub const S2_32: f64 = 0.0083333293858894631756; // 0x111110896efbb2p-59
#[allow(dead_code)]
pub const S3_32: f64 = -0.000198393348360966317347; // -0x1a00f9e2cae774p-65
#[allow(dead_code)]
pub const S4_32: f64 = 0.0000027183114939898219064; // 0x16cd878c3b46a7p-71

// =============================================================================
// f64 Constants
// =============================================================================

/// 2/π for f64 argument reduction
#[allow(dead_code)]
pub const FRAC_2_PI_64: f64 = 6.36619772367581382433e-01; // 0x3FE45F306DC9C883

/// First 33 bits of π/2 for Cody-Waite reduction (f64)
#[allow(dead_code)]
pub const PIO2_1_64: f64 = 1.57079632673412561417e+00; // 0x3FF921FB54400000

/// Second part of π/2 for Cody-Waite reduction (f64)
#[allow(dead_code)]
pub const PIO2_1T_64: f64 = 6.07710050650619224932e-11; // 0x3DD0B4611A626331

/// Third part of π/2 for very high precision (f64)
#[allow(dead_code)]
pub const PIO2_2_64: f64 = 6.07710050630396597660e-11; // 0x3DD0B4611A600000

/// Fourth part of π/2 (f64)
#[allow(dead_code)]
pub const PIO2_2T_64: f64 = 2.02226624879595063154e-21; // 0x3BA3198A2E037073

/// π/4 for range checking (f64)
#[allow(dead_code)]
pub const PIO4_64: f64 = 7.85398163397448278999e-01; // 0x3FE921FB54442D18

/// Threshold below which cos(x) ≈ 1 for f64: |x| < 2^-27 * sqrt(2)
#[allow(dead_code)]
pub const TINY_COS_64: f64 = 2.6469779601696886e-8; // roughly 2^-27 * sqrt(2)

// -----------------------------------------------------------------------------
// f64 Cosine kernel coefficients (__cos)
// Approximates cos(x) - 1 + x²/2 on [-π/4, π/4] with |error| < 2^-58
// cos(x) ≈ 1 - x²/2 + C1*x⁴ + C2*x⁶ + C3*x⁸ + C4*x¹⁰ + C5*x¹² + C6*x¹⁴
// -----------------------------------------------------------------------------

#[allow(dead_code)]
pub const C1_64: f64 = 4.16666666666666019037e-02; // 0x3FA555555555554C
#[allow(dead_code)]
pub const C2_64: f64 = -1.38888888888741095749e-03; // 0xBF56C16C16C15177
#[allow(dead_code)]
pub const C3_64: f64 = 2.48015872894767294178e-05; // 0x3EFA01A019CB1590
#[allow(dead_code)]
pub const C4_64: f64 = -2.75573143513906633035e-07; // 0xBE927E4F809C52AD
#[allow(dead_code)]
pub const C5_64: f64 = 2.08757232129817482790e-09; // 0x3E21EE9EBDB4B1C4
#[allow(dead_code)]
pub const C6_64: f64 = -1.13596475577881948265e-11; // 0xBDA8FAE9BE8838D4

// -----------------------------------------------------------------------------
// f64 Sine kernel coefficients (__sin)
// Approximates sin(x)/x - 1 on [-π/4, π/4] with |error| < 2^-58
// sin(x) ≈ x + S1*x³ + S2*x⁵ + S3*x⁷ + S4*x⁹ + S5*x¹¹ + S6*x¹³
// -----------------------------------------------------------------------------

#[allow(dead_code)]
pub const S1_64: f64 = -1.66666666666666324348e-01; // 0xBFC5555555555549
#[allow(dead_code)]
pub const S2_64: f64 = 8.33333333332248946124e-03; // 0x3F8111111110F8A6
#[allow(dead_code)]
pub const S3_64: f64 = -1.98412698298579493134e-04; // 0xBF2A01A019C161D5
#[allow(dead_code)]
pub const S4_64: f64 = 2.75573137070700676789e-06; // 0x3EC71DE357B1FE7D
#[allow(dead_code)]
pub const S5_64: f64 = -2.50507602534068634195e-08; // 0xBE5AE5E68A2B9CEB
#[allow(dead_code)]
pub const S6_64: f64 = 1.58969099521155010221e-10; // 0x3DE5D93A5ACFD57C

// =============================================================================
// Magic constants for branchless quadrant selection
// =============================================================================

/// Rounding bias for rint operation: 1.5 * 2^52
#[allow(dead_code)]
pub const TOINT: f64 = 1.5 / f64::EPSILON; // 6755399441055744.0

/// Small multiples of π/2 for direct reduction (f32)
#[allow(dead_code)]
pub const C1PIO2: f64 = 1.5707963267948966; // 1 * π/2
#[allow(dead_code)]
pub const C2PIO2: f64 = 3.141592653589793; // 2 * π/2
#[allow(dead_code)]
pub const C3PIO2: f64 = 4.71238898038469; // 3 * π/2
#[allow(dead_code)]
pub const C4PIO2: f64 = 6.283185307179586; // 4 * π/2
