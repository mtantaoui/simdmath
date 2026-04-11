//! Constants for SIMD `ln(x)` (natural logarithm) implementations.
//!
//! These constants are derived from Sun Microsystems' fdlibm `e_log.c` / `e_logf.c`
//! (via musl libc). They have specific bit patterns designed for numerical accuracy.
//!
//! # Algorithm Overview
//!
//! The natural logarithm is computed using:
//! 1. **Argument decomposition**: Decompose `x = 2^k * m` where `m ∈ [√2/2, √2]`
//! 2. **Variable substitution**: Let `f = m - 1`, `s = f / (2 + f)`
//! 3. **Polynomial approximation**: `log(1+f) = f - f²/2 + s*(f²/2 + R(s²))`
//!    where `R` is a minimax rational polynomial in `s²`
//! 4. **Reconstruction**: `ln(x) = k*ln(2) + log(1+f)`
//!
//! The substitution `s = f/(2+f)` maps `f ∈ [-0.293, 0.414]` to `|s| < 0.166`,
//! giving faster polynomial convergence than a direct Taylor expansion of `log(1+f)`.
//!
//! # Note
//!
//! `ln(2)` is split into high and low parts (`LN2_HI + LN2_LO`) so that
//! `k * LN2_HI` is computed exactly (LN2_HI has only 33 significant bits).

// Allow clippy warnings for fdlibm constants - they have specific bit patterns
#![allow(clippy::excessive_precision)]

// =============================================================================
// f64 polynomial coefficients (Lg1–Lg7) from fdlibm e_log.c
// =============================================================================

// These approximate the coefficients of 2/(2n+1) in the series:
//   log((1+s)/(1-s)) = 2s + 2s³/3 + 2s⁵/5 + ... where s = f/(2+f)
// Minimax-optimized for the range |s| < 0.1716...

/// Lg1 ≈ 2/3 (coefficient for s² term)
#[allow(dead_code)]
pub const LG1_64: f64 = f64::from_bits(0x3FE5555555555593); // 6.666666666666735130e-01

/// Lg2 ≈ 2/5 (coefficient for s⁴ term)
#[allow(dead_code)]
pub const LG2_64: f64 = f64::from_bits(0x3FD999999997FA04); // 3.999999999940941908e-01

/// Lg3 ≈ 2/7 (coefficient for s⁶ term)
#[allow(dead_code)]
pub const LG3_64: f64 = f64::from_bits(0x3FD2492494229359); // 2.857142874366239149e-01

/// Lg4 ≈ 2/9 (coefficient for s⁸ term)
#[allow(dead_code)]
pub const LG4_64: f64 = f64::from_bits(0x3FCC71C51D8E78AF); // 2.222219843214978396e-01

/// Lg5 ≈ 2/11 (coefficient for s¹⁰ term)
#[allow(dead_code)]
pub const LG5_64: f64 = f64::from_bits(0x3FC7466496CB03DE); // 1.818357216161805012e-01

/// Lg6 ≈ 2/13 (coefficient for s¹² term)
#[allow(dead_code)]
pub const LG6_64: f64 = f64::from_bits(0x3FC39A09D078C69F); // 1.531383769920937332e-01

/// Lg7 ≈ 2/15 (coefficient for s¹⁴ term)
#[allow(dead_code)]
pub const LG7_64: f64 = f64::from_bits(0x3FC2F112DF3E5244); // 1.479819860511658591e-01

// =============================================================================
// ln(2) split for extended precision
// =============================================================================

/// High part of ln(2) — only 33 significant bits so that `k * LN2_HI` is exact
/// for integer k with |k| < 2^20.
#[allow(dead_code)]
pub const LN2_HI_64: f64 = f64::from_bits(0x3FE62E42FEE00000); // 6.93147180369123816490e-01

/// Low part of ln(2): `ln(2) = LN2_HI + LN2_LO` to ~70 bits of precision.
#[allow(dead_code)]
pub const LN2_LO_64: f64 = f64::from_bits(0x3DEA39EF35793C76); // 1.90821492927058500170e-10

// =============================================================================
// Thresholds
// =============================================================================

/// √2 ≈ 1.4142135623730951 — mantissa normalization threshold.
/// If the normalized mantissa exceeds this, we halve it and increment the exponent.
#[allow(dead_code)]
pub const SQRT2_64: f64 = f64::from_bits(0x3FF6A09E667F3BCD); // 1.4142135623730951

/// 2^52 — used to normalize f64 subnormals by shifting them into the normal range.
#[allow(dead_code)]
pub const TWO52_64: f64 = (1u64 << 52) as f64; // 4503599627370496.0
