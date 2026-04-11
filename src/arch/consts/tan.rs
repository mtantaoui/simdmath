//! Constants for SIMD `tan(x)` implementations.
//!
//! These constants are derived from musl libc's `tanf.c`, `tan.c`, `__tandf.c`,
//! and `__tan.c` (which in turn come from Sun Microsystems' fdlibm).
//!
//! # Algorithm Overview
//!
//! The tangent function is computed using:
//! 1. **Argument reduction**: Reduce `x` to `y` in `[-π/4, π/4]` via `y = x - n*(π/2)`
//! 2. **Quadrant selection**: Based on `n mod 2`, compute `tan(y)` or `-1/tan(y)`
//! 3. **Polynomial approximation**: Use minimax rational approximation for the kernel
//!
//! ## Quadrant Table (n mod 2)
//!
//! | n mod 2 | tan(x)      |
//! |---------|-------------|
//! | 0       |  tan(y)     |
//! | 1       | -1/tan(y)   |
//!
//! # Note
//!
//! Unlike sin and cos which have period 2π, tan has period π. So we only need
//! to track n mod 2, not n mod 4. When n is odd, we use the cotangent identity:
//! tan(y + π/2) = -cot(y) = -1/tan(y).

// Allow clippy warnings for musl constants - they have specific bit patterns
#![allow(clippy::excessive_precision)]

// Re-export argument reduction constants from cos.rs
pub use crate::arch::consts::cos::{
    FRAC_2_PI_32, FRAC_2_PI_64, PIO2_1_32, PIO2_1_64, PIO2_1T_32, PIO2_1T_64, PIO2_2_64,
    PIO2_2T_64, TOINT,
};

// =============================================================================
// f32 Tangent kernel coefficients (__tandf)
// =============================================================================

// These coefficients approximate tan(x)/x on [-π/4, π/4].
// From musl's __tandf.c: tan(x) ≈ x + T[0]*x³ + T[1]*x⁵ + T[2]*x⁷ + T[3]*x⁹ + T[4]*x¹¹
// The polynomial is evaluated in f64 precision for f32 inputs.

/// tan(x)/x - 1 coefficient for x³ term
#[allow(dead_code)]
pub const T0_32: f64 = 0.333331395030791399758; // 0x15554d3418c99fp-54

/// tan(x)/x - 1 coefficient for x⁵ term
#[allow(dead_code)]
pub const T1_32: f64 = 0.133392002712976742718; // 0x1112fd38999f72p-55

/// tan(x)/x - 1 coefficient for x⁷ term
#[allow(dead_code)]
pub const T2_32: f64 = 0.0533812378445670393523; // 0x1b54c91d865afep-57

/// tan(x)/x - 1 coefficient for x⁹ term
#[allow(dead_code)]
pub const T3_32: f64 = 0.0245283181166547278873; // 0x191df3908c33cep-58

/// tan(x)/x - 1 coefficient for x¹¹ term
#[allow(dead_code)]
pub const T4_32: f64 = 0.00297435743359967304927; // 0x185dadfcecf44ep-61

/// tan(x)/x - 1 coefficient for x¹³ term
#[allow(dead_code)]
pub const T5_32: f64 = 0.00946564784943673166728; // 0x1362b9bf971bcdp-59

// =============================================================================
// f64 Tangent kernel coefficients (__tan)
// =============================================================================

// From musl's __tan.c: rational approximation tan(x)/x ≈ 1 + x²*P(x²)/Q(x²)
// where the reduced argument |x| < π/4.
// These hex representations are from Sun's fdlibm via musl.

/// Numerator coefficient T0 (x² term) = 3.33333333333334091986e-01
#[allow(dead_code)]
pub const T0_64: f64 = f64::from_bits(0x3FD5555555555563);

/// Numerator coefficient T1 (x⁴ term) = 1.33333333333201242699e-01
#[allow(dead_code)]
pub const T1_64: f64 = f64::from_bits(0x3FC111111110FE7A);

/// Numerator coefficient T2 (x⁶ term) = 5.39682539762260521377e-02
#[allow(dead_code)]
pub const T2_64: f64 = f64::from_bits(0x3FABA1BA1BB341FE);

/// Numerator coefficient T3 (x⁸ term) = 2.18694882948595424599e-02
#[allow(dead_code)]
pub const T3_64: f64 = f64::from_bits(0x3F9664F48406D637);

/// Numerator coefficient T4 (x¹⁰ term) = 8.86323982359930005737e-03
#[allow(dead_code)]
pub const T4_64: f64 = f64::from_bits(0x3F8226E3E96E8493);

/// Numerator coefficient T5 (x¹² term) = 3.59207910759131235356e-03
#[allow(dead_code)]
pub const T5_64: f64 = f64::from_bits(0x3F6D6D22C9560328);

/// Numerator coefficient T6 (x¹⁴ term) = 1.45620945432529025516e-03
#[allow(dead_code)]
pub const T6_64: f64 = f64::from_bits(0x3F57DBC8FEE08315);

/// Numerator coefficient T7 (x¹⁶ term) = 5.88041240820264096874e-04
#[allow(dead_code)]
pub const T7_64: f64 = f64::from_bits(0x3F4344D8F2F26501);

/// Numerator coefficient T8 (x¹⁸ term) = 2.46463134818469906812e-04
#[allow(dead_code)]
pub const T8_64: f64 = f64::from_bits(0x3F3026F71A8D1068);

/// Numerator coefficient T9 (x²⁰ term) = 7.81794442939557092300e-05
#[allow(dead_code)]
pub const T9_64: f64 = f64::from_bits(0x3F147E88A03792A6);

/// Numerator coefficient T10 (x²² term) = 7.14072491382608190305e-05
#[allow(dead_code)]
pub const T10_64: f64 = f64::from_bits(0x3F12B80F32F0A7E9);

/// Numerator coefficient T11 (x²⁴ term) = -1.85586374855275456654e-05
#[allow(dead_code)]
pub const T11_64: f64 = f64::from_bits(0xBEF375CBDB605373);

/// Numerator coefficient T12 (x²⁶ term) = 2.59073051863633712884e-05
#[allow(dead_code)]
pub const T12_64: f64 = f64::from_bits(0x3EFB2A7074BF7AD4);

/// Threshold below which tan(x) ≈ x for f64
#[allow(dead_code)]
pub const TINY_TAN_64: f64 = 1e-300;

/// Threshold for small argument optimization in f64
#[allow(dead_code)]
pub const SMALL_TAN_64: f64 = 1e-14;

/// π/4 high part for "big" argument handling
/// From musl: 0x3FE921FB, 54442D18
#[allow(dead_code)]
pub const PIO4_HI_64: f64 = f64::from_bits(0x3FE921FB54442D18);

/// π/4 low part for extended precision
/// From musl: 0x3C81A626, 33145C07
#[allow(dead_code)]
pub const PIO4_LO_64: f64 = f64::from_bits(0x3C81A62633145C07);

/// Threshold for "big" argument path: |x| >= 0.6744
/// Below this, the standard polynomial works. At and above, need special handling.
/// From musl: 0x3FE59428 => 0.67434...
#[allow(dead_code)]
pub const BIG_THRESH_64: f64 = f64::from_bits(0x3FE5942800000000);
