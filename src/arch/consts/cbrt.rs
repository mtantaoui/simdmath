//! Constants for `cbrt` (cube root) implementations.
//!
//! The cube root uses a clever bit-manipulation trick to get a rough initial
//! approximation (~5 bits), followed by Newton–Raphson iterations to refine.
//!
//! ## f32 Algorithm
//!
//! 1. **Initial approximation**: `hx/3 + B1` (or `B2` for subnormals) exploits
//!    the IEEE 754 representation to compute `cbrt(2^e * m) ≈ 2^(e/3) * m^(1/3)`.
//! 2. **Two Newton iterations** in double precision:
//!    `t = t * (x + x + r) / (x + r + r)` where `r = t³`
//! 3. Convert back to f32 (rounding is perfect in round-to-nearest mode).
//!
//! ## f64 Algorithm
//!
//! 1. **Initial approximation**: Same bit trick with 64-bit constants.
//! 2. **Polynomial refinement** (~23 bits): `t = t * P(t³/x)` where P is degree-4.
//! 3. **Rounding step**: Round `t` to 23 bits away from zero.
//! 4. **One Newton iteration** to full 53-bit precision.
//!
//! All constants are taken verbatim from **musl libc `cbrt.c` / `cbrtf.c`**
//! (FreeBSD/fdlibm descent, optimized by Bruce D. Evans).

// These are intentional musl libc constants with specific bit patterns, not approximations.
#![allow(clippy::excessive_precision)]

// ===========================================================================
// f32 Constants — from musl libc cbrtf.c
// ===========================================================================

/// Initial approximation bias for normal f32 values.
///
/// `B1 = (127 - 127.0/3 - 0.03306235651) * 2^23`
///
/// This magic constant, when added to `hx/3`, produces an initial cube root
/// estimate accurate to ~5 bits.
pub(crate) const B1_32: u32 = 709958130;

/// Initial approximation bias for subnormal f32 values.
///
/// `B2 = (127 - 127.0/3 - 24/3 - 0.03306235651) * 2^23`
///
/// After scaling subnormals by `2^24`, this bias accounts for the extra
#[allow(dead_code)]
pub(crate) const B2_32: u32 = 642849266;

/// Scale factor for subnormal f32 values: `2^24`.
///
/// Multiplying a subnormal by this brings it into the normal range.
/// IEEE 754 hex: `0x4b800000`
pub(crate) const X1P24_32: f32 = 16777216.0; // 2^24

// ===========================================================================
// f64 Constants — from musl libc cbrt.c
// ===========================================================================

/// Initial approximation bias for normal f64 values.
///
/// `B1 = (1023 - 1023/3 - 0.03306235651) * 2^20`
///
/// Applied to the upper 32 bits of the exponent field.
pub(crate) const B1_64: u32 = 715094163;

/// Initial approximation bias for subnormal f64 values.
///
/// `B2 = (1023 - 1023/3 - 54/3 - 0.03306235651) * 2^20`
///
/// After scaling subnormals by `2^54`, this bias accounts for the extra offset.
pub(crate) const B2_64: u32 = 696219795;

/// Scale factor for subnormal f64 values: `2^54`.
///
/// Multiplying a subnormal by this brings it into the normal range.
/// IEEE 754 hex: `0x43500000_00000000`
pub(crate) const X1P54_64: f64 = 18014398509481984.0; // 2^54

// ---------------------------------------------------------------------------
// Polynomial coefficients for f64 refinement
//
// These approximate `1/cbrt(r)` to within `2^-23.5` when `|r - 1| < 1/10`.
// The polynomial is: P(r) = P0 + r*P1 + r²*P2 + r³*P3 + r⁴*P4
// ---------------------------------------------------------------------------

/// P0 coefficient: 1.87595182427177009643
/// IEEE hex: `0x3ffe03e6_0f61e692`
pub(crate) const P0: f64 = 1.875_951_824_271_770_096_43;

/// P1 coefficient: -1.88497979543377169875
/// IEEE hex: `0xbffe28e0_92f02420`
pub(crate) const P1: f64 = -1.884_979_795_433_771_698_75;

/// P2 coefficient: 1.621429720105354466140
/// IEEE hex: `0x3ff9f160_4a49d6c2`
pub(crate) const P2: f64 = 1.621_429_720_105_354_466_14;

/// P3 coefficient: -0.758397934778766047437
/// IEEE hex: `0xbfe844cb_bee751d9`
pub(crate) const P3: f64 = -0.758_397_934_778_766_047_437;

/// P4 coefficient: 0.145996192886612446982
/// IEEE hex: `0x3fc2b000_d4e4edd7`
pub(crate) const P4: f64 = 0.145_996_192_886_612_446_982;

// ---------------------------------------------------------------------------
// Rounding mask for f64 Newton setup
// ---------------------------------------------------------------------------

/// Mask to round f64 to 23 significant bits (clear low 32 bits of mantissa).
///
/// Used before the final Newton iteration to ensure `t*t` is exact.
pub(crate) const ROUND_MASK_64: u64 = 0xffffffffc0000000;

/// Bias added before masking to round away from zero.
pub(crate) const ROUND_BIAS_64: u64 = 0x80000000;
