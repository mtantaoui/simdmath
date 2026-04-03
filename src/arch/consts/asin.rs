//! Constants for `asin` implementations.
//!
//! The `asin` function uses the same Padé rational approximation as `acos`,
//! so it shares the P_S* and Q_S* constants from the `acos` module. This module
//! defines only the constants unique to `asin`.
//!
//! These constants are taken verbatim from musl libc's `asinf.c` and `asin.c`
//! (which descend from Sun's fdlibm).

// These are intentional musl libc constants with specific bit patterns, not approximations.
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]

// ===========================================================================
// f32 Constants (from musl asinf.c / fdlibm)
// ===========================================================================

/// High word of π/2 in the Dekker two-sum `PIO2_HI + PIO2_LO = π/2`.
///
/// The split keeps `PIO2_HI` exact to 23 significant bits so that arithmetic
/// involving π/2 avoids catastrophic cancellation in the Dekker compensation.
/// IEEE 754 hex: 0x3FC90FDA
pub(crate) const PIO2_HI_32: f32 = 1.570_796_3;

/// Low word of π/2: `PIO2_LO = π/2 − PIO2_HI` (≈ 7.55e-8).
///
/// Adding `PIO2_LO` to a result that already includes `PIO2_HI` restores the
/// full precision of π/2.
/// IEEE 754 hex: 0x33A22168
pub(crate) const PIO2_LO_32: f32 = 7.549_789_4e-8;

/// Threshold below which `asin(x) ≈ x` (avoiding underflow).
///
/// This is `2^-12` in IEEE 754 hex format (0x39800000).
/// For |x| < 2^-12, the polynomial correction is negligible and returning
/// x directly avoids potential underflow in the intermediate computations.
pub(crate) const TINY_THRESHOLD_32: u32 = 0x3980_0000;

// ===========================================================================
// f64 Constants (from musl asin.c / fdlibm)
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

/// Threshold below which `asin(x) ≈ x` for f64.
///
/// This is `2^-27` — below this threshold, the polynomial correction
/// is smaller than 1 ULP.
pub(crate) const TINY_THRESHOLD_64: u64 = 0x3E40_0000_0000_0000;
