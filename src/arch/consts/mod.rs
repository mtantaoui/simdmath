//! Shared constants for math function implementations across all SIMD architectures.
//!
//! These constants are organized by function and taken verbatim from musl libc
//! (which descends from Sun's fdlibm). They have specific bit patterns designed
//! for numerical accuracy.

pub(crate) mod acos;
pub(crate) mod asin;
