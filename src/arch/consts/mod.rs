//! Shared constants for math function implementations across all SIMD architectures.
//!
//! These constants are organized by function and taken verbatim from musl libc
//! (which descends from Sun's fdlibm). They have specific bit patterns designed
//! for numerical accuracy.
//!
//! These constants are referenced only from the SIMD backends (AVX2, AVX-512,
//! NEON). When the crate is built for a target without a SIMD backend (e.g.
//! the scalar fallback), the constants are unused — the warnings below are
//! silenced because every constant is genuinely needed when a SIMD backend is
//! active.

#![allow(dead_code, unused_imports)]

pub(crate) mod acos;
pub(crate) mod asin;
pub(crate) mod atan;
pub(crate) mod atan2;
pub(crate) mod cbrt;
pub(crate) mod cos;
pub(crate) mod exp;
pub(crate) mod ln;
pub(crate) mod sin;
pub(crate) mod tan;
