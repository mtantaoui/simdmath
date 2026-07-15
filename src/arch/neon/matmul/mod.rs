//! BLIS-style blocked matrix multiplication for the NEON backend.
//!
//! Mirrors the x86 backends' layout with kernel shapes scaled to the 32
//! 128-bit NEON registers:
//!
//! - [`cache`] — runtime cache detection and the analytical blocking model,
//!   shared by every element type
//! - [`panels`] — packed panel storage and packing routines, generic over
//!   the element type
//! - [`f32`] — 8×12 microkernel (two 4-lane vectors × 12 columns) and driver
//! - [`f64`] — 4×12 microkernel (two 2-lane vectors × 12 columns) and driver

pub(crate) mod cache;
pub(crate) mod f32;
pub(crate) mod f64;
pub(crate) mod panels;
