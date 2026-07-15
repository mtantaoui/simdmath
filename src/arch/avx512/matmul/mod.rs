//! BLIS-style blocked matrix multiplication for the AVX-512 backend.
//!
//! Mirrors [`crate::arch`]'s AVX2 layout with kernel shapes scaled to the 32
//! ZMM registers:
//!
//! - [`cache`] — runtime cache detection and the analytical blocking model,
//!   shared by every element type
//! - [`panels`] — packed panel storage and packing routines, generic over
//!   the element type
//! - [`f32`] — 32×14 microkernel (two 16-lane ZMM vectors × 14 columns) and
//!   driver
//! - [`f64`] — 16×14 microkernel (two 8-lane ZMM vectors × 14 columns) and
//!   driver

pub(crate) mod cache;
pub(crate) mod f32;
pub(crate) mod f64;
pub(crate) mod panels;
