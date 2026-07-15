//! BLIS-style blocked matrix multiplication for the AVX2 backend.
//!
//! - [`cache`] — runtime cache detection and the analytical blocking model,
//!   shared by every element type
//! - [`panels`] — packed panel storage and packing routines, generic over
//!   the element type
//! - [`f32`] — 16×6 microkernel (two 8-lane YMM vectors × 6 columns) and
//!   driver
//! - [`f64`] — 8×6 microkernel (two 4-lane YMM vectors × 6 columns) and
//!   driver

pub(crate) mod cache;
pub(crate) mod f32;
pub(crate) mod f64;
pub(crate) mod panels;
