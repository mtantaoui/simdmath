//! BLIS-style blocked matrix multiplication for the scalar fallback backend.
//!
//! Mirrors the SIMD backends' layout with plain-Rust microkernels: the
//! constant-trip accumulator loops are written so the compiler can
//! auto-vectorise them (e.g. to SSE2 on baseline `x86_64`), but no
//! intrinsics are used, so this compiles on any architecture.
//!
//! - [`cache`] — runtime cache detection and the analytical blocking model,
//!   shared by every element type
//! - [`panels`] — packed panel storage and packing routines, generic over
//!   the element type
//! - [`f32`] — 8×6 register-tile microkernel and driver
//! - [`f64`] — 4×6 register-tile microkernel and driver

pub(crate) mod cache;
pub(crate) mod f32;
pub(crate) mod f64;
pub(crate) mod panels;
