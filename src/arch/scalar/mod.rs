//! Scalar fallback backend.
//!
//! Used when no SIMD ISA is available (or its feature flags are not enabled).
//! Implementations are straight `Iterator`-based loops; the compiler may
//! auto-vectorise them, but no hand-tuned SIMD is performed. This backend
//! exists so that a default `cargo build` succeeds on any platform.
//!
//! For `x86_64` targets where SSE4.1 (but not AVX2/AVX-512) is available, the
//! [`crate::arch::sse`] module re-uses this scalar implementation as a
//! placeholder until a true SSE backend is implemented.

pub(crate) mod vec;
