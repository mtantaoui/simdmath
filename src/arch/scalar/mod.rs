//! Scalar fallback backend.
//!
//! Used when no SIMD ISA is available (or its feature flags are not enabled).
//! Implementations are straight `Iterator`-based loops; the compiler may
//! auto-vectorise them, but no hand-tuned SIMD is performed. This backend
//! exists so that a default `cargo build` succeeds on any platform.
//!
//! SSE-only `x86_64` targets (without AVX2/AVX-512) also fall through to this
//! backend until a true SSE implementation is added.

pub(crate) mod vec;
