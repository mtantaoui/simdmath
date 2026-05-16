//! Architecture dispatch for SIMD backends.
//!
//! Selects the appropriate backend at compile time based on target features:
//! - **AVX-512** (`x86_64` + `avx512f`)
//! - **AVX2** (`x86_64` + `avx2`, no `avx512f`)
//! - **NEON** (`aarch64` + `neon`)
//! - **Scalar fallback** (all other targets — emits a `compile_error!` until implemented)

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    target_feature = "avx2"
))]
pub(crate) mod avx2;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) mod avx512;

pub(crate) mod consts;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod neon;

// Scalar fallback: compiled whenever there is no SIMD backend selected, i.e.
// either an x86_64 host without AVX2/AVX-512 (with or without SSE4.1) or any
// other non-x86/non-aarch64 target.
#[cfg(any(
    all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2"),
    ),
    not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    ))
))]
pub(crate) mod scalar;
