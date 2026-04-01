#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) use avx512 as current;

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    target_feature = "avx2"
))]
pub(crate) use avx2 as current;

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2"),
    target_feature = "sse4.1"
))]
pub(crate) use sse as current;

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2"),
    not(target_feature = "sse4.1")
))]
pub(crate) use scalar as current;

#[cfg(target_arch = "aarch64")]
pub(crate) use neon as current;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use scalar as current;

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    target_feature = "avx2"
))]
pub(crate) mod avx2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) mod avx512;
pub(crate) mod consts;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub(crate) mod scalar;
pub(crate) mod sse;
