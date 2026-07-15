//! Linear-algebra kernels (matrix multiplication).
//!
//! The backend is selected at compile time like the element-wise math
//! functions: AVX2 or AVX-512 on `x86_64` (build with e.g.
//! `RUSTFLAGS="-C target-feature=+avx2,+fma"`), NEON on `aarch64`, and a
//! cache-blocked scalar fallback everywhere else.

/// Computes `C += A × B` for column-major `f32` matrices.
///
/// Uses a BLIS-style cache-blocked algorithm with an FMA microkernel (16×6
/// on AVX2, 32×14 on AVX-512, 8×12 on NEON, 8×6 auto-vectorised scalar
/// fallback); blocking parameters are derived at runtime from the machine's
/// cache hierarchy.
///
/// # Arguments
///
/// * `a` - Matrix A in column-major format (`m × k` elements)
/// * `b` - Matrix B in column-major format (`k × n` elements)
/// * `c` - Output matrix C in column-major format (`m × n` elements); the
///   product is **accumulated** onto its existing contents
/// * `m` - Number of rows in A and C
/// * `n` - Number of columns in B and C
/// * `k` - Number of columns in A and rows in B (shared dimension)
///
/// If any dimension is `0` the function returns without touching `c`.
///
/// # Panics
///
/// Panics if a slice length does not match its `m`/`n`/`k` dimensions.
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        target_feature = "avx2"
    ))]
    crate::arch::avx2::matmul::f32::matmul_auto(a, b, c, m, n, k);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    crate::arch::avx512::matmul::f32::matmul_auto(a, b, c, m, n, k);
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    crate::arch::neon::matmul::f32::matmul_auto(a, b, c, m, n, k);
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
    crate::arch::scalar::matmul::f32::matmul_auto(a, b, c, m, n, k);
}

/// Computes `C += A × B` for column-major `f64` matrices.
///
/// Uses a BLIS-style cache-blocked algorithm with an FMA microkernel (8×6
/// on AVX2, 16×14 on AVX-512, 4×12 on NEON, 4×6 auto-vectorised scalar
/// fallback); blocking parameters are derived at runtime from the machine's
/// cache hierarchy.
///
/// # Arguments
///
/// * `a` - Matrix A in column-major format (`m × k` elements)
/// * `b` - Matrix B in column-major format (`k × n` elements)
/// * `c` - Output matrix C in column-major format (`m × n` elements); the
///   product is **accumulated** onto its existing contents
/// * `m` - Number of rows in A and C
/// * `n` - Number of columns in B and C
/// * `k` - Number of columns in A and rows in B (shared dimension)
///
/// If any dimension is `0` the function returns without touching `c`.
///
/// # Panics
///
/// Panics if a slice length does not match its `m`/`n`/`k` dimensions.
pub fn matmul_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        target_feature = "avx2"
    ))]
    crate::arch::avx2::matmul::f64::matmul_auto(a, b, c, m, n, k);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    crate::arch::avx512::matmul::f64::matmul_auto(a, b, c, m, n, k);
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    crate::arch::neon::matmul::f64::matmul_auto(a, b, c, m, n, k);
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
    crate::arch::scalar::matmul::f64::matmul_auto(a, b, c, m, n, k);
}
