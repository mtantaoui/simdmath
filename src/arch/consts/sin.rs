//! Constants for SIMD `sin(x)` implementations.
//!
//! These constants are derived from musl libc's `sinf.c`, `sin.c`, `__sindf.c`,
//! `__cosdf.c`, `__sin.c`, `__cos.c`, and `__rem_pio2f.c` (which in turn come
//! from Sun Microsystems' fdlibm).
//!
//! # Algorithm Overview
//!
//! The sine function is computed using:
//! 1. **Argument reduction**: Reduce `x` to `y` in `[-π/4, π/4]` via `y = x - n*(π/2)`
//! 2. **Quadrant selection**: Based on `n mod 4`, compute `±sin(y)` or `±cos(y)`
//! 3. **Polynomial approximation**: Use minimax polynomials for the kernel functions
//!
//! ## Quadrant Table (n mod 4)
//!
//! | n | sin(x) |
//! |---|--------|
//! | 0 |  sin(y) |
//! | 1 |  cos(y) |
//! | 2 | -sin(y) |
//! | 3 | -cos(y) |
//!
//! Note: sin and cos share the same reduction constants and kernel coefficients,
//! so we re-export from `cos.rs` and add any sin-specific constants here.

// Re-export shared constants from cos.rs
pub use super::cos::{
    // f32 constants
    C0_32,
    C1_32,
    // f64 constants
    C1_64,
    C2_32,
    C2_64,
    C3_32, // cosine kernel coefficients
    C3_64,
    C4_64,
    C5_64,
    C6_64, // cosine kernel coefficients
    FRAC_2_PI_32,
    FRAC_2_PI_64,
    PIO2_1_32,
    PIO2_1_64,
    PIO2_1T_32,
    PIO2_1T_64,
    PIO2_2_64,
    PIO2_2T_64,
    S1_32,
    S1_64,
    S2_32,
    S2_64,
    S3_32,
    S3_64,
    S4_32, // sine kernel coefficients
    S4_64,
    S5_64,
    S6_64, // sine kernel coefficients
    // Magic constants
    TOINT,
};
