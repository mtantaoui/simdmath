# Summary

[Overview](./overview.md)

# Getting Started

- [Installation and feature flags](./getting_started/installation.md)
- [Required CPU features and `RUSTFLAGS`](./getting_started/cpu_features.md)
- [Runtime vs compile-time dispatch](./getting_started/dispatch.md)

# Foundations

- [IEEE-754 in two slides](./foundations/ieee754.md)
- [ULP, faithful rounding, correct rounding](./foundations/ulp.md)
- [Why `≤ 1 ULP` and not `correctly rounded`](./foundations/why_not_correct.md)
- [Compensated arithmetic: two-sum and Dekker product](./foundations/compensated.md)
- [Polynomial evaluation: Horner, Estrin, FMA](./foundations/polynomial_evaluation.md)
- [Argument-reduction taxonomy](./foundations/argument_reduction.md)

# SIMD Backends

- [AVX2 (8×f32, 4×f64)](./backends/avx2.md)
- [AVX-512 (16×f32, 8×f64)](./backends/avx512.md)
- [NEON (4×f32, 2×f64)](./backends/neon.md)
- [Compile-time dispatch](./backends/dispatch.md)
- [The `Load` / `Store` / `Align` traits](./backends/traits.md)

# Trigonometric Functions

- [Sine `sin`](./functions/sin.md)
- [Cosine `cos`](./functions/cos.md)
- [Tangent `tan`](./functions/tan.md)

# Inverse Trigonometric Functions

- [Arc sine `asin`](./functions/asin.md)
- [Arc cosine `acos`](./functions/acos.md)
- [Arc tangent `atan`](./functions/atan.md)
- [Two-argument arc tangent `atan2`](./functions/atan2.md)

# Exponential and Logarithmic Functions

- [Natural exponential `exp`](./functions/exp.md)
- [Natural logarithm `ln`](./functions/ln.md)
- [Power `pow`](./functions/pow.md)

# Roots

- [Square root `sqrt`](./functions/sqrt.md)
- [Cube root `cbrt`](./functions/cbrt.md)

# Element-wise Arithmetic and Reductions

- [Vectorised arithmetic and reductions](./functions/arithmetic.md)

# Precision and Performance

- [ULP measurement methodology](./precision/methodology.md)
- [Per-function worst-case ULP tables](./precision/tables.md)
- [Benchmarks](./benchmarks.md)

# Appendices

- [A — Polynomial coefficient tables](./appendices/A_coefficients.md)
- [B — Constant tables (π/2, ln 2, magic seeds)](./appendices/B_constants.md)
- [C — Comparison with other SIMD math libraries](./appendices/C_comparison.md)
- [D — Glossary](./appendices/D_glossary.md)
- [E — Bibliography](./appendices/E_bibliography.md)
