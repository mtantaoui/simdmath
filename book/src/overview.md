# Overview

`simdmath` is a Rust library that provides SIMD-vectorised implementations of
common floating-point mathematical functions — trigonometric, inverse
trigonometric, exponential, logarithmic, power, root, and element-wise
arithmetic — plus a cache-blocked **matrix multiplication**, for `f32` and
`f64`, with backends for **AVX2**, **AVX-512**, and **ARM NEON**.

This book is the **mathematical reference** for the crate. It complements the
[API documentation on docs.rs](https://docs.rs/simdmath) by deriving every
algorithm from first principles, proving its error bound, and walking through
the per-backend SIMD implementation.

## Reading guide

- If you want to **use** the crate, start with
  [Getting Started → Installation](./getting_started/installation.md).
- If you want to **understand a specific function**, jump straight to its
  chapter (e.g. [`sin`](./functions/sin.md), [`exp`](./functions/exp.md)).
- If you want to **multiply matrices**, see
  [Linear Algebra → `matmul`](./linalg/matmul.md).
- If you want to **understand the precision claims**, read
  [Foundations → ULP](./foundations/ulp.md) and the
  [methodology chapter](./precision/methodology.md).
- If you want to **port a new function** or **add a new backend**, the
  [Foundations](./foundations/ieee754.md) and
  [SIMD Backends](./backends/avx2.md) chapters are prerequisites.

## Performance at a glance

Speed-up of the SIMD path over a scalar baseline, measured at \\(n = 1024\\)
elements (best of `f32` / `f64`). Full per-type tables with absolute
throughput numbers live in the [Benchmarks](./benchmarks.md) chapter.

| Function | AVX2 (Core Ultra 7 155H) | AVX-512 (Xeon w5-2555X) | NEON (Apple M1) |
|----------|:------------------------:|:-----------------------:|:---------------:|
| `sin`    | 3.2×                     | 5.5×                    | 1.6×            |
| `cos`    | 3.5×                     | 5.4×                    | 1.7×            |
| `tan`    | 6.9×                     | 11.1×                   | 2.1×            |
| `asin`   | 3.4×                     | 3.4×                    | 1.9×            |
| `acos`   | 2.8×                     | 2.7×                    | 1.6×            |
| `atan`   | 9.8×                     | 14.6×                   | 5.2×            |
| `atan2`  | 3.9×                     | 8.3×                    | 1.1×            |
| `exp`    | 2.6×                     | 4.1×                    | 1.4×            |
| `ln`     | 1.7×                     | 3.0×                    | 1.2×            |
| `pow`    | 1.3×                     | 2.0×                    | 0.6×            |
| `sqrt`   | 1.0×                     | 0.9×                    | 1.0×            |
| `cbrt`   | 4.9×                     | 5.8×                    | 2.2×            |

`sqrt` throughput is hardware-bound (single instruction on all backends).
`pow` on NEON is slower than scalar — see the [benchmarks notes](./benchmarks.md#notes-on-sub-1-results) for the explanation.

## Conventions

Throughout the book:

- Lowercase italic letters (\\(x, y, r, k\\)) denote real numbers.
- Roman \\(\hat{f}, \hat{x}\\) denote finite-precision approximations of an
  ideal real-valued \\(f, x\\).
- \\(\mathrm{ulp}(y)\\) denotes the gap between \\(y\\) and the next representable
  floating-point number at the same precision.
- \\(\varepsilon\\) denotes the unit roundoff: \\(2^{-24}\\) for `f32`,
  \\(2^{-53}\\) for `f64`.

## Status

All chapters are complete for the v0.1 release. Future sections will cover
runtime CPU dispatch (planned for v0.2) and extended input-distribution
benchmarks.
