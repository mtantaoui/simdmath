# Overview

`simdmath` is a Rust library that provides SIMD-vectorised implementations of
common floating-point mathematical functions — trigonometric, inverse
trigonometric, exponential, logarithmic, power, root, and element-wise
arithmetic — for `f32` and `f64`, with backends for **AVX2**, **AVX-512**, and
**ARM NEON**.

This book is the **mathematical reference** for the crate. It complements the
[API documentation on docs.rs](https://docs.rs/simdmath) by deriving every
algorithm from first principles, proving its error bound, and walking through
the per-backend SIMD implementation.

## Reading guide

- If you want to **use** the crate, start with
  [Getting Started → Installation](./getting_started/installation.md).
- If you want to **understand a specific function**, jump straight to its
  chapter (e.g. [`sin`](./functions/sin.md), [`exp`](./functions/exp.md)).
- If you want to **understand the precision claims**, read
  [Foundations → ULP](./foundations/ulp.md) and the
  [methodology chapter](./precision/methodology.md).
- If you want to **port a new function** or **add a new backend**, the
  [Foundations](./foundations/ieee754.md) and
  [SIMD Backends](./backends/avx2.md) chapters are prerequisites.

## Conventions

Throughout the book:

- Lowercase italic letters (\\(x, y, r, k\\)) denote real numbers.
- Roman \\(\hat{f}, \hat{x}\\) denote finite-precision approximations of an
  ideal real-valued \\(f, x\\).
- \\(\mathrm{ulp}(y)\\) denotes the gap between \\(y\\) and the next representable
  floating-point number at the same precision.
- \\(\varepsilon\\) denotes the unit roundoff: \\(2^{-24}\\) for `f32`,
  \\(2^{-53}\\) for `f64`.
- Display math uses `\\[ … \\]`; inline math uses `\\( … \\)`. Both render via
  MathJax in the rendered book (mdBook's `mathjax-support` integration).

## Status

This documentation is being written incrementally. Chapter pages that have
not yet been authored are stubs and will be filled in over the course of the
0.1.x series.
