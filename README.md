# simdmath

[![crates.io](https://img.shields.io/crates/v/simdmath.svg)](https://crates.io/crates/simdmath)
[![docs.rs](https://img.shields.io/docsrs/simdmath/latest)](https://docs.rs/simdmath)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![AVX2](https://github.com/mtantaoui/simdmath/actions/workflows/avx2.yml/badge.svg)](https://github.com/mtantaoui/simdmath/actions/workflows/avx2.yml)
[![AVX-512](https://github.com/mtantaoui/simdmath/actions/workflows/avx512.yml/badge.svg)](https://github.com/mtantaoui/simdmath/actions/workflows/avx512.yml)
[![NEON](https://github.com/mtantaoui/simdmath/actions/workflows/neon.yml/badge.svg)](https://github.com/mtantaoui/simdmath/actions/workflows/neon.yml)

**High-performance SIMD implementations of mathematical functions for Rust.**

`simdmath` provides vectorised versions of the common transcendental and
elementary functions — trigonometric, inverse trigonometric, exponential,
logarithmic, power, and roots — operating on multiple `f32` or `f64` values
in parallel using hardware SIMD instructions. Algorithms are ported from
[musl libc](https://musl.libc.org/) (which descends from Sun's fdlibm). The
worst-case ULP bound is documented per-function (the [accuracy
table](https://docs.rs/simdmath#accuracy) on docs.rs is the single source of
truth); all currently-implemented functions are ≤ 3 ULP, with most at ≤ 1 ULP
and the two hardware-correctly-rounded ops (`abs`, `sqrt`) at ≤ 0.5 ULP.

## Supported targets

| Architecture | ISA       | `f32` lanes | `f64` lanes |
|--------------|-----------|-------------|-------------|
| `x86_64`     | AVX2 + FMA | 8           | 4           |
| `x86_64`     | AVX-512   | 16          | 8           |
| `aarch64`    | NEON      | 4           | 2           |
| any          | scalar fallback | 1     | 1           |

The backend is selected at compile time from the `target-feature` flags.
When no SIMD ISA is enabled the crate falls back to a correct (but slow)
scalar implementation that delegates to the standard library. Runtime CPU
dispatch is planned for a future release.

## Installation

```toml
[dependencies]
simdmath = "0.1"
```

For SIMD speeds on `x86_64`, enable AVX2 (and optionally AVX-512) at
compile time:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
# or, to use the host CPU's full feature set
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

On `aarch64`, NEON is enabled by default on most targets. Without any of
these flags, the scalar fallback is used.

## Usage

The primary entry points are the `VecMath` trait (transcendentals) and the
`VecExt` trait (arithmetic + reductions), implemented for `Vec<f32>` and
`Vec<f64>`. Both are re-exported from the crate's `prelude`:

```rust,ignore
use simdmath::prelude::*;

let angles: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
let sines   = angles.sin();
let cosines = angles.cos();

let a = vec![1.0_f32; 1024];
let b = vec![2.0_f32; 1024];
let c = a.add(&b);          // element-wise add
let s = c.sum();            // horizontal reduce
```

## Available functions

| Family             | Functions                              |
|--------------------|----------------------------------------|
| Trigonometric      | `sin`, `cos`, `tan`                    |
| Inverse trig       | `asin`, `acos`, `atan`, `atan2`        |
| Exp / log          | `exp`, `ln`, `pow`                     |
| Roots              | `sqrt`, `cbrt`                         |
| Sign / absolute    | `abs`                                  |
| Arithmetic         | `+`, `-`, `*`, `/`, `%` (and `_assign`) |
| Reductions         | `sum`, `product`, `min`, `max`         |

## Accuracy

Every function is documented with a worst-case ULP bound verified by a
sweep test over its domain. The full table lives in the [crate
docs](https://docs.rs/simdmath#accuracy); per-function derivations, error
analyses, and figures live in the
[mathematical reference book](https://github.com/mtantaoui/simdmath/tree/main/book).
The methodology used to obtain the numbers is documented in
[`docs/ulp-methodology.md`](./docs/ulp-methodology.md).

## Documentation

- **API reference** — <https://docs.rs/simdmath>
- **Mathematical reference (book)** — derivations, error analysis, and
  one chapter per function. Source in [`book/`](./book/); render with
  `mdbook build book/` (or browse the [hosted version][book-online] once
  published).
- **ULP methodology** — [`docs/ulp-methodology.md`](./docs/ulp-methodology.md)

[book-online]: https://mtantaoui.github.io/simdmath/

## Prior art and positioning

`simdmath` exists in a crowded niche. It differs from its neighbours roughly
as follows:

| Crate                      | Approach                                     | Where `simdmath` differs                  |
|----------------------------|----------------------------------------------|-------------------------------------------|
| [`sleef`](https://crates.io/crates/sleef)         | Bindings to the C SLEEF library             | Pure Rust, no C build dependency          |
| [`pulp`](https://crates.io/crates/pulp)           | Generic SIMD with runtime dispatch          | Specific to math functions, not generic   |
| [`wide`](https://crates.io/crates/wide)           | Portable SIMD types                         | Math kernels, not just types              |
| [`std::simd`](https://doc.rust-lang.org/std/simd) | Portable SIMD primitives (nightly)          | Stable, hand-tuned per-ISA kernels        |

If you need a battle-tested C library, use SLEEF. If you need portable SIMD
arithmetic without math kernels, use `wide` or `std::simd`. `simdmath` aims
to fill the "Rust-only, ULP-documented, per-ISA-tuned, math-only" slot.

## Building the book

```sh
cargo install mdbook
mdbook build book/
mdbook serve book/        # local preview at http://localhost:3000
```

## License

MIT — see [LICENSE](./LICENSE).

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) and the [agent guidelines](./AGENTS.md)
for project conventions. The mathematical-documentation style guide is in
[`book/src/STYLE.md`](./book/src/STYLE.md).
