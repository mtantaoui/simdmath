# Compile-time dispatch

This crate selects a SIMD backend at **compile time** using `#[cfg(...)]`
attributes that key off `target_arch` and `target_feature`. There is
**no runtime CPU detection**: a binary built without `+avx512f` cannot
upgrade to the AVX-512 backend on a Skylake-X host, and a binary built
with `+avx512f` will fault on an older CPU.

This chapter walks through the actual `cfg` ladder used in the source,
explains the priority rules, and shows how to override the default
selection with `RUSTFLAGS`.

## The dispatch ladder

The selector lives at the top of [`src/arch/mod.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/mod.rs):

```rust,ignore
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
```

A parallel ladder in [`src/math/mod.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/math/mod.rs)
selects the matching `Vec<T>` `VecMath` implementation file.

The `current` alias is the single name that the rest of the crate imports
from; downstream code never names a specific backend, only `arch::current`.

## Priority order

Reading the cascade top-to-bottom yields:

| Priority | Selected backend | Required target | Required feature(s)        |
|----------|------------------|-----------------|----------------------------|
| 1        | AVX-512          | `x86_64`        | `avx512f`                  |
| 2        | AVX2             | `x86_64`        | `avx2` (and implicitly `fma`) |
| 3        | SSE 4.1          | `x86_64`        | `sse4.1` (skeleton, not full coverage) |
| 4        | Scalar fallback  | `x86_64`        | (no SIMD feature)          |
| 5        | NEON             | `aarch64`       | (always on)                |
| 6        | Scalar fallback  | other           | (no SIMD)                  |

The `not(target_feature = "...")` guards in each rule are what make the
ladder a strict priority order: the AVX2 arm only fires when AVX-512 is
*absent*, etc. Without those guards two arms would match and the build
would fail with a duplicate-name error on `current`.

## Why no runtime dispatch (yet)?

Runtime CPU dispatch — the strategy used by Intel SVML, Sleef, glibc's
`libm` IFUNCs, and the Rust `std::simd` portable shim — picks the best
available backend by inspecting `cpuid` / `getauxval(AT_HWCAP)` on first
call. It would let a single binary run optimally on Sandy Bridge, Haswell,
Ice Lake, and Sapphire Rapids.

It is **not** in the crate today for three reasons:

1. **Function-pointer indirection cost.** The transcendental kernels are
   small, mostly `inline(always)`, and their per-call overhead is in the
   single-digit nanoseconds. An IFUNC-style dispatch adds a non-inlinable
   indirect call per element-wise operation; for short slices this can
   dominate the kernel's own cost.

2. **`#[target_feature]` ABI separation.** A function annotated
   `#[target_feature(enable = "avx512f")]` cannot be called from a
   non-AVX-512 caller without an explicit safety boundary. Mixing several
   such functions behind a runtime selector requires a wrapper type or
   trait object, which negates the inlining advantage.

3. **Versioned API stability.** The matrix of `(backend, function)` is
   currently 13 functions × 4 backends = 52 entry points. Doing runtime
   dispatch correctly means producing 52 IFUNC resolvers and verifying
   their selection logic on every supported microarchitecture. That is
   tracked as future work; v0.1 keeps the surface small.

The recommended mitigation when shipping a binary today is one of:

- **Build the same binary twice**, once with `+avx2` and once with
  `+avx512f`, and pick at startup via a tiny launcher that reads `cpuid`.
- **Build with `-C target-cpu=native`** for a private deployment, locking
  the binary to the build host's microarchitecture.
- **Build a portable `+avx2,+fma` binary**, accepting the throughput
  ceiling that comes with 256-bit SIMD on a chip that supports 512.

## Forcing a specific backend

The dispatch ladder respects only the cargo / rustc target features. To
force AVX2 on a Skylake-X workstation that defaults to AVX-512 under
`-C target-cpu=native`:

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

To force AVX-512:

```bash
RUSTFLAGS="-C target-feature=+avx512f" cargo build --release
```

To exercise the scalar fallback on x86_64:

```bash
RUSTFLAGS="-C target-feature=-avx512f,-avx2,-sse4.1,-fma" \
    cargo build --release
```

Note the leading minus signs — the scalar fallback only compiles when
*all three* of `avx2`, `avx512f`, and `sse4.1` are absent. To
cross-compile to ARM and verify the NEON path:

```bash
cargo check --target aarch64-unknown-linux-gnu
```

To cross-check that nothing inadvertently breaks on a feature-poor
target:

```bash
cargo check --target wasm32-unknown-unknown
# falls through to the scalar arm
```

## Verifying which backend was chosen

There is no `cfg!(...)` evaluator on opaque module re-exports, so the
direct way to check is to pipe a stripped object through `nm`:

```bash
cargo build --release
nm target/release/libsimdmath.rlib 2>/dev/null \
    | grep -E '_(mm256|mm512|vsin)_' | head
```

`_mm256_*` symbols mean AVX2; `_mm512_*` means AVX-512; `vsin_*` /
`vfma_*` mean NEON; raw `f32::sin` / `f64::sin` calls mean the scalar
fallback. This is the most reliable check, because it observes the actual
generated code rather than relying on cargo features which can shadow the
true `target_feature` set.

## Where to look in the source

| Topic | File |
|-------|------|
| Backend `cfg` ladder | [`arch/mod.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/mod.rs) |
| `VecMath` ladder     | [`math/mod.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/math/mod.rs) |
| Scalar fallback      | [`arch/scalar/`](https://github.com/mtantaoui/simdmath/tree/main/src/arch/scalar) |

## See also

- [Required CPU features and `RUSTFLAGS`](../getting_started/cpu_features.md) — user-facing build instructions.
- [Runtime vs compile-time dispatch](../getting_started/dispatch.md) — design rationale.
- [AVX2](./avx2.md), [AVX-512](./avx512.md), [NEON](./neon.md) — the backends.
