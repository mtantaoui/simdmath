# Runtime vs compile-time dispatch

When a math library wants to support multiple SIMD instruction sets it has to
decide *when* to pick a backend: at **compile time**, baking one path into the
binary, or at **runtime**, using `CPUID` to choose at first call. `simdmath`
v0.1.x uses compile-time dispatch exclusively. This chapter explains why,
shows the actual `cfg` ladder, and outlines the planned v0.2 runtime
mechanism.

## Compile-time dispatch in one diagram

```text
       ┌─────────────────────────────┐
       │ user crate calls VecMath    │
       └──────────────┬──────────────┘
                      │
        ╔═════════════╧═════════════╗
        ║   #[cfg(target_feature)]  ║      <-- evaluated by rustc
        ╚═════════════╤═════════════╝
                      │
   ┌──────────────────┼──────────────────┐
   ▼                  ▼                  ▼
AVX-512 path      AVX2 path          NEON path
(zmm, k masks)    (ymm, blendv)      (vN, vbslq)
```

Exactly one branch is compiled into the final binary. The unused branches
are not even type-checked beyond the `mod` declaration, which lets the AVX-512
backend use intrinsics that don't exist on AArch64 toolchains and vice
versa.

## The `cfg` ladder

The canonical dispatch site lives in `src/math/mod.rs`. Stripped to its
skeleton, it reads:

```rust,ignore
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    target_feature = "avx2"
))]
mod avx2;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod neon;

// Scalar fallback: no SIMD ISA available.
#[cfg(any(
    all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2"),
    ),
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
mod scalar;
```

The arms are mutually exclusive by construction: AVX-512 wins over AVX2
because the AVX2 arm explicitly requires `not(target_feature = "avx512f")`,
and the final `any(...)` arm is the scalar fallback. Exactly one backend
module exists in any given build.

## Why compile-time?

Compile-time dispatch buys three properties that runtime dispatch cannot
match:

1. **Inlining across the boundary.** Because `sin` is an ordinary `#[inline]`
   function, the compiler can fuse it into the caller's loop, schedule
   instructions across the boundary, and elide redundant loads of
   the polynomial constants. Runtime dispatch usually requires a function
   pointer or a `match` on a global, both of which inhibit inlining.
2. **No first-call latency.** The first call to `f.sin()` does not have to
   pay for a `CPUID` probe and an indirect jump.
3. **Simpler unsafe surface.** Each kernel is `#[target_feature(enable = "…")]`
   on every entry point but never has to hand-implement the
   "feature-detect-then-trampoline" pattern, which is notoriously easy to
   get wrong (see the `multiversion` crate's caveats around
   `#[target_feature]` propagation).

## Why *not* compile-time?

The price is **portability**. A binary compiled with `+avx512f` will
SIGILL on Skylake or any AMD chip older than Zen 4. Distributors have
three options:

- Ship a "lowest common denominator" build (`+avx2,+fma`) and accept the
  AVX-512 perf-on-the-table.
- Ship multiple binaries and dispatch at the *operating-system* level (the
  glibc `hwcaps` mechanism, or `LD_LIBRARY_PATH=…/avx512`).
- Wait for `simdmath` v0.2 with runtime dispatch.

## Inspecting which backend was compiled

```rust,ignore
pub const fn active_backend() -> &'static str {
    if cfg!(all(target_arch = "x86_64",
                target_feature = "avx512f")) { "avx512" }
    else if cfg!(all(target_arch = "x86_64",
                     target_feature = "avx2")) { "avx2"   }
    else if cfg!(all(target_arch = "aarch64",
                     target_feature = "neon")) { "neon"   }
    else { "scalar" }
}
```

This compiles to a single immediate at the call site — no branching,
no `CPUID` — and is a useful sanity check in your application's startup log.

## What runtime dispatch will look like (v0.2 roadmap)

The intended public API does not change: `xs.sin()` will still compile and
run. Internally we will:

1. At crate initialisation (`std::sync::OnceLock<Backend>`), run
   `is_x86_feature_detected!("avx512f")` etc.;
2. Store a function pointer `static SIN: AtomicPtr<fn(F32x8) -> F32x8>`;
3. Trampoline the public `sin` to the chosen pointer.

The mechanism is the same one used by `std::arch::is_x86_feature_detected!`
and by the [`multiversion`](https://crates.io/crates/multiversion) crate.
The chosen kernels will be compiled with `#[target_feature]` instead of
relying on a global `RUSTFLAGS`, so a single binary will support all three
backends.

Tradeoffs at that point:

| Property                    | Compile-time (v0.1) | Runtime (v0.2 plan) |
|-----------------------------|---------------------|---------------------|
| First-call latency          | 0                   | one `CPUID` + indirect call |
| Per-call overhead           | 0                   | one indirect call (BTB-cached after warm-up) |
| Inlining at the call site   | yes                 | no, unless LTO promotes the function pointer |
| Binary size                 | one backend         | three backends + dispatcher (~3× kernels) |
| Portability                 | low                 | high                |
| Suitable for static linking | yes                 | yes                 |
| Suitable for `cdylib` ABI   | yes                 | yes                 |

If your application needs portability *today*, the recommended workaround is
to compile two crates with different `RUSTFLAGS` and dispatch in your own
code with `is_x86_feature_detected!`.

## See also

- [Installation and feature flags](./installation.md)
- [Required CPU features and `RUSTFLAGS`](./cpu_features.md)
- [Compile-time dispatch — backend internals](../backends/dispatch.md)
- The Rustonomicon, [*`#[target_feature]`*](https://doc.rust-lang.org/nomicon/target-feature.html)
- The [`multiversion`](https://docs.rs/multiversion) crate (a possible
  blueprint for the v0.2 runtime dispatcher).
