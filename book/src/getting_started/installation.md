# Installation and feature flags

This chapter covers how to add `simdmath` to a Cargo project, the minimum
toolchain version, the (small) set of feature flags, the `RUSTFLAGS` you need
on each target architecture, and a short troubleshooting guide.

## Minimum supported Rust version (MSRV)

`simdmath` requires **Rust 1.85** or newer. The MSRV is driven by:

- stabilised portable SIMD intrinsics (`std::arch::x86_64::*` for AVX2 / AVX-512
  and `std::arch::aarch64::*` for NEON);
- the FMA intrinsics (`_mm256_fmadd_ps`, `vfmaq_f32`, …) used pervasively in
  every backend;
- `const` features used inside the constant tables under `src/arch/consts/`.

Older toolchains will fail to compile with errors of the form
`error[E0658]: ... is unstable`. Upgrade with `rustup update stable`.

## Adding the dependency

Add the crate to your `Cargo.toml` as a normal dependency:

```toml
[dependencies]
simdmath = "0.1"
```

If you are tracking the development branch, point Cargo at the Git repository:

```toml
[dependencies]
simdmath = { git = "https://github.com/mtantaoui/simdmath", branch = "main" }
```

`simdmath` has no required runtime dependencies — it links only against the
platform `libm` for a handful of fallback paths and against `core::arch`
intrinsics provided by the compiler.

## Feature flags

The crate exposes only one feature flag:

| Feature                  | Default | Purpose                                                              |
|--------------------------|---------|----------------------------------------------------------------------|
| `unstable-register-api`  | off     | Reserved for the future low-level `Register<f32>` / `Register<f64>` traits. The surface is **not** stable across patch releases; opt in only if you are experimenting with custom kernels. |

There is intentionally no `std`/`no_std` feature: `simdmath` is `no_std`-compatible
out of the box and uses `core::arch` exclusively.

## Selecting a SIMD backend

The backend is chosen at **compile time** from the active `target_feature`s. See
[Runtime vs compile-time dispatch](./dispatch.md) for the full discussion. The
short version is that you control the backend through `RUSTFLAGS` (or the
equivalent `[build]` section of `.cargo/config.toml`).

### x86_64 with AVX2 + FMA

This is the most common desktop / server target since 2013 (Intel Haswell,
AMD Excavator):

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

Or persist it in `.cargo/config.toml`:

```toml
[build]
rustflags = ["-C", "target-feature=+avx2,+fma"]
```

### x86_64 with AVX-512

AVX-512 is available on Intel Skylake-X and newer Xeons, Ice Lake / Tiger Lake
client parts, and AMD Zen 4:

```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512dq,+avx512vl,+avx512bw" \
  cargo build --release
```

The four extensions form the *AVX-512 baseline* required by `simdmath`'s
16-lane f32 and 8-lane f64 kernels; see
[Required CPU features](./cpu_features.md).

### aarch64 with NEON

NEON is mandatory on every AArch64 chip, so no flag is strictly required, but
enabling FMA explicitly is good hygiene:

```bash
RUSTFLAGS="-C target-feature=+neon,+fp16" cargo build --release \
  --target aarch64-unknown-linux-gnu
```

On Apple Silicon (`aarch64-apple-darwin`) NEON and FMA are always present.

### `target-cpu=native` for maximum throughput

If the binary will only run on the build machine — typical for HPC, CI
benchmarking, or scientific computing — let the compiler use every feature
the host supports:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This is equivalent to `-march=native` in C and will pick the *best* backend
`simdmath` knows about for that CPU. **Do not** ship the resulting binary to
end users with heterogeneous hardware: it will SIGILL on older machines.

## Scalar fallback

If no SIMD `target_feature` is enabled — for example you build with the bare
`x86_64-unknown-linux-gnu` defaults, which only guarantee SSE2 — `simdmath`
falls back to a portable scalar implementation written in safe Rust. The
fallback is correct but **slow**; expect a 4–16× throughput regression
compared to the AVX2 backend.

This is a recent change. Earlier versions of the crate emitted
`compile_error!("no supported SIMD ISA enabled")`. If you see that error you
are on an old `simdmath`; update to `0.1.x`:

```bash
cargo update -p simdmath
```

## A minimal example

```rust,ignore
use simdmath::math::VecMath;

fn main() {
    let xs: [f32; 8] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75];
    let ys = xs.sin();              // dispatches to the active backend
    println!("{ys:?}");
}
```

The high-level entry point is the
[`VecMath`](https://docs.rs/simdmath/latest/simdmath/math/trait.VecMath.html)
trait. The lower-level intrinsic-style entry points (`_mm256_sin_ps`,
`vsinq_f32`, …) are also re-exported under `simdmath::arch::*` for users who
want to write their own kernels.

## Verifying the build

After compiling, you can confirm the right backend was selected by inspecting
the disassembly:

```bash
cargo asm --release simdmath::math::VecMath::sin | head -40
```

You should see `vfmadd231ps`, `vfmadd231pd` (AVX2/512) or `fmla` (NEON)
instructions in the polynomial-evaluation section.

## Troubleshooting

- **`error[E0432]: unresolved import \`std::arch::x86_64::_mm256_*\`**
  → You compiled without `+avx2`. Add it to `RUSTFLAGS` or rely on the scalar
  fallback (rebuild with `cargo clean && cargo build`).
- **SIGILL at runtime** → The binary uses an instruction the host does not
  support. Rebuild with the lowest-common-denominator features, or implement
  runtime dispatch (planned for v0.2).
- **`compile_error!: no supported SIMD ISA enabled`** → Pre-`0.1.0`
  behaviour; the scalar fallback now exists. Upgrade.
- **Slower than expected on AVX-512** → Some CPUs (Skylake-X, Ice Lake-SP)
  downclock heavily under sustained AVX-512. Profile both `+avx2` and
  `+avx512f` builds before assuming the wider ISA is faster.

## See also

- [Required CPU features and `RUSTFLAGS`](./cpu_features.md)
- [Runtime vs compile-time dispatch](./dispatch.md)
- [AVX2 backend](../backends/avx2.md), [AVX-512 backend](../backends/avx512.md),
  [NEON backend](../backends/neon.md)
- The Rust reference, [*Conditional compilation*](https://doc.rust-lang.org/reference/conditional-compilation.html#target_feature)
- The `cargo` book, [*Configuration*](https://doc.rust-lang.org/cargo/reference/config.html)
