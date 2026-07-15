# Required CPU features and `RUSTFLAGS`

`simdmath` exposes three SIMD backends — AVX2, AVX-512, and NEON — each
gated on a specific set of CPU features. This chapter explains what each
feature actually provides at the instruction-set level, how to discover what
your host CPU supports, and which `simdmath` functions need which features.

## What "AVX2 + FMA" means

The AVX2 backend assumes the union of three Intel ISA extensions, all of
which shipped together on Haswell (2013) and the AMD equivalent (Excavator,
2015):

- **AVX** — 256-bit `ymm` registers and the basic floating-point
  arithmetic on them (`vaddps`, `vmulps`, …).
- **AVX2** — 256-bit *integer* operations and gather instructions; required
  for masking tricks in `cbrt` and the integer-bias manipulation in `exp`.
- **FMA3** — fused multiply–add, three-operand form. The
  `vfmadd231ps`/`vfmadd231pd` family is the workhorse of every polynomial
  in this crate.

In Rust the corresponding `target_feature` flags are `avx`, `avx2`, and `fma`.
Setting `+avx2` does **not** automatically enable `+fma` — you must list both
explicitly. `simdmath`'s build emits a `compile_error!` if `avx2` is set
without `fma`, since every kernel assumes one-rounding multiply-adds.

## What "AVX-512 baseline" means

AVX-512 is not a single feature: it is a family of 19+ extensions. `simdmath`
requires only the **foundation extension**:

| Extension     | Rust `target_feature` | Provides                                                                 |
|---------------|-----------------------|--------------------------------------------------------------------------|
| AVX-512F      | `avx512f`             | 512-bit `zmm` registers, masked FP arithmetic, the foundation.           |

The backend selection in `src/arch/mod.rs` gates exclusively on `avx512f`;
if it is missing, the build falls back to AVX2. Every CPU that ships
AVX-512F in practice (Skylake-X onward, AMD Zen 4/5) also ships `avx512vl`,
`avx512dq`, and `avx512bw`, so enabling those too is harmless — but
`simdmath` neither checks for nor needs them.

`simdmath` does **not** require AVX-512ER, IFMA, VBMI, VPOPCNTDQ, BF16, FP16,
or any other "post-Skylake-X" extension. This keeps the code path
runnable on Cannon Lake, Ice Lake, Tiger Lake, Sapphire Rapids, and AMD
Zen 4 without further configuration.

## What "NEON" means

ARM Advanced SIMD ("NEON") is **mandatory** on AArch64 — every Armv8-A
profile chip has 128-bit `vN` registers, FMA (`fmla`), and the full f32 / f64
pipeline. There is therefore no "do I have NEON?" check on
`aarch64-*-*` targets; the `target_feature = "neon"` cfg is always true for
those triples.

Two further features are *optional* but enable extra speed:

- **`fp16`** — half-precision FP arithmetic. `simdmath` does not yet expose
  `f16` math, but the flag enables tighter scheduling on Apple Silicon.
- **`crypto`** — AES/SHA accelerators; unrelated to `simdmath` but often
  bundled in benchmark configurations.

On 32-bit ARMv7 (`armv7-*`), NEON is **not** mandatory, and `simdmath` does
not currently support that target.

## Inspecting the host CPU

### Linux

```bash
# AVX flags only
lscpu | grep -E -o 'avx[^ ]*' | sort -u

# Full feature list
cat /proc/cpuinfo | awk '/^flags/ {print; exit}' | tr ' ' '\n' | sort | column
```

Look for `avx2`, `fma`, and `avx512f`.

### macOS

```bash
sysctl -a | grep machdep.cpu.features
sysctl -a | grep machdep.cpu.leaf7_features   # AVX-512 lives here
```

On Apple Silicon, the relevant key is `hw.optional.arm.FEAT_FP16` and the
`neon` feature is implicit.

### Windows

```powershell
Get-CimInstance Win32_Processor | Format-List Name, Description, *Features*
```

For a deeper view, install Microsoft's
[`coreinfo`](https://learn.microsoft.com/sysinternals/downloads/coreinfo)
and run `coreinfo -f`.

### Programmatically from Rust

```rust,ignore
fn main() {
    println!("avx2     : {}", is_x86_feature_detected!("avx2"));
    println!("fma      : {}", is_x86_feature_detected!("fma"));
    println!("avx512f  : {}", is_x86_feature_detected!("avx512f"));
    println!("avx512vl : {}", is_x86_feature_detected!("avx512vl"));
    println!("avx512dq : {}", is_x86_feature_detected!("avx512dq"));
    println!("avx512bw : {}", is_x86_feature_detected!("avx512bw"));
}
```

The `is_x86_feature_detected!` macro performs a `CPUID` query at runtime;
the `aarch64` analogue is `is_aarch64_feature_detected!`.

## Feature → function matrix

The table below lists which CPU features each `simdmath` function relies on
(beyond the architecture default). Only **non-default** requirements are
listed; AVX2 needs `+fma` everywhere, NEON inherits FMA from the base ISA.

| Function family                  | AVX2 backend           | AVX-512 backend          | NEON backend |
|----------------------------------|------------------------|--------------------------|--------------|
| `sin`, `cos`, `tan`              | `avx2`, `fma`          | `avx512f`                | `neon`       |
| `asin`, `acos`, `atan`           | `avx2`, `fma`          | `avx512f`                | `neon`       |
| `atan2`                          | `avx2`, `fma`          | `avx512f`                | `neon`       |
| `exp`, `ln`                      | `avx2`, `fma`          | `avx512f`                | `neon`       |
| `pow`                            | `avx2`, `fma`          | `avx512f`                | `neon`       |
| `sqrt`                           | `avx`                  | `avx512f`                | `neon`       |
| `cbrt`                           | `avx2`, `fma`          | `avx512f`                | `neon`       |
| element-wise add/mul/min/max     | `avx`                  | `avx512f`                | `neon`       |

Every AVX-512 kernel — `cbrt` and `pow` included — is written against the
foundation extension only; no `avx512vl`/`dq`/`bw` instruction is emitted.

## Setting the features in `RUSTFLAGS`

For the AVX-512 backend, the canonical invocation is:

```bash
RUSTFLAGS="-C target-feature=+avx512f" \
  cargo build --release
```

For mixed environments, prefer `-C target-cpu=<exact-uarch>` over the
feature list, e.g. `-C target-cpu=icelake-server` or `-C target-cpu=znver4`.
This lets the compiler also tune *scheduling* for that micro-architecture,
not just enable the right encodings.

## Verifying that the right code path was selected

`simdmath` uses `cfg!`-style ladders, so the simplest verification is to
compile with `--emit=asm` and grep:

```bash
RUSTFLAGS="-C target-feature=+avx512f" \
  cargo rustc --release -- --emit=asm
grep -c 'zmm' target/release/deps/*.s     # >0 means AVX-512 was used
grep -c 'ymm' target/release/deps/*.s     # AVX/AVX2
grep -c 'fmla v' target/release/deps/*.s  # NEON
```

## See also

- [Installation and feature flags](./installation.md)
- [Runtime vs compile-time dispatch](./dispatch.md)
- [AVX2 backend](../backends/avx2.md), [AVX-512 backend](../backends/avx512.md),
  [NEON backend](../backends/neon.md)
- Intel, *Intel® 64 and IA-32 Architectures Software Developer's Manual*,
  Vol. 2 (instruction reference) and Vol. 1 §15 (AVX-512 overview).
- ARM, *Arm® Architecture Reference Manual for A-profile* — Section C2 (NEON).
