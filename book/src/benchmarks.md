# Benchmarks

`simdmath` ships microbenchmarks for every public operation. The harness is
[criterion.rs](https://github.com/bheisler/criterion.rs) (see
[`Cargo.toml`](https://github.com/mtantaoui/simdmath/blob/main/Cargo.toml)
`[dev-dependencies]`), one bench file per function under
[`benches/`](https://github.com/mtantaoui/simdmath/tree/main/benches), with a
shared harness in
[`benches/common.rs`](https://github.com/mtantaoui/simdmath/blob/main/benches/common.rs).

This page documents how to run them, what they measure, and — importantly —
what they do **not** measure. Reference numbers will be filled in as the
project hits stable releases; for now the focus is on making the methodology
reproducible.

## Layout

```text
benches/
├── common.rs          # SIZES_F32 / SIZES_F64 + scalar baselines
├── add.rs sub.rs mul.rs div.rs rem.rs
├── reduce.rs
├── sin.rs cos.rs tan.rs
├── asin.rs acos.rs atan.rs atan2.rs
├── exp.rs ln.rs pow.rs
├── sqrt.rs cbrt.rs
```

Cargo's auto-bench detection is **disabled** (`autobenches = false` in
`Cargo.toml`) so that `common.rs` is not picked up as a bench target. Each
real bench is registered explicitly with `harness = false`, which delegates to
criterion's `criterion_main!` macro.

A typical bench body looks like (excerpted from
[`benches/sin.rs`](https://github.com/mtantaoui/simdmath/blob/main/benches/sin.rs)):

```rust,ignore
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simdmath::math::VecMath;
use common::*;

fn bench_sin_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/sin");
    for &n in SIZES_F32 {
        let a: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 20.0 - 10.0).collect();
        g.bench_with_input(BenchmarkId::new("simd",   n), &n, |b, _| b.iter(|| black_box(a.sin())));
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| black_box(a.iter().map(|x| x.sin()).collect::<Vec<_>>()))
        });
    }
    g.finish();
}
```

The bench compares the SIMD path (`a.sin()` via the
[`VecMath`](https://docs.rs/simdmath/latest/simdmath/math/trait.VecMath.html)
trait) against a scalar baseline (`std::f32::sin` mapped over the slice) at
the input sizes declared in `common.rs`:

```rust,ignore
pub const SIZES_F32: &[usize] = &[8, 64, 256, 1024];
pub const SIZES_F64: &[usize] = &[4, 64, 256, 1024];
```

The smallest size matches one full SIMD register (8 f32 lanes for AVX2,
4 f64 lanes for AVX2). The largest size (1024) is comfortably above any L1
prefetch boundary on current Intel and AMD chips, so the L1/L2 transition is
captured.

## Running benchmarks

Build with the right `RUSTFLAGS` so that the SIMD backend you want to measure
is actually selected — see [Required CPU features](./getting_started/cpu_features.md).
Without these flags the crate falls back to the scalar implementation and the
bench is meaningless.

### Run all benches

```text
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo bench
```

### Run a single function

`cargo bench -- <regex>` filters by criterion bench-id substring. The bench
groups are `f32/<func>` and `f64/<func>`:

```text
cargo bench -- "sin"           # both f32/sin and f64/sin
cargo bench -- "f32/sin"       # only the f32 path
cargo bench -- "f32/sin/simd"  # only the SIMD lane of f32/sin
```

### Compare against a baseline

Criterion's `--save-baseline` / `--baseline` is the recommended workflow for
"did my refactor regress anything?":

```text
# Save the current main as a baseline named "main"
git checkout main
cargo bench -- --save-baseline main

# Run the same benches on a feature branch and compare
git checkout my-feature
cargo bench -- --baseline main
```

The output is a per-bench median and a percentage delta with a 95% confidence
interval. Anything outside the noise threshold (criterion's default is 5%)
shows up as a regression or improvement.

For longer-term tracking, save dated baselines (`--save-baseline 2025-q1`)
and keep the resulting `target/criterion/` directories under `.gitignore`.

## What we measure

The bench harness measures **bulk throughput** of the SIMD path: time per
iteration to apply the operation to a *sized* input slice, returning a fresh
output `Vec`. The unit of interest, derived from the criterion median and the
input size \\(n\\), is

\\[
\mathrm{throughput}
= \frac{n \\;\text{elements}}{\mathrm{median\\;time\\;per\\;iteration}}
\quad\bigl[\text{elements/s}\bigr].
\\]

This is the realistic figure of merit for the *intended* use case: applying a
math kernel to a contiguous array.

The bench harness deliberately does *not* set `criterion::Throughput` because
the absolute time is the more useful primary signal during development; the
elements/s number is computed by the reader if needed.

### Why a fresh `Vec` per iteration?

The slice version of `VecMath` allocates and returns. Reusing an output
buffer would tilt the comparison in favour of the SIMD path because the
scalar baseline can't easily reuse a buffer through `iter().map().collect()`.
Allocating in both halves keeps the comparison fair — and matches what real
callers do.

## What we do *not* measure

These are explicit non-goals for v0.1 and will not be reported as bench
results.

1. **Single-element latency.** `simdmath` is not a scalar-accelerated math
   library. There is no bench measuring the latency of `f32x8::sin` applied to
   a single 8-lane register; SIMD register-level latency is dominated by
   instruction decode on amortised loops, and the best operating point is bulk
   throughput on contiguous arrays. If you need scalar latency, use `std`
   directly.

2. **Cross-platform absolute numbers.** A criterion median measured on an
   Intel Tiger Lake laptop tells you very little about a Zen 4 server.
   Reference numbers (when published) will pin both the CPU model and the
   measured ISA features.

3. **Vector-vs-vector wins on tiny `n`.** For `n = 8` the SIMD path is one
   register-load + one kernel + one store, and any extra control flow in the
   wrapping `Vec` path swamps the actual work. The 8-element row is included
   for completeness, not as a comparison target.

## Disclaimers

Microbenchmark numbers on modern CPUs are *very* sensitive to:

- **Frequency scaling.** Turbo / SpeedShift will boost the first few seconds
  of a bench run and then settle. Criterion's warm-up phase mitigates this
  for short benches but not for very long groups. Pinning the governor
  (`cpupower frequency-set -g performance` on Linux) is recommended for
  publishable numbers.

- **Thermal throttling.** Sustained AVX-512 throughput will throttle the
  package frequency on most chips manufactured before 2022. If you see the
  AVX-512 bench *slower* than AVX2 on identical input, suspect this first.

- **AVX-512 frequency licensing.** On older Intel parts (Skylake-X, Ice Lake-X),
  using zmm registers downclocks the whole socket — so a bench process
  running concurrently with a non-AVX-512 workload will *speed up* the
  non-AVX-512 workload by reducing pressure on the AVX-512 voltage domain.
  This is the classic "AVX-512 makes everything else faster" story; it is
  visible in cross-bench measurements but not within a single bench file.

- **`black_box` semantics.** Criterion's `black_box` is a hint to the
  optimiser, not a barrier. With `-O3 -C lto=fat` the compiler can still
  hoist some of the work out of the timing loop. The bench files use
  `black_box` on both inputs and outputs, which is sufficient for stable
  numbers in `--release` but not bulletproof.

- **Sample input distribution.** The current benches use a deterministic
  ramp (`make_sin_input_*` returns evenly spaced values in `[-10, 10]`),
  which avoids data-dependent branch mispredictions but may *underestimate*
  the cost on inputs that exercise rare branches in the reducer (e.g. f32
  `sin` on inputs above \\(2^{20}\\)). Bench inputs that target specific
  performance-sensitive regions are a v0.2 item.

## Reference numbers

> **Placeholder.** Representative wall-clock numbers — measured on pinned
> hardware, with documented `RUSTFLAGS`, governor, and microcode revision —
> will appear in this section in subsequent releases. Until then the
> meaningful comparison is the *delta* between two baselines on your own
> hardware via `--save-baseline` / `--baseline` (above).

The intent is to publish a small per-release table of medians on:

- Intel Tiger Lake (AVX2+FMA, no AVX-512)
- Intel Sapphire Rapids (AVX-512F + AVX-512DQ + AVX-512VL)
- AMD Zen 4 (AVX-512 with double-pumped 256-bit data path)
- Apple M2 (NEON, 128-bit)

For each: per-function median elements/s at \\(n = 1024\\), plus the SIMD-vs-
scalar speed-up. This is the table that will eventually live in this section.

## See also

- [Per-function ULP tables](./precision/tables.md) — the precision contract
  that the benchmarks are measured against.
- [SIMD backends](./backends/avx2.md) — how the dispatch picks a backend at
  compile time.
- [Required CPU features and `RUSTFLAGS`](./getting_started/cpu_features.md) —
  how to make sure your bench actually runs the SIMD path.
