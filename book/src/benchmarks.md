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
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use simdmath::math::VecMath;
use common::*;

fn bench_sin_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/sin/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let a = make_sin_input_f32(n);
        g.bench_function("simd",   |b| b.iter(|| black_box(a.sin())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_sin_f32(black_box(&a)))));
        g.finish();
    }
}
```

One criterion group is created **per input size**, each with
`Throughput::Elements` set. This gives criterion enough information to render
clean two-bar violin plots — one for `simd`, one for `scalar` — without
crowding multiple sizes onto the same axis.

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
groups follow the pattern `<type>/<func>/<size>`:

```text
cargo bench -- "sin"              # all sin groups (f32 + f64, all sizes)
cargo bench -- "f32/sin"          # only the f32/sin groups
cargo bench -- "f32/sin/1024"     # only the 1024-element f32/sin group
cargo bench -- "f32/sin/1024/simd"  # only the SIMD variant
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

Each bench group sets `criterion::Throughput::Elements(n)`, so criterion
reports elements/s directly alongside the raw median time.

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

All numbers below are criterion **medians** at \\(n = 1024\\) elements,
reported in **Melem/s** (millions of elements per second). The speed-up column
is SIMD median divided into scalar median; values below 1× mean the SIMD path
is slower than the scalar baseline on that machine (see notes after each
table).

---

### AVX2 — Intel Core Ultra 7 155H @ 4.5 GHz

`RUSTFLAGS="-C target-feature=+avx2,+fma" cargo bench`

| Function | f32 Melem/s | f32 speed-up | f64 Melem/s | f64 speed-up |
|----------|------------:|:------------:|------------:|:------------:|
| `sin`    |         933 |    2.3×      |         651 |    3.2×      |
| `cos`    |         953 |    2.1×      |         668 |    3.5×      |
| `tan`    |        1092 |    6.9×      |         467 |    2.0×      |
| `asin`   |        1312 |    3.4×      |         379 |    1.3×      |
| `acos`   |         963 |    2.8×      |         296 |    1.0×      |
| `atan`   |        2883 |    9.8×      |         541 |    2.0×      |
| `atan2`  |         587 |    3.9×      |         182 |    1.6×      |
| `exp`    |         775 |    1.5×      |         807 |    2.6×      |
| `ln`     |         583 |    1.2×      |         620 |    1.7×      |
| `pow`    |         158 |   **0.6×**   |         172 |    1.3×      |
| `sqrt`   |        5781 |    1.0×      |        1562 |    1.0×      |
| `cbrt`   |         864 |    2.5×      |         587 |    4.9×      |

---

### AVX-512 — Intel Xeon w5-2555X @ 4.8 GHz

`RUSTFLAGS="-C target-feature=+avx512f,+avx512vl" cargo bench`

| Function | f32 Melem/s | f32 speed-up | f64 Melem/s | f64 speed-up |
|----------|------------:|:------------:|------------:|:------------:|
| `sin`    |        1591 |    5.5×      |        1074 |    5.3×      |
| `cos`    |        1644 |    4.0×      |        1038 |    5.4×      |
| `tan`    |        1610 |   11.1×      |         715 |    3.1×      |
| `asin`   |        1182 |    3.4×      |         364 |    1.3×      |
| `acos`   |         844 |    2.7×      |         248 |   **0.9×**   |
| `atan`   |        3992 |   14.6×      |         497 |    1.9×      |
| `atan2`  |        1172 |    8.3×      |         207 |    1.7×      |
| `exp`    |        1115 |    2.2×      |        1289 |    4.1×      |
| `ln`     |         956 |    1.8×      |         949 |    3.0×      |
| `pow`    |         280 |    1.0×      |         259 |    2.0×      |
| `sqrt`   |        4816 |   **0.9×**   |        1307 |   **0.9×**   |
| `cbrt`   |         937 |    2.9×      |         642 |    5.8×      |

---

### NEON — Apple Mac Pro M1

`RUSTFLAGS="-C target-feature=+neon" cargo bench` (aarch64)

| Function | f32 Melem/s | f32 speed-up | f64 Melem/s | f64 speed-up |
|----------|------------:|:------------:|------------:|:------------:|
| `sin`    |         699 |    1.6×      |         405 |    1.4×      |
| `cos`    |         720 |    1.7×      |         314 |    1.2×      |
| `tan`    |         861 |    2.1×      |         301 |    1.0×      |
| `asin`   |        1038 |    1.9×      |         379 |    1.6×      |
| `acos`   |         823 |    1.6×      |         292 |    1.1×      |
| `atan`   |        2155 |    5.2×      |         432 |    1.8×      |
| `atan2`  |         499 |    1.1×      |         126 |   **0.8×**   |
| `exp`    |         509 |   **0.8×**   |         573 |    1.4×      |
| `ln`     |         530 |    1.2×      |         400 |    1.2×      |
| `pow`    |          94 |   **0.3×**   |          82 |   **0.6×**   |
| `sqrt`   |        6244 |    1.0×      |        3056 |    1.0×      |
| `cbrt`   |         674 |    2.2×      |         379 |   **0.9×**   |

---

### Notes on sub-1× results

**`pow`** is implemented as \\(\exp(y \cdot \ln x)\\), composing two
transcendental kernels. At \\(n = 1024\\) the allocation and loop overhead of
the outer `Vec` path, combined with two sequential kernel passes, erases the
SIMD lane advantage on machines where the scalar `std::f64::powf` is
well-optimised (e.g. Apple M1 with its fast integer divider and scalar FPU
pipeline). This is a known trade-off for v0.1; a fused `pow` kernel that
avoids the intermediate `Vec` is a v0.2 item.

**`sqrt`** is a single hardware instruction (`vsqrtps` / `vsqrtpd` /
`fsqrt`). The scalar baseline compiles to the same instruction, so the
speed-up is limited to the lane-width ratio minus loop overhead — which at
\\(n = 1024\\) rounds to ≈ 1×.

**NEON lane counts** are half those of AVX2 (4×f32, 2×f64 vs 8×f32, 4×f64),
so the ceiling speed-up is correspondingly lower. Functions with speed-ups
near 1× on NEON are bottlenecked by memory bandwidth or scalar pipeline
throughput rather than compute.

## See also

- [Per-function ULP tables](./precision/tables.md) — the precision contract
  that the benchmarks are measured against.
- [SIMD backends](./backends/avx2.md) — how the dispatch picks a backend at
  compile time.
- [Required CPU features and `RUSTFLAGS`](./getting_started/cpu_features.md) —
  how to make sure your bench actually runs the SIMD path.
