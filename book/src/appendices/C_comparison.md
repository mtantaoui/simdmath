# Appendix C — Comparison with other SIMD math libraries

This appendix surveys the SIMD math landscape and positions `simdmath` against
the established alternatives. It is not a benchmark — see
[Benchmarks](../benchmarks.md) for runtime numbers — and the accuracy
columns reproduce the *advertised* claims of each project, not measurements
made by us.

The intent is to give a prospective user enough information to pick the right
tool. `simdmath` is intentionally a small, opinionated, Rust-native crate; for
many use cases one of the larger libraries is the better choice and this page
will tell you when.

## Overview table

| Library | Lang. | Accuracy claim | AVX2 | AVX-512 | NEON | SVE | Other | Runtime dispatch | License | Status |
|---------|-------|---------------|------|---------|------|-----|-------|------------------|---------|--------|
| **simdmath** (this crate) | Rust | ≤ 1–3 ULP, documented per fn | ✅ | ✅ | ✅ | ❌ | — | ❌ (compile-time only) | MIT | v0.1, active |
| [SLEEF](https://sleef.org) | C | "1 ULP" / "3.5 ULP" variants | ✅ | ✅ | ✅ | ✅ | RVV, VSX, Cuda | ✅ | BSL-1.0 | mature, very active |
| [glibc libmvec](https://sourceware.org/glibc/wiki/libmvec) | C / asm | ≤ 4 ULP (per glibc spec) | ✅ | ✅ | ✅ (limited) | — | — | ✅ (via ifunc) | LGPL-2.1+ | mature, ships in glibc |
| [Intel SVML](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/short-vector-math-library-operations-svml.html) | C / asm | "HA / LA / EP" (1 / 4 / 5+ ULP) | ✅ | ✅ | — | — | — | ✅ (via ICC `-fimf-*`) | Proprietary (Intel oneAPI) | mature |
| [Arm Optimized Routines / mathlib](https://github.com/ARM-software/optimized-routines) | C | ≤ 3.5 ULP | — | — | ✅ | ✅ | — | n/a (per-arch builds) | MIT | mature |
| [vsimd / xsimd extras](https://github.com/xtensor-stack/xsimd) | C++ | wraps SLEEF | ✅ | ✅ | ✅ | ✅ | WASM | via SLEEF | BSD-3 | mature |
| [pulp](https://crates.io/crates/pulp) | Rust | scalar-quality (delegates to `f32`/`f64` methods or libm) | ✅ | ✅ | ✅ | — | — | ✅ (via `Arch`) | MIT/Apache-2.0 | active |
| [wide](https://crates.io/crates/wide) | Rust | scalar-quality (per-lane scalar libm) | ✅ | partial | ✅ | — | — | ❌ | Zlib/Apache | active |
| [`std::simd`](https://doc.rust-lang.org/std/simd/) (portable_simd) | Rust | none — no transcendentals yet | ✅ | ✅ | ✅ | ✅ | — | n/a | dual-licensed | nightly only |

Notes on the "Accuracy claim" column:

- *SLEEF* publishes two variants per function: a "1 ULP" path (e.g.
  `Sleef_sinf_u10`) and a "3.5 ULP" path (e.g. `Sleef_sinf_u35`) that trades
  precision for speed. We report both bands.
- *libmvec* nominally targets 4 ULP; in practice it is much closer to 1 ULP
  for most functions, but the *contract* is 4.
- *Intel SVML* exposes three accuracy modes via the ICC flag
  `-fimf-precision={high,low,extended-precision}`. The "EP" mode trades down
  to roughly 11 bits and is rarely used in scientific code.
- *pulp* and *wide* delegate the math kernel to per-lane scalar operations;
  their precision is therefore the precision of `std`'s `f32::sin` (or libm)
  on the host platform — typically faithful but not guaranteed by the crate.

## Where `simdmath` fits

`simdmath` occupies a deliberately narrow slice:

> **A small, Rust-native, MIT-licensed SIMD math kernel set with a documented
> per-function ULP contract, no C dependencies, and no link-time tricks.**

It is *not* a competitor to SLEEF or libmvec on breadth. It implements 13
functions; SLEEF implements ≈ 50. It is also not a portable-SIMD shim: we
hand-write the AVX2, AVX-512 and NEON paths because the portable-SIMD
abstraction (Rust `std::simd`, the C++ `std::experimental::simd`) does not
yet expose the FMA + mask-blend + double-pumped-blend primitives we rely on
to hold the ≤ 1 ULP bound.

### Where `simdmath` wins

1. **Native Rust.** No `cc` build, no system libm, no FFI surface. Building
   `simdmath` is a `cargo build`. This matters in environments that ship a
   cross-compiled binary (embedded, WASM-adjacent, regulated builds).

2. **MIT licensing.** Both SLEEF (Boost Software Licence) and libmvec
   (LGPL-2.1+) are friendly licences, but MIT is the lowest-friction choice
   for a Cargo dependency and matches the rest of the typical Rust stack.

3. **Per-function, per-precision, documented ULP contract.** The table at the
   top of [`src/lib.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/lib.rs)
   plus [Appendix B](./B_constants.md) and the methodology in
   [precision/methodology.md](../precision/methodology.md) is the contract.
   SLEEF documents per-function ULPs but in a per-symbol-name form (`u10`
   suffix) rather than a contract; libmvec relies on a single global "≤ 4 ULP"
   number; `pulp` and `wide` have no contract.

4. **`no_std`-friendly.** No allocation in the kernel path; the slice
   wrappers do allocate (because `Vec::collect` is convenient) but the SIMD
   primitives themselves are register-only. SLEEF compiled with the default
   Makefile is also `no_std` in spirit but pulls in C runtime symbols on
   most platforms.

5. **Reviewable scope.** 13 functions × 3 ISAs × 2 precisions = 78 kernels,
   plus tests. A reader can audit the entire kernel surface in an afternoon.
   SLEEF's kernel surface is closer to 1000 functions including all ISA
   variants.

### Where `simdmath` loses

1. **No SVE or RVV.** `simdmath` only targets the three ISAs widely deployed
   on consumer hardware in 2025. SLEEF, ARM mathlib and (emerging) RVV
   ports of Glibc all cover SVE2 and RISC-V Vector. If you target
   Graviton 3 / 4 with SVE-256 in production, SLEEF is a better fit today.

2. **No runtime dispatch.** Backend selection happens at compile time via
   `target_feature`. To ship a single binary that uses AVX-512 on Sapphire
   Rapids and AVX2 on Skylake, you need to build per-target or use
   `multiversion`-style crates. This is on the v0.2 roadmap; SLEEF and
   libmvec already do it (libmvec via the glibc ifunc mechanism).

3. **Smaller function set.** No `tanh`, `sinh`, `cosh`, `expm1`, `log1p`,
   `log2`, `log10`, `exp2`, `erf`, `erfc`, gamma, Bessel. The roadmap
   targets `expm1` and `log1p` next (because `exp` and `ln` already provide
   the kernels) but full coverage is a multi-release effort.

4. **No complex-number SIMD.** SLEEF and the C++ `xsimd` extras handle
   `complex<float>` / `complex<double>` lane patterns; `simdmath` only does
   real-valued lanes.

5. **Younger.** SLEEF dates to 2010; `simdmath` is at v0.1. The kernel
   choices are well-trodden (musl / fdlibm), but the surface is new and the
   ecosystem integration (cargo features, `multiversion` interop, criterion
   baselines on public CI) is still being built out.

## When to pick what

A fast-and-loose decision tree:

- **You need every transcendental imaginable (Bessel, Gamma, complex
  trig)** → SLEEF. Nothing else has the breadth.
- **You ship a single x86_64 binary that must run on Skylake *and*
  Sapphire Rapids and your distro is Linux** → libmvec. Glibc ifuncs do
  the dispatch for you, free.
- **You write Rust, your target is AVX2/AVX-512/NEON, and you want MIT-
  licensed dependencies with documented ULPs** → `simdmath`.
- **You write Rust, your target is wider than AVX2/AVX-512/NEON, and
  scalar-quality math is acceptable** → `pulp` or `wide`.
- **You target Apple Silicon only and want to use Accelerate.framework's
  vForce** → not a `simdmath` use case; bind to vForce directly.
- **You target SVE on Graviton/Fujitsu A64FX/RVV servers** → ARM mathlib or
  SLEEF (RVV port in flight).

## Honest disclosures

- **We have not independently re-verified the ULP claims of SLEEF, libmvec,
  SVML, ARM mathlib, pulp, wide, or `std::simd`.** The numbers in the
  overview table are reproduced from each project's own documentation.
  Empirical comparisons require a fixed test harness and a fixed input
  distribution, which is out of scope for this appendix.
- **Performance is not part of this comparison.** A Rust crate calling
  hand-written assembly with runtime dispatch (libmvec) can outpace any
  intrinsics-based crate on the right hardware. For Rust-vs-Rust
  comparisons on identical inputs, see [Benchmarks](../benchmarks.md) once
  the v0.2 reference numbers land.
- **License compatibility cuts both ways.** MIT is liberal but does not
  protect against license-laundering of derived works; users who care about
  copyleft propagation should read each library's LICENSE file directly.

## References and links

- SLEEF: <https://sleef.org> — Shibata, N. *SLEEF: A Portable Vectorized
  Library of C Standard Mathematical Functions*, IEEE TPDS, 2020.
- glibc libmvec: <https://sourceware.org/glibc/wiki/libmvec>
- Intel SVML: part of Intel oneAPI / Intel C++ Compiler.
  <https://www.intel.com/content/www/us/en/developer/tools/oneapi/svml.html>
- ARM Optimized Routines / mathlib:
  <https://github.com/ARM-software/optimized-routines>
- xsimd: <https://github.com/xtensor-stack/xsimd>
- pulp: <https://crates.io/crates/pulp>
- wide: <https://crates.io/crates/wide>
- portable-simd RFC: <https://github.com/rust-lang/portable-simd>

For the formal citations of the *algorithms* (not the libraries), see
[Appendix E — Bibliography](./E_bibliography.md).
