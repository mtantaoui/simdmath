# AVX2 backend

This chapter documents the AVX2 + FMA backend. AVX2 is the workhorse SIMD ISA
on the x86_64 desktop, server, and laptop market: every Intel chip from
Haswell (2013) onward and every AMD chip from Excavator (2015) onward
implements the full instruction set used here.

The backend lives under [`src/arch/avx2/`](https://github.com/mtantaoui/simdmath/tree/main/src/arch/avx2).

## Lane counts and register types

AVX2 widens the SSE 128-bit XMM register file to 256-bit YMM registers. The
two SIMD vector types this crate exposes for the backend are:

| Type | Underlying | Element | Lane count | Width |
|------|------------|---------|------------|-------|
| `F32x8` | `__m256`  | `f32`   | 8          | 256 b |
| `F64x4` | `__m256d` | `f64`   | 4          | 256 b |

Both are wrapped in a thin POD struct that carries an active-lane count for
tail handling:

```rust,ignore
#[derive(Copy, Clone, Debug)]
pub struct F32x8 {
    pub(crate) size: usize,
    pub(crate) elements: __m256,
}
```

The `size` field is **not** the SIMD width: it is the number of *logically
active* lanes, used by [`store_at_partial`](./traits.md) to suppress writes
to padding lanes when a slice's length is not a multiple of 8 (or 4 for
`F64x4`).

## Required CPU features

Two compile-time feature flags must be enabled:

- `avx2`  — 256-bit integer and float operations on YMM registers.
- `fma`   — three-operand fused multiply-add (`vfmadd*`/`vfnmadd*`/`vfmsub*`).

Both are enabled together by:

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
# or simply target-cpu=native on a Haswell-or-newer host
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

FMA is mandatory because every transcendental kernel in this crate evaluates
its polynomial in Horner form using `_mm256_fmadd_*`. Replacing those with
separate `mul` + `add` adds an extra rounding step per term, which alone is
enough to push the worst-case error past the `≤ 1 ULP` envelope for the
inverse trigonometric functions.

## Alignment

256-bit YMM loads and stores are most efficient at **32-byte alignment**:

```rust,ignore
impl Align<f32> for F32x8 {
    fn is_aligned(ptr: *const f32) -> bool {
        (ptr as usize).is_multiple_of(core::mem::align_of::<__m256>())
    }
}
```

`align_of::<__m256>()` is 32 bytes. The crate dispatches automatically to
`_mm256_load_ps` / `_mm256_loadu_ps` based on the result. On Haswell and
newer microarchitectures the unaligned-vs-aligned penalty is negligible
when the address happens to be aligned at run-time, so the primary
motivation for the alignment check is avoiding the **fault** that
`_mm256_load_ps` raises on a misaligned pointer.

## Key intrinsics used

The implementations rely on a small, well-tuned subset of the AVX2 + FMA
intrinsic surface. Below is a representative list with the role each
intrinsic plays in the kernels.

### Arithmetic / FMA

| Intrinsic | Computes | Used in |
|-----------|----------|---------|
| `_mm256_fmadd_ps`   | \\(a \cdot b + c\\)  | Horner step in every polynomial |
| `_mm256_fnmadd_ps`  | \\(-a \cdot b + c\\) | Cody-Waite reduction `y = x - n·π/2` |
| `_mm256_fmsub_ps`   | \\(a \cdot b - c\\)  | acos compensation step |
| `_mm256_mul_ps`     | \\(a \cdot b\\)      | \\(z = x^2\\), \\(w = z^2\\), etc. |
| `_mm256_div_ps`     | \\(a / b\\)          | atan reduction, asin Padé |
| `_mm256_sqrt_ps`    | \\(\sqrt{a}\\)       | asin half-angle |

The 3-operand FMA is the single most performance-critical primitive: a
degree-9 odd polynomial like atan(t) collapses to 9 dependent FMAs with no
intermediate rounding.

### Comparisons and blends

AVX2 comparisons (`_mm256_cmp_ps`) return a vector mask whose lanes are all-1
or all-0. Result selection is done with `_mm256_blendv_ps`, which picks the
lane from the second source where the mask's sign bit is set:

```rust,ignore
// asin tail: |x| == 1 returns ±π/2, |x| > 1 returns NaN
let result_ge_1 = _mm256_blendv_ps(nan, result_eq_1, is_abs_eq_1);
```

This branchless idiom is used everywhere in the trig and inverse-trig
kernels: every range case is computed unconditionally, and a chain of
`blendv` instructions selects the final result. See the asin code excerpt
in [the asin chapter](../functions/asin.md) for a full example.

### Bit manipulation

The `__m256i` integer-domain intrinsics are reused on float bit patterns
via `_mm256_castps_si256` / `_mm256_castsi256_ps`. The Dekker split in asin
masks the low 12 mantissa bits of \\(\sqrt{z}\\) to produce an exact high part:

```rust,ignore
let df = _mm256_castsi256_ps(_mm256_and_si256(
    _mm256_castps_si256(s_large),
    _mm256_set1_epi32(0xfffff000_u32 as i32),
));
```

The cast intrinsics emit no instructions — they are pure type re-tags.

### Masked load/store

Tail handling for slices whose length is not a multiple of the lane count
uses `_mm256_maskload_ps` / `_mm256_maskstore_ps` driven by a precomputed
sign-bit table:

```rust,ignore
pub static MASK: [[i32; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0, 0, 0],
    /* … */
];
```

Indexing `MASK[size]` produces a register with `size` lanes active. This is
the AVX2-specific encoding of the [`Load::load_partial`](./traits.md)
trait method.

### Lane-promotion for f32 transcendentals

A subtle but important pattern: the f32 transcendental kernels (sin, cos,
tan) compute their kernel polynomials in **f64 internally**. The 8-lane
input `__m256` is split into two `__m128` halves and promoted to `__m256d`
via `_mm256_cvtps_pd`, processed in parallel as two 4-lane f64 kernels,
and then narrowed back to `__m256` via `_mm256_cvtpd_ps`:

```rust,ignore
let x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
let x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
let sin_lo = sin_ps_in_f64(x_lo);
let sin_hi = sin_ps_in_f64(x_hi);
let result_lo = _mm256_cvtpd_ps(sin_lo);
let result_hi = _mm256_cvtpd_ps(sin_hi);
_mm256_insertf128_ps(_mm256_castps128_ps256(result_lo), result_hi, 1)
```

This costs two `cvtps_pd`, two `cvtpd_ps`, and a 128-bit `insertf128`, but
guarantees that the worst-case argument-reduction cancellation error for
small-but-not-tiny `|x|` near multiples of `π/2` is absorbed by the extra
f64 mantissa bits. The end result is `≤ 2 ULP` in single precision, which
matches what musl's `sinf`/`cosf` kernels achieve and would not be possible
in pure f32.

## Tail handling

For a slice of length `N`, the per-vector loop processes
`N / LANE_COUNT` full registers, then a single `load_partial` /
`store_at_partial` pair handles the remaining `N % LANE_COUNT` lanes. The
inactive lanes of the partial register are zeroed on load (so they don't
generate spurious infinities or NaNs in arithmetic) and masked on store
(so they don't overwrite memory the caller does not own).

## Where to look in the source

| Topic | File |
|-------|------|
| Register type, load/store, operators | [`avx2/f32x8.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/f32x8.rs), [`avx2/f64x4.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/f64x4.rs) |
| `VecMath` register impl   | [`avx2/math.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/math.rs) |
| Trig kernels              | [`avx2/sin.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/sin.rs), [`avx2/cos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/cos.rs), [`avx2/tan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/tan.rs) |
| Inverse trig kernels      | [`avx2/asin.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/asin.rs), [`avx2/acos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/acos.rs), [`avx2/atan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/atan.rs) |
| Vec-level loop / chunking | [`math/avx2.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/math/avx2.rs) |

## See also

- [AVX-512 backend](./avx512.md) — the wider-register companion, with mask
  registers instead of vector blends.
- [NEON backend](./neon.md) — the aarch64 counterpart.
- [Compile-time dispatch](./dispatch.md) — how the AVX2 path is selected.
- [`Load`/`Store`/`Align` traits](./traits.md) — the abstract contract every
  backend implements.
