# NEON backend

This chapter documents the Advanced SIMD ("NEON") backend used on
`aarch64`. Where AVX2 and AVX-512 are compile-time gated by an explicit
target feature, NEON is **always on** for AArch64: the Armv8-A architecture
specification mandates the full Advanced SIMD register file and instruction
set in every conformant aarch64 implementation.

The backend lives under [`src/arch/neon/`](https://github.com/mtantaoui/simdmath/tree/main/src/arch/neon).

## Lane counts and register types

NEON's vector registers are 128 bits wide. Compared with AVX2's 256-bit YMM
this halves the lane count for both precisions:

| Type    | Underlying      | Element | Lane count | Width  |
|---------|-----------------|---------|------------|--------|
| `F32x4` | `float32x4_t`   | `f32`   | 4          | 128 b  |
| `F64x2` | `float64x2_t`   | `f64`   | 2          | 128 b  |

The struct shape mirrors the x86 backends; the `size` field is again the
count of *logically active* lanes, not the SIMD width:

```rust,ignore
#[derive(Copy, Clone, Debug)]
pub struct F32x4 {
    pub(crate) size: usize,
    pub(crate) elements: float32x4_t,
}
```

## Always-on nature on aarch64

The crate compiles the NEON backend whenever `target_arch = "aarch64"`
without any feature gate. This is in contrast to the x86 backends, which
require explicit `target-feature` flags. The trade-off is that the
`aarch64-unknown-linux-gnu` and `aarch64-apple-darwin` targets have no
"scalar fallback" mode at all in this crate — there is no aarch64 chip on
which NEON is missing.

A practical consequence: cross-compiling to aarch64 requires nothing
beyond the target itself:

```bash
cargo check --target aarch64-unknown-linux-gnu
# (no RUSTFLAGS needed)
```

## f64 NEON: aarch64-only

A subtle but important caveat: the f64 path (`F64x2`, `vsin_f64`,
`_mm256d`-equivalents on NEON) only exists on **64-bit ARM**. The original
ARMv7-A NEON specification is single-precision only; the f64 lanes
(`vaddq_f64`, `vmulq_f64`, …) were added in ARMv8-A and are not available
on 32-bit ARM. This crate therefore gates the entire NEON backend on
`target_arch = "aarch64"` rather than `target_feature = "neon"`, which
would also trigger on ARMv7.

If a 32-bit ARM target is selected, the build falls through to the scalar
fallback path described in the [dispatch](./dispatch.md) chapter.

## Alignment

NEON's `vld1q_*` / `vst1q_*` intrinsics handle aligned and unaligned
addresses transparently:

```rust,ignore
impl Align<f32> for F32x4 {
    fn is_aligned(ptr: *const f32) -> bool {
        (ptr as usize).is_multiple_of(core::mem::align_of::<float32x4_t>())
    }
}
```

`align_of::<float32x4_t>()` is **16 bytes**, half the AVX2 requirement.
Modern Cortex-A cores (A53 and later) and all Apple silicon execute
unaligned 128-bit loads with no measurable penalty when the address
happens to land on a 16-byte boundary, so the alignment check is mostly
defensive — there is no "aligned faults if misaligned" instruction at
this width.

## Ordering and naming quirks

The NEON intrinsic surface differs from x86 in three places that matter
for porting algorithms across backends. The differences are deliberately
called out by the algorithm comments in the source.

### 1. `vbslq` argument order vs `_mm256_blendv_ps`

The bit-select intrinsic `vbslq_f32(mask, a, b)` returns `a` where the
mask bits are 1 and `b` where they are 0. The x86 `_mm256_blendv_ps(a, b,
mask)` inverts both the mask convention *and* the operand order: it picks
`b` where the high bit of the mask lane is 1 and `a` otherwise.

| Backend | Intrinsic                           | "Picked when mask is true" |
|---------|--------------------------------------|----------------------------|
| AVX2    | `_mm256_blendv_ps(false_val, true_val, mask)` | second operand    |
| NEON    | `vbslq_f32(mask, true_val, false_val)`         | second operand    |

Both end up with `(true_val, false_val)` as the trailing pair, but the
mask is the **first** argument on NEON and the **last** argument on AVX2.
This is the single most common porting bug across the two backends; the
crate's NEON sources mark every blend explicitly:

```rust,ignore
// Select sin or cos kernel using vbslq (mask, true_val, false_val)
let kernel_result = vbslq_f64(use_cos, cos_y, sin_y);
```

### 2. FMA accumulator-first

The fused-multiply-add intrinsics on NEON take the **accumulator first**:

```rust,ignore
vfmaq_f64(c, a, b)   // computes a*b + c
vfmsq_f64(c, a, b)   // computes c - a*b
```

Compare with x86, where the accumulator is *last*:

```rust,ignore
_mm256_fmadd_pd(a, b, c)   // computes a*b + c
```

This affects every Horner-step transcription. The NEON sin kernel reads:

```rust,ignore
let r       = vfmaq_f64(s3, z, s4);          // s3 + z*s4
let inner   = vfmaq_f64(s1, z, s2);          // s1 + z*s2
let term1   = vfmaq_f64(x, s, inner);        // x + s*(s1 + z*s2)
vfmaq_f64(term1, sw, r)                      // ... + s*w*r
```

### 3. No `vmvnq_u64` — emulate with XOR

Bitwise-NOT on 64-bit integer lanes does not have a dedicated NEON
intrinsic. The standard idiom is XOR with all-ones:

```rust,ignore
let all_ones = vreinterpretq_u64_s64(vdupq_n_s64(-1));
let inv_mask = veorq_u64(mask, all_ones);
```

This shows up in the f64 sin kernel where the negation mask for quadrants
2 and 3 is built by sign-bit XOR rather than by inverting a comparison
mask:

```rust,ignore
let negated = vreinterpretq_f64_u64(veorq_u64(
    vreinterpretq_u64_f64(kernel_result),
    sign_bit_mask,
));
```

## Common intrinsics by role

| Role          | NEON                          | AVX2 equivalent       |
|---------------|-------------------------------|-----------------------|
| Load aligned  | `vld1q_f32`                   | `_mm256_load_ps`      |
| Store         | `vst1q_f32`                   | `_mm256_store_ps`     |
| Broadcast     | `vdupq_n_f32`                 | `_mm256_set1_ps`      |
| FMA           | `vfmaq_f32(c, a, b)`          | `_mm256_fmadd_ps(a, b, c)` |
| Compare LT    | `vcltq_f32`                   | `_mm256_cmp_ps(_, _, _CMP_LT_OQ)` |
| Blend         | `vbslq_f32(mask, t, f)`       | `_mm256_blendv_ps(f, t, mask)` |
| Sqrt          | `vsqrtq_f32`                  | `_mm256_sqrt_ps`      |
| Reciprocal    | `vdivq_f32`                   | `_mm256_div_ps`       |
| Bit AND/OR/XOR| `vandq_u32` / `vorrq_u32` / `veorq_u32` | `_mm256_and_si256` / `_or_` / `_xor_` |

## Tail handling

Unlike AVX-512's mask register or AVX2's pre-computed sign-bit mask
table, NEON has no native masked load/store at the 128-bit width. The
crate's tail strategy is a small element-wise scalar copy into a
register-sized buffer, then a normal `vld1q_*` load. This is invisible
to the calling code; see [`f32x4.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/f32x4.rs) and the parallel `f64x2.rs`.

## Where to look in the source

| Topic | File |
|-------|------|
| Register type, load/store, operators | [`neon/f32x4.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/f32x4.rs), [`neon/f64x2.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/f64x2.rs) |
| `VecMath` register impl    | [`neon/math.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/math.rs) |
| Trig kernels               | [`neon/sin.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/sin.rs), [`neon/cos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/cos.rs), [`neon/tan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/tan.rs) |
| Inverse trig kernels       | [`neon/asin.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/asin.rs), [`neon/acos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/acos.rs), [`neon/atan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/atan.rs) |

## See also

- [AVX2 backend](./avx2.md) — for the corresponding x86_64 lane-count and
  intrinsic family.
- [AVX-512 backend](./avx512.md) — wider x86_64 backend, mask-register
  predication.
- [Compile-time dispatch](./dispatch.md) — how aarch64 always picks NEON.
- [`Load`/`Store`/`Align` traits](./traits.md) — the abstract contract.
