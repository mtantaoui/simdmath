# Square root `sqrt`

`sqrt(x) = ` \\(\sqrt{x}\\) is the only function in this crate whose
implementation is *literally one instruction per backend*. IEEE 754
mandates that square root be **correctly rounded** — the unique
representable value closest to the exact mathematical result — and
every modern CPU exposes a hardware instruction that delivers exactly
that. There is therefore no algorithm to derive, no polynomial to tune,
no argument reduction to perform.

## 1. Mathematical definition

For \\(x \ge 0\\),

\\[
\sqrt{x} \\;=\\; \text{the unique } y \ge 0 \text{ with } y^2 = x.
\\]

For \\(x < 0\\), \\(\sqrt{x}\\) is not real and the IEEE 754 specification
returns \\(\text{NaN}\\). The function is the inverse of \\(y \mapsto y^2\\)
restricted to \\([0, \infty)\\).

## 2. Domain and range

| input            | output         |
|------------------|----------------|
| \\(x \ge 0\\)        | \\(\sqrt{x} \in [0, +\infty)\\) |
| \\(-0\\)             | \\(-0\\)           |
| \\(x < 0\\)          | \\(\text{NaN}\\) (and signals invalid-operation) |
| \\(+\infty\\)        | \\(+\infty\\)      |
| \\(\text{NaN}\\)     | \\(\text{NaN}\\)   |

The mathematical range is \\([0, +\infty)\\). Note that \\(\sqrt{-0} = -0\\) is
preserved as a *signed zero* by all listed instructions, in line with
the IEEE 754 sign-preservation rule.

## 3. Special values (IEEE 754 Sec. 5.4.1)

| input         | output       | flag         |
|---------------|--------------|--------------|
| \\(+0\\)          | \\(+0\\)         | none         |
| \\(-0\\)          | \\(-0\\)         | none         |
| \\(+\infty\\)     | \\(+\infty\\)    | none         |
| \\(-\infty\\)     | \\(\text{NaN}\\) | invalid      |
| any \\(x < 0\\)   | \\(\text{NaN}\\) | invalid      |
| \\(\text{NaN}\\)  | \\(\text{NaN}\\) | (q→q, s→invalid) |

These behaviours are produced *by the hardware itself* — the SIMD
implementations contain no special-case code.

## 4. Algorithm overview

`sqrt` is one of the five **basic operations** of IEEE 754 (alongside
\\(+, -, \times, \div\\)) for which the standard requires *correct
rounding*. Every IEEE 754 conforming processor therefore provides a
single dedicated instruction, fully pipelined and producing the same
bit-for-bit result regardless of microarchitecture:

| ISA           | f32 instruction      | f64 instruction      |
|---------------|----------------------|----------------------|
| x86 SSE2      | `sqrtps` / `sqrtss`  | `sqrtpd` / `sqrtsd`  |
| x86 AVX/AVX2  | `vsqrtps` (256-bit)  | `vsqrtpd` (256-bit)  |
| x86 AVX-512   | `vsqrtps` (512-bit)  | `vsqrtpd` (512-bit)  |
| AArch64 NEON  | `fsqrt` / `vsqrtq.f32` | `fsqrt` / `vsqrtq.f64` |

The Rust intrinsics that wrap these instructions are exactly what the
backend `sqrt` methods call:

| backend  | f32 intrinsic    | f64 intrinsic    |
|----------|------------------|------------------|
| AVX2     | `_mm256_sqrt_ps` | `_mm256_sqrt_pd` |
| AVX-512  | `_mm512_sqrt_ps` | `_mm512_sqrt_pd` |
| NEON     | `vsqrtq_f32`     | `vsqrtq_f64`     |

## 5. Argument reduction

There is none. The hardware operates directly on the IEEE 754 bit
representation. Internally each implementation uses something like a
SRT or Newton-Raphson iterator with a final round-to-nearest-even
adjustment, but that is a CPU-internal concern; from software's point
of view `sqrt` is atomic.

## 6. Polynomial / kernel approximation

There is none. The hardware delivers the correctly-rounded result.

If a software fallback were ever needed (e.g. for an exotic architecture
without hardware sqrt) the standard recipe is:

1. Decompose \\(x = 2^{2k} m\\) with \\(m \in [1, 4)\\).
2. Approximate \\(1/\sqrt{m}\\) via a low-degree polynomial seed
   (sometimes a lookup table).
3. Refine with Newton's iteration \\(y\_{n+1} = \tfrac{1}{2} y_n (3 - x y_n^2)\\)
   for the reciprocal sqrt, or one-step Heron \\(y\_{n+1} = \tfrac12(y_n + x/y_n)\\).
4. Apply a final Tuckerman-style sticky-bit correction to nail the round.

This crate does **not** implement any of that; it would only slow down
the SIMD loops while delivering identical results.

## 7. Reconstruction

There is none. The output of the hardware instruction *is* the final
result.

The only thing the wrapper does is forward the register through the
crate's `F32x8` / `F64x4` / `F32x16` / `F64x8` / `F32x4` / `F64x2`
struct so that it composes uniformly with the rest of the math API.

## 8. Per-precision differences (f32 vs f64)

| aspect             | f32        | f64        |
|--------------------|------------|------------|
| latency (Skylake)  | 12 cycles  | 18 cycles  |
| throughput (Skylake)| 1/3 cycles| 1/6 cycles |
| ULP                | \\(\le 0.5\\) (correctly rounded) | \\(\le 0.5\\) |

Both precisions deliver the same correctness contract. The only
difference is throughput: the f64 version processes half as many lanes
(per register width) and the underlying FP unit takes longer because the
mantissa is twice as wide.

## 9. Per-backend differences (AVX2 / AVX-512 / NEON)

| backend  | f32 lanes | f64 lanes | intrinsic                        |
|----------|-----------|-----------|----------------------------------|
| AVX2     | 8         | 4         | `_mm256_sqrt_ps` / `_mm256_sqrt_pd` |
| AVX-512  | 16        | 8         | `_mm512_sqrt_ps` / `_mm512_sqrt_pd` |
| NEON     | 4         | 2         | `vsqrtq_f32` / `vsqrtq_f64`         |

Per-cycle throughput scales with the lane count, so AVX-512 nominally
processes twice as many lanes per cycle as AVX2 — though in practice
the actual speed-up is reduced by the AVX-512 frequency-throttling on
older Intel parts. NEON's `vsqrtq_*` is fully pipelined on Apple Silicon
and on most ARM Cortex-A7x cores; on smaller in-order cores it may
serialise.

## 10. Error analysis

By IEEE 754 construction:

\\[
\widehat{\sqrt{x}} \\;=\\; \mathrm{round}\_{\mathrm{N}}(\sqrt{x}),
\\]

where \\(\mathrm{round}\_{\mathrm{N}}\\) is round-to-nearest-even on the
target precision. The error is therefore at most half a ULP of the
*result*:

\\[
|\widehat{\sqrt{x}} - \sqrt{x}| \\;\le\\; \tfrac{1}{2}\\,\mathrm{ulp}(\sqrt{x}).
\\]

This is the gold standard — strictly tighter than the \\(\le 1\\) ULP
contract of all the polynomial-based functions in this crate. The
`src/lib.rs` ULP table accordingly lists `sqrt` as **≤ 0.5 ULP
(correctly rounded)**.

There is no input-dependent worst case; every output is correctly
rounded by hardware mandate.

## 11. Code excerpt

From [`src/arch/avx2/math.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/math.rs):

```rust,ignore
/// Square root of every lane via `vsqrtps`.
#[inline(always)]
fn sqrt(&self) -> F32x8 {
    F32x8 {
        size: self.size,
        elements: unsafe { _mm256_sqrt_ps(self.elements) },
    }
}
```

From [`src/arch/neon/math.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/math.rs):

```rust,ignore
/// Square root of every lane via `vsqrtq_f32`.
#[inline(always)]
fn sqrt(&self) -> F32x4 {
    F32x4 {
        size: self.size,
        elements: unsafe { vsqrtq_f32(self.elements) },
    }
}
```

The f64 counterparts are identical modulo the suffix and lane count.

## 12. References

- IEEE 754-2008, §5.4.1 (*formatOf operations*) — mandates correctly-rounded `squareRoot` for every supported format.
- Intel® 64 and IA-32 Architectures Software Developer's Manual, vol. 2: descriptions of `sqrtps`, `sqrtpd`, `vsqrtps`, `vsqrtpd`.
- ARM Architecture Reference Manual: `FSQRT (vector)` for AArch64.
- Markstein, *IA-64 and Elementary Functions*, chapter 9: how a hardware sqrt is actually implemented (SRT vs. Newton).
- Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed., §3.5 — discussion of correctly-rounded basic operations.
- Repo source: [`src/arch/avx2/math.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/math.rs), [`src/arch/avx512/math.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx512/math.rs), [`src/arch/neon/math.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/math.rs).

## See also

- [Cube root `cbrt`](./cbrt.md) — by contrast, lacks a hardware
  instruction and so requires a full bit-trick + Newton derivation.
- [Arc sine `asin`](./asin.md) and [Arc cosine `acos`](./acos.md) — both
  consume `sqrt` as a sub-step in their argument reduction.
- [ULP, faithful rounding, correct rounding](../foundations/ulp.md) for
  the formal definition of "correctly rounded".
