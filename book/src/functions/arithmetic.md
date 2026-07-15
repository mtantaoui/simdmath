# Vectorised arithmetic and reductions

This chapter covers the *non-transcendental* part of the public API:
the [`VecExt`] trait, which extends `Vec<T>` (for `T = f32, f64`) with
element-wise arithmetic, scalar broadcasts, in-place updates, and the
four reductions `sum` / `product` / `min` / `max`. It is the entry point
for users who want SIMD-accelerated array math without touching the
math kernels.

[`VecExt`]: https://github.com/mtantaoui/simdmath/blob/main/src/ops/vec.rs

## 1. Mathematical definition

Given two equal-length arrays \\(a, b \in \mathbb{R}^n\\) and a scalar
\\(s \in \mathbb{R}\\):

\\[
\begin{aligned}
(a + b)_i &= a_i + b_i, & (a - b)_i &= a_i - b_i, \\\\
(a \cdot b)_i &= a_i b_i, & (a / b)_i &= a_i / b_i, \\\\
(a \% b)_i &= a_i \bmod b_i \quad &\text{(Rust scalar semantics)} & \\\\
(a + s)_i &= a_i + s, & (a \cdot s)_i &= a_i s, \\\\
\mathrm{sum}(a) &= \sum_i a_i, & \mathrm{product}(a) &= \prod_i a_i, \\\\
\min(a) &= \min_i a_i, & \max(a) &= \max_i a_i.
\end{aligned}
\\]

The `%` operator follows Rust's `f32`/`f64` semantics: the result has
the sign of the dividend, e.g. \\(-7 \% 3 = -1\\) (not \\(2\\)).

## 2. Domain and range

| operation | domain                          | range / notes                        |
|-----------|---------------------------------|--------------------------------------|
| `add` / `sub` / `mul` | any pair of finite or infinite floats | follows IEEE 754 |
| `div`     | divisor may be zero             | `±0` divisor → `±∞` (sign-preserved) |
| `rem`     | divisor must be non-zero        | `0` divisor → `NaN`                  |
| `*_scalar`| any scalar                      | broadcast then apply                 |
| `sum`     | any array (incl. NaN, ±∞)       | NaN propagates, additive identity is \\(0\\) |
| `product` | any array                       | NaN propagates, multiplicative identity is \\(1\\) |
| `min` / `max` | any array                   | empty input → \\(+\infty\\) / \\(-\infty\\) |

NaN handling for `min`/`max` follows the IEEE 754 *minNum*/*maxNum*
behaviour: a single NaN is ignored unless the entire array is NaN.

## 3. Special values (IEEE 754 / Rust scalar semantics)

The element-wise operations are exactly the corresponding scalar
operations applied lane-by-lane. The IEEE 754 contract for \\(+, -, \times,
\div\\) — correctly-rounded result, NaN propagation, sign-preserving zero
— is preserved by the SIMD versions because the underlying instructions
(`vaddps`, `vmulps`, `vdivps`, etc.) are themselves correctly-rounded.

| input                        | output      |
|------------------------------|-------------|
| `add( x, NaN)`               | NaN         |
| `add(+∞, -∞)` (sub)          | NaN         |
| `mul( 0, ±∞)`                | NaN         |
| `div( x,  0)`, \\(x \neq 0\\)    | \\(\pm\infty\\) |
| `div( 0,  0)`                | NaN         |
| `rem( x,  0)`                | NaN         |
| empty array `.sum()`         | \\(+0\\)        |
| empty array `.product()`     | \\(+1\\)        |
| empty array `.min()`         | \\(+\infty\\)   |
| empty array `.max()`         | \\(-\infty\\)   |

## 4. Algorithm overview

There is no algorithm in the elementary-function sense. Each method:

1. Pads or chunks the input slice to multiples of the lane count.
2. Loads each chunk into a SIMD register.
3. Applies the corresponding native instruction
   (`_mm256_add_ps`, `vaddq_f32`, `_mm512_max_pd`, …).
4. For reductions, accumulates a running register, then folds it at the
   end by storing the lanes to a small stack array and summing scalar-wise.
5. Stores back to a `Vec<T>` (for elementwise) or returns the scalar
   (for reductions).

The chunk-then-tail pattern is implemented once in
[`src/ops/vec.rs`][src-ops-vec] and shared by every backend through the
generic helpers `binary_op`, `scalar_op`, `binary_op_inplace`,
`scalar_op_inplace`, and `unary_op`.

[src-ops-vec]: https://github.com/mtantaoui/simdmath/blob/main/src/ops/vec.rs

## 5. Argument reduction (chunking and tail handling)

Given a slice of length \\(n\\) and a backend lane count \\(L\\):

```text
   ┌──────────┬──────────┬──────────┬──────────┬─────────┐
   │  chunk 0 │  chunk 1 │  chunk 2 │   ...    │  tail   │
   └──────────┴──────────┴──────────┴──────────┴─────────┘
   ◄─── L lanes each (full SIMD load/store) ──►◄ < L  lanes ►
   full_chunks = n / L                          tail = n % L
```

The full chunks are processed by the unrolled fast path:

```rust,ignore
for i in 0..full_chunks {
    let offset = i * L;
    let a = unsafe { S::load(lhs.as_ptr().add(offset), L) };
    let b = unsafe { S::load(rhs.as_ptr().add(offset), L) };
    let result = op(a, b);
    unsafe { result.store_at(out.as_mut_ptr().add(offset)) };
}
```

The trailing partial chunk uses *partial* loads that zero the inactive
lanes:

```rust,ignore
if tail > 0 {
    let offset = full_chunks * L;
    let a = unsafe { S::load_partial(lhs.as_ptr().add(offset), tail) };
    let b = unsafe { S::load_partial(rhs.as_ptr().add(offset), tail) };
    let result = op(a, b);
    unsafe { result.store_at_partial(out.as_mut_ptr().add(offset)) };
}
```

`load_partial` and `store_at_partial` are part of the [`Load`] / [`Store`]
traits in [`src/ops/simd.rs`][src-ops-simd]; each backend implements
them via masked loads (AVX-512), masked moves through scratch buffers
(AVX2), or scalar fall-back loops (NEON for f64).

[`Load`]: https://github.com/mtantaoui/simdmath/blob/main/src/ops/simd.rs
[`Store`]: https://github.com/mtantaoui/simdmath/blob/main/src/ops/simd.rs
[src-ops-simd]: https://github.com/mtantaoui/simdmath/blob/main/src/ops/simd.rs

The crucial bit is that the *result* register inherits its `size` from
the loaded vector, so `store_at_partial` writes exactly `tail`
elements — never more, never less. This eliminates buffer overruns
without requiring the caller to pad inputs.

## 6. Polynomial / kernel approximation

There is none. The operations are basic-arithmetic native instructions:
`vaddps`, `vsubps`, `vmulps`, `vdivps`, `vminps`, `vmaxps`, plus the f64
counterparts and the NEON / AVX-512 variants. Each is correctly-rounded
by IEEE 754.

## 7. Reconstruction

**Element-wise.** No reconstruction — the SIMD register is stored
directly to the output `Vec`.

**Reductions** combine partial accumulators in lane-parallel form, then
do a *horizontal* reduction at the end:

```text
   acc = identity_register
   for chunk in chunks:
       acc = op(acc, chunk)        // 8 / 4 / 16 / 8 / 4 / 2 partial sums
   return horizontal_reduce(acc)   // 1 scalar
```

The horizontal step is deliberately *not* done with shuffle-based
intrinsics (`hadd`, `_mm512_reduce_*`, `vaddvq_*`): every backend stores
the accumulator register to a stack array and folds it scalar-wise
(`arr.iter().copied().sum()`), which the compiler lowers efficiently and
which keeps the three implementations line-for-line identical.

`min` / `max` use the corresponding vertical `min*` / `max*` instructions
for the accumulation (`_mm256_min_ps`, `vminq_f32`, …), with the same
store-and-fold pattern for the final horizontal step.

## 8. Per-precision differences (f32 vs f64)

| operation       | f32                                  | f64       |
|-----------------|--------------------------------------|-----------|
| add/sub/mul/div | correctly rounded (IEEE 754)         | same      |
| sum / product   | up to \\(n - 1\\) ULP cumulative drift   | same      |
| min / max       | exact                                | exact     |
| ULP per element | 0 (basic ops are correctly rounded)  | 0         |

The accuracy contract for *single-element* operations is identical at
both precisions — they are all hardware-correct. Reductions are where
precision diverges between the precisions, *not* because of any
algorithmic difference but because f32 has fewer mantissa bits in which
to absorb accumulation noise.

## 9. Per-backend differences (AVX2 / AVX-512 / NEON)

| backend   | f32 lanes | f64 lanes | tail mechanism            |
|-----------|-----------|-----------|---------------------------|
| AVX2      | 8         | 4         | `_mm256_maskload_ps` (or scratch buffer) |
| AVX-512   | 16        | 8         | `__mmask16` / `__mmask8`  |
| NEON      | 4         | 2         | scalar fall-back for tail |

AVX-512's masked loads make the chunked-plus-tail loop literally
*one* loop body — the mask register encodes "all lanes" for full chunks
and "first \\(k\\) lanes" for the tail. AVX2 has masked loads for f32 and
f64 from AVX-512VL onwards but the AVX2-only path uses an aligned
scratch buffer. NEON has no masked load on AArch64 prior to SVE, so the
tail elements are loaded one by one.

The per-backend code lives in:

- AVX2:   [`src/arch/avx2/vec.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/vec.rs)
- AVX-512:[`src/arch/avx512/vec.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx512/vec.rs)
- NEON:   [`src/arch/neon/vec.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/vec.rs)

In each one the arithmetic lives in `impl SliceExt<f32> for [f32]` /
`impl SliceExt<f64> for [f64]`, which wires the backend's lane count
through the generic helpers in [`src/ops/vec.rs`][src-ops-vec]; the
`VecExt` impls are thin delegators (generated by `impl_vecext_delegate!`)
that forward to the slice impls.

## 10. Error analysis

**Element-wise operations.** Each output is correctly rounded
(\\(\le 0.5\\) ULP), inherited directly from the underlying IEEE 754
instructions. No ULP claim needs to be tracked beyond what the hardware
already provides.

**Reductions** are *not* correctly rounded. With a left-fold
accumulator,

\\[
\mathrm{err}(\widehat{\mathrm{sum}}(a))
\\;\le\\;
(n - 1)\\,\varepsilon \cdot \sum_i |a_i|,
\\]

where \\(\varepsilon = 2^{-24}\\) for f32 or \\(2^{-53}\\) for f64. SIMD lane
parallelism *helps a constant factor*: an \\(L\\)-wide accumulator shortens
each element's chain of rounded additions from \\(n\\) to
\\(\lceil n/L \rceil + \log_2 L\\) (the final horizontal combine adds the
\\(\log_2 L\\) term), so the worst-case bound tightens to roughly
\\((n/L + \log_2 L)\\,\varepsilon \sum |a_i|\\). Note what this is and isn't:
it is a shorter *dependency depth*, not compensation — no rounding error is
ever recovered, the bound still grows linearly in \\(n\\), and for large
\\(n\\) the \\(L\\)-fold improvement is a constant factor, not a change in
behaviour.

**Future work.** Kahan compensated summation reduces the bound to
\\(O(\varepsilon)\sum|a_i|\\) — independent of \\(n\\) — at the cost of two
extra adds per element. A Neumaier or Kahan-Babuška variant would be
the natural future enhancement; see Higham, *Accuracy and Stability of
Numerical Algorithms*, §4.3 for a detailed comparison.

For now, applications that need bit-stable reductions over millions of
elements should call `.sum()` on subblocks and combine the partial sums
themselves, or wait for a future `kahan_sum` API.

## 11. Code excerpt

The single source of truth for the chunk-and-tail loop, from
[`src/ops/vec.rs`][src-ops-vec]:

```rust,ignore
pub(crate) fn binary_op<T, S>(
    lhs: &[T], rhs: &[T], lane_count: usize,
    op: impl Fn(S, S) -> S,
) -> Vec<T>
where
    T: Copy,
    S: Load<T, Output = S> + Store<T> + Copy,
{
    let n = lhs.len();
    let full_chunks = n / lane_count;
    let tail = n % lane_count;

    let mut out: Vec<T> = Vec::with_capacity(n);

    for i in 0..full_chunks {
        let offset = i * lane_count;
        let a = unsafe { S::load(lhs.as_ptr().add(offset), lane_count) };
        let b = unsafe { S::load(rhs.as_ptr().add(offset), lane_count) };
        let result = op(a, b);
        unsafe { result.store_at(out.as_mut_ptr().add(offset)) };
    }

    if tail > 0 {
        let offset = full_chunks * lane_count;
        let a = unsafe { S::load_partial(lhs.as_ptr().add(offset), tail) };
        let b = unsafe { S::load_partial(rhs.as_ptr().add(offset), tail) };
        let result = op(a, b);
        unsafe { result.store_at_partial(out.as_mut_ptr().add(offset)) };
    }

    unsafe { out.set_len(n) };
    out
}
```

A backend wires this up by passing its own SIMD vector type as `S` and
its own lane count, e.g. for AVX2 f32:

```rust,ignore
impl SliceExt<f32> for [f32] {
    fn add(&self, rhs: &[f32]) -> Vec<f32> {
        assert_eq!(self.len(), rhs.len());
        binary_op::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a + b)
    }
    // ...
}
// `VecExt` then simply forwards: self.as_slice().add(rhs)
```

## 12. References

- IEEE 754-2008, §5 (basic-arithmetic correctness contract).
- Higham, N. J., *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, chapter 4 — error analysis of summation and products, including Kahan compensated summation.
- Goldberg, *What every computer scientist should know about floating-point arithmetic*, ACM Comp. Surveys 23(1), 1991, §3 — basics of accumulator drift.
- Intel Intrinsics Guide — entries for `_mm256_*_ps` / `_mm256_*_pd` / `_mm512_reduce_*`.
- ARM Neon Intrinsics Reference — `vaddq_*`, `vmulq_*`, `vminq_*`, `vmaxq_*`, `vaddvq_*`.
- Repo source: [`src/ops/vec.rs`][src-ops-vec], [`src/ops/simd.rs`][src-ops-simd], and the per-arch `vec.rs` files linked above.

## See also

- [The `Load` / `Store` / `Align` traits](../backends/traits.md) — the
  abstract contract that backends implement so the chunk-and-tail loop
  can be generic.
- [AVX2](../backends/avx2.md), [AVX-512](../backends/avx512.md),
  [NEON](../backends/neon.md) — backend chapters describing the lane
  counts and intrinsics referenced here.
- [Compile-time dispatch](../backends/dispatch.md) — how the right
  `impl VecExt` is selected for the build target.
