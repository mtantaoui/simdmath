# Power `pow`

`pow(x, y)` computes \\(x^y\\) for arbitrary real exponent \\(y\\). It is by far
the most numerically delicate of the elementary functions implemented in
this crate: it composes [`ln`](./ln.md) and [`exp`](./exp.md), each
of which already has \\(\le 2\\) ULP error, but a naive composition would
amplify those errors by a factor of \\(|y|\\). Compensated (a.k.a. *double-double*)
arithmetic on the intermediate \\(\ln(|x|)\\) value is what keeps the final
result inside \\(\le 2\\) ULP.

## 1. Mathematical definition

For \\(x > 0\\) and \\(y \in \mathbb{R}\\),

\\[
x^y \\;=\\; \exp\\!\bigl(y \cdot \ln x\bigr).
\\]

For \\(x < 0\\), \\(x^y\\) is well defined only when \\(y\\) is an integer, in which
case the sign is \\((-1)^y\\). For \\(x < 0\\) and non-integer \\(y\\), the result is
\\(\text{NaN}\\). For \\(x = 0\\) the limit depends on the sign of \\(y\\) and the
parity of \\(y\\) when \\(y\\) is an integer; the IEEE 754 special-value table
in Sec. 3 enumerates every case.

## 2. Domain and range

| input set                          | output             |
|------------------------------------|--------------------|
| \\(x > 0\\), any \\(y\\)                   | finite or \\(0/\infty\\) depending on magnitude |
| \\(x < 0\\), \\(y\\) integer               | \\((-1)^y \cdot |x|^y\\) |
| \\(x < 0\\), \\(y\\) non-integer           | \\(\text{NaN}\\)       |
| \\(x = \pm 0\\)                        | per parity table   |
| any input contains NaN (except `1^anything` and `anything^0`) | \\(\text{NaN}\\) |

The mathematical range is \\([0, +\infty]\\) for \\(x \ge 0\\) and
\\((-\infty, +\infty)\\) for \\(x < 0\\).

## 3. Special values (IEEE 754-2008 Sec. 9.2.1)

The thirteen mandated cases, applied **in priority order** so that
overlapping rules are resolved deterministically:

| # | input                              | result   |
|---|------------------------------------|----------|
| 1 | `pow(x, ±0)`                       | \\(1\\) (any \\(x\\), even NaN) |
| 2 | `pow(1, y)`                        | \\(1\\) (any \\(y\\), even NaN) |
| 3 | `pow(-1, ±∞)`                      | \\(1\\) |
| 4 | `pow(x, y), x < 0, y` non-integer  | \\(\text{NaN}\\) |
| 5 | `pow(±0, y), y` odd integer \\(< 0\\)  | \\(\pm\infty\\) |
| 6 | `pow(±0, y), y < 0, y` not odd int | \\(+\infty\\) |
| 7 | `pow(±0, y), y` odd integer \\(> 0\\)  | \\(\pm 0\\) |
| 8 | `pow(±0, y), y > 0, y` not odd int | \\(+0\\) |
| 9 | `pow(-∞, y), y` odd integer \\(< 0\\)  | \\(-0\\) |
| 10| `pow(-∞, y), y < 0`, not odd int   | \\(+0\\) |
| 11| `pow(-∞, y), y` odd integer \\(> 0\\)  | \\(-\infty\\) |
| 12| `pow(-∞, y), y > 0`, not odd int   | \\(+\infty\\) |
| 13| `pow(±∞, y), y \gtrless 0`         | \\(\pm\infty\\) or \\(\pm 0\\) as appropriate |

The full table is encoded as a sequence of mask blends in
[`src/arch/avx2/pow.rs`][src-avx2] starting around line 440 (search for
`y_is_odd_int`).

[src-avx2]: https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/pow.rs

## 4. Algorithm overview

```text
                                compensated chain
   x ─►  ln_hi, ln_lo  ──┐
                         ├─►  TwoProd(y, ln_hi, ln_lo)  ──►  e_hi, e_lo
   y  ───────────────────┘                                       │
                                                                 ▼
                                          exp_compensated(e_hi + e_lo)
                                                                 │
                                                                 ▼
                                              sign correction (if x < 0)
                                                                 │
                                                                 ▼
                                                IEEE 754 special-case overrides
```

Five stages:

1. **Compensated logarithm.** `ln_hilo(|x|)` returns a hi/lo pair
   \\((\ell_\text{hi}, \ell_\text{lo})\\) such that
   \\(\ln|x| = \ell_\text{hi} + \ell_\text{lo}\\) to about 105 bits.
2. **Dekker product.** Compute \\(y \cdot (\ell_\text{hi} + \ell_\text{lo})\\)
   as \\((e_\text{hi}, e_\text{lo})\\) via FMA.
3. **Compensated exp.** Fold \\(e_\text{lo}\\) into the
   argument-reduction remainder so that the polynomial evaluates
   \\(\exp(r + e_\text{lo})\\), not just \\(\exp(r)\\).
4. **Sign correction.** If \\(x < 0\\) and \\(y\\) is an odd integer, negate.
5. **Special-case overlay.** Mask-blend the 13 IEEE 754 cases.

## 5. Argument reduction (input decomposition)

Pow has *two* inputs to inspect:

- **Magnitude of \\(x\\)**: the work piece for `ln`. Subnormal \\(x\\) is
  scaled by \\(2^{52}\\) before the IEEE 754 exponent extraction; the
  scaling is later undone in \\(k\\).
- **Integer-ness and parity of \\(y\\)**: classify \\(y\\) as
  `not-integer / even-integer / odd-integer` to drive the special-case
  table. Done with `_mm256_round_pd::<{ TO_NEAREST }>` and a comparison
  against the original \\(y\\):

  ```rust,ignore
  let y_trunc       = _mm256_round_pd::<TRUNC>(y);
  let y_is_integer  = _mm256_cmp_pd(y, y_trunc, _CMP_EQ_OQ);
  let y_half        = _mm256_mul_pd(y, _mm256_set1_pd(0.5));
  let y_half_is_int = _mm256_cmp_pd(y_half, _mm256_round_pd::<TRUNC>(y_half), _CMP_EQ_OQ);
  let y_is_odd_int  = _mm256_andnot_pd(y_half_is_int, y_is_integer);
  ```

The `not-odd-integer` mask (relevant for several rows of the special-value
table) is `andnot(y_is_odd_int, y_is_integer)` ∪ `not y_is_integer`.

## 6. Polynomial / kernel approximation

`pow` does not add a polynomial of its own — it composes the polynomials
of `ln` and `exp` (see [Sec. 6 of the `ln` chapter](./ln.md#6-polynomial--kernel-approximation)
and [Sec. 6 of the `exp` chapter](./exp.md#6-polynomial--kernel-approximation)).

The crucial detail is that the `ln` polynomial is evaluated in *hi/lo*
form. Where the standalone `ln` returns

\\[
\widehat{\ln x} \\;=\\; k\cdot\mathtt{LN2\\_HI} - ((\mathrm{hfsq} - (s(\mathrm{hfsq}+R)+k\cdot\mathtt{LN2\\_LO})) - f),
\\]

`ln_hilo` instead splits the reconstruction with Knuth's 2Sum so that
the rounding residual of the final subtraction is captured in the lo
component:

```rust,ignore
let core = /* the bracketed quantity above */;
let hi   = _mm256_sub_pd(k_ln2hi, core);
let err  = _mm256_sub_pd(_mm256_sub_pd(k_ln2hi, hi), core); // 2Sum residual
let lo   = err;                                              // up to ~15 extra bits
```

The same `LN2_HI / LN2_LO` and `LG1..LG7` constants from
[`src/arch/consts/ln.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/ln.rs)
are reused, and so are the `P1..P5` polynomial and `LN2_INV` reciprocal
from [`src/arch/consts/exp.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/exp.rs):

```rust,ignore
use crate::arch::consts::exp::{
    LN2_HI_64 as EXP_LN2_HI, LN2_INV_64, LN2_LO_64 as EXP_LN2_LO,
    OVERFLOW_THRESH_64, P1_64, P2_64, P3_64, P4_64, P5_64, UNDERFLOW_THRESH_64,
};
use crate::arch::consts::ln::{
    LG1_64, LG2_64, LG3_64, LG4_64, LG5_64, LG6_64, LG7_64,
    LN2_HI_64, LN2_LO_64, SQRT2_64, TWO52_64,
};
```

## 7. Reconstruction

**Dekker product.** Given \\((\ell_\text{hi}, \ell_\text{lo})\\) with
\\(\ell_\text{lo} \approx 0\\),

\\[
y \cdot (\ell_\text{hi} + \ell_\text{lo})
\\;=\\;
\underbrace{y\\,\ell_\text{hi}}\_{e_\text{hi}^\circ}
\\;+\\;
\underbrace{\mathrm{fma}(y, \ell_\text{hi}, -e_\text{hi}^\circ) + y\\,\ell_\text{lo}}\_{e_\text{lo}}.
\\]

The `fma` term is the *exact* rounding error of the first product, so
\\(e_\text{hi}^\circ + e_\text{lo} = y\\,\ell_\text{hi} + y\\,\ell_\text{lo}\\) to
full double-double precision.

**Compensated exp.** Run the standard `exp` reduction
\\(x = k\ln 2 + r\\) on \\(e_\text{hi}^\circ\\), then *replace* \\(r\\) by
\\(r + e_\text{lo}\\) before evaluating the polynomial. This effectively
asks the polynomial to evaluate \\(\exp(r + e_\text{lo})\\) at the cost of
one extra add — and because \\(|e_\text{lo}|\\) is tiny, the polynomial
neighbourhood is unchanged.

**Sign correction.** For negative bases:

```rust,ignore
let should_negate = _mm256_and_pd(x_is_neg, y_is_odd_int);
let result        = _mm256_blendv_pd(result, _mm256_xor_pd(result, sign_bit), should_negate);
```

**Overflow / underflow.** Inherited from `exp`:
\\(y\ln|x| > \\) `OVERFLOW_THRESH_64` \\(\Rightarrow +\infty\\), etc.

## 8. Per-precision differences (f32 vs f64)

| aspect              | f32                                       | f64        |
|---------------------|-------------------------------------------|------------|
| internal precision  | f64 (8-lane f32 split into 2× f64 halves) | native f64 |
| ln/exp constants    | shared with [`ln`](./ln.md) / [`exp`](./exp.md) | shared |
| compensated arithmetic | always (Dekker product + 2Sum split)   | always     |
| ULP                 | \\(\le 2\\)                                   | \\(\le 2\\)    |

f32 promotion is essential here: a pure-f32 chain through `ln_f32` then
`y * ln_f32` then `exp_f32` would amplify the 2 ULP error of each step
by \\(|y|\\), giving wildly inaccurate results for moderate-magnitude
exponents. Promoting to f64 buys 29 extra mantissa bits of headroom — far
more than \\(|y|\\) can spend.

## 9. Per-backend differences (AVX2 / AVX-512 / NEON)

| backend  | f32 lanes | f64 lanes | special-case mask machinery |
|----------|-----------|-----------|------------------------------|
| AVX2     | 8 (via 2× f64) | 4    | full-vector masks + `_mm256_blendv_pd` |
| AVX-512  | 16 (via 2× f64) | 8   | `__mmask8` integer masks + `_mm512_mask_blend_pd` |
| NEON     | 4 (via 2× f64)  | 2   | `vbslq_f64`, no-`vmvnq_u64` workaround |

All three backends produce bitwise-identical results given the same
input — they differ only in how the special-case priority pyramid is
laid out (vector blend vs. integer mask AND/OR).

## 10. Error analysis

Decomposing the chain:

\\[
\mathrm{err}(\widehat{\mathrm{pow}}(x, y))
\\;\le\\;
|y|\cdot \mathrm{err}(\widehat{\ln}|x|)
\\;+\\;
\mathrm{err}\_{\text{Dekker}}
\\;+\\;
\mathrm{err}(\widehat{\exp}(\\,\cdot\\,)).
\\]

Without compensation, the first term alone is
\\(|y| \cdot 2\\,\mathrm{ulp}\\) — for \\(y = 100\\), that is already 200 ULP.
The hi/lo split *does not change* the value of \\(\widehat{\ln}|x|\\), but
it *splits* its error into two parts that the Dekker product can
preserve separately:

\\[
y \cdot \widehat{\ln}|x|
\\;=\\; y(\ell_\text{hi} + \ell_\text{lo} + \delta_\ln)
\\;=\\; (e_\text{hi} + e_\text{lo}) + y\delta_\ln.
\\]

Now only the **secondary** error \\(y\delta_\ln\\) amplifies; \\(\delta_\ln\\) is
the *post-compensation* residual, which is ~\\(2^{-105}\\) rather than
\\(2^{-52}\\). For \\(|y| < 2^{40}\\) this stays well under one ULP.

**Worst-case observed.** \\(\le 2\\) ULP for both f32 and f64 across the
representative test sweep; the ULP peaks tend to occur near
\\(y\ln|x| \approx \pm 700\\), where the `exp` reconstruction is closest to
overflow.

## 11. Code excerpt

From [`src/arch/avx2/pow.rs`][src-avx2] — Dekker product + compensated
exp prologue:

```rust,ignore
// Stage 1: ln(|x|) as (hi, lo)
let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
let (ln_hi, ln_lo) = ln_hilo(abs_x);

// Stage 2: Dekker product  e_hi + e_lo = y * (ln_hi + ln_lo)
let e_hi    = _mm256_mul_pd(y, ln_hi);
let e_lo    = _mm256_fmadd_pd(y, ln_hi, _mm256_sub_pd(_mm256_setzero_pd(), e_hi));
let e_lo    = _mm256_fmadd_pd(y, ln_lo, e_lo);

// Stage 3: range-reduce e_hi by ln(2), then fold e_lo into r
let k_f64   = _mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(
              _mm256_fmadd_pd(e_hi, _mm256_set1_pd(LN2_INV_64),
                              _mm256_or_pd(_mm256_set1_pd(0.5),
                                           _mm256_and_pd(e_hi, _mm256_set1_pd(-0.0)))));
let r       = _mm256_fnmadd_pd(k_f64, _mm256_set1_pd(EXP_LN2_LO),
              _mm256_fnmadd_pd(k_f64, _mm256_set1_pd(EXP_LN2_HI), e_hi));
let r       = _mm256_add_pd(r, e_lo); // *** the compensation ***

// Stage 4: standard exp polynomial on r, then 2^k scaling (as in exp chapter)
```

The full file is just over 1100 lines; the bulk of it is the special-case
mask construction.

## 12. References

- musl libc: [`src/math/pow.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/pow.c) and [`src/math/powf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/powf.c).
- IEEE 754-2008, §9.2.1, table of *recommendedFunctions* — the special-value table reproduced in §3.
- ISO/IEC 9899:2018 (C18), §F.10.4.4 — same table for `pow`.
- Dekker, T. J., *A floating-point technique for extending the available precision*, Numer. Math. 18, 1971 — original double-double product.
- Markstein, *IA-64 and Elementary Functions*, Prentice-Hall 2000, chapter 11 — compensated `pow` derivation.
- Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed., §11.5.
- Repo source: [`src/arch/avx2/pow.rs`][src-avx2], [`src/arch/avx512/pow.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx512/pow.rs), [`src/arch/neon/pow.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/pow.rs).

## See also

- [Natural logarithm `ln`](./ln.md) — compensated front-end.
- [Natural exponential `exp`](./exp.md) — compensated back-end.
- [Compensated arithmetic: two-sum and Dekker product](../foundations/compensated.md)
  — the algebraic identities behind the `(hi, lo)` representation.
