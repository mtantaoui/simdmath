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
[`src/arch/avx2/math/pow.rs`][src-avx2] (search for `y_is_odd_int`).

[src-avx2]: https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/math/pow.rs

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

That diagram describes the **f64 path**. The **f32 path** promotes to f64
and then deliberately *skips* the compensation machinery: the raw chain is
selected by a const generic, `pow_raw_f64::<PRECISE>`, and the f32 wrapper
instantiates `PRECISE = false`, which

- uses the plain single-`f64` logarithm (no 2Sum split — `ln_hilo` returns
  `(hi, 0)`),
- multiplies \\(y \cdot \ell_\text{hi}\\) with one ordinary `mul` (no Dekker
  product),
- evaluates `exp` with a division-free degree-10 Taylor polynomial instead
  of the fdlibm Padé form, and
- skips the subnormal pre-scaling (a promoted f32 is never f64-subnormal).

This is sound because the f32 error budget is enormous in f64 terms: the
absolute error of \\(y\ln|x|\\) computed in plain f64 is
\\(\approx |y\ln|x|| \cdot 2^{-52} \le 89 \cdot 2^{-52}\\) (finite f32
results require \\(|y\ln|x|| \lesssim 89\\)), orders of magnitude below half
an f32 ULP (\\(2^{-24}\\) relative). The IEEE special-case overlay (stages
4–5) then runs **once on the full-width f32 vectors**
(`pow_special_cases_ps`) instead of twice on the promoted f64 halves —
the cascade is pure mask logic, so halving its width halves its cost.

## 5. Argument reduction (input decomposition)

Pow has *two* inputs to inspect:

- **Magnitude of \\(x\\)**: the work piece for `ln`. Subnormal \\(x\\) is
  scaled by \\(2^{52}\\) before the IEEE 754 exponent extraction; the
  scaling is later undone in \\(k\\). (f64 path only — the f32 path skips
  this entirely, since any f32 value promoted to f64 is normal.)
- **Integer-ness and parity of \\(y\\)**: classify \\(y\\) as
  `not-integer / even-integer / odd-integer` to drive the special-case
  table. Done with truncation (`_mm256_round_pd::<TRUNC>`) and a comparison
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

The crucial detail — **on the f64 path** — is that the `ln` polynomial is
evaluated in *hi/lo* form. (The f32 path uses the same polynomial but keeps
only the plain `hi` reconstruction, and swaps the divided Padé `exp` form
for a division-free Taylor polynomial; see §4 and §8.) Where the standalone
`ln` returns

\\[
\widehat{\ln x} \\;=\\; k\cdot\mathtt{LN2\\_HI} - ((\mathrm{hfsq} - (s(\mathrm{hfsq}+R)+k\cdot\mathtt{LN2\\_LO})) - f),
\\]

`ln_hilo` instead splits the reconstruction with Knuth's 2Sum so that
the rounding residual of the final subtraction is captured in the lo
component:

```rust,ignore
// val_hi = f - hfsq (may round); val_lo recovers its rounding error and
// accumulates the small terms s*(hfsq+R) + k*ln2_lo.
let val_hi = _mm256_sub_pd(f, hfsq);
let val_lo = /* (f - val_hi) - hfsq + s*(hfsq+R) + k*ln2_lo */;

// Knuth 2Sum on the *addition* val_hi + k*ln2_hi:
let hi     = _mm256_add_pd(val_hi, k_ln2_hi);
let b_virt = _mm256_sub_pd(hi, val_hi);
let a_virt = _mm256_sub_pd(hi, b_virt);
let b_err  = _mm256_sub_pd(k_ln2_hi, b_virt);
let a_err  = _mm256_sub_pd(val_hi, a_virt);
let lo     = _mm256_add_pd(_mm256_add_pd(a_err, b_err), val_lo);
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
| internal precision  | f64 (f32 lanes split into 2× f64 halves)  | native f64 |
| compensated arithmetic | **none** (`pow_raw_f64::<false>`: plain ln → mul → exp) | Dekker product + 2Sum split |
| exp kernel          | division-free degree-10 Taylor (even/odd split in \\(r^2\\)) | fdlibm Padé form (one `vdivpd`) |
| subnormal pre-scaling | skipped (promoted f32 is never f64-subnormal) | yes |
| special-case overlay | once, on the full-width f32 vectors      | on the f64 vectors |
| ULP                 | \\(\le 2\\)                                   | \\(\le 2\\)    |

f32 *promotion* is essential here: a pure-f32 chain through `ln_f32` then
`y * ln_f32` then `exp_f32` would amplify the 2 ULP error of each step
by \\(|y|\\), giving wildly inaccurate results for moderate-magnitude
exponents. Promoting to f64 buys 29 extra mantissa bits of headroom — far
more than \\(|y|\\) can spend. But f32 *compensation* is overkill: plain f64
arithmetic already lands \\(\sim 2^{21}\\) times below half an f32 ULP (see
§4), so the f32 path drops the double-double machinery and pockets the
speed.

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

From [`src/arch/avx2/math/pow.rs`][src-avx2] — the raw kernel, generic over the
precision mode. `pow_core_f64` (the f64 entry) instantiates
`pow_raw_f64::<true>`; `_mm256_pow_ps` instantiates `pow_raw_f64::<false>`
on each promoted half and finishes with `pow_special_cases_ps` on the
full-width f32 vectors:

```rust,ignore
unsafe fn pow_raw_f64<const PRECISE: bool>(x_abs: __m256d, y: __m256d) -> __m256d {
    // Stage 1: ln(|x|) — (hi, lo) pair when PRECISE, (hi, 0) otherwise
    let (ln_hi, ln_lo) = ln_hilo::<PRECISE>(x_abs);

    if PRECISE {
        // Stage 2: Dekker product  e_hi + e_lo = y * (ln_hi + ln_lo)
        let e_hi = _mm256_mul_pd(y, ln_hi);
        let e_lo = _mm256_fmadd_pd(y, ln_lo, _mm256_fmsub_pd(y, ln_hi, e_hi));
        // Stage 3: exp reduction folds e_lo into r before the polynomial
        exp_compensated::<true>(e_hi, e_lo)
    } else {
        // f32 mode: one plain product, no compensation tail
        exp_compensated::<false>(_mm256_mul_pd(y, ln_hi), ln_lo)
    }
}
```

Inside `exp_compensated::<true>`, the reduction remainder picks up the
tail — `let r = _mm256_add_pd(r, e_lo);` — before the Padé polynomial;
the `<false>` instantiation skips that add and evaluates a division-free
degree-10 Taylor polynomial instead. The bulk of the file is the
special-case mask construction, which exists twice: once in f64
(`pow_core_f64`, phase 3) and once in f32 (`pow_special_cases_ps`).

## 12. References

- musl libc: [`src/math/pow.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/pow.c) and [`src/math/powf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/powf.c).
- IEEE 754-2008, §9.2.1, table of *recommendedFunctions* — the special-value table reproduced in §3.
- ISO/IEC 9899:2018 (C18), §F.10.4.4 — same table for `pow`.
- Dekker, T. J., *A floating-point technique for extending the available precision*, Numer. Math. 18, 1971 — original double-double product.
- Markstein, *IA-64 and Elementary Functions*, Prentice-Hall 2000, chapter 11 — compensated `pow` derivation.
- Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed., §11.5.
- Repo source: [`src/arch/avx2/math/pow.rs`][src-avx2], [`src/arch/avx512/math/pow.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx512/math/pow.rs), [`src/arch/neon/math/pow.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/math/pow.rs).

## See also

- [Natural logarithm `ln`](./ln.md) — compensated front-end.
- [Natural exponential `exp`](./exp.md) — compensated back-end.
- [Compensated arithmetic: two-sum and Dekker product](../foundations/compensated.md)
  — the algebraic identities behind the `(hi, lo)` representation.
