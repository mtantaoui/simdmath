# Cube root `cbrt`

Unlike [`sqrt`](./sqrt.md), the cube root has no hardware instruction.
The implementation here follows musl libc's `cbrt.c` / `cbrtf.c` (which
descend from FreeBSD/fdlibm work by Bruce D. Evans): a bit-level magic
seed gets an initial 5-bit estimate in zero arithmetic operations, and
two or three Newton refinements bring it to \\(\le 1\\) ULP.

## 1. Mathematical definition

For all real \\(x\\),

\\[
\sqrt[3]{x} \\;=\\; \text{the unique real } y \text{ with } y^3 = x.
\\]

The cube root, unlike the square root, is defined for *all* reals
(including negatives) because \\(y^3\\) is bijective on \\(\mathbb{R}\\).

## 2. Domain and range

| input          | output            |
|----------------|-------------------|
| any \\(x \in \mathbb{R}\\) | \\(\sqrt[3]{x} \in \mathbb{R}\\) |
| \\(\pm 0\\)        | \\(\pm 0\\)           |
| \\(\pm\infty\\)    | \\(\pm\infty\\)       |
| \\(\text{NaN}\\)   | \\(\text{NaN}\\)      |
| subnormal \\(x\\)  | full precision (after \\(2^{24}\\) / \\(2^{54}\\) pre-scaling) |

The mathematical range is all of \\(\mathbb{R}\\). Every finite f32 / f64
input has a finite cube root that is also representable, so there is no
overflow or underflow regime to worry about.

## 3. Special values (IEEE 754 / C99 Sec. F.10.4.1)

| input          | output         |
|----------------|----------------|
| \\(+0\\)           | \\(+0\\)           |
| \\(-0\\)           | \\(-0\\)           |
| \\(+\infty\\)      | \\(+\infty\\)      |
| \\(-\infty\\)      | \\(-\infty\\)      |
| \\(\text{NaN}\\)   | \\(\text{NaN}\\)   |
| \\(8.0\\)          | \\(2.0\\) (exact)  |
| \\(-8.0\\)         | \\(-2.0\\) (exact) |

There are no edge-case domain errors: \\(\sqrt[3]{x}\\) always produces a
finite result for every finite \\(x\\). The only special-value handling in
the implementation is propagation of \\(\pm\infty\\) and NaN, plus
preservation of the input sign for zeros.

## 4. Algorithm overview

The decomposition is *multiplicative*:

\\[
x \\;=\\; \mathrm{sign}(x) \cdot 2^{e} \cdot m, \qquad
\sqrt[3]{x} \\;=\\; \mathrm{sign}(x) \cdot 2^{e/3} \cdot \sqrt[3]{m}.
\\]

The clever bit is that \\(e/3\\) — when applied directly to the *biased*
exponent in the IEEE 754 bit pattern — can be approximated by an integer
divide-by-3, plus a bias correction \\(B_1\\) that absorbs both the IEEE
exponent bias and the rounding error. The result of *that* is already a
5-bit-accurate approximation of \\(\sqrt[3]{x}\\), completely cooked from the
input bits with no floating-point arithmetic.

Newton's iteration for \\(\sqrt[3]{x}\\) — the root of \\(f(t) = t^3 - x\\) — is

\\[
t\_{n+1} \\;=\\; t_n - \frac{t_n^3 - x}{3 t_n^2}
\\;=\\; \frac{t_n(2x + t_n^3)}{x + 2 t_n^3}.
\\]

The cubic-Newton iteration *triples* the number of correct bits per
step. Five bits → fifteen → forty-five — so two iterations suffice for
f32 (24 mantissa bits) and three for f64 (53 mantissa bits).

## 5. Argument reduction (input decomposition)

Five preliminaries are performed before the magic seed:

1. **Special masks.** Detect \\(\pm 0\\), \\(\pm\infty\\), NaN; capture the sign
   bit for restoration.
2. **Subnormal scaling.** If \\(|x| < 2^{-126}\\) (f32) or \\(2^{-1022}\\)
   (f64), multiply by `X1P24_32` \\(= 2^{24}\\) or `X1P54_64` \\(= 2^{54}\\) to
   push into the normal range. The bias constant `B2` is used in place
   of `B1` so that the eventual reconstruction undoes the scaling.
3. **Strip the sign.** Work on \\(|x|\\) and re-apply \\(\mathrm{sign}(x)\\) at
   the end.
4. **Magic seed.** Compute \\(\mathrm{hx}/3 + B\\), where \\(\mathrm{hx}\\) is
   the high 32 bits of the f64 representation (or the full 32 bits of
   f32). The constants from
   [`src/arch/consts/cbrt.rs`][src-consts]:

```rust,ignore
pub(crate) const B1_32: u32 = 709_958_130;   // (127 - 127/3 - 0.03306) * 2^23
pub(crate) const B2_32: u32 = 642_849_266;   // B1 - (24/3) * 2^23
pub(crate) const X1P24_32: f32 = 16_777_216.0;  // 2^24

pub(crate) const B1_64: u32 = 715_094_163;   // (1023 - 1023/3 - 0.03306) * 2^20
pub(crate) const B2_64: u32 = 696_219_795;   // B1 - (54/3) * 2^20
pub(crate) const X1P54_64: f64 = 18_014_398_509_481_984.0; // 2^54
```

[src-consts]: https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/cbrt.rs

The "\\(0.03306\ldots\\)" fudge factor is the empirically-best value that
makes the *worst-case* relative error of \\(\mathrm{hx}/3 + B\\) equal in
both directions; without it, the seed would lean ~7% high.

5. **SIMD integer divide-by-3.** AVX2 has no integer division, so
   \\(\mathrm{hx}/3\\) is computed via the *multiply-by-magic-constant*
   trick:

\\[
\frac{\mathrm{hx}}{3} \\;=\\; \left\lfloor \mathrm{hx} \cdot \mathtt{0xAAAAAAAB} \right\rfloor \\!\gg\\! 33,
\\]

implemented as `mulhi(hx, 0xAAAAAAAB) >> 1`. Even and odd lanes are
processed separately because `_mm256_mul_epu32` only multiplies
even-indexed 32-bit elements.

## 6. Polynomial / kernel approximation

For f32, no polynomial — just two Newton iterations.

For f64, the implementation refines the 5-bit seed by a degree-4
*minimax* polynomial in \\(r = t^3/x\\) that approximates \\(1/\sqrt[3]{r}\\) on
\\(|r - 1| < 0.1\\), lifting accuracy from 5 bits to ~23 bits. Coefficients
from [`src/arch/consts/cbrt.rs`][src-consts]:

```rust,ignore
pub(crate) const P0: f64 =  1.875_951_824_271_770_096_43;  // 0x3FFE03E60F61E692
pub(crate) const P1: f64 = -1.884_979_795_433_771_698_75;  // 0xBFFE28E092F02420
pub(crate) const P2: f64 =  1.621_429_720_105_354_466_14;  // 0x3FF9F1604A49D6C2
pub(crate) const P3: f64 = -0.758_397_934_778_766_047_437; // 0xBFE844CBBEE751D9
pub(crate) const P4: f64 =  0.145_996_192_886_612_446_982; // 0x3FC2B000D4E4EDD7
```

The polynomial is evaluated by Horner

\\[
P(r) \\;=\\; P_0 + r(P_1 + r(P_2 + r(P_3 + r P_4))),
\\]

then \\(t \leftarrow t \cdot P(r)\\) produces the 23-bit estimate.

## 7. Reconstruction

**f32 path.**

```text
seed:           t ≈ x^{1/3} to 5 bits        (one integer add)
Newton 1 (f64): t ← t (2x + r)/(x + 2r),  r = t³            (~15 bits)
Newton 2 (f64): same formula                              (~45 bits)
narrow back:    f64 → f32 round-to-nearest                (24 bits exact)
sign restore:   OR the saved sign bit
```

The two-iteration Newton formula in the code is the algebraically-
rearranged form

\\[
t\_{n+1} \\;=\\; t_n \cdot \frac{2x + t_n^3}{x + 2 t_n^3}
\\]

which avoids subtraction in either numerator or denominator and so is
robust against cancellation when \\(t_n^3 \approx x\\).

**f64 path.**

```text
seed:                t ≈ x^{1/3} to 5 bits
poly refine:         t ← t · P(t³/x)                      (~23 bits)
round to 23 bits:    bias-and-mask trick (ROUND_MASK_64,
                     ROUND_BIAS_64) to get t exact to 23 bits
                     so that t² is exact in f64
final Newton:        t ← t + t (x/t² - t)/(2t + x/t²)     (~53 bits)
sign restore
```

The "round to 23 bits away from zero" step is the key precision trick:
after it, \\(t \cdot t\\) has at most 46 mantissa bits and so is *exactly
representable* in f64 (which carries 53). The final Newton step is
therefore performed in *exact* arithmetic except for the final divide,
yielding a faithfully-rounded answer.

```rust,ignore
pub(crate) const ROUND_MASK_64: u64 = 0xffffffff_c0000000; // keep top 23 bits
pub(crate) const ROUND_BIAS_64: u64 = 0x00000000_80000000; // round away from zero
```

## 8. Per-precision differences (f32 vs f64)

| aspect           | f32                              | f64                                    |
|------------------|----------------------------------|----------------------------------------|
| seed bias        | `B1_32` / `B2_32`                | `B1_64` / `B2_64`                      |
| seed accuracy    | ~5 bits                          | ~5 bits                                |
| refinement       | 2× Newton in f64                 | poly + round + 1× Newton (all in f64)  |
| subnormal scale  | \\(\times 2^{24}\\)                  | \\(\times 2^{54}\\)                        |
| ULP              | \\(\le 1\\)                          | \\(\le 1\\)                                |

The f32 path runs the entire refinement *in f64* — half the lanes, twice
the precision per lane — and rounds once at the end. This is cheaper
than re-deriving a separate f32 polynomial and is what makes the f32
ULP bound match f64's.

## 9. Per-backend differences (AVX2 / AVX-512 / NEON)

| backend  | f32 lanes | f64 lanes | integer divide-by-3                |
|----------|-----------|-----------|-------------------------------------|
| AVX2     | 8 (refined in 2× f64) | 4 | `mulhi(x, 0xAAAAAAAB) >> 1` (even/odd lanes split) |
| AVX-512  | 16 (refined in 2× f64) | 8 | same magic-constant trick on 16-bit-wide masked ops |
| NEON     | 4 (refined in 2× f64)  | 2 | scalar magic-constant or `vmulq_n_u32` form |

NEON's f32 cbrt uses the same Newton structure but expresses the divide
via `vmlaq_f64` (multiply-add). All three backends produce bit-identical
outputs.

## 10. Error analysis

The error budget is bounded by the final Newton step:

\\[
\mathrm{err}(t\_{n+1}) \\;\le\\; C \cdot \mathrm{err}(t_n)^3
\\]

for cubic convergence. Starting at 5 bits, two iterations give
\\(5 \cdot 3^2 = 45\\) bits — beyond f32's 24-bit mantissa, with margin for
the f64 → f32 rounding step. Three iterations (or poly + 1 Newton)
give ~53 bits for f64.

Empirically, the **worst-case ULP is \\(\le 1\\)** for both precisions
across an exhaustive sweep of f32 and a \\(2^{30}\\)-point sample of f64.
The peak error tends to occur for inputs whose mantissa is on a Newton
iteration boundary, i.e. where the polynomial seed is one bit
asymmetric.

## 11. Code excerpt

From [`src/arch/avx2/cbrt.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/cbrt.rs)
— the f32 inner loop:

```rust,ignore
// Magic seed: hx/3 + B1 (or +B2 for subnormal)
let hx_normal    = _mm256_add_epi32(div_by_3_epi32(hx),                  b1);
let hx_subnormal = _mm256_add_epi32(div_by_3_epi32(hx_scaled),           b2);
let hx_approx    = _mm256_blendv_epi8(hx_normal, hx_subnormal, is_subnormal);
let t_f32        = _mm256_castsi256_ps(_mm256_or_si256(sign_bits, hx_approx));

// Newton iterations in f64 (low 4 lanes shown; high 4 lanes mirror)
let x_low = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
let t_low = _mm256_cvtps_pd(_mm256_castps256_ps128(t_f32));

// First iteration: t ← t (2x + t³) / (x + 2 t³)
let r_low   = _mm256_mul_pd(t_low, _mm256_mul_pd(t_low, t_low));
let t_low   = _mm256_mul_pd(t_low,
              _mm256_div_pd(_mm256_add_pd(_mm256_add_pd(x_low, x_low), r_low),
                            _mm256_add_pd(x_low, _mm256_add_pd(r_low, r_low))));
// Second iteration: same formula
let r_low   = _mm256_mul_pd(t_low, _mm256_mul_pd(t_low, t_low));
let t_low_f = _mm256_mul_pd(t_low,
              _mm256_div_pd(_mm256_add_pd(_mm256_add_pd(x_low, x_low), r_low),
                            _mm256_add_pd(x_low, _mm256_add_pd(r_low, r_low))));
```

The f64 path has the analogous structure, with the polynomial refinement
and the 23-bit rounding step inserted between the seed and the final
Newton iteration.

## 12. References

- musl libc: [`src/math/cbrt.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/cbrt.c) and [`src/math/cbrtf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/cbrtf.c) (FreeBSD / fdlibm, B. D. Evans).
- Kahan, W., *Computing a real cube root*, manuscript, 1991 — the original derivation of the magic-constant seed.
- Markstein, *IA-64 and Elementary Functions*, Prentice-Hall 2000, §10.4.
- Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed., §11.3 (general theory of Newton iteration for elementary functions).
- Repo source: [`src/arch/avx2/cbrt.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/cbrt.rs), [`src/arch/avx512/cbrt.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx512/cbrt.rs), [`src/arch/neon/cbrt.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/cbrt.rs), [`src/arch/consts/cbrt.rs`][src-consts].

## See also

- [Square root `sqrt`](./sqrt.md) — by contrast, has a hardware
  instruction that makes all of this unnecessary.
- [Compensated arithmetic: two-sum and Dekker product](../foundations/compensated.md)
  for the precision-preservation tricks used in the f64 round-to-23-bits
  step.
- [Polynomial evaluation: Horner, Estrin, FMA](../foundations/polynomial_evaluation.md)
  for how the degree-4 minimax refinement is laid out.
