# Natural logarithm `ln`

The natural logarithm \\(\ln(x) = \log_e(x)\\) is the inverse of
[`exp`](./exp.md). The fdlibm algorithm decomposes \\(x\\) into an exact
power of two times a mantissa near \\(1\\), then evaluates a clever rational
substitution that converges much faster than the textbook
\\(\log(1+f)\\) Taylor series.

## 1. Mathematical definition

\\[
\ln(x) \\;=\\; \int_1^x \frac{dt}{t}, \qquad x > 0.
\\]

Equivalently, \\(\ln(x) = y\\) iff \\(\exp(y) = x\\). The Taylor expansion at
\\(x = 1\\) is

\\[
\ln(1 + f) \\;=\\; f - \tfrac{f^2}{2} + \tfrac{f^3}{3} - \cdots
\\]

which converges only for \\(|f| < 1\\) and converges *slowly* near
\\(|f| = 1\\).

## 2. Domain and range

| input | output |
|-------|--------|
| \\(x > 0\\)         | \\(\ln(x) \in \mathbb{R}\\) |
| \\(x = +0\\) or \\(-0\\)| \\(-\infty\\)              |
| \\(x < 0\\)         | \\(\text{NaN}\\)           |
| \\(x = +\infty\\)   | \\(+\infty\\)              |
| \\(x = \text{NaN}\\)| \\(\text{NaN}\\)           |

The mathematical range is all of \\(\mathbb{R}\\), finitely-representable in
floating point as \\(\ln(\mathtt{f64::MIN\\_POSITIVE}) \approx -744.4\\) at
the bottom and \\(\ln(\mathtt{f64::MAX}) \approx 709.8\\) at the top.

## 3. Special values (IEEE 754 / C99 Sec. F.10.3.7)

| input    | output       |
|----------|--------------|
| \\(+0\\)     | \\(-\infty\\)    |
| \\(-0\\)     | \\(-\infty\\)    |
| \\(1.0\\)    | \\(+0\\)         |
| \\(e\\)      | \\(1.0\\) (within ≤ 1 ULP) |
| \\(+\infty\\)| \\(+\infty\\)    |
| \\(x < 0\\)  | \\(\text{NaN}\\) |
| \\(\text{NaN}\\) | \\(\text{NaN}\\) |

Reproduced verbatim in the doc comments of [`src/arch/avx2/ln.rs`][src-avx2].

[src-avx2]: https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/ln.rs

## 4. Algorithm overview

```text
   x        ─split───►  (k, m)          ─substitute───►  (s, f)
                        x = 2^k · m                       s = f / (2 + f)
                        m ∈ [√2/2, √2]

   (s, f)   ─poly────►  R(s²)           ─reconstruct──►  ln(x)
                        ≈ ln((1+s)/(1-s))                = k·ln(2) + log(1+f)
                          − 2s
```

Five steps:

1. **Decompose.** Extract the IEEE 754 exponent into an integer \\(k\\) and
   a mantissa \\(m \in [1, 2)\\). If \\(m > \sqrt{2}\\), halve \\(m\\) and increment
   \\(k\\), so that \\(m\\) ends up in \\([\sqrt{2}/2, \sqrt{2}]\\).
2. **Substitute.** Set \\(f = m - 1\\), \\(s = f/(2+f)\\). Then
   \\(|f| < 0.414\\) and \\(|s| < 0.166\\) — much smaller than for plain \\(f\\).
3. **Polynomial.** Evaluate a degree-7 minimax polynomial
   \\(R(s^2)\\) that approximates \\(2 \mathrm{atanh}(s) - 2s\\).
4. **Reconstruct** \\(\ln(x)\\) using the two-sum split
   \\(\ln 2 = \mathtt{LN2\\_HI} + \mathtt{LN2\\_LO}\\).
5. **Special cases** via mask blending.

## 5. Input decomposition

The fastest way to extract \\((k, m)\\) from \\(x\\) is to manipulate IEEE 754
bits: \\(k\\) is the biased exponent minus 1023 (f64) or 127 (f32); \\(m\\) is
obtained by clearing the exponent bits and OR-ing in the bias of \\(1.0\\).

For subnormal inputs (whose exponent field is zero), \\(m\\) would be \\(< 1\\),
so the algorithm first multiplies by \\(2^{52}\\) (f64) to push the value
into the normal range and subtracts 52 from \\(k\\) to compensate.

The "\\(m \in [\sqrt{2}/2, \sqrt{2}]\\)" normalization is then a single
compare against `SQRT2_64` from [`src/arch/consts/ln.rs`][src-consts]:

```rust,ignore
pub const SQRT2_64: f64 = f64::from_bits(0x3FF6A09E667F3BCD); // 1.4142135623730951
pub const TWO52_64: f64 = (1u64 << 52) as f64;
```

[src-consts]: https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/ln.rs

If \\(m > \sqrt 2\\), replace \\((k, m)\\) by \\((k+1, m/2)\\).

## 6. Polynomial / kernel approximation

With \\(f = m - 1\\), \\(s = f/(2+f)\\), and \\(z = s^2\\), the polynomial is:

\\[
R(z) \\;=\\; \mathtt{Lg_1}\\,z + \mathtt{Lg_2}\\,z^2 + \mathtt{Lg_3}\\,z^3 + \mathtt{Lg_4}\\,z^4 + \mathtt{Lg_5}\\,z^5 + \mathtt{Lg_6}\\,z^6 + \mathtt{Lg_7}\\,z^7.
\\]

The coefficients approximate \\(2/(2n+1)\\) — the Taylor coefficients of
\\(\log\\!\left(\tfrac{1+s}{1-s}\right) = 2s + \tfrac{2s^3}{3} + \tfrac{2s^5}{5} + \cdots\\) —
but minimax-tuned to flatten the error curve over \\(|s| < 0.1716\\):

```rust,ignore
pub const LG1_64: f64 = f64::from_bits(0x3FE5555555555593); // 6.666_666_666_666_735e-1  ≈ 2/3
pub const LG2_64: f64 = f64::from_bits(0x3FD999999997FA04); // 3.999_999_999_940_942e-1  ≈ 2/5
pub const LG3_64: f64 = f64::from_bits(0x3FD2492494229359); // 2.857_142_874_366_239e-1  ≈ 2/7
pub const LG4_64: f64 = f64::from_bits(0x3FCC71C51D8E78AF); // 2.222_219_843_214_978e-1  ≈ 2/9
pub const LG5_64: f64 = f64::from_bits(0x3FC7466496CB03DE); // 1.818_357_216_161_805e-1  ≈ 2/11
pub const LG6_64: f64 = f64::from_bits(0x3FC39A09D078C69F); // 1.531_383_769_920_937e-1  ≈ 2/13
pub const LG7_64: f64 = f64::from_bits(0x3FC2F112DF3E5244); // 1.479_819_860_511_658e-1  ≈ 2/15
```

The substitution \\(s = f/(2+f)\\) maps the multiplicative interval
\\([1/\sqrt 2, \sqrt 2]\\) for \\(m\\) to the additive interval \\(|s| < 0.1716\\).
This converts a slowly-converging Taylor series in \\(f\\) into a *cubically*
faster one in \\(s\\).

## 7. Reconstruction

The identity used is

\\[
\ln\\!\left(\frac{1+s}{1-s}\right) \\;=\\; 2s + 2s\cdot R(s^2)/(\dots),
\\]

which fdlibm rewrites in the numerically-friendly form

\\[
\ln(x) \\;=\\; k\cdot\mathtt{LN2\\_HI}
\\;-\\;
\Bigl[\bigl(\mathrm{hfsq} - (s\cdot(\mathrm{hfsq} + R) + k\cdot\mathtt{LN2\\_LO})\bigr) - f\Bigr],
\\]

where \\(\mathrm{hfsq} = \tfrac{1}{2}f^2\\).

Three observations:

- \\(k\cdot\mathtt{LN2\\_HI}\\) is **exact** because `LN2_HI_64` was rounded to
  33 bits (its low 20 bits are zero) and \\(|k| \le 1024 < 2^{20}\\).
- \\(f\\) contributes the leading-order term \\(\ln(1+f) \approx f\\).
- \\(\mathrm{hfsq} - s\cdot(\mathrm{hfsq} + R + k\cdot\mathtt{LN2\\_LO})\\)
  collects all the second-order corrections, including the catch-up term
  \\(k\cdot\mathtt{LN2\\_LO}\\) that compensates for the truncated `LN2_HI`.

The same `LN2_HI / LN2_LO` constants are used by [`exp`](./exp.md), so
both directions of the inverse pair share their rounding behaviour.

## 8. Per-precision differences (f32 vs f64)

| aspect              | f32                                       | f64        |
|---------------------|-------------------------------------------|------------|
| internal precision  | f64 (8-lane f32 split into 2× f64 halves) | native f64 |
| polynomial          | shared `LG1..LG7`                         | shared     |
| subnormal handling  | inherited from the f64 kernel (×\\(2^{52}\\)) | ×\\(2^{52}\\)  |
| ULP                 | \\(\le 2\\)                                   | \\(\le 2\\)    |

As with [`exp`](./exp.md), the f32 path promotes to f64, runs the same
`ln_core_f64`, and converts back. This trades two cvt instructions for
the simplicity of a single shared coefficient table.

## 9. Per-backend differences (AVX2 / AVX-512 / NEON)

| backend  | f32 lanes | f64 lanes | exponent extraction                |
|----------|-----------|-----------|-------------------------------------|
| AVX2     | 8 (via 2× f64) | 4    | `_mm256_srli_epi64(bits, 52)` then mask |
| AVX-512  | 16 (via 2× f64) | 8   | `_mm512_getexp_pd` (direct hardware) |
| NEON     | 4 (via 2× f64)  | 2   | `vshrq_n_u64(bits, 52)` then mask    |

AVX-512 has a dedicated `vgetexppd` / `vgetmantpd` pair that does the
\\(x = 2^k \cdot m\\) split in hardware; the AVX2 and NEON paths emulate it
with integer shifts. The polynomial evaluation is identical — eight FMAs
on \\(z = s^2\\) in Horner order from \\(\mathtt{Lg_7}\\) down to \\(\mathtt{Lg_1}\\).

## 10. Error analysis

The total error is

\\[
\mathrm{err}(\widehat{\ln}(x))
\\;\le\\;
\underbrace{|k|\cdot\mathrm{err}\_{\ln 2}}\_{\le\\,2^{-95}}
\\;+\\;
\underbrace{\mathrm{err}\_{R}(s)}\_{\le\\,2^{-58}}
\\;+\\;
\underbrace{\mathrm{err}\_{\text{recon}}}\_{\le\\,1\text{ ULP}}.
\\]

The polynomial's intrinsic error is dominated by the round-off of
the eight FMAs evaluating \\(R(z)\\), totalling roughly \\(8\\,\varepsilon\\)
where \\(\varepsilon = 2^{-52}\\). The reconstruction add can lose up to one
extra ULP near \\(x = 1\\), where \\(\ln(x) \to 0\\) and any rounding noise in
\\(s\cdot(\mathrm{hfsq} + R)\\) becomes the leading-order term. The
implementation therefore special-cases \\(x\\) very close to \\(1\\) via the
small-\\(f\\) branch.

**Worst-case observed.** \\(\le 2\\) ULP at both precisions, attained near
the boundaries of the \\([\sqrt 2/2, \sqrt 2]\\) interval where the
polynomial residual peaks.

## 11. Code excerpt

From [`src/arch/avx2/ln.rs`][src-avx2] (abridged Horner evaluation of
\\(R(s^2)\\)):

```rust,ignore
// Step 2: substitute f = m - 1, s = f / (2 + f)
let f = _mm256_sub_pd(m, one);
let s = _mm256_div_pd(f, _mm256_add_pd(two, f));
let z = _mm256_mul_pd(s, s);

// Step 3: polynomial R(z)
let r = _mm256_fmadd_pd(z, _mm256_set1_pd(LG7_64), _mm256_set1_pd(LG6_64));
let r = _mm256_fmadd_pd(z, r, _mm256_set1_pd(LG5_64));
let r = _mm256_fmadd_pd(z, r, _mm256_set1_pd(LG4_64));
let r = _mm256_fmadd_pd(z, r, _mm256_set1_pd(LG3_64));
let r = _mm256_fmadd_pd(z, r, _mm256_set1_pd(LG2_64));
let r = _mm256_fmadd_pd(z, r, _mm256_set1_pd(LG1_64));
let r = _mm256_mul_pd(r, z);

// Step 4: reconstruct
let hfsq    = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(f, f));
let k_ln2hi = _mm256_mul_pd(k_f64, _mm256_set1_pd(LN2_HI_64));
let k_ln2lo = _mm256_mul_pd(k_f64, _mm256_set1_pd(LN2_LO_64));
// ln(x) = k*LN2_HI - ((hfsq - (s*(hfsq+R) + k*LN2_LO)) - f)
let inner   = _mm256_sub_pd(hfsq,
              _mm256_fmadd_pd(s, _mm256_add_pd(hfsq, r), k_ln2lo));
let result  = _mm256_sub_pd(k_ln2hi, _mm256_sub_pd(inner, f));
```

## 12. References

- musl libc: [`src/math/log.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/log.c) and [`src/math/logf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/logf.c) (fdlibm `e_log.c`, Sun Microsystems / K. C. Ng).
- Tang, P. T. P., *Table-driven implementation of the logarithm function in IEEE floating-point arithmetic*, ACM TOMS 16(4), 1990.
- Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed., §11.2 (logarithm-specific reduction tricks).
- Goldberg, *What every computer scientist should know about floating-point arithmetic*, ACM Comp. Surveys 23(1), 1991, §3.2 — for why \\(\log(1+x)\\) requires special care near \\(x = 0\\).
- Repo source: [`src/arch/avx2/ln.rs`][src-avx2], [`src/arch/avx512/ln.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx512/ln.rs), [`src/arch/neon/ln.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/ln.rs), [`src/arch/consts/ln.rs`][src-consts].

## See also

- [Natural exponential `exp`](./exp.md) — the inverse pair.
- [Power `pow`](./pow.md) — uses a hi/lo-split version of this kernel.
- [Compensated arithmetic: two-sum and Dekker product](../foundations/compensated.md)
  for the general theory behind the `LN2_HI` / `LN2_LO` split.
