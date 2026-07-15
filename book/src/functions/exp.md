# Natural exponential `exp`

The natural exponential \\(\exp(x) = e^x\\) is computed by the classical
*range-reduction → polynomial → reconstruction* pipeline of fdlibm,
ported in the file [`src/arch/avx2/math/exp.rs`][src-avx2] (and its AVX-512 /
NEON twins).

[src-avx2]: https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/math/exp.rs

## 1. Mathematical definition

For real \\(x\\),

\\[
\exp(x) \\;=\\; \sum\_{n=0}^{\infty} \frac{x^n}{n!}
        \\;=\\; \lim\_{n\to\infty}\left(1 + \frac{x}{n}\right)^{\\!n}.
\\]

It is the unique solution of \\(f'(x) = f(x)\\), \\(f(0) = 1\\), and the inverse
of \\(\ln\\): \\(\exp(\ln x) = x\\) for \\(x > 0\\), \\(\ln(\exp x) = x\\) for all real
\\(x\\).

## 2. Domain and range

Mathematically \\(\exp:\mathbb{R}\to(0,\infty)\\). In binary floating-point
the *useful* domain is bounded by the dynamic range of the result type:

| type | overflow at | underflow at | range |
|------|-------------|--------------|-------|
| f32  | \\(x \gtrsim 88.72\\)  | \\(x \lesssim -103.97\\) | \\([0, +\infty]\\) |
| f64  | \\(x \gtrsim 709.78\\) | \\(x \lesssim -745.13\\) | \\([0, +\infty]\\) |

The exact f64 thresholds live in [`src/arch/consts/exp.rs`][src-consts]:

```rust,ignore
/// = 7.09782712893383973096e+02, the largest x with exp(x) finite
pub const OVERFLOW_THRESH_64: f64 = f64::from_bits(0x40862E42FEFA39EF);
/// = -7.45133219101941108420e+02, below which exp(x) underflows to 0
pub const UNDERFLOW_THRESH_64: f64 = f64::from_bits(0xC0874910D52D3051);
```

[src-consts]: https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/exp.rs

## 3. Special values (IEEE 754 / C99 Sec. F.10.3.1)

| input        | output                  |
|--------------|-------------------------|
| \\(+0\\)         | \\(1.0\\)                   |
| \\(-0\\)         | \\(1.0\\)                   |
| \\(+\infty\\)    | \\(+\infty\\)               |
| \\(-\infty\\)    | \\(+0.0\\)                  |
| \\(\text{NaN}\\) | \\(\text{NaN}\\)            |
| \\(x > \\) overflow threshold | \\(+\infty\\) (overflow)         |
| \\(x < \\) underflow threshold | \\(+0.0\\) (gradual underflow) |

These are produced branchlessly with `_mm256_blendv_pd`. The \\(\pm\infty\\)
rows need no compares of their own — \\(+\infty\\) already exceeds the
overflow threshold and \\(-\infty\\) already undershoots the underflow
threshold — so only three masks exist. Overflow and underflow are disjoint,
which lets their two blends collapse into one: a combined
`overflow | underflow` mask selects a pre-blended `+∞`/`+0` value, and a
final blend applies NaN:

```text
generic  →  (overflow | underflow → ±special)  →  NaN
```

## 4. Algorithm overview

The exponential cannot be polynomial-approximated globally — its dynamic
range is too wide. The standard trick reduces to a *bounded* argument:

1. **Range reduction**. Find an integer \\(k\\) and a small reduced argument
   \\(r\\) such that \\(x = k\ln 2 + r\\), \\(|r| \le \tfrac{1}{2}\ln 2 \approx 0.347\\).
2. **Polynomial kernel**. Approximate \\(\exp(r)\\) on this narrow interval
   by a degree-5 minimax polynomial \\(P(r^2)\\) wrapped in a Padé form.
3. **Reconstruction**. Multiply by \\(2^k\\), an exact operation: just add
   \\(k\\) to the IEEE 754 exponent field of the polynomial result.
4. **Special-case overrides**.

Because \\(2^k\\) is constructed bit-wise rather than by another `exp` call,
the only rounding errors are the polynomial residual and the reduction
residual.

## 5. Argument reduction

Choose \\(k = \mathrm{round}(x / \ln 2)\\), then \\(r = x - k\ln 2\\).

A naive implementation evaluates \\(r = x - k \cdot \mathtt{LN2}\\) with a
single rounded constant for \\(\ln 2\\). That loses most of the bits of \\(x\\)
when \\(|x|\\) is large because \\(k\ln 2 \approx x\\), and the cancellation
exposes the rounding error of `LN2` itself.

The fix is to split \\(\ln 2\\) across two doubles whose sum is far closer to
the exact value than either summand alone. From
[`src/arch/consts/exp.rs`][src-consts]:

```rust,ignore
/// = 6.93147180369123816490e-01 (33 leading bits, low bits zero)
pub const LN2_HI_64: f64 = f64::from_bits(0x3FE62E42FEE00000);
/// = 1.90821492927058500170e-10 (residual)
pub const LN2_LO_64: f64 = f64::from_bits(0x3DEA39EF35793C76);
/// = 1.44269504088896338700e+00 (1/ln 2 = log2 e)
pub const LN2_INV_64: f64 = f64::from_bits(0x3FF71547652B82FE);
```

`LN2_HI_64` is rounded to 33 bits (its low 20 mantissa bits are zero), so
that the product \\(k \cdot \mathtt{LN2\\_HI}\\) is exact for \\(|k| < 2^{20}\\).
The reduction is then

\\[
r \\;=\\; (x - k \cdot \mathtt{LN2\\_HI}) - k \cdot \mathtt{LN2\\_LO},
\\]

evaluated as two FMAs. The second subtraction does *not* cancel because
\\(|k \cdot \mathtt{LN2\\_LO}|\\) is many orders of magnitude smaller than the
first residual.

```rust,ignore
let k_f64 = _mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(
    _mm256_fmadd_pd(x, ln2_inv, sign_half));
let r = _mm256_fnmadd_pd(k_f64, ln2_lo, _mm256_fnmadd_pd(k_f64, ln2_hi, x));
```

`sign_half = copysign(0.5, x)` plus truncation gives `round-to-nearest` in
a single rounding step.

## 6. Polynomial / kernel approximation

On \\(|r| \le \tfrac{1}{2}\ln 2\\), fdlibm's `e_exp.c` evaluates a degree-5
minimax polynomial in \\(r^2\\):

\\[
P(r^2) \\;=\\; P_1 + r^2\\!\left(P_2 + r^2\\!\left(P_3 + r^2(P_4 + r^2 P_5)\right)\right).
\\]

The coefficients (taken verbatim from
[`src/arch/consts/exp.rs`][src-consts], shown with their exact bit
patterns) are:

```rust,ignore
pub const P1_64: f64 = f64::from_bits(0x3FC555555555553E); //  1.666_666_666_666_660_19e-1
pub const P2_64: f64 = f64::from_bits(0xBF66C16C16BEBD93); // -2.777_777_777_701_559_34e-3
pub const P3_64: f64 = f64::from_bits(0x3F11566AAF25DE2C); //  6.613_756_321_437_934_36e-5
pub const P4_64: f64 = f64::from_bits(0xBEBBBD41C5D26BF1); // -1.653_390_220_542_185_15e-6
pub const P5_64: f64 = f64::from_bits(0x3E66376972BEA4D0); //  4.138_136_797_057_238_46e-8
```

These approximate \\(1/k!\\) for \\(k = 2, 4, 6, 8, 10\\) but are minimax-tuned to
reduce the maximum error rather than the Taylor truncation error.

The reduced exponential is reconstructed via the Padé-like form

\\[
c \\;=\\; r - r^2 P(r^2), \qquad
\exp(r) \\;=\\; 1 + r + \frac{r\\,c}{2 - c}.
\\]

The fraction \\(rc/(2-c)\\) has a small numerator (near zero) and a denominator
near 2, so neither cancellation nor overflow occurs.

## 7. Reconstruction

Once \\(\exp(r)\\) is in hand,

\\[
\exp(x) \\;=\\; 2^k \cdot \exp(r).
\\]

\\(2^k\\) is built directly in the IEEE 754 exponent field — no multiplier
table is needed. With \\(k\\) stored as a 64-bit integer:

```rust,ignore
let k_shifted = _mm256_slli_epi64(k_i64, 52);
let one_bits  = _mm256_set1_epi64x(0x3FF0000000000000_u64 as i64);
let scale     = _mm256_castsi256_pd(_mm256_add_epi64(k_shifted, one_bits));
let result    = _mm256_mul_pd(exp_r, scale);
```

For very large or very small \\(k\\), the exponent field can saturate. The
bit trick itself runs branch-free on every lane; the
`OVERFLOW_THRESH_64` / `UNDERFLOW_THRESH_64` masks are computed up front
and the saturated results (\\(+\\infty\\) / \\(0\\)) are blended over the
garbage lanes afterwards, so the trick's output for out-of-range inputs is
never observed.

## 8. Per-precision differences (f32 vs f64)

| aspect              | f32                                    | f64                  |
|---------------------|----------------------------------------|----------------------|
| internal kernel     | f64 (each 8-lane f32 vector is split into two f64 halves) | f64 native |
| polynomial          | shared `P1_64..P5_64`                  | shared               |
| overflow threshold  | \\(\sim 88.72\\)  (representable f32 limit) | \\(\sim 709.78\\) |
| underflow threshold | \\(\sim -103.97\\)                         | \\(\sim -745.13\\) |
| ULP                 | \\(\le 2\\)                                | \\(\le 2\\)              |

f32 is computed *via promotion to f64* — `_mm256_cvtps_pd` of each 128-bit
half, run `exp_core_f64` on it, then `_mm256_cvtpd_ps` and re-merge. This
costs two extra cvts per 8-lane vector but eliminates the need for a
separate f32 polynomial and constant set.

## 9. Per-backend differences (AVX2 / AVX-512 / NEON)

| backend  | f32 lanes | f64 lanes | round-to-nearest                  |
|----------|-----------|-----------|-----------------------------------|
| AVX2     | 8 (via 2× f64) | 4   | `_mm256_round_pd` w/ truncation + copysign-0.5 |
| AVX-512  | 16 (via 2× f64) | 8 | `_mm512_roundscale_pd` w/ truncation + copysign-0.5 |
| NEON     | 4 (via 2× f64) | 2  | `vrndq_f64` (truncation) + copysign-0.5 |

The AVX-512 backend can use mask registers directly, avoiding the
`blendv` chain; instead of computing all special-case results
unconditionally and selecting at the end, it applies them with masked
moves. The math is otherwise unchanged.

NEON's f64 path operates on 2-lane `float64x2_t` vectors, calling
`vfmaq_f64` for the FMA-based polynomial evaluation. The argument order
`vfmaq(c, a, b)` = \\(a \cdot b + c\\) differs from x86 — the accumulator
comes *first*.

## 10. Error analysis

Total error is the sum of three contributions:

\\[
\mathrm{err}(\widehat{\exp}(x))
\\;\le\\;
\underbrace{|k|\cdot \mathrm{err}\_{\ln 2}}\_{\text{reduction}}
\\;+\\;
\underbrace{\mathrm{err}\_{P}(r)}\_{\text{polynomial}}
\\;+\\;
\underbrace{\mathrm{err}\_{\times 2^k}}\_{\text{reconstruction (exact)}}.
\\]

- The reconstruction is **exact** because \\(2^k\\) is built bit-wise.
- The polynomial residual on \\(|r| \le \tfrac{\ln 2}{2}\\) is bounded by
  approximately \\(2^{-56}\\) relative — well under 1 ULP for f64 results.
- The reduction residual is the dominant term: it is bounded by
  \\(|k| \cdot \mathrm{ulp}(\mathtt{LN2\\_LO}) \approx 2^{-105}\\) per unit of
  \\(k\\), which for the worst-case \\(k \approx 1024\\) is about
  \\(2^{-95}\\) — still well below 1 ULP.

Empirically the worst-case error is \\(\le 2\\) ULP for both f32 and f64,
attained near the overflow boundary where the residual exponent
construction begins to interact with the polynomial-derived mantissa.

## 11. Code excerpt

From [`src/arch/avx2/math/exp.rs`][src-avx2] (the core kernel, abridged):

```rust,ignore
// Step 1: range reduction
let sign_half = _mm256_or_pd(half, _mm256_and_pd(x, sign_bit));
let k_f64 = _mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(
    _mm256_fmadd_pd(x, ln2_inv, sign_half));
let r = _mm256_fnmadd_pd(k_f64, ln2_lo,
        _mm256_fnmadd_pd(k_f64, ln2_hi, x));

// Step 2: polynomial P(r²)
let r2 = _mm256_mul_pd(r, r);
let p  = _mm256_fmadd_pd(r2, p5, p4);
let p  = _mm256_fmadd_pd(r2, p,  p3);
let p  = _mm256_fmadd_pd(r2, p,  p2);
let p  = _mm256_fmadd_pd(r2, p,  p1);

// c = r - r²·P,   exp(r) = 1 + r + r·c/(2-c)
let c     = _mm256_sub_pd(r, _mm256_mul_pd(r2, p));
let exp_r = _mm256_sub_pd(one,
            _mm256_sub_pd(_mm256_div_pd(_mm256_mul_pd(r, c),
                                        _mm256_sub_pd(c, two)), r));

// Step 3: 2^k via exponent field
let k_i64     = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(k_f64));
let k_shifted = _mm256_slli_epi64(k_i64, 52);
let scale     = _mm256_castsi256_pd(_mm256_add_epi64(k_shifted, one_bits));
let result    = _mm256_mul_pd(exp_r, scale);
```

## 12. References

- musl libc: [`src/math/exp.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/exp.c) and [`src/math/expf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/expf.c) (fdlibm `e_exp.c`, Sun Microsystems / K. C. Ng).
- Tang, P. T. P., *Table-driven implementation of the exponential function in IEEE floating-point arithmetic*, ACM TOMS 15(2), 1989.
- Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed., chapter 12 (elementary-function reduction-and-reconstruction recipes).
- Cody & Waite, *Software Manual for the Elementary Functions*, Prentice-Hall 1980 — the original source of the hi/lo \\(\ln 2\\) split.
- Repo source: [`src/arch/avx2/math/exp.rs`][src-avx2], [`src/arch/avx512/math/exp.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx512/math/exp.rs), [`src/arch/neon/math/exp.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/math/exp.rs), [`src/arch/consts/exp.rs`][src-consts].

## See also

- [Natural logarithm `ln`](./ln.md) — inverse function, shares the
  \\(\ln 2\\) hi/lo split.
- [Power `pow`](./pow.md) — built on top of `exp` and `ln` via
  \\(x^y = \exp(y \ln x)\\).
- [Argument-reduction taxonomy](../foundations/argument_reduction.md) for
  the general theory behind the \\(x = k\ln 2 + r\\) decomposition.
