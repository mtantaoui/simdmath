# Compensated arithmetic: two-sum and Dekker product

Floating-point addition and multiplication each lose, at most, one rounding
worth of information per operation. **Compensated arithmetic** is the
collection of tricks that *recover* that lost rounding as a second, smaller
floating-point number, so that the pair \\((s_\text{hi}, s_\text{lo})\\)
together represents twice the working precision.

`simdmath` uses these techniques sparingly but crucially: the tail of `pow`,
the high-precision split of \\(\pi/2\\) in trig argument reduction, and the
double-double multiplications inside `cbrt` all rely on the algorithms in
this chapter.

## TwoSum: exact addition

**Knuth's TwoSum** (Knuth, 1969, §4.2.2) computes a representable pair
\\((s, e)\\) such that

\\[
a + b \\;=\\; s + e, \qquad |e| \le \tfrac{1}{2}\\,\mathrm{ulp}(s),
\\]

with \\(s = \mathrm{fl}(a + b)\\) being the ordinary floating-point sum and
\\(e\\) being the exact rounding error. There are no preconditions on the
inputs other than no overflow.

```rust
/// Knuth's TwoSum: a + b = s + e exactly (in real arithmetic), where
/// s is the rounded sum and e is the rounding error.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s  = a + b;
    let bp = s - a;
    let ap = s - bp;
    let eb = b - bp;
    let ea = a - ap;
    (s, ea + eb)
}
```

Six operations, no branches, no comparisons, vectorises perfectly across
all three of `simdmath`'s backends. The proof of correctness (Knuth, 1969)
fits on half a page and relies only on RN-TE rounding.

A faster variant, **FastTwoSum** (Dekker, 1971), shaves three operations
but requires \\(|a| \ge |b|\\):

```rust
#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    debug_assert!(a.abs() >= b.abs() || a == 0.0);
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}
```

`simdmath` uses FastTwoSum only when one operand is a known constant
(like \\(\pi/2_\text{hi}\\)) so the precondition is statically checkable.

## The Veltkamp split

Before TwoProd we need to **split** a 53-bit `f64` into two 26-bit halves
whose unrounded product fits exactly in a single `f64`. Veltkamp's split
(Veltkamp, 1968; popularised by Dekker, 1971) does this in three operations:

```rust
/// Veltkamp split for f64. Returns (a_hi, a_lo) with
///   a = a_hi + a_lo
///   a_hi has at most 27 leading bits of mantissa
///   a_lo has at most 26 trailing bits.
#[inline]
fn split(a: f64) -> (f64, f64) {
    const C: f64 = (1u64 << 27) as f64 + 1.0;   // 2^27 + 1
    let t   = C * a;
    let ahi = t - (t - a);
    let alo = a - ahi;
    (ahi, alo)
}
```

The magic constant \\(C = 2^{\lceil p/2 \rceil} + 1\\) is \\(2^{27} + 1\\) for
`f64` and \\(2^{12} + 1\\) for `f32`. The 27 + 26 split is *deliberately*
asymmetric to leave one bit of headroom when multiplying.

Veltkamp's split fails on inputs near the overflow threshold (the
intermediate \\(C \cdot a\\) overflows), so `simdmath` only invokes it on
intermediate quantities of bounded magnitude.

## TwoProd: exact multiplication

Given Veltkamp's split, **Dekker's exact product** (Dekker, 1971) computes
\\((p, e)\\) with \\(a \cdot b = p + e\\) exactly:

```rust
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let p = a * b;
    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, e)
}
```

That is **17 floating-point operations** including the two splits — costly
enough that you only invoke it where you need it.

### The FMA shortcut

Every architecture `simdmath` supports has a fused multiply-add. With FMA,
the entire ceremony collapses into:

\\[
e \\;=\\; \mathrm{fma}(a, b, -p), \qquad p = a \cdot b
\\]

because \\(\mathrm{fma}(a, b, -p)\\) computes \\(a \cdot b - p\\) with a *single*
rounding, and the rounding error of the original \\(a \cdot b\\) is itself
representable. So:

```rust
#[inline]
fn two_prod_fma(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);   // == fma(a, b, -p)
    (p, e)
}
```

Three operations instead of seventeen. `simdmath` uses
`two_prod_fma` everywhere — the AVX2/AVX-512 backends call
`_mm256_fmadd_pd(a, b, _mm256_xor_pd(p, signmask))` and the NEON
backend calls `vfmaq_f64(neg_p, a, b)`.

## Double-double arithmetic

A "double-double" (DD) is an unevaluated pair \\((x_\text{hi}, x_\text{lo})\\)
of `f64`s with \\(|x_\text{lo}| \le \tfrac{1}{2} \mathrm{ulp}(x_\text{hi})\\).
Together they represent a number to ≈ 106 bits of precision. The
operations on DDs are built from TwoSum and TwoProd; here is DD multiplication
(Linnainmaa, 1981; Bailey, 1995):

```text
DD-mul(a_hi, a_lo, b_hi, b_lo):
    (p, e) = two_prod_fma(a_hi, b_hi)
    e      = e + a_hi * b_lo + a_lo * b_hi
    (s_hi, s_lo) = fast_two_sum(p, e)
    return (s_hi, s_lo)
```

The trailing \\(a_\text{lo} \cdot b_\text{lo}\\) contribution is dropped — it is
of order \\(2^{-106}\\) and would add three more FMAs for no observable gain.

## How `simdmath` uses these

### `pow(x, y)`

The hardest function in the library. For positive \\(x\\) we evaluate

\\[
x^y \\;=\\; \exp\bigl(y \cdot \ln x\bigr).
\\]

The intermediate \\(y \cdot \ln x\\) must be computed to *more* than `f64`
precision, otherwise a 1-ULP error in \\(\ln x\\) gets amplified by \\(|y|\\) —
disastrous when \\(|y|\\) is, say, \\(10^{15}\\). The recipe in the AVX-512 backend
is:

1. Compute \\(\ln x\\) as a double-double \\((\ell_\text{hi}, \ell_\text{lo})\\)
   using a hi/lo split of \\(\ln 2\\) and TwoProd-FMA inside the polynomial.
2. Form the product \\(y \cdot \ell\\) as a DD via the algorithm above.
3. Reduce the DD result modulo \\(\ln 2\\) for `exp`, splitting again so the
   integer part of \\(y \ell / \ln 2\\) goes into the exponent of the final
   answer and the fraction stays as a DD.

The result is ≤ 1 ULP across the full domain \\(x \in (0, \infty),\
y \in [-1024, 1024]\\).

### Trigonometric argument reduction

In [Argument-reduction taxonomy](./argument_reduction.md) we will see that
reducing \\(\hat x \mod \pi/2\\) requires representing \\(\pi/2\\) to *more* than
`f64` precision. The Cody-Waite trick is just a manual application of
TwoSum-style cancellation: split \\(\pi/2 = c_1 + c_2 + c_3\\) where \\(c_1\\) has
17 zero trailing bits, \\(c_2\\) has 30 zero trailing bits, and the residual
\\(c_3\\) captures the rest. Subtracting \\(k c_1\\), \\(k c_2\\), \\(k c_3\\) in
descending order of magnitude is equivalent to a hand-unrolled DD subtract.

### `cbrt` and `sqrt` correction step

Both root functions perform one Newton iteration in DD precision to convert
a Halley- or Newton-quality starting estimate (≈ 28 bits of precision)
into a final ≤ 0.5 ULP result. The DD multiply-and-add inside the iteration
is the hot loop; it amortises four FMAs per element and is the reason
`simdmath`'s `cbrt` is competitive with `vcbrtps` from Intel SVML.

## Vectorisation notes

All of TwoSum, FastTwoSum, Veltkamp split, and TwoProd-FMA are
*data-parallel*: they contain no branches, no comparisons, and no
cross-lane operations. They map one-to-one onto AVX2 (`vfmadd231pd`,
`vsubpd`), AVX-512 (zmm equivalents), and NEON (`vfmaq_f64`, `vsubq_f64`).
The only thing that does **not** vectorise is the implicit "if your inputs
are subnormal, the magic constants overflow" assumption, which we
discharge by clamping inputs before the first split.

## See also

- [IEEE-754 in two slides](./ieee754.md)
- [Polynomial evaluation: Horner, Estrin, FMA](./polynomial_evaluation.md)
- [Argument-reduction taxonomy](./argument_reduction.md)
- [Power `pow`](../functions/pow.md)
- Dekker, T. J. (1971), *A floating-point technique for extending the
  available precision*, Numer. Math. 18(3), 224–242.
- Knuth, D. E. (1969), *The Art of Computer Programming*, Vol. 2, §4.2.2.
- Veltkamp, G. W. (1968), *ALGOL procedure for working with double-length
  arithmetic*, EWD-track report.
- Bailey, D. H. (1995), *A Fortran-90 double-double library*.
- Boldo, S., Muller, J.-M. (2011), *Exact and approximated error of the
  FMA*, IEEE TC 60(2), 157–164.
- Hida, Y., Li, X. S., Bailey, D. H. (2001), *Algorithms for quad-double
  precision floating-point arithmetic*, ARITH-15.
