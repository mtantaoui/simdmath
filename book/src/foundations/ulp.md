# ULP, faithful rounding, correct rounding

This chapter pins down the three vocabulary words that show up in every
precision claim in `simdmath`'s documentation: **ULP**, **faithful**, and
**correctly rounded**. Get these right and the rest of the book reads
cleanly; get them wrong and the per-function ≤ 1 ULP guarantees will sound
either too strong or too weak.

## Definition of ULP

For a real number \\(x \neq 0\\) with binary exponent \\(e\\) (so that
\\(2^{e} \le |x| < 2^{e+1}\\)), and a floating-point format with precision \\(p\\)
(24 for `f32`, 53 for `f64`),

\\[
\mathrm{ulp}(x) \\;=\\; 2^{\\,e - p + 1}.
\\]

In words: \\(\mathrm{ulp}(x)\\) is the *spacing* between adjacent representable
numbers in the binade containing \\(x\\). It doubles every time \\(|x|\\) crosses a
power of two upward, and halves on the way down. The smallest positive
normal `f32` has \\(\mathrm{ulp}(2^{-126}) = 2^{-149}\\) (one bit of mantissa).

Some references define \\(\mathrm{ulp}\\) on the *floating-point* number rather
than the real input, which gives the same value except at exact powers of
two; see Muller (2018), §2.6 for the half-page of hair-splitting.

## ULP error of an approximation

Let \\(f\\) be the mathematical (real-valued) function we are approximating and
\\(\hat{f}\\) the floating-point routine that `simdmath` actually computes.
For an input \\(\hat{x}\\), the **ULP error of approximation** is

\\[
\mathcal{E}\_{\mathrm{ulp}}(\hat{x}) \\;=\\;
\frac{\bigl|f(\hat{x}) - \hat{f}(\hat{x})\bigr|}
     {\mathrm{ulp}\bigl(f(\hat{x})\bigr)}.
\\]

It says: "how many representable steps are we away from the true value?".
Two endpoints anchor the scale:

- \\(\mathcal{E}\_{\mathrm{ulp}} = 0\\) means the result is bit-for-bit identical
  to the rounded mathematical result.
- \\(\mathcal{E}\_{\mathrm{ulp}} = 0.5\\) is the *best possible* error of any
  rounding procedure: the rounded result lies on the boundary between two
  representable numbers, and rounding either way is "tied".

## Faithful rounding

A routine is **faithfully rounded** if for every input \\(\hat{x}\\),

\\[
\mathcal{E}\_{\mathrm{ulp}}(\hat{x}) \\;<\\; 1.
\\]

Equivalently, \\(\hat{f}(\hat{x})\\) is one of the two floating-point neighbours
of the true value \\(f(\hat{x})\\). Faithful rounding is the *engineering
sweet spot*: cheap to achieve with one or two clever tricks (Cody-Waite
reduction plus a high-degree polynomial), useful enough that no real-world
client can distinguish it from correctly-rounded.

## Correctly rounded

A routine is **correctly rounded** (in round-to-nearest-even) if

\\[
\mathcal{E}\_{\mathrm{ulp}}(\hat{x}) \\;\le\\; 0.5
\\]

with ties broken to the even mantissa. There is no faster algorithm in
existence: the result is *the* floating-point number nearest to the true
real value. This is what the C99 `sqrt` and the IEEE-754 §9 elementary
functions promise. Achieving it for transcendentals is dramatically harder
than faithful rounding; the next chapter explains why `simdmath` does not
attempt it.

## The table-maker's dilemma

Why is correctly-rounded so much harder? Because the worst case is
*adversarial*: there exist inputs \\(\hat{x}\\) for which \\(f(\hat{x})\\) falls
arbitrarily close to the midpoint between two floating-point numbers, and
deciding which side of the midpoint requires evaluating \\(f\\) to *more* than
twice the working precision. This phenomenon is known as the
**table-maker's dilemma** (Kahan, 1973; popularised by Muller, 2016).

For `binary64` transcendentals the empirical worst cases are within
\\(2^{-110}\\) of a tie — a margin you can only resolve with > 110 bits of
internal precision. The CRlibm and CR-LIBM-SIMD projects (de Dinechin,
2007; Lefèvre & Muller, 2001) precomputed these worst cases for the
common transcendentals over the `binary64` domain; their tables drive the
"second step" of Ziv's strategy.

## Computing ULP error in Rust

The cleanest way to measure ULP error empirically is to exploit the
lexicographic ordering of the IEEE-754 bit encoding (see
[IEEE-754 in two slides](./ieee754.md)):

```rust
/// ULP distance between two finite, same-sign f64 values.
fn ulps_between(a: f64, b: f64) -> u64 {
    debug_assert!(a.is_finite() && b.is_finite());
    debug_assert!(a.signum() == b.signum() || a == 0.0 || b == 0.0);

    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    ai.abs_diff(bi)
}

/// ULP error of `approx` against the high-precision oracle `truth`.
fn ulp_error(approx: f64, truth: f64) -> f64 {
    if approx == truth { return 0.0; }
    if approx.is_nan() || truth.is_nan() { return f64::INFINITY; }

    // ulp(truth) reconstructed from the exponent of `truth`.
    let bits = truth.abs().to_bits();
    let exp  = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let ulp_truth = f64::from_bits(((exp - 52 + 1023) as u64) << 52);

    (approx - truth).abs() / ulp_truth
}
```

The first function returns an integer ULP distance for finite same-sign
inputs and is the right tool for ranking errors across a sweep. The second
is appropriate when comparing against a higher-precision oracle (say, an
`f128` implementation or a `mpfr` reference).

For the f32 routines we evaluate the same comparison with an `f64` oracle
and divide by `ulp(truth_as_f32)`; the f64 routines need an arbitrary
precision oracle (we use `rug::Float` with 256-bit working precision in the
sweep tests).

## Worked numerical example

Consider \\(f(x) = \sqrt{2}\\) as an `f32`. The exact value is

\\[
\sqrt{2} \\;=\\; 1.41421356237\ldots
\\]

The two nearest `f32` neighbours are

```text
0x3FB504F3 = 1.4142134189605713...   <- below
0x3FB504F4 = 1.4142135381698608...   <- above
```

The midpoint between them is \\(1.41421347856\ldots\\), which is *less* than
\\(\sqrt 2\\). Therefore the correctly-rounded `f32` result is the upper
neighbour, `0x3FB504F4`. A faithful routine is permitted to return either
of the two — the lower neighbour misses by \\(\approx 0.99\\,\mathrm{ulp}\\),
the upper neighbour by \\(\approx 0.005\\,\mathrm{ulp}\\). Both satisfy
"\\(\le 1\\) ULP" but only the upper is correctly rounded.

## Caveats and pitfalls

- **ULP at zero is not defined** by the formula above; conventions vary.
  `simdmath`'s sweep tests treat the smallest subnormal's ULP as \\(2^{-149}\\)
  (`f32`) or \\(2^{-1074}\\) (`f64`) for the purpose of the denominator.
- **Subnormal results** have fewer mantissa bits, so a "1 ULP" bound on a
  subnormal output is a *much* weaker absolute bound than 1 ULP on a normal
  output. Beware when comparing routines that disagree on subnormal
  handling.
- **Crossing a power of two**: if the true value is \\(1.0\\) and the
  approximation rounds to \\(1.0 - 2^{-25}\\), the integer-bit distance is
  \\(1\\) but the *real* distance is half an ULP because the binade just
  changed. The `ulp_error` function above uses \\(\mathrm{ulp}(\mathrm{truth})\\)
  consistently, which is the convention `simdmath` reports.
- **Definition drift in the literature**: Goldberg (1991) and Kahan (2004)
  use slightly different ULP conventions near powers of two (called
  \\(\mathrm{ulp}^*\\) vs \\(\mathrm{ulp}\\) by Muller). The discrepancy is
  \\(\le 0.5\\) in the answer; we ignore it.

## See also

- [IEEE-754 in two slides](./ieee754.md)
- [Why `≤ 1 ULP` and not `correctly rounded`](./why_not_correct.md)
- [ULP measurement methodology](../precision/methodology.md)
- Goldberg, D. (1991), *What Every Computer Scientist Should Know About
  Floating-Point Arithmetic*.
- Muller, J.-M. (2016), *Elementary Functions: Algorithms and
  Implementation*, 3rd ed., Birkhäuser, §3.
- Muller, J.-M. *et al.* (2018), *Handbook of Floating-Point Arithmetic*,
  2nd ed., §2.6 (ULP) and §12 (correctly rounded transcendentals).
- de Dinechin, F., Lauter, C., Muller, J.-M. (2007), *Fast and correctly
  rounded logarithms in double-precision*, RAIRO Theor. Inform. Appl.
