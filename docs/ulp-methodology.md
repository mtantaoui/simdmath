# ULP measurement methodology

This crate's accuracy claims (≤ N ULP) are obtained by direct comparison against
a reference oracle on a curated input distribution. This document defines what
"≤ N ULP" means here and how each function is measured, so the numbers in
function-level documentation are reproducible.

## Definition

For a real-valued function $f : D \to \mathbb{R}$ and a finite-precision
implementation $\hat{f}$, the **unit in the last place** error at input $x$ is

$$
\mathrm{ulp\text{-}error}(x)
= \left\lvert \frac{\hat{f}(x) - f(x)}{\mathrm{ulp}(f(x))} \right\rvert,
$$

where $\mathrm{ulp}(y)$ is the gap between $y$ and the next representable
floating-point number of the same precision (Goldberg, 1991; Muller, 2016).
A claim of "≤ N ULP" in this crate means

$$
\max_{x \in S} \mathrm{ulp\text{-}error}(x) \le N
$$

over a finite sample $S \subset D$ described below.

## Reference oracle

By default the reference $f(x)$ is taken from Rust's standard library
(`f32::sin`, `f64::exp`, …), which delegates to the platform `libm` and is
typically faithfully rounded (≤ 1 ULP). Where higher confidence is required —
e.g. to validate a sub-1-ULP claim — a higher-precision oracle is used:

- **Optional MPFR oracle**: a small helper crate (planned, see
  `tools/mpfr-oracle`) computes the round-to-nearest IEEE-754 result by
  evaluating $f$ in arbitrary precision and rounding once. Tests can opt in
  with the `mpfr-oracle` cargo feature.
- For functions where Rust std and musl libm agree to within 1 ULP across the
  sample, we treat the std value as the reference.

## Sample distribution

Tests use a layered sample $S = S_\text{uniform} \cup S_\text{boundary} \cup
S_\text{singular}$:

1. **Uniform sweep** — $\sim 10^5$ points uniformly distributed in the function's
   primary domain (e.g. $[-2\pi, 2\pi]$ for trig, $[-50, 50]$ for $\exp$).
2. **Argument-reduction boundaries** — dense clusters around each $k\pi/2$
   for trig functions, around $k \ln 2$ for $\exp$, around $1$ for $\ln$. A
   logarithmic ladder $\{x_0 + 2^{-i}\}_{i=0}^{52}$ exposes catastrophic
   cancellation near these points.
3. **Special values** — $\pm 0$, $\pm 1$, $\pm \infty$, NaN, the smallest
   subnormal, the largest finite, and the IEEE-754 boundary cases listed in
   the C99 standard for each function (`§7.12`).

## Reporting

Per-function precision tables report the **worst-case** ULP error observed
over $S$, not the mean. Worst-case is the relevant guarantee for
numerical-analysis users; mean error is reported only as a supplementary
diagnostic when relevant.

## Reproduction

The sweep tests live in `#[cfg(test)] mod tests` blocks at the bottom of each
`src/arch/<backend>/<func>.rs` file. They can be re-run with:

```text
cargo test --release -- --include-ignored ulp_sweep
```

The MPFR-backed oracle (when enabled) requires the `gmp` and `mpfr` system
libraries.

## References

- D. Goldberg, *What Every Computer Scientist Should Know About
  Floating-Point Arithmetic*, ACM Computing Surveys, 1991.
- J.-M. Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed.,
  Birkhäuser, 2018.
- J.-M. Muller, *Elementary Functions: Algorithms and Implementation*,
  3rd ed., Birkhäuser, 2016.
- ISO/IEC 9899:1999 (C99), §7.12 *Mathematics `<math.h>`*.
