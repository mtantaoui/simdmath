# ULP measurement methodology

This chapter is the long-form companion to
[`docs/ulp-methodology.md`](https://github.com/mtantaoui/simdmath/blob/main/docs/ulp-methodology.md).
The short form is the contract; this chapter explains the *reasoning* behind it
and the practical machinery that backs every "≤ N ULP" claim in the book and in
the rustdoc of `simdmath`.

The single source of truth for the numbers themselves is the table at the top
of [`src/lib.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/lib.rs)
and its mirror in [Per-function tables](./tables.md). If the prose in this
chapter ever appears to disagree with that table, the table wins and this
chapter is wrong.

## What "ULP error" means here

For an ideal real-valued function \\(f : D \to \mathbb{R}\\) and a finite-precision
implementation \\(\hat{f}\\), the **unit-in-the-last-place error** at a point
\\(\hat{x}\\) is

\\[
\mathrm{ulp\text{-}error}(\hat{x})
= \left\lvert
    \frac{\hat{f}(\hat{x}) - f(\hat{x})}{\mathrm{ulp}\bigl(f(\hat{x})\bigr)}
  \right\rvert,
\\]

where \\(\mathrm{ulp}(y)\\) is the gap between \\(y\\) and its nearest distinct
floating-point neighbour of the same precision. This is the metric defined in
[Foundations: ULP, faithful rounding, correct rounding](../foundations/ulp.md);
the formal definition is due to Goldberg (1991) and Muller (2016, *Elementary
Functions*, ch. 2).

Three immediate consequences set the tone for everything that follows.

1. **0.5 ULP is the floor.** Any function whose mathematical value \\(f(\hat{x})\\)
   is not exactly representable can at best be rounded to the nearest float, and
   that rounded value already differs from the truth by up to 0.5 ULP. So a
   "≤ 1 ULP" claim says: the implementation is at most one floating-point step
   away from the correctly-rounded result, i.e. *faithful rounding*.

2. **ULP error is asymmetric near boundaries.** When \\(f(\hat{x})\\) is close to a
   power of 2, the ULP changes by a factor of 2 across the boundary. A relative
   error of \\(2^{-23}\\) that costs 0.6 ULP just below the boundary costs 1.2 ULP
   just above it. Sweep tests must therefore *cluster* samples around such
   boundaries, not just sample uniformly.

3. **NaN compares equal to NaN.** Mathematically there is no "ULP distance" to
   a NaN. By convention this crate reports a ULP distance of `0` when both the
   reference and the candidate are NaN, and `u32::MAX` / `u64::MAX` when only
   one of them is. This matches the tests in
   [`src/arch/avx2/cos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/cos.rs).

## The oracle policy for v0.1

Every measurement is a comparison \\(\hat{f}(\hat{x})\\) versus a reference
\\(f^{\star}(\hat{x})\\). The reference is what we call the **oracle**. The choice
of oracle is the most important methodological knob because it sets the
*ceiling* on the precision we can prove. For v0.1 the policy is deliberately
conservative:

| Tier | Oracle for f32 | Oracle for f64 | Used when |
|------|----------------|----------------|-----------|
| 1 | `f64::sin(x as f64) as f32` (etc.) | `f64::sin(x)` (etc.) | Default: claims of ≤ 2 ULP |
| 2 | `f64` libm reference rounded once to f32 | MPFR (planned, `mpfr-oracle` feature) | Sub-1-ULP investigations |
| 3 | Hand-computed exact value | Hand-computed exact value | Special points: 0, ±1, ±∞, NaN |

The default tier — Rust `std` math, which on Unix delegates to the platform
`libm` (glibc / musl / Apple libm) — is *not* correctly rounded, but for the
13 functions in this crate it is empirically faithful (≤ 1 ULP) on every
platform we test. That is just precise enough to validate ≤ 2 ULP claims with
margin: if our implementation differs from the oracle by ≤ 1 ULP and the
oracle differs from the truth by ≤ 1 ULP, then by the triangle inequality

\\[
\bigl\lvert \hat{f}(\hat{x}) - f(\hat{x}) \bigr\rvert
\\;\le\\; \bigl\lvert \hat{f}(\hat{x}) - f^{\star}(\hat{x}) \bigr\rvert
       + \bigl\lvert f^{\star}(\hat{x}) - f(\hat{x}) \bigr\rvert
\\;\le\\; 2\\,\mathrm{ulp}.
\\]

For the few sub-1-ULP claims (`sqrt`: correctly rounded by hardware, `abs`:
exact) the std-`libm` oracle is not strong enough and we fall back to either
hardware semantics (IEEE 754 §5.4.1 for `sqrt`) or bit-for-bit equality
(`abs`). A future tier-2 MPFR oracle, planned in `tools/mpfr-oracle`, will
allow promoting any function to a sub-1-ULP claim by computing \\(f(\hat{x})\\) in
arbitrary precision and rounding *once* to the target format. The opt-in cargo
feature is reserved as `mpfr-oracle`.

## Sample distribution

A pure uniform sweep over \\(D\\) is a poor test. Worst-case ULP almost never
occurs at a uniform-random point — it occurs at *structural* points where
argument reduction or reconstruction stresses cancellation. The crate uses a
three-layer sample \\(S = S_\text{uniform} \cup S_\text{boundary}
\cup S_\text{singular}\\) for every function:

1. **Uniform sweep** — typically \\(10^4\\) to \\(10^5\\) points uniformly distributed
   across the function's natural domain. Concrete ranges per function:

   | Function | Sweep range |
   |----------|-------------|
   | `sin`, `cos`, `tan` | \\([-2\pi,\\,2\pi]\\) and \\([-100,\\,100]\\) |
   | `asin`, `acos` | \\([-1,\\,1]\\) |
   | `atan` | \\([-100,\\,100]\\) |
   | `atan2` | \\([-10,\\,10]^2\\) grid |
   | `exp` | \\([-50,\\,50]\\) for f32, \\([-700,\\,700]\\) for f64 |
   | `ln` | logarithmic grid in \\((2^{-50},\\,2^{50})\\) |
   | `cbrt` | \\([-10^6,\\,10^6]\\) |

2. **Argument-reduction boundaries** — dense clusters around each
   \\(\hat{x}\\) where the reducer changes branch. These are exactly the points
   where catastrophic cancellation in \\(\hat{x} - k\\,(\pi/2)\\) or
   \\(\hat{x} - k\\,\ln 2\\) has the most "room" to inflate. The pattern used is a
   logarithmic ladder

   \\[
   x_0 \pm 2^{-i}, \qquad i = 0, 1, 2, \ldots, p,
   \\]

   where \\(p\\) is the working mantissa width (24 for f32, 53 for f64). Each
   ladder rung exercises a different number of cancelled bits in the reducer.
   Concrete boundary points:

   | Function family | Boundary points \\(x_0\\) |
   |-----------------|-----------------------|
   | `sin`, `cos`, `tan` | \\(0,\\;\pm\pi/4,\\;\pm\pi/2,\\;\pm\pi,\\;\pm 3\pi/2\\) |
   | `asin`, `acos` | \\(0,\\;\pm 0.5,\\;\pm 1\\) |
   | `atan` | \\(0,\\;\pm 7/16,\\;\pm 11/16,\\;\pm 19/16,\\;\pm 39/16\\) (the musl breakpoints) |
   | `exp` | \\(0,\\;\pm \ln 2 / 2,\\;\pm k\\,\ln 2\\) |
   | `ln` | \\(1,\\;\sqrt{2},\\;1/\sqrt{2}\\) and powers of 2 |
   | `cbrt` | \\(0,\\;\pm 1\\), ±powers of 8 |

3. **Special values** — the IEEE 754 / C99 §7.12 mandated points:
   \\(\pm 0\\), \\(\pm 1\\), \\(\pm \infty\\), signalling and quiet NaN, the smallest
   subnormal (`f32::MIN_POSITIVE`, `f64::MIN_POSITIVE`), the largest finite
   (`f32::MAX`, `f64::MAX`), and any function-specific singular point (e.g.
   \\(\pi/2 + k\pi\\) for `tan`).

The union \\(S\\) contains on the order of \\(10^5\\) – \\(10^6\\) points per function,
which is comfortably enough to bound the *empirical* worst case but
intentionally not enough to *prove* a tight bound — provability is what the
MPFR oracle and exhaustive f32 sweeps are for, and that is on the v0.2
roadmap.

## How the sweep tests are wired

Every backend file under
[`src/arch/<backend>/<func>.rs`](https://github.com/mtantaoui/simdmath/tree/main/src/arch)
ends in a `#[cfg(test)] mod tests` block. The conventions are:

1. **Two ULP helpers per file** — one for f32, one for f64 — using
   bitwise distance:

   ```rust
   /// Compute ULP difference between two f32 values
   fn ulp_diff_f32(a: f32, b: f32) -> u32 {
       if a.is_nan() && b.is_nan() {
           return 0;
       }
       if a.is_nan() || b.is_nan() {
           return u32::MAX;
       }
       let a_bits = a.to_bits() as i32;
       let b_bits = b.to_bits() as i32;
       (a_bits.wrapping_sub(b_bits)).unsigned_abs()
   }
   ```

   This is the *binade-aware* ULP distance: the integer reinterpretation of
   IEEE 754 has the property that adjacent representable floats differ by 1
   in this integer view, modulo the sign bit. The `as i32` cast plus
   `wrapping_sub` handles the sign-bit kink at zero correctly.

2. **One sweep test per (backend, function, precision)** — named
   `<func>_ps_ulp_sweep` for f32 and `<func>_pd_ulp_sweep` for f64. They
   share the layered sample described above and assert that the maximum
   `ulp_diff_*` over the sample is `<= MAX_ULP`.

3. **Lane independence** — every SIMD test fills a register with \\(L\\)
   *different* inputs (where \\(L\\) is the lane count for that backend) and
   checks each lane against the scalar oracle individually. A bug that
   broadcasts lane 0's result across the register would otherwise pass.

The sweep tests are tagged `#[ignore]` when they take more than a second so
that `cargo test` stays fast; the full suite is run with

```text
cargo test --release -- --include-ignored ulp_sweep
```

## Acceptance criteria

A claim of "≤ N ULP" lands in
[`src/lib.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/lib.rs) only
when **all** of the following hold:

1. The sweep test for every backend (AVX2, AVX-512, NEON) passes with
   `MAX_ULP <= N` over the layered sample \\(S\\) defined above.
2. The same test passes with the same bound on **both** precisions of that
   backend (8-lane f32 and 4-lane f64 for AVX2, etc.).
3. Special values match IEEE 754 / C99 exactly (including sign-of-zero and
   NaN propagation).
4. The lane-independence test passes.
5. The number of samples is at least \\(10^4\\) for f32 sweeps and \\(10^4\\) for f64
   sweeps; in addition, the boundary clusters of part 2 above are present.

If a backend fails any of these, its claim is *demoted* to the next integer
ULP bound and the table in `lib.rs` is updated. The current state of the table
already reflects this: `atan f32` is ≤ 3 ULP because one AVX-512 boundary
cluster touches 2.7 ULP and we round up. There is no separate per-backend
table — the worst backend wins.

## Why the bound is *worst case*, not *mean*

Numerical analysis users compose primitives. If \\(g\\) has mean error 0.3 ULP but
worst-case error 50 ULP, an iterated map \\(g^n\\) can land anywhere; only the
worst case feeds into a Wilkinson-style backward error analysis. We therefore
report the worst case observed and use the mean only as a *diagnostic* signal
when investigating regressions (a sudden jump in mean error usually flags a
miscompiled FMA).

## Reproduction

To reproduce any number in [tables.md](./tables.md):

```text
git checkout v0.1.0
cargo test --release -- --include-ignored ulp_sweep
```

The release flag matters: debug builds disable FMA fusion in the inliner and
introduce extra rounding steps that *worsen* the empirical ULP. All claims in
this crate are stated for `--release` builds with the documented `RUSTFLAGS`
target features (see
[Required CPU features and `RUSTFLAGS`](../getting_started/cpu_features.md)).

## References

- D. Goldberg, *What Every Computer Scientist Should Know About
  Floating-Point Arithmetic*, ACM Computing Surveys 23 (1), 1991.
  See [`E_bibliography.md`](../appendices/E_bibliography.md) entry [1].
- J.-M. Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed.,
  Birkhäuser, 2018. Entry [2].
- J.-M. Muller, *Elementary Functions: Algorithms and Implementation*,
  3rd ed., Birkhäuser, 2016.
- ISO/IEC 9899:1999 (C99), §7.12 *Mathematics `<math.h>`*.
- IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019.
