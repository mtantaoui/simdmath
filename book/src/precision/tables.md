# Per-function worst-case ULP tables

This page reproduces the precision contract from the top of
[`src/lib.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/lib.rs) and
annotates each row with the *measurement context* — the oracle used, the size
of the sample, and the input region where the worst case shows up.

> **Single source of truth.** The `lib.rs` table is canonical. If the values on
> this page disagree with `lib.rs`, treat `lib.rs` as authoritative and open an
> issue to fix this page. The two are intended to be regenerated from the same
> sweep results.

For the methodology and definition of "ULP error" see
[ULP measurement methodology](./methodology.md). For oracle policy see the same
chapter, *§ The oracle policy for v0.1*.

## Headline contract

| Function | f32 (worst-case ULP) | f64 (worst-case ULP) | Domain |
|----------|----------------------|----------------------|--------|
| `abs`    | 0 (exact)            | 0 (exact)            | all finite + ±∞ + NaN |
| `sqrt`   | ≤ 0.5 (correctly rounded) | ≤ 0.5 (correctly rounded) | \\(x \ge 0\\) |
| `cbrt`   | ≤ 1                  | ≤ 1                  | all finite |
| `sin`    | ≤ 2                  | ≤ 2                  | all finite |
| `cos`    | ≤ 2                  | ≤ 2                  | all finite |
| `tan`    | ≤ 2                  | ≤ 2                  | excluding \\(\pi/2 + k\pi\\) |
| `asin`   | ≤ 1                  | ≤ 1                  | \\([-1,\\,1]\\) |
| `acos`   | ≤ 1                  | ≤ 1                  | \\([-1,\\,1]\\) |
| `atan`   | ≤ 3                  | ≤ 1                  | all finite |
| `atan2`  | ≤ 3                  | ≤ 2                  | all finite × all finite |
| `exp`    | ≤ 2                  | ≤ 2                  | all finite (clamped at over/underflow) |
| `ln`     | ≤ 2                  | ≤ 2                  | \\(x > 0\\) |
| `pow`    | ≤ 2                  | ≤ 2                  | per IEEE 754-2008 special cases |

## Annotated table

Each row records the *empirical* worst case (one number) plus the context in
which it was obtained. The oracle column refers to the tier names from
[methodology.md](./methodology.md#the-oracle-policy-for-v01).

### f32

| Function | Domain | Oracle | Sample size | Worst case occurs near | Empirical max | Reported claim |
|----------|--------|--------|-------------|-------------------------|---------------|----------------|
| `abs`    | all finite + ±∞ + NaN | bit-exact | full IEEE pattern + special | `n/a` (bitwise equal) | 0 | 0 (exact) |
| `sqrt`   | \\(x \ge 0\\) | hardware (IEEE 754 §5.4.1) | \\(\sim 10^5\\) uniform on \\([0, 10^{38}]\\) + boundary | n/a — hardware-rounded | 0.5 | ≤ 0.5 (correctly rounded) |
| `cbrt`   | all finite | tier 1 (`f64::cbrt as f32`) | \\(\sim 10^5\\) on \\([-10^6, 10^6]\\) + powers of 8 | binade boundary \\(\pm 8^k\\) | 1 | ≤ 1 |
| `sin`    | all finite | tier 1 (`f64::sin as f32`) | \\(\sim 10^5\\) uniform on \\([-2\pi, 2\pi]\\) + \\([-100,100]\\) + ladder near \\(k\pi/2\\) | dense ladder around \\(k\pi\\) for moderate \\(k\\) | 2 | ≤ 2 |
| `cos`    | all finite | tier 1 | same | dense ladder around \\(\pi/2 + k\pi\\) | 2 | ≤ 2 |
| `tan`    | excludes \\(\pi/2 + k\pi\\) | tier 1 | \\(\sim 10^5\\) on \\([-2\pi, 2\pi]\\) excluding ε-balls of \\(\pi/2 + k\pi\\) | within \\(\sim 2^{-12}\\) of \\(\pi/2\\) | 2 | ≤ 2 |
| `asin`   | \\([-1, 1]\\) | tier 1 | \\(\sim 10^5\\) uniform + ladder near \\(\pm 1\\) | \\(\lvert x \rvert \to 1\\) | 1 | ≤ 1 |
| `acos`   | \\([-1, 1]\\) | tier 1 | same | \\(\lvert x \rvert \to 1\\) (sqrt cancellation) | 1 | ≤ 1 |
| `atan`   | all finite | tier 1 | \\(\sim 10^5\\) on \\([-100, 100]\\) + ladder near 7/16, 11/16, 19/16, 39/16 | range boundary 11/16 | 2.7 (rounded up) | ≤ 3 |
| `atan2`  | all finite² | tier 1 | \\(\sim 10^4 \times 10^4\\) grid + axis clusters | quadrant III near \\(-\pi\\) | 2.6 (rounded up) | ≤ 3 |
| `exp`    | clamped | tier 1 (`f64::exp as f32`) | \\(\sim 10^5\\) on \\([-50, 50]\\) + ladder at \\(k\ln 2\\) | high-magnitude \\(\hat{x}\\) | 2 | ≤ 2 |
| `ln`     | \\(x > 0\\) | tier 1 | \\(\sim 10^5\\) log-uniform on \\((2^{-50}, 2^{50})\\) + ladder at 1, \\(\sqrt{2}\\) | \\(x \to 1\\) (cancellation) | 2 | ≤ 2 |
| `pow`    | per C99 | tier 1 | grid on \\((0, 100) \times (-50, 50)\\) + special-value matrix | edge of overflow ramp | 2 | ≤ 2 |

### f64

| Function | Domain | Oracle | Sample size | Worst case occurs near | Empirical max | Reported claim |
|----------|--------|--------|-------------|-------------------------|---------------|----------------|
| `abs`    | all finite + ±∞ + NaN | bit-exact | full IEEE pattern + special | n/a | 0 | 0 (exact) |
| `sqrt`   | \\(x \ge 0\\) | hardware | \\(\sim 10^5\\) on \\([0, 10^{300}]\\) + boundary | n/a — hardware | 0.5 | ≤ 0.5 (correctly rounded) |
| `cbrt`   | all finite | tier 1 (`f64::cbrt`) | \\(\sim 10^5\\) on \\([-10^{18}, 10^{18}]\\) + powers of 8 | binade boundary | 1 | ≤ 1 |
| `sin`    | all finite | tier 1 (`f64::sin`) | \\(\sim 10^5\\) on \\([-2\pi, 2\pi]\\) + \\([-100,100]\\) + ladder near \\(k\pi/2\\) | \\(k\pi\\) for \\(k \approx 30\\) | 2 | ≤ 2 |
| `cos`    | all finite | tier 1 | same | \\(\pi/2 + k\pi\\) | 2 | ≤ 2 |
| `tan`    | excludes \\(\pi/2 + k\pi\\) | tier 1 | \\(\sim 10^5\\) on \\([-2\pi, 2\pi]\\) excluding singularities | within \\(\sim 2^{-26}\\) of \\(\pi/2\\) | 2 | ≤ 2 |
| `asin`   | \\([-1, 1]\\) | tier 1 (`f64::asin`) | \\(\sim 10^5\\) uniform + ladder near \\(\pm 1\\) | \\(\lvert x \rvert \to 1\\) | 1 | ≤ 1 |
| `acos`   | \\([-1, 1]\\) | tier 1 | same | \\(\lvert x \rvert \to 1\\) | 1 | ≤ 1 |
| `atan`   | all finite | tier 1 | \\(\sim 10^5\\) + ladder at 7/16, 11/16, 19/16, 39/16 | none — uniform within 1 ULP | 1 | ≤ 1 |
| `atan2`  | all finite² | tier 1 | grid + axis clusters | \\(y \to 0^-\\) on negative \\(x\\) axis | 2 | ≤ 2 |
| `exp`    | clamped | tier 1 (`f64::exp`) | \\(\sim 10^5\\) on \\([-700, 700]\\) + ladder at \\(k\ln 2\\) | \\(\hat{x}\\) near OVERFLOW_THRESH_64 | 2 | ≤ 2 |
| `ln`     | \\(x > 0\\) | tier 1 | \\(\sim 10^5\\) log-uniform on \\((2^{-1000}, 2^{1000})\\) + 1, \\(\sqrt{2}\\) | \\(x \to 1\\) | 2 | ≤ 2 |
| `pow`    | per C99 | tier 1 | grid + special-value matrix | edge of overflow ramp | 2 | ≤ 2 |

## Reading the "worst case occurs near" column

The locations listed are the *family* of inputs where the empirical maximum
clusters, not a single point. They reflect the structural error sources:

- **`sin`, `cos`, `tan`** worsen near multiples of \\(\pi/2\\) because Cody-Waite
  argument reduction loses bits to cancellation as \\(\hat{x} - k\\,(\pi/2)\\)
  becomes small. The worst rounding for f32 is at relatively large \\(k\\) (around
  \\(10^4\\)) where the second tail term `PIO2_1T_32` starts to matter; for f64
  the Cody-Waite triple-tail handles up to \\(k \approx 2^{20}\\) before Payne-
  Hanek would be needed (and which this crate does not yet implement — see
  [Argument-reduction taxonomy](../foundations/argument_reduction.md)).
- **`asin`, `acos`** worsen near \\(\lvert x \rvert = 1\\) because the kernel
  uses \\(\sqrt{1 - x^2}\\) and the subtraction cancels. The Padé / Dekker split
  in [`acos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/acos.rs)
  is what holds the bound to ≤ 1 ULP.
- **`atan`** for f32 has its worst case at the musl breakpoints because each
  range uses its own two-sum offset (`ATANHI_i + ATANLO_i`); the f32 reducer
  uses a single polynomial across the whole domain rather than the four-range
  reduction that f64 uses, hence the looser ≤ 3 ULP bound. f64 with
  four-range reduction holds ≤ 1 ULP.
- **`exp`** worsens for \\(\hat{x}\\) near `OVERFLOW_THRESH_64` because the
  reconstruction \\(2^k \cdot \exp(r)\\) stresses the exponent field; below the
  threshold, error is uniformly under 1 ULP.
- **`ln`** worsens at \\(x \to 1\\) because \\(\ln(1) = 0\\) and tiny absolute errors
  blow up in the relative metric; the \\(s = f/(2+f)\\) substitution in
  [`ln.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/ln.rs)
  is precisely what bounds this.

## Per-backend numbers

Per-backend results are *not* tabulated here because the contract is
"the worst backend wins" — the reported claim already covers AVX2, AVX-512
and NEON simultaneously. Per-backend regression numbers are kept in the test
output and the criterion baselines, not in the published precision contract.

## See also

- [Methodology](./methodology.md) — how the numbers in this table are
  produced and what they entail.
- [Foundations: ULP](../foundations/ulp.md) — the underlying definition.
- [Appendix A — coefficient tables](../appendices/A_coefficients.md) —
  the polynomials whose error budgets feed into these ULP numbers.
- [Appendix B — constants](../appendices/B_constants.md) — the two-sum
  splits that hold the boundary error in check.
