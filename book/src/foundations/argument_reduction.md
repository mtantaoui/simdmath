# Argument-reduction taxonomy

A polynomial only approximates a function well over a *small* interval. To
evaluate \\(\sin(\hat x)\\) for \\(|\hat x| \le 10^{18}\\) we cannot fit a single
polynomial; instead we **reduce** the input to a small canonical interval
\\(r\\), evaluate the polynomial there, and **reconstruct** the answer using
identities like

\\[
\sin(\hat x) \\;=\\; \sin\bigl(r + k\\,\tfrac{\pi}{2}\bigr)
\\]

for integer \\(k\\) and \\(|r| \le \pi/4\\).

The art of argument reduction is performing the modular subtraction without
losing precision when \\(\hat x\\) is huge. This chapter is a taxonomy of the
three reduction strategies that appear in `simdmath`.

## Why naïve reduction fails

For a real \\(\hat x\\) near \\(10^{12}\\) and an `f64`-rounded constant
\\(\hat C \approx \pi/2\\), the computation

\\[
r \\;=\\; \hat x - k\\,\hat C, \qquad k = \mathrm{round}(\hat x / \hat C)
\\]

suffers because \\(\hat C\\) is wrong by \\(\le \tfrac{1}{2}\\,\mathrm{ulp}(\hat C)
\approx 2^{-53}\\), and that error is multiplied by \\(|k| \approx 10^{12} / 1.57
\approx 6.4 \cdot 10^{11}\\). The relative error in \\(r\\) is

\\[
\frac{|r - r_\text{true}|}{|r|}
\\;\approx\\;
|k| \cdot 2^{-53}
\\;\approx\\;
2^{-13}
\\]

— **only 13 bits of accuracy** in the reduced argument. Garbage in,
garbage out.

The fix is to represent \\(\pi/2\\) to *more* than `f64` precision, perform
the subtraction in extended precision, and round only at the end.

## Strategy 1: Cody-Waite reduction

**Cody-Waite** (Cody & Waite, 1980) splits the constant into two or three
floating-point parts:

\\[
C \\;=\\; C_\text{hi} + C_\text{mid} + C_\text{lo}
\\]

with \\(C_\text{hi}\\) chosen so that its low bits are *zero* — typically the
upper 30 bits of \\(\pi/2\\) as an `f64` with the bottom 23 bits zeroed out.
That makes \\(k \cdot C_\text{hi}\\) **exact** in floating point (provided
\\(|k| \le 2^{23}\\) for `f64`), so the only rounding error comes from the
much-smaller \\(k \cdot C_\text{lo}\\) term. The reduction becomes:

```text
k = round(x / C)                       # integer, possibly via cvt
r = ((x - k * C_hi) - k * C_mid) - k * C_lo
```

Each subtraction is FMA-fusible (`fnmadd(k, C_*, accumulator)`), giving a
total of three FMAs and one round-to-integer per element. The resulting
\\(r\\) is accurate to ≤ 1 ULP for \\(|x| \le\\) (cardinality of representable
integers in the high constant), typically \\(|x| \le 2^{50}\\) or so for
`f64` \\(\pi/2\\).

`simdmath` uses Cody-Waite **2-part** for `f32` trig (input clamp
\\(|x| \le 2^{18}\\)) and **3-part** for `f64` trig (input clamp
\\(|x| \le 2^{50}\\)). Inputs beyond that range hit Strategy 2.

### Worked numerics for `f64` \\(\pi/2\\) split

The constants actually used (defined in `src/arch/consts/cos.rs`, shared by
the trig kernels, and taken verbatim from musl) form a *four-part* ladder —
two truncated "hi" chunks, each paired with its full-precision tail:

```text
PIO2_1_64  = 0x3FF921FB54400000  = 1.57079632673412561417e+00   # top 33 bits of π/2 (low bits zero)
PIO2_1T_64 = 0x3DD0B4611A626331  = 6.07710050650619224932e-11   # π/2 − PIO2_1, rounded
PIO2_2_64  = 0x3DD0B4611A600000  = 6.07710050630396597660e-11   # top 33 bits of that tail
PIO2_2T_64 = 0x3BA3198A2E037073  = 2.02226624879595063154e-21   # remainder
```

Because `PIO2_1` and `PIO2_2` end in a run of zero bits, the products
\\(k \cdot \mathtt{PIO2\\_1}\\) and \\(k \cdot \mathtt{PIO2\\_2}\\) are exact for
the \\(k\\) range allowed by the input clamp, so all the rounding lives in the
tiny `*T` tail terms. The `f64` kernels subtract \\(k \cdot \mathtt{PIO2\\_1}\\),
then \\(k \cdot \mathtt{PIO2\\_2}\\) and \\(k \cdot \mathtt{PIO2\\_2T}\\) with an
explicit cancellation-recovery step (musl's "second-iteration" `__rem_pio2`
form); the `f32`-via-`f64` paths get away with the two-part
`(PIO2_1_32, PIO2_1T_32)` pair.

## Strategy 2: Payne-Hanek reduction

For inputs of *truly* huge magnitude — anywhere near \\(10^{15}\\) for `f64` —
the Cody-Waite three-part split runs out of bits. **Payne-Hanek**
(Payne & Hanek, 1983) takes the structural view: storing \\(2/\pi\\) as a
*precomputed table* of 1024 (or so) bits, indexing into the table starting
at the bit position determined by the *exponent* of \\(\hat x\\), and
multiplying out the relevant 200 bits with full extended precision.

Algorithm sketch:

1. Let \\(e\\) be the binary exponent of \\(\hat x\\).
2. Look up bits \\([e - 70, e + 130)\\) of \\(2/\pi\\) from a hardcoded table
   (Bailey's "long \\(2/\pi\\)" with ~ 1024 hex digits).
3. Multiply the 53-bit mantissa of \\(\hat x\\) by the 200-bit table chunk
   into a multi-word integer.
4. Take the low 64 bits of the integer part to get \\(k \mod 4\\) (which
   quadrant) and the high 64 bits of the *fractional* part to get \\(r\\).
5. Multiply \\(r\\) by \\(\pi/2\\) in double-double to get the reduced argument
   in radians.

This is far more expensive than Cody-Waite (≈ 30 cycles of scalar work)
and `simdmath` invokes it on only the rare inputs that need it. The hot
path tests \\(|\hat x| \le 2^{50}\\) with a single mask compare; on the
common case the Payne-Hanek slow path is dead-code-eliminated. On the
rare-case lane we **gather** the table chunks in a scalar fall-back —
the SIMD backends do not vectorise Payne-Hanek, since the input
distribution it caters to does not arise in batch form.

## Strategy 3: Hi/lo split of `ln 2` for `exp` and `ln`

`exp` and `ln` use the relation \\(\exp(x) = 2^{x / \ln 2}\\). The same
hi/lo trick used for trig applies, but the constant is \\(\ln 2\\) instead
of \\(\pi/2\\):

```text
LN2_HI = 0x3FE62E42FEFA3800   = 0.69314718055994528622...   (low 22 bits zero)
LN2_LO = 0x3D2EF35793C76730   = 5.497923018708371e-12
```

For `exp`, the integer part of \\(\hat x / \ln 2\\) becomes the binary
exponent of the answer (a free `vscalefps` on AVX-512!), and the
residue \\(r = \hat x - k \cdot \mathrm{LN2}_\text{hi} - k \cdot
\mathrm{LN2}_\text{lo}\\) is fed into a polynomial approximation of
\\(2^{r/\ln 2}\\) over \\(|r| \le \tfrac{\ln 2}{2}\\).

For `ln`, we go the other direction: extract the binary exponent \\(e\\) of
the input, divide the mantissa \\(m\\) by 1, fit \\(\ln(m)\\) for \\(m \in
[\sqrt 2 / 2, \sqrt 2]\\), and reconstruct
\\(\ln(\hat x) = e \cdot \ln 2 + \ln(m)\\) — using **the same hi/lo split of
\\(\ln 2\\)** in the multiplication \\(e \cdot \ln 2\\).

## Strategy 4: Bit-level scaling for `cbrt`

`cbrt` is a special case: instead of subtracting a constant, we exploit the
identity

\\[
\sqrt[3]{x} \\;=\\; 2^{e/3}\\,\sqrt[3]{m}, \qquad
x = 2^{e}\\,m,\quad m \in [1, 2),\quad e = 3q + s,\\;\\; s \in \{0, 1, 2\}.
\\]

Reduction is performed at the **bit level**: extract the biased exponent,
divide by 3 (an integer divide that vectorises on AVX-512 and emulates
cheaply on AVX2/NEON via the multiply-by-magic-constant trick), and
recombine into a new `f64` whose exponent field is \\(\lfloor e/3 \rfloor +
\mathrm{bias}\\) and whose mantissa is the original mantissa multiplied by
the appropriate \\(2^{s/3}\\) factor. The polynomial then approximates
\\(\sqrt[3]{m \cdot 2^s}\\) on \\([1, 8)\\), a tiny interval where degree 7 is
overkill.

There is no Cody-Waite split here because the "constant" is \\(2^{1/3}\\)
itself, multiplied as part of the polynomial fit. The reduction is exact
in real arithmetic (it is just bit shuffling), so \\(r\\) enters the
polynomial at full precision.

## Reduction strategy by function

| Function     | Strategy                              | Reduced interval       | Special-case threshold |
|--------------|---------------------------------------|------------------------|------------------------|
| `sin`, `cos` | Cody-Waite 3-part by \\(\pi/2\\)          | \\([-\pi/4, \pi/4]\\)      | Payne-Hanek for \\(\|x\| > 2^{50}\\) |
| `tan`        | Cody-Waite 3-part by \\(\pi/2\\)          | \\([-\pi/4, \pi/4]\\)      | Payne-Hanek for \\(\|x\| > 2^{50}\\) |
| `asin`,`acos`| Two-arm split at \\(\pm 0.5\\)            | \\([0, 0.5]\\) or \\([0.5, 1]\\) | none, finite domain  |
| `atan`       | Range partition at \\(1\\), \\(\sqrt{3}/3\\)  | \\([0, \tan(\pi/12)]\\)    | none, finite "infinity-side" |
| `atan2`      | Quadrant + `atan` reduction           | as `atan`              | \\(\pm 0\\), \\(\pm\infty\\) paths |
| `exp`        | Hi/lo split by \\(\ln 2\\)                | \\([-\ln 2 / 2, \ln 2 / 2]\\) | over/underflow guard |
| `ln`         | Mantissa/exponent split with hi/lo \\(\ln 2\\) | \\([\sqrt 2/2, \sqrt 2]\\) | \\(\le 0\\) → NaN/-∞     |
| `pow`        | DD `ln(x)` then DD `× y` then `exp`   | as `exp`               | many edge cases       |
| `sqrt`       | Hardware `vsqrtps` / `fsqrt`          | exact                  | none                  |
| `cbrt`       | Bit-level exponent ÷ 3                | \\([1, 8)\\)               | \\(0, \pm \infty\\) direct |

## Precision cost summary

For each strategy, here is the *additional* error contributed by the
reduction step itself, on top of the polynomial error:

| Strategy                    | Reduction error (typical)       |
|-----------------------------|---------------------------------|
| Cody-Waite 2-part (`f32`)   | \\(\le 0.05\\) ULP, \\(\|x\| \le 2^{18}\\) |
| Cody-Waite 3-part (`f64`)   | \\(\le 0.1\\) ULP, \\(\|x\| \le 2^{50}\\)  |
| Payne-Hanek (rare path)     | \\(\le 0.5\\) ULP for any finite \\(x\\) |
| Hi/lo split of \\(\ln 2\\)      | \\(\le 0.05\\) ULP, \\(\|x\| \le 1024\\) |
| Range partition (`atan`)    | exact (table lookup)            |
| Bit-level scaling (`cbrt`)  | exact (no rounding)             |

The reduction step is therefore **never** the dominant error source in any
of `simdmath`'s functions. The end-to-end ≤ 1 ULP guarantee is set by the
polynomial evaluation, not by reduction.

## SIMD considerations

Cody-Waite vectorises trivially: three FMAs per element, no branches, lane-
independent. The only non-trivial step is computing \\(k = \mathrm{round}(\hat
x / C)\\), which on AVX-512 is `_mm512_cvtps_epi32(_mm512_mul_round_ps(...))`
in a single instruction.

Payne-Hanek does **not** vectorise: the table lookup is exponent-dependent
(per-lane), the multiply is a multi-word scalar operation, and the
distribution of inputs that need it is sparse. `simdmath` handles this with
a mask and a scalar fall-back loop.

Bit-level scaling for `cbrt` vectorises perfectly. The integer "÷ 3" is
\\(\lfloor (e + 2) \cdot \mathtt{0xAAAAAAAB} \rfloor\\) — a single multiply-
high — and the exponent reassembly is one `vorps` per element.

## See also

- [Compensated arithmetic: two-sum and Dekker product](./compensated.md)
- [Polynomial evaluation: Horner, Estrin, FMA](./polynomial_evaluation.md)
- [Sine `sin`](../functions/sin.md), [Natural exponential `exp`](../functions/exp.md),
  [Cube root `cbrt`](../functions/cbrt.md)
- [Constant tables (π/2, ln 2, magic seeds)](../appendices/B_constants.md)
- Cody, W. J., Waite, W. M. (1980), *Software Manual for the Elementary
  Functions*, Prentice-Hall.
- Payne, M. H., Hanek, R. N. (1983), *Radian reduction for trigonometric
  functions*, ACM SIGNUM Newsletter 18(1).
- Ng, K. C. (1992), *Argument reduction for huge arguments: good to the
  last bit*, SunPro Tech. Report.
- Muller, J.-M. (2016), *Elementary Functions*, 3rd ed., §11.
