# Appendix B — Constant tables (π/2, ln 2, magic seeds)

This appendix lists the *non-polynomial* constants used by the kernels:
two-sum splits of irrational reference values, magic bit-trick seeds for
initial approximations, threshold values that select branches, and the
rounding-bias constants used for branchless integer extraction.

Each constant is reproduced from
[`src/arch/consts/`](https://github.com/mtantaoui/simdmath/tree/main/src/arch/consts)
and links to its source. For the polynomial coefficients see
[Appendix A](./A_coefficients.md).

A two-sum split (denoted \\(a_\text{hi} + a_\text{lo} = a\\), with
\\(\lvert a_\text{lo} \rvert \le \tfrac{1}{2}\\,\mathrm{ulp}(a_\text{hi})\\)) lets
the kernel multiply \\(a\\) by an integer \\(k\\) as the *exact* product
\\(k \cdot a_\text{hi}\\) followed by a small correction \\(k \cdot a_\text{lo}\\);
this is how Cody-Waite-style argument reduction holds onto bits that a single
\\(f64\\) cannot represent.

## Sine / cosine / tangent — argument reduction

The trig kernels reduce \\(\hat{x}\\) to \\(y \in [-\pi/4,\\,\pi/4]\\) via
\\(y = \hat{x} - n\\,(\pi/2)\\) where \\(n = \mathrm{round}\bigl(\hat{x} \cdot
\tfrac{2}{\pi}\bigr)\\). The constants below provide \\(\tfrac{2}{\pi}\\), \\(\pi/2\\)
in two-sum form, and the rounding bias used in the branchless integer cast.

Source:
[`consts/cos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/cos.rs)
(re-exported by `consts/sin.rs` and `consts/tan.rs`).

### f32 path (kernel evaluated in f64)

| Constant | Value | Hex | Role |
|----------|-------|-----|------|
| `FRAC_2_PI_32` | `0.6366197723675814` (f64) | `0x3FE45F306DC9C883` | \\(2/\pi\\) for \\(n = \mathrm{round}(\hat{x} \cdot 2/\pi)\\) |
| `PIO2_1_32`    | `1.5707963109016418` (f64) | `0x3FF921FB50000000` | High part of \\(\pi/2\\), exact to 25 bits |
| `PIO2_1T_32`   | `1.5893254773528197e-08` (f64) | `0x3E5110B4611A6263` | Tail: \\(\pi/2 - \\) `PIO2_1_32` |
| `PIO4_32`      | `0.7853981633974483` (f64) | `0x3FE921FB54442D18` | \\(\pi/4\\) for range checks |
| `TINY_COS_32`  | `2.44140625e-4` (f32)      | `0x39800000` | \\(2^{-12}\\): below this, \\(\cos(\hat{x}) \approx 1\\) |
| `MEDIUM_32`    | bit pattern `0x4DC90FDB` | — | \\(2^{28}\\,\pi/2\\): cutoff for medium-size reducer |
| `TOINT`        | `1.5 / f64::EPSILON = 6755399441055744.0` | `0x4338000000000000` | \\(1.5 \cdot 2^{52}\\): rounding bias for branchless `f64 → i64` |

### f64 path

| Constant | Value | Hex | Role |
|----------|-------|-----|------|
| `FRAC_2_PI_64` | `6.36619772367581382433e-01` | `0x3FE45F306DC9C883` | \\(2/\pi\\) |
| `PIO2_1_64`    | `1.57079632673412561417e+00` | `0x3FF921FB54400000` | First 33 bits of \\(\pi/2\\) |
| `PIO2_1T_64`   | `6.07710050650619224932e-11` | `0x3DD0B4611A626331` | Second part of \\(\pi/2\\) |
| `PIO2_2_64`    | `6.07710050630396597660e-11` | `0x3DD0B4611A600000` | Third part of \\(\pi/2\\) |
| `PIO2_2T_64`   | `2.02226624879595063154e-21` | `0x3BA3198A2E037073` | Fourth part of \\(\pi/2\\) |
| `PIO4_64`      | `7.85398163397448278999e-01` | `0x3FE921FB54442D18` | \\(\pi/4\\) |
| `TINY_COS_64`  | `2.6469779601696886e-8`      | — | \\(\sim 2^{-27}\sqrt{2}\\): below this, \\(\cos(\hat{x}) \approx 1\\) |

The four-part split of \\(\pi/2\\) gives roughly 33 + 53 + 53 + 70 = 209 effective
bits, which is enough headroom for \\(\lvert n \rvert\\) up to \\(2^{20}\\) before
the slower Payne-Hanek reducer would be needed (Payne-Hanek is not yet
implemented; see
[Argument-reduction taxonomy](../foundations/argument_reduction.md)).

### Tangent — extras

Source: [`consts/tan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/tan.rs).

| Constant | Value | Role |
|----------|-------|------|
| `TINY_TAN_64`    | `1e-300` | Below this, \\(\tan(\hat{x}) \approx \hat{x}\\) |
| `SMALL_TAN_64`   | `1e-14`  | Below this, low-order polynomial path is taken |
| `PIO4_HI_64`     | `f64::from_bits(0x3FE921FB54442D18)` | \\(\pi/4\\) high part |
| `PIO4_LO_64`     | `f64::from_bits(0x3C81A62633145C07)` | \\(\pi/4\\) low part |
| `BIG_THRESH_64`  | `f64::from_bits(0x3FE5942800000000) = 0.6743…` | Switches to "big argument" reconstruction |

### Direct multiples of \\(\pi/2\\) (small-argument shortcut)

For \\(\lvert \hat{x} \rvert\\) small enough that \\(n \in \{1,2,3,4\}\\) the reducer
can subtract a precomputed multiple directly without rounding:

| Constant | Value | Role |
|----------|-------|------|
| `C1PIO2` | `1.5707963267948966` | \\(1 \cdot \pi/2\\) |
| `C2PIO2` | `3.141592653589793`  | \\(2 \cdot \pi/2 = \pi\\) |
| `C3PIO2` | `4.71238898038469`   | \\(3 \cdot \pi/2\\) |
| `C4PIO2` | `6.283185307179586`  | \\(4 \cdot \pi/2 = 2\pi\\) |

## Arc sine / arc cosine

Source: [`consts/acos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/acos.rs)
and [`consts/asin.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/asin.rs).

### f32

| Constant | Value | Hex | Role |
|----------|-------|-----|------|
| `PIO2_HI_32` | `1.570_796_3`    | `0x3FC90FDA` | High word of \\(\pi/2\\), exact to 23 bits |
| `PIO2_LO_32` | `7.549_789_4e-8` | `0x33A22168` | Low word: \\(\pi/2 - \\) `PIO2_HI_32` |
| `TINY_THRESHOLD_32` | bit pattern `0x39800000` | — | \\(2^{-12}\\): below this, \\(\arcsin(\hat{x}) \approx \hat{x}\\) |
| `X1P_120_32` | `1.175_494_4e-38` | smallest normal `f32` | Forces IEEE *inexact* flag at \\(\hat{x} = -1\\) for `acos` |

### f64

| Constant | Value | Role |
|----------|-------|------|
| `PIO2_HI_64` | `1.570_796_326_794_896_558_00e+00` | High word of \\(\pi/2\\) |
| `PIO2_LO_64` | `6.123_233_995_736_766_035_87e-17` | Low word of \\(\pi/2\\) |
| `TINY_THRESHOLD_64` | bit pattern `0x3E40_0000_0000_0000` | \\(2^{-27}\\): below this, \\(\arcsin(\hat{x}) \approx \hat{x}\\) |

The two-sum split of \\(\pi/2\\) is what allows the reflection
\\(\arccos(\hat{x}) = \pi/2 - \arcsin(\hat{x})\\) to retain full precision when
\\(\hat{x}\\) is small (the cancellation in the subtraction is absorbed into the
hi/lo accumulator).

## Arc tangent

Source: [`consts/atan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/atan.rs).

### f32

| Constant | Value | Role |
|----------|-------|------|
| `FRAC_PI_2_32` | `1.570_796_326_794_896_619_231_3` (f32) | \\(\pi/2\\) for the \\(\hat{x} \to 1/\hat{x}\\) reflection |

### f64 — four-range reduction breakpoints

The f64 `atan` selects one of four reduction ranges based on
\\(\lvert \hat{x} \rvert\\) and adds the corresponding \\(\arctan\\) of the breakpoint
back at reconstruction. Each \\(\arctan\\) value is stored as a two-sum:

| Range id | Breakpoint \\(\lvert \hat{x} \rvert\\) | `ATANHI_i` | `ATANLO_i` |
|----------|-------------------------------------|------------|-----------|
| 0 | \\(7/16 \le \cdot < 11/16\\) | `4.636_476_090_008_060_935_15e-01` (\\(=\arctan(0.5)\_{\text{hi}}\\)) | `2.269_877_745_296_168_709_24e-17` |
| 1 | \\(11/16 \le \cdot < 19/16\\) | `7.853_981_633_974_482_789_99e-01` (\\(=\pi/4\\) hi) | `3.061_616_997_868_383_017_93e-17` |
| 2 | \\(19/16 \le \cdot < 39/16\\) | `9.827_937_232_473_290_540_82e-01` (\\(=\arctan(1.5)\_{\text{hi}}\\)) | `1.390_331_103_123_099_845_16e-17` |
| 3 | \\(\cdot \ge 39/16\\) | `1.570_796_326_794_896_558_00e+00` (\\(=\pi/2\\) hi) | `6.123_233_995_736_766_035_87e-17` |

| Constant | Value | Role |
|----------|-------|------|
| `ATAN_THRESH_0` | `7/16 = 0.4375` | Lower boundary of range 0 (below: identity) |
| `ATAN_THRESH_1` | `11/16 = 0.6875` | Upper boundary of range 0 |
| `ATAN_THRESH_2` | `19/16 = 1.1875` | Upper boundary of range 1 |
| `ATAN_THRESH_3` | `39/16 = 2.4375` | Upper boundary of range 2 |

The 7/16, 11/16, 19/16, 39/16 break-points are exact in IEEE 754 (they have
finite binary expansions), so their use as range tests does not introduce
rounding error.

## Arc tangent of two arguments — `atan2`

Source: [`consts/atan2.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/atan2.rs).

### f32

| Constant | Value | Hex | Role |
|----------|-------|-----|------|
| `PI_HI_32`     | `3.141_592_741_01`  | `0x40490FDB` | High part of \\(\pi\\) |
| `PI_LO_32`     | `-8.742_278e-8`     | `0xB3BBBD2E` | Low part: \\(\pi - \\) `PI_HI_32` |
| `FRAC_PI_2_32` | `1.570_796_370_51`  | — | \\(\pi/2\\) |
| `FRAC_PI_4_32` | `0.785_398_185_25`  | — | \\(\pi/4\\) |
| `FRAC_3_PI_4_32` | `2.356_194_496_155` | — | \\(3\pi/4\\) |
| `HUGE_RATIO_THRESHOLD_32` | `26 << 23 = 218103808` | — | \\(\lvert y/x \rvert > 2^{26}\\) ⇒ result \\(\approx \pm\pi/2\\) |

### f64

| Constant | Value | Role |
|----------|-------|------|
| `PI_HI_64` | `3.141_592_653_589_793_115_998` | High part of \\(\pi\\) |
| `PI_LO_64` | `1.224_646_799_147_353_207_17e-16` | Low part of \\(\pi\\) |
| `FRAC_PI_2_64` | `1.570_796_326_794_896_558_00` | \\(\pi/2\\) |
| `FRAC_PI_4_64` | `0.785_398_163_397_448_279_00` | \\(\pi/4\\) |
| `FRAC_3_PI_4_64` | `2.356_194_490_192_344_836_91` | \\(3\pi/4\\) |
| `HUGE_RATIO_THRESHOLD_64` | `60 << 52` | \\(\lvert y/x \rvert > 2^{60}\\) ⇒ \\(\approx \pm\pi/2\\) |

## Exponential

Source: [`consts/exp.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/exp.rs).

The `exp` reducer writes \\(\hat{x} = k\\,\ln(2) + r\\) with
\\(\lvert r \rvert \le \ln(2)/2\\) and reconstructs \\(2^k \exp(r)\\). The
\\(\ln(2)\\) split is the trig analogue: a two-sum keeps \\(k\\,\ln(2)\\) exact for
\\(\lvert k \rvert < 2^{20}\\).

| Constant | Value | Hex | Role |
|----------|-------|-----|------|
| `LN2_HI_64` | `6.93147180369123816490e-01` | `0x3FE62E42FEE00000` | High part of \\(\ln 2\\) — only 33 significant bits, so \\(k\\,\ln\_{\text{hi}} 2\\) is exact for \\(\lvert k \rvert < 2^{20}\\) |
| `LN2_LO_64` | `1.90821492927058500170e-10` | `0x3DEA39EF35793C76` | Low part: \\(\ln 2 - \\) `LN2_HI_64` |
| `LN2_INV_64` | `1.44269504088896338700e+00` | `0x3FF71547652B82FE` | \\(1/\ln 2 = \log_2 e\\) for the initial scaling \\(k = \mathrm{round}(\hat{x} \cdot \log_2 e)\\) |
| `HALF_LN2_64` | `6.93147180559945286227e-01` | `0x3FE62E42FEFA39EF` | \\(\ln 2\\) to full f64 precision (used for the rounding-bias trick) |
| `OVERFLOW_THRESH_64`  | `7.09782712893383973096e+02` | `0x40862E42FEFA39EF` | \\(\hat{x} >\\) this ⇒ \\(+\infty\\). Equals \\(\ln(2^{1024})\\) |
| `UNDERFLOW_THRESH_64` | `-7.45133219101941108420e+02` | `0xC0874910D52D3051` | \\(\hat{x} <\\) this ⇒ \\(0\\). Equals \\(\ln(2^{-1075})\\) |
| `TINY_THRESH_64`  | \\(2^{-54} \approx 5.55\text{e-}17\\) | `0x3C90000000000000` | Below this, \\(\exp(\hat{x}) - 1 \approx \hat{x}\\) |

## Logarithm

Source: [`consts/ln.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/ln.rs).

`ln` reuses the same `LN2_HI_64` / `LN2_LO_64` split (for the
\\(\ln 2\\) scaling step) plus:

| Constant | Value | Role |
|----------|-------|------|
| `LN2_HI_64` | `6.93147180369123816490e-01` | (same as `exp`) |
| `LN2_LO_64` | `1.90821492927058500170e-10` | (same as `exp`) |
| `SQRT2_64`  | `1.4142135623730951` | Mantissa-normalisation threshold: if normalised mantissa \\(> \sqrt{2}\\), halve and bump exponent |
| `TWO52_64`  | \\(2^{52} = 4503599627370496.0\\) | Subnormal-rescaling factor |

## Cube root

Source: [`consts/cbrt.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/cbrt.rs).

The cube-root kernel famously starts from a *bit trick* — divide the IEEE 754
exponent field by 3 and add a magic constant — to get a 5-bit estimate of
\\(\sqrt[3]{x}\\) with one integer add, no division. The magic constants are:

### f32 magic seeds

| Constant | Value | Role |
|----------|-------|------|
| `B1_32` | `709958130` (`0x2A5119F2`) | Normal-input bias: roughly \\((127 - 127/3 - 0.0331) \cdot 2^{23}\\) |
| `B2_32` | `642849266` (`0x265119F2`) | Subnormal-input bias: same trick after a \\(2^{24}\\) pre-scale |
| `X1P24_32` | `16777216.0` (`0x4B800000`) | \\(2^{24}\\): pre-scale for f32 subnormals |

### f64 magic seeds

| Constant | Value | Role |
|----------|-------|------|
| `B1_64` | `715094163` (`0x2A9F7893`) | Normal-input bias for the upper 32 bits of an f64 |
| `B2_64` | `696219795` (`0x297F7893`) | Subnormal-input bias |
| `X1P54_64` | `18014398509481984.0` (`0x4350000000000000`) | \\(2^{54}\\): pre-scale for f64 subnormals |
| `ROUND_MASK_64` | `0xFFFFFFFFC0000000` | Truncates a f64 to 23 significant bits before the last Newton step (so \\(t \cdot t\\) is exact) |
| `ROUND_BIAS_64` | `0x80000000` | Bias added before masking, to round-away-from-zero rather than truncate |

The seed `B1_64 = 715094163 = 0x2A9F7893` is the f64 analogue of the famous
*0x5F3759DF* fast-inverse-square-root trick: a single integer add to the
exponent half of the mantissa, exploiting that
\\(\log_2 \sqrt[3]{x} = \tfrac{1}{3}\log_2 x\\).

## See also

- [Appendix A](./A_coefficients.md) — the polynomial coefficients that
  consume these constants.
- [Foundations: compensated arithmetic](../foundations/compensated.md) —
  why two-sum splits work and why the *low* word is *not* `f64::EPSILON`.
- [Foundations: argument reduction](../foundations/argument_reduction.md) —
  how `LN2_HI/LO` and `PIO2_HI/LO` plug into Cody-Waite.
