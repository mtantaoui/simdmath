# Appendix A — Polynomial coefficient tables

This appendix tabulates every polynomial coefficient used by the kernels in
`simdmath`, grouped by function. Each table is reproduced from a single source
file under
[`src/arch/consts/`](https://github.com/mtantaoui/simdmath/tree/main/src/arch/consts);
if the values here ever drift from the source, the source wins.

The coefficients are mostly ports of the minimax polynomials from
**musl libc** (which descend from Sun's fdlibm), with the same bit patterns.
Do **not** substitute `std::f32::consts` or `std::f64::consts` for them — the
hand-tuned bit patterns are what makes the ≤ 1 ULP / ≤ 2 ULP claims hold.

For the role of these polynomials in the overall algorithm, see the
per-function chapters:

- [`sin`](../functions/sin.md), [`cos`](../functions/cos.md),
  [`tan`](../functions/tan.md)
- [`asin`](../functions/asin.md), [`acos`](../functions/acos.md),
  [`atan`](../functions/atan.md), [`atan2`](../functions/atan2.md)
- [`exp`](../functions/exp.md), [`ln`](../functions/ln.md)
- [`cbrt`](../functions/cbrt.md)

## Sine — `S1..S6`

The sine kernel approximates \\(\sin(y)/y\\) on \\(y \in [-\pi/4,\\,\pi/4]\\) as an
*odd* minimax polynomial in \\(y^2\\):

\\[
\sin(y) \approx y + S_1\\,y^3 + S_2\\,y^5 + S_3\\,y^7 + S_4\\,y^9
              + S_5\\,y^{11} + S_6\\,y^{13}.
\\]

The odd structure is exact (sine is an odd function), so all coefficients of
even powers are identically zero and not stored. The f32 kernel uses the
first four coefficients in f64 working precision; the f64 kernel uses all six.

Source:
[`src/arch/consts/cos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/cos.rs)
(re-exported by `consts/sin.rs`).

| Coeff | f32 (stored as `f64`) | Approx |
|-------|-----------------------|--------|
| \\(S_1\\) | `-0.166666666416265235595` | \\(\approx -1/6\\) |
| \\(S_2\\) | `0.0083333293858894631756` | \\(\approx 1/120\\) |
| \\(S_3\\) | `-0.000198393348360966317347` | \\(\approx -1/5040\\) |
| \\(S_4\\) | `0.0000027183114939898219064` | \\(\approx 1/362880\\) |

| Coeff | f64 | Approx |
|-------|-----|--------|
| \\(S_1\\) | `-1.66666666666666324348e-01` | \\(-1/6\\) |
| \\(S_2\\) | `8.33333333332248946124e-03`  | \\(1/120\\) |
| \\(S_3\\) | `-1.98412698298579493134e-04` | \\(-1/5040\\) |
| \\(S_4\\) | `2.75573137070700676789e-06`  | \\(1/362880\\) |
| \\(S_5\\) | `-2.50507602534068634195e-08` | \\(-1/39916800\\) |
| \\(S_6\\) | `1.58969099521155010221e-10`  | \\(1/6227020800\\) |

The "Approx" column shows the coefficient that the leading Taylor expansion
would give. The minimax-fitted coefficients deviate slightly from these in
their lower bits — that small bias is what redistributes the error uniformly
over \\([-\pi/4,\\,\pi/4]\\) rather than concentrating it at the endpoints.

## Cosine — `C0/C1..C6`

Two related kernels. The f32 cosine kernel approximates \\(\cos(y)\\) directly on
\\([-\pi/4,\\,\pi/4]\\):

\\[
\cos(y) \approx 1 + C_0\\,y^2 + C_1\\,y^4 + C_2\\,y^6 + C_3\\,y^8
\qquad\text{(f32 path)}.
\\]

The f64 kernel approximates \\(\cos(y) - 1 + y^2/2\\) for sub-1-ULP accuracy near
\\(y = 0\\), where the leading \\(-y^2/2\\) is exact-ish and the polynomial supplies
the higher-order correction:

\\[
\cos(y) \approx 1 - \tfrac{y^2}{2} + C_1\\,y^4 + C_2\\,y^6 + C_3\\,y^8
              + C_4\\,y^{10} + C_5\\,y^{12} + C_6\\,y^{14}
\qquad\text{(f64 path)}.
\\]

Source: [`consts/cos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/cos.rs).

| Coeff | f32 path (`f64` storage) |
|-------|--------------------------|
| \\(C_0\\) | `-0.499999997251031003120` |
| \\(C_1\\) | `0.0416666233237390631894` |
| \\(C_2\\) | `-0.00138867637746099294692` |
| \\(C_3\\) | `0.0000243904487962774090654` |

| Coeff | f64 path | Approx |
|-------|----------|--------|
| \\(C_1\\) | `4.16666666666666019037e-02`  | \\(1/24\\) |
| \\(C_2\\) | `-1.38888888888741095749e-03` | \\(-1/720\\) |
| \\(C_3\\) | `2.48015872894767294178e-05`  | \\(1/40320\\) |
| \\(C_4\\) | `-2.75573143513906633035e-07` | \\(-1/3628800\\) |
| \\(C_5\\) | `2.08757232129817482790e-09`  | \\(1/479001600\\) |
| \\(C_6\\) | `-1.13596475577881948265e-11` | \\(-1/87178291200\\) |

## Tangent — `T0..T12`

The tangent kernel is a high-degree polynomial approximating
\\((\tan(y) - y)/y^3\\) on \\([-\pi/4,\\,\pi/4]\\). Tangent has higher curvature than
sine or cosine and needs more terms.

f32 path (degree 13 in \\(y\\), evaluated in f64). Source:
[`consts/tan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/tan.rs):

\\[
\tan(y) \approx y + T_0\\,y^3 + T_1\\,y^5 + T_2\\,y^7 + T_3\\,y^9
              + T_4\\,y^{11} + T_5\\,y^{13}.
\\]

| Coeff | f32 path |
|-------|----------|
| \\(T_0\\) | `0.333331395030791399758` |
| \\(T_1\\) | `0.133392002712976742718` |
| \\(T_2\\) | `0.0533812378445670393523` |
| \\(T_3\\) | `0.0245283181166547278873` |
| \\(T_4\\) | `0.00297435743359967304927` |
| \\(T_5\\) | `0.00946564784943673166728` |

f64 path (degree 27 in \\(y\\), used as a numerator with a denominator factor
elsewhere — see [`tan`](../functions/tan.md) for reconstruction):

| Coeff | f64 path | Hex |
|-------|----------|-----|
| \\(T_0\\)  | `3.33333333333334091986e-01`   | `0x3FD5555555555563` |
| \\(T_1\\)  | `1.33333333333201242699e-01`   | `0x3FC111111110FE7A` |
| \\(T_2\\)  | `5.39682539762260521377e-02`   | `0x3FABA1BA1BB341FE` |
| \\(T_3\\)  | `2.18694882948595424599e-02`   | `0x3F9664F48406D637` |
| \\(T_4\\)  | `8.86323982359930005737e-03`   | `0x3F8226E3E96E8493` |
| \\(T_5\\)  | `3.59207910759131235356e-03`   | `0x3F6D6D22C9560328` |
| \\(T_6\\)  | `1.45620945432529025516e-03`   | `0x3F57DBC8FEE08315` |
| \\(T_7\\)  | `5.88041240820264096874e-04`   | `0x3F4344D8F2F26501` |
| \\(T_8\\)  | `2.46463134818469906812e-04`   | `0x3F3026F71A8D1068` |
| \\(T_9\\)  | `7.81794442939557092300e-05`   | `0x3F147E88A03792A6` |
| \\(T\_{10}\\) | `7.14072491382608190305e-05` | `0x3F12B80F32F0A7E9` |
| \\(T\_{11}\\) | `-1.85586374855275456654e-05` | `0xBEF375CBDB605373` |
| \\(T\_{12}\\) | `2.59073051863633712884e-05`  | `0x3EFB2A7074BF7AD4` |

## Exponential — `P1..P5`

The `exp` kernel rebuilds \\(\exp(r)\\) on \\(\lvert r \rvert \le \ln(2)/2\\) from the
identity

\\[
\exp(r) - 1 \\;=\\; \frac{r \cdot R(r^2)}{R(r^2) - 2},
\qquad
R(r^2) = 2 - r \cdot P(r^2),
\\]

where \\(P\\) is the minimax polynomial below. This rational form is more stable
than a direct polynomial in \\(r\\) because the leading 1 is restored *exactly*
via division.

Source: [`consts/exp.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/exp.rs).

\\[
P(z) = P_1 + z\\,(P_2 + z\\,(P_3 + z\\,(P_4 + z\\,P_5)))
\qquad\text{with } z = r^2.
\\]

| Coeff | f64 | Hex | Approx |
|-------|-----|-----|--------|
| \\(P_1\\) | `1.66666666666666019037e-01`  | `0x3FC555555555553E` | \\(1/6\\) |
| \\(P_2\\) | `-2.77777777770155933842e-03` | `0xBF66C16C16BEBD93` | \\(-1/360\\) |
| \\(P_3\\) | `6.61375632143793436117e-05`  | `0x3F11566AAF25DE2C` | \\(1/15120\\) |
| \\(P_4\\) | `-1.65339022054218515266e-06` | `0xBEBBBD41C5D26BF1` | \\(-1/604800\\) |
| \\(P_5\\) | `4.13813679705723846039e-08`  | `0x3E66376972BEA4D0` | \\(\approx 1/24192000\\) |

## Logarithm — `Lg1..Lg7`

The `ln` kernel uses the substitution \\(s = f / (2 + f)\\) with \\(f = m - 1\\) for
the normalized mantissa \\(m \in [\sqrt{2}/2,\\,\sqrt{2}]\\). With this substitution

\\[
\ln\\!\left(\frac{1+s}{1-s}\right)
= 2\\,\bigl(s + L_1\\,s^3 + L_2\\,s^5 + \cdots + L_7\\,s^{15}\bigr),
\\]

so the implementation evaluates \\(L_1\\,s^2 + L_2\\,s^4 + \ldots\\) as a polynomial
in \\(s^2\\) and folds in the leading \\(s\\) outside the polynomial.

Source: [`consts/ln.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/ln.rs).

| Coeff | f64 | Hex | Approx |
|-------|-----|-----|--------|
| \\(L_1\\) | `6.666666666666735130e-01` | `0x3FE5555555555593` | \\(2/3\\) |
| \\(L_2\\) | `3.999999999940941908e-01` | `0x3FD999999997FA04` | \\(2/5\\) |
| \\(L_3\\) | `2.857142874366239149e-01` | `0x3FD2492494229359` | \\(2/7\\) |
| \\(L_4\\) | `2.222219843214978396e-01` | `0x3FCC71C51D8E78AF` | \\(2/9\\) |
| \\(L_5\\) | `1.818357216161805012e-01` | `0x3FC7466496CB03DE` | \\(2/11\\) |
| \\(L_6\\) | `1.531383769920937332e-01` | `0x3FC39A09D078C69F` | \\(2/13\\) |
| \\(L_7\\) | `1.479819860511658591e-01` | `0x3FC2F112DF3E5244` | \\(\approx 2/15\\) |

## Arc sine and arc cosine — shared `P_S* / Q_S*`

`asin` and `acos` share a Padé rational approximation \\(r(z) = z\\,p(z)/q(z)\\)
where \\(z = (1 - \lvert x \rvert)/2\\) for \\(\lvert x \rvert\\) close to 1, and
\\(z = x^2\\) otherwise. The numerator \\(p\\) and denominator \\(q\\) are the
\\(P\_{S_i}\\) / \\(Q\_{S_i}\\) tables below.

Source: [`consts/acos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/acos.rs)
(used by both `acos` and `asin`).

### f32 (degree-2 numerator, degree-1 denominator)

| Coeff | f32 |
|-------|-----|
| \\(P\_{S_0}\\) | `1.666_658_7e-1` |
| \\(P\_{S_1}\\) | `-4.274_342_2e-2` |
| \\(P\_{S_2}\\) | `-8.656_363e-3` |
| \\(Q\_{S_1}\\) | `-7.066_296_3e-1` |

### f64 (degree-5 numerator, degree-4 denominator)

| Coeff | f64 |
|-------|-----|
| \\(P\_{S_0}\\) | `1.666_666_666_666_666_574_15e-01` |
| \\(P\_{S_1}\\) | `-3.255_658_186_224_009_154_05e-01` |
| \\(P\_{S_2}\\) | `2.012_125_321_348_629_258_81e-01` |
| \\(P\_{S_3}\\) | `-4.005_553_450_067_941_140_27e-02` |
| \\(P\_{S_4}\\) | `7.915_349_942_898_145_321_76e-04` |
| \\(P\_{S_5}\\) | `3.479_331_075_960_211_675_70e-05` |
| \\(Q\_{S_1}\\) | `-2.403_394_911_734_414_218_78e+00` |
| \\(Q\_{S_2}\\) | `2.020_945_760_233_505_694_71e+00` |
| \\(Q\_{S_3}\\) | `-6.882_839_716_054_532_930_30e-01` |
| \\(Q\_{S_4}\\) | `7.703_815_055_590_193_527_91e-02` |

## Arc tangent — `ATAN_P*` (f32) and `aT[0..10]` (f64)

The f32 path uses a single odd polynomial of degree 17:

\\[
\arctan(t) \approx \sum\_{k=0}^{8} P_k \\, t^{2k+1}
\qquad\text{for } t \in [-1,\\,1].
\\]

The f64 path splits the polynomial into odd-indexed and even-indexed parts
(`s1` and `s2` in the source) for ILP and combines them on a four-range
reduction (see [`atan` chapter](../functions/atan.md)).

Source: [`consts/atan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/atan.rs).

### f32 — `ATAN_P0..P8`

| Coeff | Value |
|-------|-------|
| \\(P_0\\) | `0.999_999_871_164` |
| \\(P_1\\) | `-0.333_325_240_026` |
| \\(P_2\\) | `0.199_848_846_856` |
| \\(P_3\\) | `-0.141_548_060_419` |
| \\(P_4\\) | `0.104_775_391_987` |
| \\(P_5\\) | `-0.071_943_845_424_6` |
| \\(P_6\\) | `0.039_345_413_147_9` |
| \\(P_7\\) | `-0.014_152_348_036_2` |
| \\(P_8\\) | `0.002_398_139_012_51` |

### f64 — `aT[0..10]`

The f64 polynomial uses \\(z = t^2\\) and \\(w = z^2\\):

\\[
\begin{aligned}
s_1 &= z\\,(aT_0 + w\\,(aT_2 + w\\,(aT_4 + w\\,(aT_6 + w\\,(aT_8 + w\\,aT\_{10}))))),\\\\
s_2 &= w\\,(aT_1 + w\\,(aT_3 + w\\,(aT_5 + w\\,(aT_7 + w\\,aT_9)))),\\\\
\arctan(t) &\approx t \cdot (s_1 + s_2)\quad\text{(plus reduction offset)}.
\end{aligned}
\\]

| Coeff | Value | Approx |
|-------|-------|--------|
| \\(aT_0\\)  | `3.333_333_333_333_293_180_27e-01`  | \\(1/3\\) |
| \\(aT_1\\)  | `-1.999_999_999_987_648_324_76e-01` | \\(-1/5\\) |
| \\(aT_2\\)  | `1.428_571_427_250_346_637_11e-01`  | \\(1/7\\) |
| \\(aT_3\\)  | `-1.111_111_040_546_235_578_80e-01` | \\(-1/9\\) |
| \\(aT_4\\)  | `9.090_887_133_436_506_561_96e-02`  | \\(1/11\\) |
| \\(aT_5\\)  | `-7.691_876_205_044_829_994_95e-02` | \\(-1/13\\) |
| \\(aT_6\\)  | `6.661_073_137_387_531_206_69e-02`  | \\(1/15\\) |
| \\(aT_7\\)  | `-5.833_570_133_790_573_486_45e-02` | \\(-1/17\\) |
| \\(aT_8\\)  | `4.976_877_994_615_932_360_17e-02`  | \\(1/19\\) |
| \\(aT_9\\)  | `-3.653_157_274_421_691_552_70e-02` | \\(-1/21\\) |
| \\(aT\_{10}\\) | `1.628_582_011_536_578_236_23e-02` | \\(\approx 1/23\\) |

The deviation of \\(aT\_{10}\\) from \\(1/23 \approx 0.04348\\) is the
*minimax* signature: the higher-degree terms absorb part of the truncation
residual to flatten the error across the four-range reduction window.

## Cube root — `P0..P4`

The f64 cube root refines an initial 5-bit bit-trick estimate via a degree-4
polynomial in \\(r = (t^3 / x)\\) that approximates \\(1/\sqrt[3]{r}\\) to within
\\(2^{-23.5}\\) for \\(\lvert r - 1 \rvert < 0.1\\):

\\[
P(r) = P_0 + r\\,(P_1 + r\\,(P_2 + r\\,(P_3 + r\\,P_4))).
\\]

The f32 path uses no polynomial — Newton iterations alone hold the bound.

Source: [`consts/cbrt.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/cbrt.rs).

| Coeff | f64 | Hex |
|-------|-----|-----|
| \\(P_0\\) | `1.875_951_824_271_770_096_43`   | `0x3FFE03E60F61E692` |
| \\(P_1\\) | `-1.884_979_795_433_771_698_75`  | `0xBFFE28E092F02420` |
| \\(P_2\\) | `1.621_429_720_105_354_466_14`   | `0x3FF9F1604A49D6C2` |
| \\(P_3\\) | `-0.758_397_934_778_766_047_437` | `0xBFE844CBBEE751D9` |
| \\(P_4\\) | `0.145_996_192_886_612_446_982`  | `0x3FC2B000D4E4EDD7` |

## Cross-references

- [Foundations: polynomial evaluation](../foundations/polynomial_evaluation.md)
  — Horner, Estrin, FMA folding.
- [Foundations: argument reduction](../foundations/argument_reduction.md) —
  why the polynomials live on \\([-\pi/4,\\,\pi/4]\\) etc. and not on the full
  domain.
- [Appendix B](./B_constants.md) — the *non*-polynomial constants
  (`PIO2_HI/LO`, `LN2_HI/LO`, magic seeds) that complement these tables.
- [Appendix E](./E_bibliography.md) — citations for the original musl /
  fdlibm references.
