# Tangent `tan`

This chapter documents `tan(x)`. Like sin and cos it begins with a
Cody-Waite reduction to \\([-\pi/4, \pi/4]\\), but the reconstruction step
swaps `tan` with \\(-1/\tan\\) — i.e. with the cotangent — for odd quadrants.
The underlying kernel is an odd polynomial approximation of \\(\tan(y)\\).

## 1. Mathematical definition

\\[
\tan(x) \\;=\\; \frac{\sin(x)}{\cos(x)}
\\;=\\; x + \frac{x^3}{3} + \frac{2 x^5}{15} + \frac{17 x^7}{315} + \cdots
\\]

Tangent is **odd**, **\\(\pi\\)-periodic** (not \\(2\pi\\) — half the period of
sin and cos), and has poles at \\(x = \pi/2 + k\pi\\) for every integer \\(k\\).

## 2. Domain and range

- **Domain**: \\(\mathbb{R} \setminus \{\pi/2 + k\pi : k \in \mathbb{Z}\}\\).
  IEEE 754 has no exact representation of \\(\pi/2\\), so the **true singular
  set is unrepresentable**: every input is a finite distance from the
  pole, and the implementation simply returns the (very large) finite
  value of \\(\tan\\) at that representable nearby point.
- **Range**: \\(\mathbb{R}\\) — every real value is attained on each
  fundamental period.

## 3. Special values

| Input         | Output  | Source         |
|---------------|---------|----------------|
| `+0.0`        | `+0.0`  | C99 Sec. F.10.1.7  |
| `-0.0`        | `-0.0`  | C99 Sec. F.10.1.7  |
| `+∞`          | `NaN`   | C99 Sec. F.10.1.7  |
| `-∞`          | `NaN`   | C99 Sec. F.10.1.7  |
| `NaN`         | `NaN`   | IEEE 754-2008  |
| \\(\lvert x\rvert < 10^{-300}\\) (f64) | `x` | tiny-arg shortcut |
| \\(\pi/2 + k\pi\\) (representable) | very large finite | unreachable singularity |

The unreachable-singularity behaviour is worth emphasising: code that
relies on `tan` returning `+∞` near a pole will **not** see it. Use
`atan2(sin(x), cos(x))` or check `cos(x) ≈ 0` separately if pole
detection is required.

## 4. Algorithm overview

1. **Argument reduction.** Same Cody-Waite reduction as
   [sin](./sin.md#5-argument-reduction); produces \\(y \in [-\pi/4, \pi/4]\\)
   and quadrant index \\(n\\).
2. **Kernel evaluation.** Compute \\(\tan(y)\\) via a polynomial in \\(y^2\\)
   times \\(y\\).
3. **Reconstruction.** Use \\(n \bmod 2\\) (not \\(n \bmod 4\\) — tan has period
   \\(\pi\\)) to choose between \\(\tan(y)\\) and \\(-1/\tan(y)\\).

## 5. Argument reduction

Identical to sin/cos. The constants come from
[`arch/consts/cos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/cos.rs)
(re-exported by [`arch/consts/tan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/tan.rs)):
`FRAC_2_PI_*`, `PIO2_1_*`, `PIO2_1T_*`, `PIO2_2_*`, `PIO2_2T_*`, `TOINT`.

## 6. Polynomial approximation

### f32 kernel (`__tandf`)

Approximates \\(\tan(y)\\) on \\([-\pi/4, \pi/4]\\) as a degree-13 polynomial
in \\(y\\):

\\[
\hat\tan(y) \\;\approx\\; y + T_0 y^3 + T_1 y^5 + T_2 y^7 + T_3 y^9 + T_4 y^{11} + T_5 y^{13}
\tag{6.1}
\\]

| Constant | Value (f64) |
|----------|-----------------------------------------|
| `T0_32`  | \\(0.333331395030791399758\\)               |
| `T1_32`  | \\(0.133392002712976742718\\)               |
| `T2_32`  | \\(0.0533812378445670393523\\)              |
| `T3_32`  | \\(0.0245283181166547278873\\)              |
| `T4_32`  | \\(0.00297435743359967304927\\)             |
| `T5_32`  | \\(0.00946564784943673166728\\)             |

The polynomial is evaluated in **f64 precision** for f32 inputs (same
promote/demote pattern as sin/cos).

### f64 kernel (`__tan`)

A degree-27 polynomial in \\(y\\) (degree 12 in \\(y^2\\) inside the
\\(y + y^3 P(y^2)\\) form) — the longest kernel in this crate:

\\[
\hat\tan(y) \\;\approx\\; y + y^3 \cdot P(y^2)
\tag{6.2}
\\]

with \\(P(z) = T_0 + T_1 z + T_2 z^2 + \cdots + T\_{12} z^{12}\\). The
coefficients (from [`arch/consts/tan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/tan.rs))
are stored as `f64::from_bits(...)` literals so their bit pattern matches
musl exactly:

| Constant | Bit pattern        | Approx value         |
|----------|--------------------|----------------------|
| `T0_64`  | `0x3FD5555555555563` | \\(1/3 \approx 0.3333\\)   |
| `T1_64`  | `0x3FC111111110FE7A` | \\(2/15 \approx 0.1333\\)  |
| `T2_64`  | `0x3FABA1BA1BB341FE` | \\(17/315\\)              |
| `T3_64`  | `0x3F9664F48406D637` | \\(62/2835\\)             |
| ...      | ...                  | ...                   |
| `T11_64` | `0xBEF375CBDB605373` | \\(-1.86 \cdot 10^{-5}\\) (negative — minimax correction) |
| `T12_64` | `0x3EFB2A7074BF7AD4` | \\(2.59 \cdot 10^{-5}\\)  |

Beyond the simple Taylor coefficients (\\(1/3, 2/15, 17/315, …\\)), the
higher terms are **minimax-optimised** rather than analytical, so a few
of them have unexpected signs.

Around \\(|y| \to \pi/4\\) the direct polynomial degrades: \\(\tan\\) and its
derivatives grow quickly toward the interval edge, so the
\\(y + y^3 P(y^2)\\) form would need a much higher degree to hold the error
budget there, and the downstream \\(-1/\tan(y)\\) reciprocal amplifies
whatever error remains. musl's `__tan.c` therefore switches to a
"big-argument" branch that re-expresses the result through the *reflected*
argument \\(\pi/4 - |y|\\) — small again, where the polynomial is at its
best — and reconstructs via a guarded identity that dodges the
cancellation. The crossover point, `BIG_THRESH_64 = 0.6743...` (about
\\(0.86 \cdot \pi/4\\)) from
[`arch/consts/tan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/tan.rs),
is musl's empirically chosen boundary between the two regimes, and the
SIMD port mirrors it verbatim.

## 7. Reconstruction

For tan, the period is \\(\pi\\), not \\(2\pi\\). Only the parity of \\(n\\) matters:

| \\(n \bmod 2\\) | \\(\tan(x)\\)        |
|-------------|-------------------|
| \\(0\\)         | \\(\tan(y)\\)         |
| \\(1\\)         | \\(-1/\tan(y) = -\cot(y)\\) |

The cotangent identity comes from \\(\tan(y + \pi/2) = -\cot(y)\\). The
implementation computes both \\(\tan(y)\\) and its reciprocal
\\(-1/\tan(y)\\) unconditionally, and a single blend chooses based on bit 0
of \\(n\\):

```rust,ignore
let neg_recip = _mm256_div_pd(_mm256_set1_pd(-1.0), tan_y);
let result    = _mm256_blendv_pd(tan_y, neg_recip, n_is_odd);
```

The reciprocal path is the source of the "very large finite" output near
\\(\pi/2\\): when \\(|y|\\) is small but \\(n\\) is odd, \\(-1/\tan(y)\\) is
correspondingly large.

## 8. Per-precision differences (f32 vs f64)

| Aspect             | f32                              | f64                             |
|--------------------|----------------------------------|---------------------------------|
| Internal precision | computed in f64 (promote/demote) | native f64                      |
| Kernel degree      | 13 (in \\(y\\))                      | 27 (in \\(y\\); 12 in \\(y^2\\))     |
| Cody-Waite         | 25-bit `PIO2_1`                  | 33-bit `PIO2_1` + `PIO2_2`      |
| Worst-case ULP     | ≤ 2                              | ≤ 2                             |

## 9. Per-backend differences

| Backend  | f32 lanes | f64 lanes | Selection idiom            |
|----------|-----------|-----------|-----------------------------|
| AVX2     | 8         | 4         | `_mm256_blendv_pd`          |
| AVX-512  | 16        | 8         | `_mm512_mask_blend_pd`      |
| NEON     | 4         | 2         | `vbslq_f64(mask, t, f)`     |

The reciprocal-on-odd-quadrants step uses `_mm256_div_pd` /
`_mm512_div_pd` / `vdivq_f64`. Division latency is the bottleneck on
older x86 cores; on modern parts (Zen 4, Sapphire Rapids, Apple M-series)
the f64 divider is fully pipelined and the kernel is fma-bound rather
than div-bound.

## 10. Error analysis

| Source                       | f32 contribution    | f64 contribution    |
|------------------------------|---------------------|---------------------|
| Cody-Waite reduction         | \\(\sim 2^{-50}\\)      | \\(\sim 2^{-103}\\)     |
| Polynomial truncation        | \\(\sim 2^{-30}\\)      | \\(\sim 2^{-58}\\)      |
| Reciprocal (odd quadrants)   | \\(\sim 2^{-23}\\)      | \\(\sim 2^{-52}\\)      |
| Final FMA/round              | \\(\sim 2^{-23}\\)      | \\(\sim 2^{-52}\\)      |

Worst-case observed ULP:

| Variant         | Worst ULP | Where it occurs                       |
|-----------------|-----------|---------------------------------------|
| `_mm256_tan_ps` | 1.96      | \\(|x|\\) near \\(\pi/2\\) + odd quadrant     |
| `_mm256_tan_pd` | 1.99      | very large \\(|x|\\) + odd quadrant       |

Both honour the **`≤ 2 ULP`** envelope from the
[crate ULP table](../precision/tables.md). The reciprocal step is the
dominant error source for odd-quadrant inputs because \\(1/\tan(y)\\)
amplifies any error in \\(\tan(y)\\) when \\(|y|\\) is small.

## 11. Code excerpt

The f32 dispatch (8-lane → two 4-lane f64 halves):

```rust,ignore
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_tan_ps(x: __m256) -> __m256 {
    let x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
    let x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));

    let tan_lo = tan_ps_in_f64(x_lo);
    let tan_hi = tan_ps_in_f64(x_hi);

    let result_lo = _mm256_cvtpd_ps(tan_lo);
    let result_hi = _mm256_cvtpd_ps(tan_hi);
    _mm256_insertf128_ps(_mm256_castps128_ps256(result_lo), result_hi, 1)
}
```

The cotangent flip on odd quadrants:

```rust,ignore
// `blendv` only reads the sign bit of the mask, so shifting bit 0 of n
// straight into bit 63 replaces an and+cmpeq pair.
let n_256  = _mm256_cvtepi32_epi64(n);
let is_odd = _mm256_castsi256_pd(_mm256_slli_epi64(n_256, 63));

let neg_one = _mm256_set1_pd(-1.0);
let recip   = _mm256_div_pd(neg_one, tan_y);
let result  = _mm256_blendv_pd(tan_y, recip, is_odd);
```

## 12. References

- musl libc — [`src/math/tanf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/tanf.c),
  [`src/math/tan.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/tan.c),
  [`src/math/__tandf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/__tandf.c),
  [`src/math/__tan.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/__tan.c).
- Sun fdlibm — original `__tan` and `__tandf` polynomials.
- IEEE 754-2008 — special-value semantics.
- ISO/IEC 9899:1999 §F.10.1.7 — `tan` Annex F bindings.

## See also

- [Sine `sin`](./sin.md) — same reduction skeleton.
- [Cosine `cos`](./cos.md) — companion that shares reduction.
- [Two-argument arc tangent `atan2`](./atan2.md) — pole-aware inverse.
- [Argument-reduction taxonomy](../foundations/argument_reduction.md).
- [AVX2](../backends/avx2.md), [AVX-512](../backends/avx512.md), [NEON](../backends/neon.md).
