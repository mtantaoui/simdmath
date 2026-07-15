# Arc tangent `atan`

Arc tangent has no \\(|x|=1\\) singularity to compensate, but it has the
**opposite** problem: the function flattens out toward \\(\pm \pi/2\\) as
\\(|x| \to \infty\\), so a polynomial fit must be defined either on a bounded
range (with a per-range offset) or with an explicit reciprocal step.

The crate uses two different strategies:

- **f32**: a single-range reduction at \\(|x| = 1\\), plus a degree-9 odd
  minimax polynomial.
- **f64**: musl libc's four-range reduction at breakpoints
  \\(\{7/16, 11/16, 19/16, 39/16\}\\), with per-range two-sum offsets and a
  degree-11 polynomial split into odd/even halves.

## 1. Mathematical definition

\\[
\operatorname{atan}(x)
\\;=\\; \int_0^{x} \frac{dt}{1 + t^2}
\\;=\\; x - \frac{x^3}{3} + \frac{x^5}{5} - \frac{x^7}{7} + \cdots
\\]

Atan is **odd** and analytic on \\(\mathbb{R}\\), with horizontal asymptotes
at \\(\pm \pi/2\\).

## 2. Domain and range

- **Domain**: all \\(x \in \mathbb{R} \cup \{-\infty, +\infty\}\\).
- **Range**: \\((-\pi/2, +\pi/2)\\), with \\(\operatorname{atan}(\pm\infty) = \pm\pi/2\\).

## 3. Special values

| Input  | Output | Source         |
|--------|--------|----------------|
| `+0.0` | `+0.0` | C99 Sec. F.10.1.3  |
| `-0.0` | `-0.0` | C99 Sec. F.10.1.3  |
| `+1.0` | `+π/4` | exact via offset |
| `-1.0` | `-π/4` | exact via offset |
| `+∞`   | `+π/2` | C99 Sec. F.10.1.3  |
| `-∞`   | `-π/2` | C99 Sec. F.10.1.3  |
| `NaN`  | `NaN`  | IEEE 754-2008  |

The negative-zero case is preserved by extracting the sign bit at the
start and XOR-restoring it at the end (rather than negating with a
comparison-based mask), matching the IEEE 754 sign-preservation rule.

## 4. Algorithm overview

The skeleton is "extract sign, reduce, polynomial, add offset, restore
sign":

1. \\(|x|\\) and the sign bit are split.
2. \\(|x|\\) is mapped to a small reduced argument \\(t\\) on a fixed interval.
3. A minimax polynomial is evaluated at \\(t\\).
4. A per-range constant `atanhi[i] + atanlo[i]` is added (two-sum
   compensation).
5. The original sign bit is XOR-merged back in.

## 5. Argument reduction

### f32: single-range

Only one breakpoint at \\(|x| = 1\\):

| Range          | \\(t\\)         | Offset  |
|----------------|-------------|---------|
| \\(|x| \le 1\\)    | \\(t = |x|\\)   | \\(0\\)     |
| \\(|x| > 1\\)      | \\(t = 1/|x|\\) | \\(\pi/2 - \cdot\\) |

For \\(|x| > 1\\) the identity used is
\\(\operatorname{atan}(|x|) = \pi/2 - \operatorname{atan}(1/|x|)\\). The
constant `FRAC_PI_2_32 = 1.5707963` from
[`arch/consts/atan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/atan.rs)
is the offset.

### f64: four ranges (musl libc)

Five sub-domains separated by four breakpoints:

| Range id | Condition              | \\(t\\)                            | \\((\mathrm{hi}, \mathrm{lo})\\)            |
|----------|------------------------|--------------------------------|-----------------------------------------|
| \\(-1\\)     | \\(|x| < 7/16\\)           | \\(t = |x|\\)                      | \\((0, 0)\\)                                |
| \\(0\\)      | \\(7/16 \le |x| < 11/16\\) | \\(t = (2|x| - 1)/(2 + |x|)\\)     | \\(\operatorname{atan}(0.5) = (\mathrm{ATANHI}\_{\mathrm{0}}, \mathrm{ATANLO}\_{\mathrm{0}})\\) |
| \\(1\\)      | \\(11/16 \le |x| < 19/16\\) | \\(t = (|x| - 1)/(|x| + 1)\\)     | \\(\operatorname{atan}(1) = (\mathrm{ATANHI}\_{\mathrm{1}}, \mathrm{ATANLO}\_{\mathrm{1}})\\) |
| \\(2\\)      | \\(19/16 \le |x| < 39/16\\) | \\(t = (2|x| - 3)/(2 + 3|x|)\\)   | \\(\operatorname{atan}(1.5) = (\mathrm{ATANHI}\_{\mathrm{2}}, \mathrm{ATANLO}\_{\mathrm{2}})\\) |
| \\(3\\)      | \\(|x| \ge 39/16\\)        | \\(t = -1/|x|\\)                  | \\(\operatorname{atan}(\infty) = \pi/2 = (\mathrm{ATANHI}\_{\mathrm{3}}, \mathrm{ATANLO}\_{\mathrm{3}})\\) |

Each \\((\mathrm{hi}, \mathrm{lo})\\) pair is a Dekker-style two-sum split of
\\(\operatorname{atan}\\) at the range's "centre point" \\(\{0.5, 1.0, 1.5,
\infty\}\\). The numerical values from
[`arch/consts/atan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/atan.rs):

| Constant     | Value                                  |
|--------------|----------------------------------------|
| `ATANHI_0`   | \\(4.636_476_090_008_060_935 \cdot 10^{-1}\\) |
| `ATANLO_0`   | \\(2.269_877_745_296_168_709 \cdot 10^{-17}\\) |
| `ATANHI_1`   | \\(7.853_981_633_974_482_790 \cdot 10^{-1}\\) |
| `ATANLO_1`   | \\(3.061_616_997_868_383_018 \cdot 10^{-17}\\) |
| `ATANHI_2`   | \\(9.827_937_232_473_290_541 \cdot 10^{-1}\\) |
| `ATANLO_2`   | \\(1.390_331_103_123_099_845 \cdot 10^{-17}\\) |
| `ATANHI_3`   | \\(1.570_796_326_794_896_558\\)              |
| `ATANLO_3`   | \\(6.123_233_995_736_766_036 \cdot 10^{-17}\\) |

The implementation computes **all five reduced arguments unconditionally**
and selects with a four-blend cascade (priority high → low), then applies
the matching \\((\mathrm{hi}, \mathrm{lo})\\) pair the same way.

## 6. Polynomial approximation

### f32 polynomial

Degree-17 odd polynomial (in \\(t\\); the bracket has 9 coefficients, degree 8 in \\(t^2\\)):

\\[
\operatorname{atan}(t)
\\;\approx\\; t \cdot \big(P_0 + t^2 (P_1 + t^2 (P_2 + \cdots))\big)
\tag{6.1}
\\]

with constants from [`arch/consts/atan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/atan.rs):

| Constant      | Coefficient | Value (f32)            |
|---------------|-------------|------------------------|
| `ATAN_P0_32`  | \\(t\\)         | \\(\phantom{-}0.999_999_871_164\\)  |
| `ATAN_P1_32`  | \\(t^3\\)       | \\(-0.333_325_240_026\\)            |
| `ATAN_P2_32`  | \\(t^5\\)       | \\(\phantom{-}0.199_848_846_856\\)  |
| `ATAN_P3_32`  | \\(t^7\\)       | \\(-0.141_548_060_419\\)            |
| `ATAN_P4_32`  | \\(t^9\\)       | \\(\phantom{-}0.104_775_391_987\\)  |
| `ATAN_P5_32`  | \\(t^{11}\\)    | \\(-0.071_943_845_424_6\\)          |
| `ATAN_P6_32`  | \\(t^{13}\\)    | \\(\phantom{-}0.039_345_413_147_9\\) |
| `ATAN_P7_32`  | \\(t^{15}\\)    | \\(-0.014_152_348_036_2\\)          |
| `ATAN_P8_32`  | \\(t^{17}\\)    | \\(\phantom{-}0.002_398_139_012_51\\) |

Note the leading coefficient is **slightly less than 1**: the minimax
solver shrinks it from the analytical \\(1\\) to compensate for higher-order
truncation, which gains a fraction of a ULP at the cost of an exact
zero of \\(\operatorname{atan}(0)\\). The implementation handles
\\(\operatorname{atan}(0) = 0\\) correctly because the polynomial
\\(\operatorname{poly}(0) = 0\\) regardless.

### f64 polynomial (musl `__atan`)

Degree-23 polynomial in \\(t\\) (the correction is degree-11 in
\\(t^2\\)), split into odd and even halves with \\(z = t^2\\),
\\(w = z^2\\):

\\[
\begin{aligned}
s_1 &= z \cdot (a\_{T_0} + w (a\_{T_2} + w (a\_{T_4} + w (a\_{T_6} + w (a\_{T_8} + w \cdot a\_{T\_{10}}))))) \\\\
s_2 &= w \cdot (a\_{T_1} + w (a\_{T_3} + w (a\_{T_5} + w (a\_{T_7} + w \cdot a\_{T_9}))))
\end{aligned}
\tag{6.2}
\\]

The split improves instruction-level parallelism: \\(s_1\\) and \\(s_2\\) have
independent dependency chains and can be computed simultaneously by the
two FMA ports on a modern x86 core.

Coefficients (from
[`arch/consts/atan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/atan.rs)):

| Constant | Coefficient | Approx value           |
|----------|-------------|------------------------|
| `AT0`    | \\(1/3\\)       | \\(\phantom{-}0.333_333_333_333_293_180\\)  |
| `AT1`    | \\(-1/5\\)      | \\(-0.199_999_999_987_648_325\\)            |
| `AT2`    | \\(1/7\\)       | \\(\phantom{-}0.142_857_142_725_034_664\\)  |
| `AT3`    | \\(-1/9\\)      | \\(-0.111_111_104_054_623_558\\)            |
| `AT4`    | \\(1/11\\)      | \\(\phantom{-}0.090_908_871_334_365_066\\)  |
| `AT5`    | \\(-1/13\\)     | \\(-0.076_918_762_050_448_300\\)            |
| `AT6`    | \\(1/15\\)      | \\(\phantom{-}0.066_610_731_373_875_312\\)  |
| `AT7`    | \\(-1/17\\)     | \\(-0.058_335_701_337_905_735\\)            |
| `AT8`    | \\(1/19\\)      | \\(\phantom{-}0.049_768_779_946_159_324\\)  |
| `AT9`    | \\(-1/21\\)     | \\(-0.036_531_572_744_216_916\\)            |
| `AT10`   | \\(1/23\\)      | \\(\phantom{-}0.016_285_820_115_365_782\\)  |

These are the analytical Taylor coefficients **slightly perturbed** by
minimax — close enough that the original \\(1/(2k+1)\\) structure is
recognisable.

## 7. Reconstruction

For range id \\(i \in \{0, 1, 2, 3\}\\) the corrected result is:

\\[
\operatorname{atan}(|x|)
\\;=\\; \mathrm{hi}_i \\;+\\; \mathrm{lo}_i \\;+\\; \big(t - t \cdot (s_1 + s_2)\big)
\\]

In simpler form: `result = hi + (lo + t·poly(t²) - 0)` where the last
"−0" depends on the range. For id \\(-1\\) (no reduction) the entire result
is just \\(t = |x|\\) corrected by the polynomial.

The original sign bit is reattached by XOR:

```rust,ignore
_mm256_xor_ps(abs_result, sign_bits)
```

This is what makes \\(\operatorname{atan}(-0) = -0\\): a comparison-based
"if x < 0 negate" approach would map \\(-0\\) to \\(+0\\), but XOR with the
extracted sign bit propagates the input's sign-bit pattern unconditionally.

## 8. Per-precision differences (f32 vs f64)

| Aspect                  | f32                    | f64                          |
|-------------------------|------------------------|------------------------------|
| Reduction breakpoints   | 1                      | 4                            |
| Polynomial degree (in \\(t\\)) | 17 (degree-8 in \\(t^2\\)) | 23 (degree-11 in \\(t^2\\))  |
| Per-range two-sum offset | no                    | yes (`ATANHI_*` / `ATANLO_*`) |
| Worst-case ULP          | ≤ 3                    | ≤ 1                          |

The f32 path's `≤ 3 ULP` envelope reflects two cumulative roundings (the
final \\(t \cdot \mathrm{poly}\\) multiplication, plus the reduction
divide \\(1/|x|\\)); the f64 path achieves `≤ 1 ULP` because each of its four
ranges is small enough that the polynomial truncation error is several
bits below half-ULP, and the two-sum offset absorbs the constant-term
rounding exactly.

## 9. Per-backend differences

| Backend  | f32 lanes | f64 lanes | Selection idiom            |
|----------|-----------|-----------|-----------------------------|
| AVX2     | 8         | 4         | `_mm256_blendv_ps/pd`       |
| AVX-512  | 16        | 8         | `_mm512_mask_blend_ps/pd`   |
| NEON     | 4         | 2         | `vbslq_f32/f64(mask, t, f)` |

The f64 path's four-range reduction performs **four** blends per
result (one per range boundary). On AVX-512 these become four
mask-domain `mask_blend` instructions that retire on the mask port,
leaving the FMA ports for the polynomial. On AVX2 each blend is a full
`blendv_pd` on the FMA pipeline, which is the main reason the AVX-512
backend is materially faster than 2× the AVX2 throughput here even
though the lane count is exactly 2×.

## 10. Error analysis

### f32 (≤ 3 ULP)

| Source                          | Worst contribution |
|---------------------------------|--------------------|
| Reciprocal \\(1/|x|\\) for \\(|x| > 1\\) | \\(\le 1\\) ULP        |
| Polynomial truncation (eq 6.1)  | \\(\le 1.5\\) ULP      |
| Final \\(\pi/2 - \cdot\\) subtraction | \\(\le 1\\) ULP      |

Worst observed: 2.78 ULP near \\(|x| = 1\\) where the reduction crosses over.

### f64 (≤ 1 ULP)

| Source                                | Worst contribution |
|---------------------------------------|--------------------|
| Range division (e.g. \\((2|x|-1)/(2+|x|)\\)) | \\(\le 0.5\\) ULP   |
| Polynomial truncation (eq 6.2)        | \\(\le 0.5\\) ULP      |
| Two-sum offset assembly               | exact (Dekker)     |

Worst observed: 0.97 ULP near \\(|x| = 39/16\\) (range 2 / range 3 boundary).

Both honour the **`f32 ≤ 3 ULP, f64 ≤ 1 ULP`** envelopes from the
[crate ULP table](../precision/tables.md).

## 11. Code excerpt

The f64 four-range cascading blend from
[`src/arch/avx2/math/atan.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/math/atan.rs)
(the `t0..t3` range values and the cascading blend):

```rust,ignore
// Compute all 4 reduced arguments unconditionally
let t0 = _mm256_div_pd(
    _mm256_sub_pd(_mm256_mul_pd(two, abs_x), one),
    _mm256_add_pd(two, abs_x));                          // (2x − 1)/(2 + x)
let t1 = _mm256_div_pd(_mm256_sub_pd(abs_x, one),
                       _mm256_add_pd(abs_x, one));        // (x − 1)/(x + 1)
let t2 = _mm256_div_pd(
    _mm256_sub_pd(_mm256_mul_pd(two, abs_x), three),
    _mm256_add_pd(two, _mm256_mul_pd(three, abs_x)));    // (2x − 3)/(2 + 3x)
let t3 = _mm256_div_pd(neg_one, abs_x);                   // −1/x

// Cascade: priority highest first
let t = {
    let t = _mm256_blendv_pd(t3, t2, is_lt_thr3);
    let t = _mm256_blendv_pd(t,  t1, is_lt_thr2);
    let t = _mm256_blendv_pd(t,  t0, is_lt_thr1);
    _mm256_blendv_pd(t, abs_x, is_lt_thr0)               // id = -1: no reduction
};
```

The odd/even split that recovers the polynomial value:

```rust,ignore
let z = _mm256_mul_pd(t, t);     // t²
let w = _mm256_mul_pd(z, z);     // t⁴

// s1 = z · (aT0 + w·(aT2 + w·(aT4 + w·(aT6 + w·(aT8 + w·aT10)))))
let s1 = _mm256_mul_pd(z,
    _mm256_fmadd_pd(w,
        _mm256_fmadd_pd(w,
            _mm256_fmadd_pd(w,
                _mm256_fmadd_pd(w,
                    _mm256_fmadd_pd(w, at10, at8),
                at6),
            at4),
        at2),
    at0));

// s2 = w · (aT1 + w·(aT3 + w·(aT5 + w·(aT7 + w·aT9))))
let s2 = _mm256_mul_pd(w,
    _mm256_fmadd_pd(w,
        _mm256_fmadd_pd(w,
            _mm256_fmadd_pd(w, _mm256_fmadd_pd(w, at9, at7), at5),
        at3),
    at1));
```

## 12. References

- musl libc — [`src/math/atanf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/atanf.c),
  [`src/math/atan.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/atan.c).
- Sun fdlibm — original four-range atan algorithm.
- Cody, W. J.; Waite, W. — *Software Manual for the Elementary Functions*, 1980 (chapter 11, atan reduction).
- Muller, J.-M. et al. — *Handbook of Floating-Point Arithmetic*, 2nd ed., Birkhäuser, 2018 — discussion of two-sum offsets.
- IEEE 754-2008 — special-value semantics.
- ISO/IEC 9899:1999 §F.10.1.3 — `atan` Annex F bindings.

## See also

- [Two-argument arc tangent `atan2`](./atan2.md) — uses the same kernel with quadrant disambiguation.
- [Arc sine `asin`](./asin.md), [Arc cosine `acos`](./acos.md) — sibling inverse-trig functions.
- [Argument-reduction taxonomy](../foundations/argument_reduction.md).
- [AVX2](../backends/avx2.md), [AVX-512](../backends/avx512.md), [NEON](../backends/neon.md).
