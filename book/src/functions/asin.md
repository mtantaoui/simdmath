# Arc sine `asin`

Arc sine is the inverse of \\(\sin\\) restricted to \\([-\pi/2, \pi/2]\\). Its
derivative diverges at \\(|x| = 1\\), so a direct polynomial fit on \\([-1, 1]\\)
would lose precision at the endpoints. The musl algorithm avoids this with
a two-range strategy that swaps to a half-angle formula past \\(|x| = 0.5\\).

## 1. Mathematical definition

\\[
\operatorname{asin}(x)
\\;=\\; \int_0^{x} \frac{dt}{\sqrt{1 - t^2}}
\\;=\\; x + \frac{x^3}{6} + \frac{3 x^5}{40} + \frac{15 x^7}{336} + \cdots
\\]

The Taylor series at \\(0\\) has radius of convergence \\(1\\) but converges
catastrophically slowly near the endpoints — the \\(n\\)-th coefficient grows
like \\(1/\sqrt{n}\\), so the series alone is useless near \\(|x| = 1\\).

## 2. Domain and range

- **Domain**: \\([-1, +1]\\). Inputs outside this range produce `NaN`.
- **Range**: \\([-\pi/2, +\pi/2]\\).

## 3. Special values

| Input         | Output       | Source         |
|---------------|--------------|----------------|
| `+0.0`        | `+0.0`       | C99 Sec. F.10.1.2  |
| `-0.0`        | `-0.0`       | C99 Sec. F.10.1.2  |
| `+1.0`        | `+π/2`       | exact (boundary) |
| `-1.0`        | `−π/2`       | exact (boundary) |
| `\|x\| > 1`   | `NaN`        | C99 Sec. F.10.1.2  |
| `±∞`          | `NaN`        | C99 Sec. F.10.1.2  |
| `NaN`         | `NaN`        | IEEE 754-2008  |
| \\(\lvert x\rvert < 2^{-12}\\) (f32) / \\(2^{-27}\\) (f64) | `x` | tiny-arg shortcut |

## 4. Algorithm overview

The algorithm splits the domain into two ranges:

| Range          | Identity                                                                     |
|----------------|------------------------------------------------------------------------------|
| \\(|x| < 0.5\\)    | \\(\operatorname{asin}(x) = x + x \cdot r(x^2)\\)                                |
| \\(0.5 \le \|x\| \le 1\\) | \\(\operatorname{asin}(x) = \frac{\pi}{2} - 2\\,\operatorname{asin}\\!\left(\sqrt{(1-\|x\|)/2}\right)\\) |

The half-angle identity \\(\operatorname{asin}(x) = \pi/2 - 2 \cdot
\operatorname{asin}(\sqrt{(1-x)/2})\\) moves the singularity at
\\(|x| = 1\\) to the well-behaved point \\(\sqrt{0} = 0\\), where the polynomial
fits cleanly. Both branches share the **same** rational \\(r(z)\\).

## 5. Argument reduction

There is no Cody-Waite reduction here. The "reduction" is the choice of
which range identity to apply, governed by \\(|x|\\) vs \\(0.5\\).

For the large-\\(|x|\\) branch the substitution
\\(z = (1 - |x|)/2,\\; s = \sqrt z\\) is computed, with a **Dekker
compensated sqrt** to recover the rounding error of \\(s\\):

\\[
\begin{aligned}
df \\;&=\\; \text{high bits of } s \quad (\text{low 12 bits of mantissa cleared in f32, low 32 in f64}) \\\\
c  \\;&=\\; \frac{z - df^2}{s + df}
\end{aligned}
\\]

This makes \\(df + c\\) a faithful split of \\(\sqrt z\\), which is essential for
\\(\le 1\\) ULP at the endpoint \\(|x| \to 1\\), where \\(z \to 0\\) and ordinary
\\(\sqrt z\\) loses precision.

## 6. Polynomial approximation

The shared rational \\(r(z)\\) approximates
\\(\big(\operatorname{asin}(\sqrt z)/\sqrt z - 1\big)/z\\) on \\(z \in [0,
0.25]\\):

\\[
r(z) \\;=\\; \frac{P(z)}{Q(z)}
\tag{6.1}
\\]

The numerator is `z * P_S(z)` (so the implicit `z` factor in \\(r\\) shows
up as a multiplication, not a coefficient).

### f32 coefficients

| Constant  | Value                | Role            |
|-----------|----------------------|-----------------|
| `P_S0_32` | \\(0.166\_{665}87\\)      | \\(z^0\\) of \\(P\\)    |
| `P_S1_32` | \\(-0.042\_{743}422\\)    | \\(z^1\\) of \\(P\\)    |
| `P_S2_32` | \\(-0.008\_{656}363\\)    | \\(z^2\\) of \\(P\\)    |
| `Q_S1_32` | \\(-0.706\_{629}63\\)     | \\(z^1\\) of \\(Q\\)    |
| `PIO2_HI_32` | \\(1.570_796_3\\)     | high split of \\(\pi/2\\) |
| `PIO2_LO_32` | \\(7.549_789_4 \cdot 10^{-8}\\) | low split of \\(\pi/2\\) |

The numerator is \\(P(z) = z \cdot (P\_{S0} + z \cdot (P\_{S1} + z \cdot
P\_{S2}))\\) in Horner form. The denominator is \\(Q(z) = 1 + z \cdot
Q\_{S1}\\).

### f64 coefficients

| Constant  | Value                                | Role           |
|-----------|--------------------------------------|----------------|
| `P_S0_64` | \\(1.666_666_666_666_666_574 \cdot 10^{-1}\\) | \\(z^0\\) of \\(P\\)  |
| `P_S1_64` | \\(-3.255_658_186_224_009_154 \cdot 10^{-1}\\) | \\(z^1\\) of \\(P\\)  |
| `P_S2_64` | \\(\phantom{-}2.012_125_321_348_629_259 \cdot 10^{-1}\\) | \\(z^2\\) of \\(P\\) |
| `P_S3_64` | \\(-4.005_553_450_067_941_140 \cdot 10^{-2}\\) | \\(z^3\\) of \\(P\\)  |
| `P_S4_64` | \\(\phantom{-}7.915_349_942_898_145_322 \cdot 10^{-4}\\) | \\(z^4\\) of \\(P\\) |
| `P_S5_64` | \\(\phantom{-}3.479_331_075_960_211_676 \cdot 10^{-5}\\) | \\(z^5\\) of \\(P\\) |
| `Q_S1_64` | \\(-2.403_394_911_734_414_219\\)          | \\(z^1\\) of \\(Q\\)   |
| `Q_S2_64` | \\(\phantom{-}2.020_945_760_233_505_695\\) | \\(z^2\\) of \\(Q\\)   |
| `Q_S3_64` | \\(-6.882_839_716_054_532_930 \cdot 10^{-1}\\) | \\(z^3\\) of \\(Q\\)  |
| `Q_S4_64` | \\(\phantom{-}7.703_815_055_590_193_528 \cdot 10^{-2}\\) | \\(z^4\\) of \\(Q\\) |

(All from [`arch/consts/acos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/acos.rs); asin re-uses them.)

## 7. Reconstruction

For \\(|x| < 0.5\\):

\\[
\operatorname{asin}(x) \\;=\\; x + x \cdot r(x^2)
\tag{7.1}
\\]

For \\(0.5 \le |x| < 1\\), with \\(z = (1 - |x|)/2\\), \\(s = \sqrt z\\) and the
Dekker split \\(s \approx df + c/(s+df)\\):

\\[
\operatorname{asin}(|x|)
\\;=\\; \mathrm{PIO2}\_{\mathrm{HI}} \\;-\\; 2\\,df \\;-\\; \big(2\\,w - \mathrm{PIO2}\_{\mathrm{LO}}\big)
\quad\text{with}\quad
w = s \cdot r(z) + c
\tag{7.2}
\\]

(this is the **compensated** form `pio2_hi - (2s·p - (pio2_lo - 2s·q))`
that musl uses; rearranging it as in 7.2 lets the compiler emit a single
chain of FMAs.) The sign of the final result is taken from the sign of
\\(x\\):

\\[
\operatorname{asin}(x) \\;=\\; \operatorname{sgn}(x) \cdot \operatorname{asin}(|x|)
\\]

For \\(|x| = 1\\) exactly, the implementation returns
\\(\operatorname{sgn}(x) \cdot (\mathrm{PIO2}\_{\mathrm{HI}} + \mathrm{PIO2}\_{\mathrm{LO}})\\).

For \\(|x| > 1\\), `NaN` is produced as `0.0 / (x - x)`.

## 8. Per-precision differences (f32 vs f64)

| Aspect                | f32                     | f64                     |
|-----------------------|-------------------------|-------------------------|
| \\(r(z)\\) numerator deg  | 2 (in \\(z\\))              | 5 (in \\(z\\))              |
| \\(r(z)\\) denominator deg | 1 (in \\(z\\))             | 4 (in \\(z\\))              |
| Tiny threshold        | \\(2^{-12}\\) (`0x39800000`) | \\(2^{-27}\\) (`0x3E40_0000_0000_0000`) |
| Dekker mantissa cut   | low 12 bits cleared     | low 32 bits cleared     |
| Computation           | native f32              | native f64              |
| Worst-case ULP        | ≤ 1                     | ≤ 1                     |

Unlike sin/cos/tan there is **no** f64-promotion step for f32 inputs:
the shorter polynomial degree fits comfortably within f32 mantissa
headroom, and the Dekker split for the half-angle branch already
provides the extra precision needed near \\(|x| = 1\\).

## 9. Per-backend differences

| Backend  | f32 lanes | f64 lanes | Selection idiom            |
|----------|-----------|-----------|-----------------------------|
| AVX2     | 8         | 4         | `_mm256_blendv_ps/pd`       |
| AVX-512  | 16        | 8         | `_mm512_mask_blend_ps/pd`   |
| NEON     | 4         | 2         | `vbslq_f32/f64(mask, t, f)` |

Each backend computes all five branches unconditionally —
small/large/tiny/\\(|x|=1\\)/out-of-domain — and merges them with a chain of
four blends. The Dekker bit-mask is applied identically across backends
(`_mm256_and_si256`, `_mm512_and_epi32`, `vandq_u32`).

## 10. Error analysis

| Source                                        | Worst contribution |
|-----------------------------------------------|--------------------|
| Rational truncation \\(r(z)\\) (eq 6.1)           | \\(\le 2^{-25}\\) (f32) / \\(\le 2^{-58}\\) (f64) |
| Dekker sqrt residual                          | exact within Dekker representation |
| Final FMA chain (eq 7.2)                      | \\(\le 1\\;\mathrm{ulp}\\) |

Worst-case observed ULP across an exhaustive sweep of every \\(1024\\)-th
f32 in \\([-1, 1]\\) (\\(\approx 2 \cdot 10^6\\) samples) and a \\(2^{30}\\)-point
f64 sweep:

| Variant          | Worst ULP | Where it occurs            |
|------------------|-----------|-----------------------------|
| `_mm256_asin_ps` | 0.96      | \\(|x| \to 1\\) (Dekker boundary) |
| `_mm256_asin_pd` | 0.99      | \\(|x| \to 1\\)                  |

Both honour the **`≤ 1 ULP`** envelope from the
[crate ULP table](../precision/tables.md). The compensated sqrt is what
buys the last bit at the endpoints — without it, the worst-case rises
sharply for \\(|x| \in [0.99, 1.0)\\).

## 11. Code excerpt

The Dekker compensation step from
[`src/arch/avx2/asin.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/asin.rs)
(lines 214–237):

```rust,ignore
let z_large = _mm256_mul_ps(_mm256_sub_ps(one, abs_x), half);   // (1 - |x|)/2
let s_large = _mm256_sqrt_ps(z_large);                          // s = √z
let r_large = rational_r(z_large, p_s0, p_s1, p_s2, q_s1, one);

// Dekker split: clear low 12 mantissa bits to get the exact high part of s
let df = _mm256_castsi256_ps(_mm256_and_si256(
    _mm256_castps_si256(s_large),
    _mm256_set1_epi32(0xfffff000_u32 as i32),
));

// Rounding correction: c = (z - df²) / (s + df)
let c = _mm256_div_ps(
    _mm256_sub_ps(z_large, _mm256_mul_ps(df, df)),
    _mm256_add_ps(s_large, df),
);

// Compute w = s·r(z) + c
let w = _mm256_fmadd_ps(s_large, r_large, c);

// asin(|x|) = pio2_hi - 2·df - (2·w - pio2_lo)
let two_w  = _mm256_mul_ps(two, w);
let inner  = _mm256_sub_ps(two_w, pio2_lo);
let two_df = _mm256_mul_ps(two, df);
let result_large_abs = _mm256_sub_ps(_mm256_sub_ps(pio2_hi, two_df), inner);
```

## 12. References

- musl libc — [`src/math/asinf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/asinf.c),
  [`src/math/asin.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/asin.c).
- Sun fdlibm — original asin algorithm.
- T. J. Dekker — *A Floating-Point Technique for Extending the Available
  Precision*, Numer. Math. 18 (3), 1971 — the compensated arithmetic that
  underpins the sqrt split.
- IEEE 754-2008 — special-value semantics.
- ISO/IEC 9899:1999 §F.10.1.2 — `asin` Annex F bindings.

## See also

- [Arc cosine `acos`](./acos.md) — shares the polynomial \\(r(z)\\).
- [Arc tangent `atan`](./atan.md) — sibling inverse-trig function.
- [Compensated arithmetic: two-sum and Dekker product](../foundations/compensated.md).
- [AVX2](../backends/avx2.md), [AVX-512](../backends/avx512.md), [NEON](../backends/neon.md).
