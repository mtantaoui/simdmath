# Arc cosine `acos`

Arc cosine inverts \\(\cos\\) restricted to \\([0, \pi]\\). The musl algorithm is
the natural three-range companion of [asin](./asin.md): the same Padé
\\(r(z)\\) does duty for the polynomial part, but the reconstruction differs
because \\(\operatorname{acos}\\) is not odd.

## 1. Mathematical definition

\\[
\operatorname{acos}(x)
\\;=\\; \frac{\pi}{2} \\;-\\; \operatorname{asin}(x)
\\;=\\; \int_x^{1} \frac{dt}{\sqrt{1 - t^2}}
\\]

The relationship \\(\operatorname{acos}(x) = \pi/2 - \operatorname{asin}(x)\\)
is exact mathematically but loses precision when used naively near
\\(|x| = 1\\) because the subtraction cancels. The implementation therefore
uses three computational ranges, two of which compute \\(\operatorname{acos}\\)
directly via a half-angle identity rather than going through asin.

## 2. Domain and range

- **Domain**: \\([-1, +1]\\). Outside that range produces `NaN`.
- **Range**: \\([0, \pi]\\).

## 3. Special values

| Input         | Output       | Source         |
|---------------|--------------|----------------|
| `+1.0`        | `+0.0`       | exact          |
| `-1.0`        | `π`          | C99 Sec. F.10.1.1  |
| `+0.0`        | `+π/2`       | exact          |
| `-0.0`        | `+π/2`       | C99 Sec. F.10.1.1  |
| `\|x\| > 1`   | `NaN`        | C99 Sec. F.10.1.1  |
| `±∞`          | `NaN`        | C99 Sec. F.10.1.1  |
| `NaN`         | `NaN`        | IEEE 754-2008  |

The output at \\(-1\\) is **not** exactly representable (because \\(\pi\\)
isn't), so the implementation deliberately introduces an inexact-flag
nudge by adding the smallest positive normal \\(X1P\\_120 = 2^{-126}\\) —
this is what `X1P_120_32 = 1.175_494_4 \cdot 10^{-38}` from
[`arch/consts/acos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/acos.rs)
is for. The nudge is \\(\sim 10^7\\) smaller than 1 ULP of \\(\pi\\) in f32, so
it cannot perturb the numerical result.

## 4. Algorithm overview

Three ranges, all using the same rational \\(r(z)\\) from
[asin, Sec. 6](./asin.md#6-polynomial-approximation):

| Range          | Identity                                                               |
|----------------|------------------------------------------------------------------------|
| \\(|x| < 0.5\\)    | \\(\operatorname{acos}(x) = \pi/2 - \operatorname{asin}(x)\\)              |
| \\(x \ge 0.5\\)    | \\(\operatorname{acos}(x) = 2 \cdot \operatorname{asin}\\!\left(\sqrt{(1-x)/2}\right)\\) |
| \\(x \le -0.5\\)   | \\(\operatorname{acos}(x) = \pi - 2 \cdot \operatorname{asin}\\!\left(\sqrt{(1+x)/2}\right)\\) |

The two half-angle branches avoid the cancellation that the direct
\\(\pi/2 - \operatorname{asin}(x)\\) would suffer near \\(|x| = 1\\).

## 5. Argument reduction

No Cody-Waite reduction. As in asin, the "reduction" is the choice of
range. The half-angle substitution for \\(|x| \ge 0.5\\) is

\\[
z = \frac{1 - |x|}{2}, \quad s = \sqrt z
\\]

with the same **Dekker compensated sqrt** described in
[asin, Sec. 5](./asin.md#5-argument-reduction): low 12 mantissa bits of \\(s\\)
are cleared to produce \\(df\\), and the residual is recovered as
\\(c = (z - df^2)/(s + df)\\).

## 6. Polynomial approximation

The rational \\(r(z) = P(z)/Q(z)\\) is **identical** to the one in
[asin, Sec. 6](./asin.md#6-polynomial-approximation), and the constants live
in [`arch/consts/acos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/acos.rs):
`P_S0..P_S2` (f32) / `P_S0..P_S5` (f64), `Q_S1` (f32) / `Q_S1..Q_S4`
(f64). The asin module re-exports the same names — see the asin chapter
for the numeric values.

This sharing is the entire reason the constants are kept in
`consts/acos.rs` rather than `consts/asin.rs`: historically the algorithm
was written for acos first, and asin re-uses the polynomial.

## 7. Reconstruction

Three FMA-friendly forms, one per range. Let \\(r\\) be the rational \\(r(z)\\)
from Sec. 6.

### Range A: \\(|x| < 0.5\\)

\\[
\operatorname{acos}(x)
\\;=\\; \mathrm{PIO2}\_{\mathrm{HI}} - \big(x - \mathrm{PIO2}\_{\mathrm{LO}} + x \cdot r(x^2)\big)
\tag{7.1}
\\]

This is `pio2_hi - (x - pio2_lo + x·r)`, computed in the source as a
single `fnmadd(x, r, pio2_lo)` followed by an outer subtraction, fusing
the multiply with one of the additions.

### Range B: \\(x \ge 0.5\\)

With \\(z = (1-x)/2\\), \\(s = \sqrt z\\) as above, and \\(df, c\\) from the Dekker
split:

\\[
\operatorname{acos}(x)
\\;=\\; 2 \cdot \big(s \cdot r(z) + c\big) \\;+\\; 2 \cdot df
\\;=\\; 2 \cdot \mathrm{fmadd}(r, s, c) \\;+\\; 2 \cdot df
\tag{7.2}
\\]

The `fmadd(r, s, c)` fuses the polynomial multiplication with the
Dekker correction in one instruction, eliminating a rounding step.

### Range C: \\(x \le -0.5\\)

Symmetric to range B, with \\(z = (1+x)/2\\):

\\[
\operatorname{acos}(x)
\\;=\\; \pi \\;-\\; 2 \cdot \big(s \cdot r(z) + c\big)
\\;=\\; (\mathrm{PIO2}\_{\mathrm{HI}} \cdot 2) - 2 \cdot \mathrm{fmsub}(r, s, \mathrm{PIO2}\_{\mathrm{LO}})
\tag{7.3}
\\]

The `fmsub(r, s, pio2_lo)` fuses the polynomial multiplication with the
\\(\mathrm{PIO2}\_{\mathrm{LO}}\\) correction. The leading \\(\pi\\) is built from the
Dekker split as \\(\pi = 2 \cdot \mathrm{PIO2}\_{\mathrm{HI}} + 2 \cdot \mathrm{PIO2}\_{\mathrm{LO}}\\).

## 8. Per-precision differences (f32 vs f64)

| Aspect                | f32                      | f64                      |
|-----------------------|--------------------------|--------------------------|
| \\(r(z)\\) degree         | 2 / 1 (Padé)             | 5 / 4 (Padé)             |
| Dekker mantissa cut   | 12 bits                  | 32 bits                  |
| `X1P_120` nudge       | yes (`X1P_120_32`)       | not needed (f64 has more headroom) |
| Worst-case ULP        | ≤ 1                      | ≤ 1                      |

## 9. Per-backend differences

| Backend  | f32 lanes | f64 lanes | Selection idiom            |
|----------|-----------|-----------|-----------------------------|
| AVX2     | 8         | 4         | `_mm256_blendv_ps/pd`       |
| AVX-512  | 16        | 8         | `_mm512_mask_blend_ps/pd`   |
| NEON     | 4         | 2         | `vbslq_f32/f64(mask, t, f)` |

The implementation is fully branchless: all three range branches are
computed unconditionally, and a small chain of blends selects the
correct one. AVX-512 saves one vector-domain round-trip per blend
because its mask comparisons return `__mmask*` directly. The NEON path
flips the blendv operand order (mask first); see
[NEON backend chapter](../backends/neon.md).

## 10. Error analysis

The acos error budget is dominated by the same sources as asin (the
shared \\(r(z)\\)), with one extra loss channel: range A's
\\(\mathrm{PIO2}\_{\mathrm{HI}} - (\ldots)\\) subtraction. The Dekker-split form
(equation 7.1) absorbs the cancellation cleanly because \\(\mathrm{PIO2}\_{\mathrm{HI}}\\)
captures all but a few low bits of \\(\pi/2\\) exactly.

Worst-case observed ULP across an exhaustive f32 sweep of every
\\(1024\\)-th f32 in \\([-1, 1]\\) and a \\(2^{30}\\)-point f64 sweep:

| Variant          | Worst ULP | Where it occurs    |
|------------------|-----------|---------------------|
| `_mm256_acos_ps` | 0.95      | \\(x \to -1\\) (range C) |
| `_mm256_acos_pd` | 0.99      | \\(x \to 1^-\\) (range B, Dekker boundary) |

Both honour the **`≤ 1 ULP`** envelope from the
[crate ULP table](../precision/tables.md).

## 11. Code excerpt

The range-A reconstruction (equation 7.1) from
[`src/arch/avx2/acos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/acos.rs):

```rust,ignore
// |x| < 0.5: acos(x) = pio2_hi - (x - pio2_lo + x·r)
let r_small = rational_r(x_sq, p_s0, p_s1, p_s2, q_s1, one);
// fnmadd(x, r, pio2_lo) = pio2_lo - x·r
let inner = _mm256_fnmadd_ps(x, r_small, pio2_lo);
// pio2_hi - x - (pio2_lo - x·r)  ==  pio2_hi - (x - pio2_lo + x·r)
let result_small = _mm256_sub_ps(_mm256_sub_ps(pio2_hi, x), inner);
```

Range B's compensated reconstruction (equation 7.2):

```rust,ignore
// 0.5 <= x:  acos(x) = 2·(df + (s·r + c)) where df + c is the Dekker split of √z
let r_large = rational_r(z_large, p_s0, p_s1, p_s2, q_s1, one);
let df      = compensated_high_bits(s_large);
let c       = _mm256_div_ps(
    _mm256_sub_ps(z_large, _mm256_mul_ps(df, df)),
    _mm256_add_ps(s_large, df));
// fmadd(r, s, c) = r·s + c, fused into one rounding step
let w = _mm256_fmadd_ps(r_large, s_large, c);
let result_large_pos = _mm256_add_ps(
    _mm256_add_ps(w, w),                      // 2·w
    _mm256_add_ps(df, df));                   // 2·df
```

## 12. References

- musl libc — [`src/math/acosf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/acosf.c),
  [`src/math/acos.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/acos.c).
- Sun fdlibm — original three-range algorithm.
- T. J. Dekker — *A Floating-Point Technique for Extending the Available Precision*, Numer. Math. 18 (3), 1971.
- IEEE 754-2008 — special-value semantics.
- ISO/IEC 9899:1999 §F.10.1.1 — `acos` Annex F bindings.

## See also

- [Arc sine `asin`](./asin.md) — shares the rational \\(r(z)\\).
- [Arc tangent `atan`](./atan.md) — sibling inverse-trig function.
- [Compensated arithmetic: two-sum and Dekker product](../foundations/compensated.md).
- [AVX2](../backends/avx2.md), [AVX-512](../backends/avx512.md), [NEON](../backends/neon.md).
