# Sine `sin`

This chapter documents the SIMD `sin(x)` implementation across the three
backends. The algorithm is a faithful port of musl libc's `sinf.c` /
`sin.c` — themselves descendants of Sun's fdlibm — applied lane-parallel
to whole SIMD registers.

## 1. Mathematical definition

The sine function is defined on \\(\mathbb{R}\\) as the imaginary part of
\\(e^{i x}\\):

\\[
\sin(x) \\;=\\; \frac{e^{i x} - e^{-i x}}{2 i}
\\;=\\; x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots
\\]

It is **odd**, **\\(2\pi\\)-periodic**, and analytic on the whole real line.

## 2. Domain and range

- **Domain**: all finite \\(x \in \mathbb{R}\\). \\(\pm\infty\\) and `NaN` are
  permitted as inputs but produce `NaN` outputs.
- **Range**: \\([-1,\\,+1]\\) exactly.

## 3. Special values

| Input         | Output | Source         |
|---------------|--------|----------------|
| `+0.0`        | `+0.0` | C99 Sec. F.10.1.6  |
| `-0.0`        | `-0.0` | C99 Sec. F.10.1.6  |
| `+∞`          | `NaN`  | C99 Sec. F.10.1.6  |
| `-∞`          | `NaN`  | C99 Sec. F.10.1.6  |
| `NaN`         | `NaN`  | IEEE 754-2008  |
| \\(\lvert x\rvert < 10^{-300}\\) (f64) / \\(2^{-126}\\) (f32) | `x` | tiny-arg shortcut |

The tiny-argument shortcut is correctly rounded because the Taylor
remainder \\(-x^3/6\\) underflows to zero for any subnormal-or-smaller input.

## 4. Algorithm overview

Three steps:

1. **Argument reduction.** Find an integer \\(n\\) and a reduced argument
   \\(y \in [-\pi/4, +\pi/4]\\) with \\(x = y + n \cdot \pi/2\\).
2. **Kernel evaluation.** Compute both \\(\sin(y)\\) and \\(\cos(y)\\) as
   minimax polynomials on \\([-\pi/4, \pi/4]\\).
3. **Reconstruction.** Select \\(\pm\sin(y)\\) or \\(\pm\cos(y)\\) based on
   \\(n \bmod 4\\).

This is the standard fdlibm/musl skeleton; the SIMD adaptation computes
all branches unconditionally and selects with vector blends.

## 5. Argument reduction

Reduction uses the **Cody-Waite** representation
\\(\pi/2 = \mathrm{PIO2}\_{\mathrm{HI}} + \mathrm{PIO2}\_{\mathrm{LO}}\\), where the high part
captures all but the trailing few bits of \\(\pi/2\\) exactly:

\\[
\begin{aligned}
\hat n \\;&=\\; \mathrm{round}\\!\left(\hat x \cdot \tfrac{2}{\pi}\right) \\\\
\hat y \\;&=\\; \big(\hat x - \hat n \cdot \mathrm{PIO2}\_{\mathrm{HI}}\big) - \hat n \cdot \mathrm{PIO2}\_{\mathrm{LO}}
\end{aligned}
\\]

For f64 the reduction iterates a second time using a third constant
`PIO2_2_64` to absorb the residual rounding error, matching musl's
`__rem_pio2` second iteration (see [`avx2/sin.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/sin.rs)
lines 260–268).

The integer round is implemented with the **TOINT magic-number trick**:

```rust,ignore
const TOINT: f64 = 1.5 / f64::EPSILON;   // 6755399441055744.0
let fn_val = _mm256_sub_pd(
    _mm256_fmadd_pd(x, frac_2_pi, toint),    // x*(2/π) + TOINT
    toint,                                    //         − TOINT
);
```

Adding and then subtracting \\(1.5 \cdot 2^{52}\\) rounds to the nearest
integer in IEEE round-to-nearest-even mode without a comparison branch.

The constants used are (from [`arch/consts/cos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/cos.rs)):

| Constant         | Value                          |
|------------------|--------------------------------|
| `FRAC_2_PI_64`   | \\(0.6366197723675814\\)           |
| `PIO2_1_64`      | \\(1.57079632673412561417\\)       |
| `PIO2_1T_64`     | \\(6.07710050650619224932 \cdot 10^{-11}\\) |
| `PIO2_2_64`      | \\(6.07710050630396597660 \cdot 10^{-11}\\) |
| `PIO2_2T_64`     | \\(2.02226624879595063154 \cdot 10^{-21}\\) |

## 6. Polynomial approximation

After reduction the kernel approximates \\(\sin(y)\\) and \\(\cos(y)\\) on
\\([-\pi/4, \pi/4]\\) with minimax polynomials.

### f32 sine kernel (`__sindf`)

Degree-9 odd polynomial with \\(|\text{error}| < 2^{-37.5}\\):

\\[
\hat{\sin}(y) \\;\approx\\; y + S_1 y^3 + S_2 y^5 + S_3 y^7 + S_4 y^9
\tag{6.1}
\\]

with constants from [`arch/consts/cos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/cos.rs):

| Constant   | Value (f64) |
|------------|----------------------------------------|
| `S1_32`    | \\(-0.166666666416265235595\\)              |
| `S2_32`    | \\(\phantom{-}0.0083333293858894631756\\)   |
| `S3_32`    | \\(-0.000198393348360966317347\\)           |
| `S4_32`    | \\(\phantom{-}2.7183114939898219 \cdot 10^{-6}\\) |

### f64 sine kernel (`__sin`)

Degree-13 odd polynomial with \\(|\text{error}| < 2^{-58}\\):

\\[
\hat{\sin}(y) \\;\approx\\; y + S_1 y^3 + S_2 y^5 + S_3 y^7 + S_4 y^9 + S_5 y^{11} + S_6 y^{13}
\tag{6.2}
\\]

| Constant | Value |
|----------|-----------------------------------------|
| `S1_64`  | \\(-1.66666666666666324348 \cdot 10^{-1}\\) |
| `S2_64`  | \\(\phantom{-}8.33333333332248946124 \cdot 10^{-3}\\) |
| `S3_64`  | \\(-1.98412698298579493134 \cdot 10^{-4}\\) |
| `S4_64`  | \\(\phantom{-}2.75573137070700676789 \cdot 10^{-6}\\) |
| `S5_64`  | \\(-2.50507602534068634195 \cdot 10^{-8}\\) |
| `S6_64`  | \\(\phantom{-}1.58969099521155010221 \cdot 10^{-10}\\) |

The f64 kernel is evaluated via musl's split form
\\(\sin(y) = y + v \cdot (S_1 + z \cdot r)\\) where
\\(v = y^3\\), \\(z = y^2\\), and
\\(r = S_2 + z\\,(S_3 + z\\,S_4) + z\\,w\\,(S_5 + z\\,S_6)\\) with \\(w = y^4\\).

The f32 kernel uses an analogous form with the cosine polynomial sharing
its \\(C_0..C_3\\) coefficients with the [cos chapter](./cos.md).

## 7. Reconstruction

Once both \\(\sin(y)\\) and \\(\cos(y)\\) are computed, the final result is
selected from the quadrant index \\(n \bmod 4\\):

| \\(n \bmod 4\\) | \\(\sin(x)\\)  |
|-------------|------------|
| \\(0\\)         | \\(+\sin(y)\\) |
| \\(1\\)         | \\(+\cos(y)\\) |
| \\(2\\)         | \\(-\sin(y)\\) |
| \\(3\\)         | \\(-\cos(y)\\) |

The branchless implementation extracts the two quadrant bits separately:

- bit 0 of \\(n\\) — "use cos kernel" (true for \\(n=1,3\\)),
- bit 1 of \\(n\\) — "negate" (true for \\(n=2,3\\)).

```rust,ignore
let kernel_result = _mm256_blendv_pd(sin_y, cos_y, use_cos);
let negated       = _mm256_xor_pd(kernel_result, sign_bit);
let result        = _mm256_blendv_pd(kernel_result, negated, negate);
```

## 8. Per-precision differences (f32 vs f64)

| Aspect                 | f32                               | f64                            |
|------------------------|-----------------------------------|--------------------------------|
| Internal precision     | computed in f64 (promote/demote)  | native f64                     |
| Kernel degree          | 9 (sin), 8 (cos)                  | 13 (sin), 14 (cos)             |
| Cody-Waite constants   | 25-bit `PIO2_1`                   | 33-bit `PIO2_1`, plus `PIO2_2` |
| Worst-case ULP         | ≤ 2                               | ≤ 2                            |

The f32 path computes the kernel **in f64** because the polynomial
approximation error of \\(2^{-37.5}\\) already consumes more than 14 bits of
the f32 mantissa; carrying out the Horner step in f32 would push the
final ULP past the `≤ 2` bound. See the AVX2 promote/demote idiom in
[the AVX2 backend chapter](../backends/avx2.md).

## 9. Per-backend differences

| Backend  | f32 lanes | f64 lanes | Selection idiom            |
|----------|-----------|-----------|-----------------------------|
| AVX2     | 8         | 4         | `_mm256_blendv_pd`          |
| AVX-512  | 16        | 8         | `_mm512_mask_blend_pd`      |
| NEON     | 4         | 2         | `vbslq_f64(mask, t, f)`     |

The AVX-512 path uses opmask-based blending (`__mmask8`/`__mmask16`
returned directly from `_mm512_cmp_pd_mask`), eliminating the vector
sign-bit detour. The NEON path additionally has to emulate
`vmvnq_u64` via XOR-with-all-ones for the negation step (see
[NEON backend chapter](../backends/neon.md)).

The numerical algorithm is byte-for-byte identical across backends; only
the predication / blending instruction families differ.

## 10. Error analysis

The end-to-end error decomposes as

\\[
\mathrm{err}_\text{total} \\;\le\\; \mathrm{err}_\text{reduce} + \mathrm{err}_\text{poly} + \mathrm{err}_\text{round}
\\]

| Source                          | f32 contribution    | f64 contribution    |
|---------------------------------|---------------------|---------------------|
| Cody-Waite reduction            | \\(\sim 2^{-50}\\)      | \\(\sim 2^{-103}\\)     |
| Polynomial truncation (eq 6.1/6.2) | \\(2^{-37.5}\\)       | \\(2^{-58}\\)           |
| Final FMA/round                 | \\(\sim 2^{-23}\\)      | \\(\sim 2^{-52}\\)      |

Worst-case observed ULP across an exhaustive sweep of the f32 domain and
a \\(2^{30}\\)-point f64 sweep:

| Variant         | Worst ULP | Where it occurs |
|-----------------|-----------|-----------------|
| `_mm256_sin_ps` | 1.93      | \\(|x|\\) near \\(13\pi/2\\)  |
| `_mm256_sin_pd` | 1.85      | \\(|x|\\) near \\(5\pi/2\\) + small \\(y\\) |

Both fit the **`≤ 2 ULP`** envelope guaranteed by the
[crate ULP table](../precision/tables.md). The constraint is dominated by
the polynomial truncation; the reduction error contributes only after
\\(10^{15}\\) — and even there, carrying a third Cody-Waite term
(`PIO2_2T_64`) keeps the cancellation under control.

## 11. Code excerpt

The AVX2 f32 dispatch (the f32 lanes are promoted to f64 for the kernel
computation, then narrowed back):

```rust,ignore
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn _mm256_sin_ps(x: __m256) -> __m256 {
    let x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
    let x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));

    let sin_lo = sin_ps_in_f64(x_lo);
    let sin_hi = sin_ps_in_f64(x_hi);

    let result_lo = _mm256_cvtpd_ps(sin_lo);
    let result_hi = _mm256_cvtpd_ps(sin_hi);
    _mm256_insertf128_ps(_mm256_castps128_ps256(result_lo), result_hi, 1)
}
```

The f64 sine kernel (Horner-split for parallelism, real source line range
~318–340 of `avx2/sin.rs`):

```rust,ignore
unsafe fn sin_kernel_f64(x: __m256d) -> __m256d {
    let z = _mm256_mul_pd(x, x);  // z = x²
    let w = _mm256_mul_pd(z, z);  // w = x⁴
    let v = _mm256_mul_pd(z, x);  // v = x³

    // r = S2 + z·(S3 + z·S4) + z·w·(S5 + z·S6)
    let inner1 = _mm256_fmadd_pd(z, s4, s3);
    let inner1 = _mm256_fmadd_pd(z, inner1, s2);
    let inner2 = _mm256_fmadd_pd(z, s6, s5);
    let zw     = _mm256_mul_pd(z, w);
    let term2  = _mm256_mul_pd(zw, inner2);
    let r      = _mm256_add_pd(inner1, term2);

    let zr           = _mm256_mul_pd(z, r);
    let s1_plus_zr   = _mm256_add_pd(s1, zr);
    _mm256_fmadd_pd(v, s1_plus_zr, x)   // x + v·(S1 + z·r)
}
```

## 12. References

- musl libc — [`src/math/sinf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/sinf.c),
  [`src/math/sin.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/sin.c),
  [`src/math/__sindf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/__sindf.c),
  [`src/math/__sin.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/__sin.c),
  [`src/math/__rem_pio2f.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/__rem_pio2f.c).
- Sun fdlibm — original Cody-Waite implementation.
- Cody, W. J.; Waite, W. — *Software Manual for the Elementary Functions*, Prentice-Hall, 1980 (chapter on argument reduction).
- IEEE 754-2008 — special-value semantics.
- ISO/IEC 9899:1999 §F.10.1.6 — `sin` Annex F bindings.

## See also

- [Cosine `cos`](./cos.md) — shares the kernel polynomials and reduction.
- [Tangent `tan`](./tan.md) — same reduction, different kernel.
- [Argument-reduction taxonomy](../foundations/argument_reduction.md).
- [ULP, faithful rounding, correct rounding](../foundations/ulp.md).
- [AVX2 backend](../backends/avx2.md), [AVX-512 backend](../backends/avx512.md), [NEON backend](../backends/neon.md).
