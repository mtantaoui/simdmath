# Cosine `cos`

Cosine shares the entire argument-reduction infrastructure with [sin](./sin.md);
only the quadrant-selection table and the kernel coefficients differ. This
chapter focuses on those differences and refers back to the sin chapter for
the shared machinery.

## 1. Mathematical definition

\\[
\cos(x) \\;=\\; \frac{e^{i x} + e^{-i x}}{2}
\\;=\\; 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots
\\]

Cosine is **even**, **\\(2\pi\\)-periodic**, and analytic on \\(\mathbb{R}\\).

## 2. Domain and range

- **Domain**: all finite \\(x\\). \\(\pm\infty\\) produce `NaN`.
- **Range**: \\([-1, +1]\\).

## 3. Special values

| Input           | Output | Source        |
|-----------------|--------|---------------|
| `+0.0`          | `+1.0` | C99 Sec. F.10.1.5 |
| `-0.0`          | `+1.0` | C99 Sec. F.10.1.5 |
| `+∞`            | `NaN`  | C99 Sec. F.10.1.5 |
| `-∞`            | `NaN`  | C99 Sec. F.10.1.5 |
| `NaN`           | `NaN`  | IEEE 754-2008 |
| \\(\lvert x\rvert < 2^{-27}\sqrt 2\\) (f64) | `1.0` | tiny-arg shortcut |

## 4. Algorithm overview

Identical to [sin](./sin.md):

1. Cody-Waite reduction of \\(x\\) to \\(y \in [-\pi/4, \pi/4]\\) with quadrant
   index \\(n\\).
2. Compute both \\(\sin(y)\\) and \\(\cos(y)\\) via minimax polynomials.
3. Reconstruct \\(\cos(x)\\) from the quadrant.

## 5. Argument reduction

Same constants and same Cody-Waite skeleton as sin (see
[Sine, Sec. 5](./sin.md#5-argument-reduction)). The constants module
[`arch/consts/cos.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/cos.rs)
is the canonical home of `FRAC_2_PI_*`, `PIO2_1_*`, `PIO2_1T_*`, and
`PIO2_2_*`/`PIO2_2T_*`; `consts/sin.rs` re-exports from there.

## 6. Polynomial approximation

### f32 cosine kernel (`__cosdf`)

Degree-8 even polynomial with \\(|\text{error}| < 2^{-34.1}\\):

\\[
\hat{\cos}(y) \\;\approx\\; 1 + C_0 y^2 + C_1 y^4 + C_2 y^6 + C_3 y^8
\tag{6.1}
\\]

| Constant | Value (f64) |
|----------|-----------------------------------------|
| `C0_32`  | \\(-0.499999997251031003120\\)              |
| `C1_32`  | \\(\phantom{-}0.0416666233237390631894\\)   |
| `C2_32`  | \\(-0.00138867637746099294692\\)            |
| `C3_32`  | \\(\phantom{-}2.43904487962774090654 \cdot 10^{-5}\\) |

### f64 cosine kernel (`__cos`)

Degree-14 polynomial with \\(|\text{error}| < 2^{-58}\\), written as
\\(\cos(y) \approx 1 - y^2/2 + C_1 y^4 + \cdots + C_6 y^{14}\\):

| Constant | Value |
|----------|-----------------------------------------|
| `C1_64`  | \\(\phantom{-}4.16666666666666019037 \cdot 10^{-2}\\) |
| `C2_64`  | \\(-1.38888888888741095749 \cdot 10^{-3}\\) |
| `C3_64`  | \\(\phantom{-}2.48015872894767294178 \cdot 10^{-5}\\) |
| `C4_64`  | \\(-2.75573143513906633035 \cdot 10^{-7}\\) |
| `C5_64`  | \\(\phantom{-}2.08757232129817482790 \cdot 10^{-9}\\) |
| `C6_64`  | \\(-1.13596475577881948265 \cdot 10^{-11}\\) |

The leading \\(-y^2/2\\) is computed exactly (no FMA rounding because
\\(y^2 / 2\\) is just a sign-bit toggle on the exponent), then the
higher-order corrections are folded in by Horner FMA.

The sin kernel coefficients \\(S_1..S_6\\) are the same ones documented in
the [sin chapter](./sin.md#6-polynomial-approximation); cosine needs
both because some quadrants flip from `cos(y)` to `sin(y)` (see Sec. 7).

## 7. Reconstruction

The quadrant-selection table for cosine flips the roles relative to sin:

| \\(n \bmod 4\\) | \\(\cos(x)\\)  |
|-------------|------------|
| \\(0\\)         | \\(+\cos(y)\\) |
| \\(1\\)         | \\(-\sin(y)\\) |
| \\(2\\)         | \\(-\cos(y)\\) |
| \\(3\\)         | \\(+\sin(y)\\) |

Decoded into bits:

- bit 0 of \\(n\\) — "use sin kernel" (true for \\(n=1,3\\)),
- the negation pattern is \\(\text{negate} = ((n+1) \bmod 4) \in \{2,3\}\\),
  equivalently \\(((n+1) \\,\&\\, 2) \neq 0\\).

This is implemented as the same two-blend pattern as sin, but with the
indexing offset by one quadrant.

## 8. Per-precision differences (f32 vs f64)

| Aspect              | f32                              | f64                            |
|---------------------|----------------------------------|--------------------------------|
| Internal precision  | computed in f64 (promote/demote) | native f64                     |
| Kernel degree       | 8 (cos), 9 (sin)                 | 14 (cos), 13 (sin)             |
| Cody-Waite          | 25-bit `PIO2_1`                  | 33-bit `PIO2_1` + `PIO2_2`     |
| Worst-case ULP      | ≤ 2                              | ≤ 2                            |

The same f32-via-f64 evaluation strategy used in [sin](./sin.md) applies
here, for the same reason: the polynomial truncation alone consumes more
than the f32 mantissa headroom.

## 9. Per-backend differences

| Backend  | f32 lanes | f64 lanes | Selection idiom            |
|----------|-----------|-----------|-----------------------------|
| AVX2     | 8         | 4         | `_mm256_blendv_pd`          |
| AVX-512  | 16        | 8         | `_mm512_mask_blend_pd`      |
| NEON     | 4         | 2         | `vbslq_f64(mask, t, f)`     |

See [AVX2](../backends/avx2.md), [AVX-512](../backends/avx512.md), and
[NEON](../backends/neon.md) for backend-specific details.

## 10. Error analysis

| Source                          | f32 contribution | f64 contribution |
|---------------------------------|-------------------|-------------------|
| Cody-Waite reduction            | \\(\sim 2^{-50}\\)    | \\(\sim 2^{-103}\\)   |
| Polynomial truncation (eq 6.1)  | \\(2^{-34.1}\\)       | \\(2^{-58}\\)         |
| Final FMA/round                 | \\(\sim 2^{-23}\\)    | \\(\sim 2^{-52}\\)    |

Worst-case observed ULP:

| Variant         | Worst ULP | Where it occurs                 |
|-----------------|-----------|---------------------------------|
| `_mm256_cos_ps` | 1.97      | \\(y\\) near \\(\pi/4\\) (boundary)     |
| `_mm256_cos_pd` | 1.92      | very large \\(|x|\\), post-reduction |

Both variants honour the **`≤ 2 ULP`** envelope from the
[crate ULP table](../precision/tables.md).

The "boundary" mode at \\(y \approx \pi/4\\) is interesting: at that point
\\(\cos(y) \approx \sin(y) \approx 1/\sqrt 2\\), which means the choice
between the cos kernel and the sin kernel is on the brink of crossing
over. The minimax polynomials happen to disagree by ~1 ULP at the
crossover, and which one is used in any given lane depends on rounding
of \\(\hat n = \text{round}(\hat x \cdot 2/\pi)\\).

## 11. Code excerpt

The cos f64 quadrant-decode portion (compare with the sin variant):

```rust,ignore
// n mod 4: 0→cos(y), 1→-sin(y), 2→-cos(y), 3→sin(y)
let n_and_1 = _mm256_and_si256(n_256, one);   // bit 0: use sin kernel
let n_plus_1 = _mm256_add_epi64(n_256, one);  // shift quadrant by 1
let n_p1_and_2 = _mm256_and_si256(n_plus_1, two);

let use_sin = _mm256_cmpeq_epi64(n_and_1, one);
let negate  = _mm256_cmpeq_epi64(n_p1_and_2, two);

let kernel_result = _mm256_blendv_pd(cos_y, sin_y, _mm256_castsi256_pd(use_sin));
let negated       = _mm256_xor_pd(kernel_result, _mm256_set1_pd(-0.0));
let result        = _mm256_blendv_pd(kernel_result,
                                     negated,
                                     _mm256_castsi256_pd(negate));
```

The shared kernel polynomials are the same `sin_kernel_f64` /
`cos_kernel_f64` helpers used by [sin](./sin.md).

## 12. References

- musl libc — [`src/math/cosf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/cosf.c),
  [`src/math/cos.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/cos.c),
  [`src/math/__cosdf.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/__cosdf.c),
  [`src/math/__cos.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/__cos.c).
- Sun fdlibm — original `__cos` and `__cosdf` polynomial coefficients.
- IEEE 754-2008 — special-value semantics.
- ISO/IEC 9899:1999 §F.10.1.5 — `cos` Annex F bindings.

## See also

- [Sine `sin`](./sin.md) — companion function, shares reduction and kernel coefficients.
- [Tangent `tan`](./tan.md) — same reduction, ratio of sin to cos.
- [Argument-reduction taxonomy](../foundations/argument_reduction.md).
- [AVX2](../backends/avx2.md), [AVX-512](../backends/avx512.md), [NEON](../backends/neon.md).
