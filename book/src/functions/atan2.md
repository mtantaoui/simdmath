# Two-argument arc tangent `atan2`

`atan2(y, x)` returns the angle, in radians, of the point \\((x, y)\\) measured
counter-clockwise from the positive \\(x\\)-axis. Unlike the single-argument
\\(\mathrm{atan}(y/x)\\), it inspects both signs and so resolves the quadrant
ambiguity, giving a continuous angle on the full circle.

## 1. Mathematical definition

For real \\((x, y) \neq (0, 0)\\), \\(\mathrm{atan2}\\) is the unique
\\(\theta \in (-\pi, \pi]\\) such that

\\[
x = r\cos\theta, \qquad y = r\sin\theta, \qquad r = \sqrt{x^2 + y^2}.
\\]

Equivalently, away from the negative real axis,

\\[
\mathrm{atan2}(y, x) \\;=\\; 2\\,\mathrm{atan}\\!\left(\frac{y}{r + x}\right).
\\]

The function is the principal-branch argument of the complex number
\\(x + iy\\), and is the natural inverse of the polar-coordinate transform
\\((r, \theta) \mapsto (r\cos\theta,\\, r\sin\theta)\\).

## 2. Domain and range

| Input | Domain |
|-------|--------|
| \\(x\\)   | \\(\mathbb{R} \cup \{\pm\infty, \text{NaN}\}\\) |
| \\(y\\)   | \\(\mathbb{R} \cup \{\pm\infty, \text{NaN}\}\\) |

The mathematical range is \\((-\pi,\pi]\\). Because \\(\pi\\) is not exactly
representable in binary floating point, the implementation actually returns
values in \\([-\pi_\text{fp},\\, \pi_\text{fp}]\\) where \\(\pi_\text{fp}\\) is the
nearest representable value to \\(\pi\\). The two-sum split \\(\pi = \pi_\text{hi}
+ \pi_\text{lo}\\) from [`src/arch/consts/atan2.rs`][src-consts] is used so that
the angle obtained near the negative real axis is faithfully rounded.

[src-consts]: https://github.com/mtantaoui/simdmath/blob/main/src/arch/consts/atan2.rs

For f32, `PI_HI_32 = 3.141_592_741_01` and
`PI_LO_32 = -8.742_278e-8`; for f64, `PI_HI_64 ≈ 3.141_592_653_589_793_1`
and `PI_LO_64 ≈ 1.224_646_799e-16`.

## 3. Special values (IEEE 754 / C99 Sec. F.10.1.4)

| \\(y\\)         | \\(x\\)         | result   |
|-------------|-------------|----------|
| \\(\pm 0\\)     | \\(+0\\)        | \\(\pm 0\\)  |
| \\(\pm 0\\)     | \\(-0\\)        | \\(\pm\pi\\) |
| \\(\pm 0\\)     | \\(x > 0\\)     | \\(\pm 0\\)  |
| \\(\pm 0\\)     | \\(x < 0\\)     | \\(\pm\pi\\) |
| \\(y > 0\\)     | \\(\pm 0\\)     | \\(+\pi/2\\) |
| \\(y < 0\\)     | \\(\pm 0\\)     | \\(-\pi/2\\) |
| \\(\pm\infty\\) | \\(+\infty\\)   | \\(\pm\pi/4\\) |
| \\(\pm\infty\\) | \\(-\infty\\)   | \\(\pm 3\pi/4\\) |
| \\(\pm\infty\\) | finite      | \\(\pm\pi/2\\) |
| finite      | \\(+\infty\\)   | \\(\pm 0\\)  |
| finite      | \\(-\infty\\)   | \\(\pm\pi\\) |
| \\(\text{NaN}\\)| any         | \\(\text{NaN}\\) |
| any         | \\(\text{NaN}\\)| \\(\text{NaN}\\) |

This is exactly the table reproduced verbatim in the doc comment of
[`src/arch/avx2/atan2.rs`][src-avx2]. The `±` in the result follows
\\(\mathrm{sign}(y)\\) in every row.

[src-avx2]: https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/atan2.rs

## 4. Algorithm overview

`atan2` is a separate function — not just a wrapper around \\(\mathrm{atan}\\) —
because the single-argument arctangent collapses both quadrants \\(\{x>0, y\}\\)
and \\(\{x<0, -y\}\\) to the same value. `atan2` re-introduces the lost sign of
\\(x\\) as an additive correction \\(\pm\pi\\).

The implementation follows musl libc's `atan2.c` / `atan2f.c` and proceeds
in four branchless steps:

1. **Quadrant encoding.** Combine the sign bits of \\(x\\) and \\(y\\) into a
   2-bit code \\(m = 2\\,\mathrm{sign}(x) + \mathrm{sign}(y)\\).
2. **Special-case detection.** Build masks for NaN, \\(x = 0\\), \\(y = 0\\),
   \\(x = \pm\infty\\), and the *huge ratio* case \\(|y/x|\\) above
   \\(2^{26}\\) (f32) or \\(2^{60}\\) (f64), where the result is indistinguishable
   from \\(\pm\pi/2\\).
3. **Generic case.** Compute \\(\alpha = \mathrm{atan}(|y/x|)\\) using the
   already-vectorised single-argument `atan` kernel.
4. **Quadrant correction.** Apply the \\(m\\)-dependent affine map to \\(\alpha\\),
   then blend against the special-case constants in priority order.

| \\(m\\) | \\(\mathrm{sign}(x)\\) | \\(\mathrm{sign}(y)\\) | quadrant | result |
|-----|--------------------|--------------------|----------|--------|
| 0   | \\(+\\) | \\(+\\) | I   | \\(\alpha\\) |
| 1   | \\(+\\) | \\(-\\) | IV  | \\(-\alpha\\) |
| 2   | \\(-\\) | \\(+\\) | II  | \\(\pi - \alpha\\) |
| 3   | \\(-\\) | \\(-\\) | III | \\(\alpha - \pi\\) |

## 5. Argument reduction

There is no polynomial of its own here: `atan2` reuses [`atan`](./atan.md)
for the radial component. The reduction it performs is purely *categorical*:

- Replace \\((y, x)\\) by \\((|y|, |x|)\\), then divide. Sign is restored later.
- If \\(|y/x|\\) would overflow, replace it by the saturated value
  \\(\pm\infty\\), which `atan` maps to \\(\pm\pi/2\\) exactly.

The threshold `HUGE_RATIO_THRESHOLD_32 = 26 << 23` (and the f64 equivalent
`60 << 52`) is compared against the *integer* exponent difference
\\(\mathrm{exp}(|y|) - \mathrm{exp}(|x|)\\), so the test is a single integer
subtraction and compare per lane.

## 6. Polynomial / kernel approximation

The polynomial work is delegated to [`atan`](./atan.md). For the dedicated
\\(\pi/4\\) and \\(3\pi/4\\) branches the constants come from
[`src/arch/consts/atan2.rs`][src-consts]:

```rust,ignore
pub(crate) const FRAC_PI_2_32:    f32 = 1.570_796_370_51;
pub(crate) const FRAC_PI_4_32:    f32 = 0.785_398_185_25;
pub(crate) const FRAC_3_PI_4_32:  f32 = 2.356_194_496_155;
pub(crate) const PI_HI_32:        f32 = 3.141_592_741_01;
pub(crate) const PI_LO_32:        f32 = -8.742_278e-8;

pub(crate) const FRAC_PI_2_64:    f64 = 1.570_796_326_794_896_558_00;
pub(crate) const FRAC_PI_4_64:    f64 = 0.785_398_163_397_448_279_00;
pub(crate) const FRAC_3_PI_4_64:  f64 = 2.356_194_490_192_344_836_91;
pub(crate) const PI_HI_64:        f64 = 3.141_592_653_589_793_115_998;
pub(crate) const PI_LO_64:        f64 = 1.224_646_799_147_353_207_17e-16;
```

The two-sum split \\(\pi_\text{hi} + \pi_\text{lo} = \pi\\) guarantees that the
quadrant-correction add \\(\alpha + \pi_\text{hi} + \pi_\text{lo}\\) retains
≤ 1 ULP near the negative real axis, where catastrophic cancellation would
otherwise dominate.

## 7. Reconstruction

Let \\(\alpha = \mathrm{atan}(|y/x|) \in [0, \pi/2]\\). The full reconstruction
is performed branchlessly by computing all four quadrant results in
parallel and selecting via `_mm256_blendv_*` / `vbslq_*`:

\\[
\theta \\;=\\;
\begin{cases}
\phantom{-}\alpha,                 & m = 0 \\\\
-\alpha,                           & m = 1 \\\\
\phantom{-}\pi_\text{hi} - \alpha + \pi_\text{lo}, & m = 2 \\\\
\phantom{-}\alpha - \pi_\text{hi} - \pi_\text{lo}, & m = 3
\end{cases}
\\]

Special-case overrides are then applied in priority order:

```text
NaN  →  x == ±∞  →  y == 0  →  x == 0  →  huge ratio  →  generic
```

The order matters: e.g. `atan2(±∞, ±∞)` would otherwise be classified
both as "x == ±∞" and "huge ratio"; the more specific branch must win.

## 8. Per-precision differences (f32 vs f64)

| aspect             | f32                      | f64                      |
|--------------------|--------------------------|--------------------------|
| huge-ratio cutoff  | \\(2^{26}\\)                 | \\(2^{60}\\)                 |
| \\(\pi\\) split        | hi = `0x40490FDB`, lo ≈ \\(-8.74\cdot10^{-8}\\) | hi = `0x400921FB54442D18`, lo ≈ \\(1.22\cdot10^{-16}\\) |
| inner kernel       | `atan_f32` (f32)         | `atan_f64` (f64)         |
| worst-case ULP     | \\(\le 3\\)                  | \\(\le 2\\)                  |

The f32 worst case is dominated by `atan_f32`'s own \\(\le 3\\) ULP. f64 is
strictly tighter because the underlying `atan_f64` is \\(\le 1\\) ULP and the
quadrant correction adds at most one further ULP via the
\\(\pi_\text{hi} + \pi_\text{lo}\\) compensated subtraction.

## 9. Per-backend differences (AVX2 / AVX-512 / NEON)

| backend  | f32 lanes | f64 lanes | mask blend                          |
|----------|-----------|-----------|--------------------------------------|
| AVX2     | 8         | 4         | `_mm256_blendv_ps` / `_mm256_blendv_pd` |
| AVX-512  | 16        | 8         | `_mm512_mask_blend_ps` (integer mask) |
| NEON     | 4         | 2         | `vbslq_f32` / `vbslq_f64` (arg order: mask, true, false) |

NEON has no direct equivalent of `_mm256_movemask_ps`; the special-case
masks are kept as full-width vectors. There is also no `vmvnq_u64`, so the
"not" used in priority blending is emulated as
`veorq_u64(x, all_ones)`. AVX-512 stores the masks as `__mmask8` /
`__mmask16` integers, which lets the priority-blend pyramid use plain
bitwise `&` and `|` instead of vector ops.

The arithmetic is otherwise identical across the three backends — they all
call into the corresponding `atan` kernel for the heavy lifting.

## 10. Error analysis

The error budget decomposes as

\\[
\mathrm{err}(\hat{\theta}) \\;\le\\;
\underbrace{\mathrm{err}(\hat\alpha)}\_{\le\\,3\text{ ULP (f32)}}
\\;+\\;
\underbrace{\mathrm{err}_\text{div}(|y|/|x|)}\_{\le\\,0.5\text{ ULP}}
\\;+\\;
\underbrace{\mathrm{err}_\text{quad}}\_{\le\\,1\text{ ULP}}.
\\]

The dominant term is the inner `atan` evaluation. The division
\\(|y|/|x|\\) is correctly rounded, contributing only its rounding error.
The quadrant correction \\(\pi_\text{hi} - \alpha + \pi_\text{lo}\\) is the
delicate piece: when \\(\alpha\\) is close to \\(\pi\\), naive
\\(\pi - \alpha\\) would lose all leading bits. The hi/lo split deflects
that cancellation into the lo term, which is then a rounding-noise-level
correction.

**Worst-case observed.** f32 reaches 3 ULP near \\(y/x \approx \tan(1)\\) with
\\(x < 0\\), where the quadrant correction stacks on top of `atan_f32`'s peak
error. f64 stays at \\(\le 2\\) ULP everywhere.

## 11. Code excerpt

From [`src/arch/avx2/atan2.rs`][src-avx2]:

```rust,ignore
// Quadrant encoding: m = 2*sign(x) + sign(y)
let sx = _mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(0x80000000_u32 as i32));
let sy = _mm256_and_si256(_mm256_castps_si256(y), _mm256_set1_epi32(0x80000000_u32 as i32));

// |y| / |x|
let ax = _mm256_andnot_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000_u32 as i32)), x);
let ay = _mm256_andnot_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000_u32 as i32)), y);
let alpha = _mm256_atan_ps(_mm256_div_ps(ay, ax));

// Four branchless quadrant results
let pi_hi = _mm256_set1_ps(PI_HI_32);
let pi_lo = _mm256_set1_ps(PI_LO_32);
let q2 = _mm256_add_ps(_mm256_sub_ps(pi_hi, alpha), pi_lo); // π - α
let q3 = _mm256_sub_ps(_mm256_sub_ps(alpha, pi_hi), pi_lo); // α - π
// Sign restoration via XOR with sx, sy as appropriate; full code is in the file.
```

The NEON variant differs only in the intrinsics used (`vsubq_f32`,
`vbslq_f32`); the math is line-for-line the same.

## 12. References

- musl libc: [`src/math/atan2.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/atan2.c) and [`src/math/atan2f.c`](https://git.musl-libc.org/cgit/musl/tree/src/math/atan2f.c) (fdlibm descent, Sun Microsystems 1993).
- IEEE 754-2008, §9.2.1: recommended operations, including the special-value table for `atan2`.
- ISO/IEC 9899:2018 (C18), §F.10.1.4: same special-value table.
- Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed., §11.4.2: argument-reduction techniques for inverse trigonometry.
- Repo source: [`src/arch/avx2/atan2.rs`][src-avx2], [`src/arch/avx512/atan2.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx512/atan2.rs), [`src/arch/neon/atan2.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/atan2.rs), [`src/arch/consts/atan2.rs`][src-consts].

## See also

- [Arc tangent `atan`](./atan.md) — single-argument inner kernel.
- [Sine `sin`](./sin.md), [Cosine `cos`](./cos.md) — the forward maps that
  `atan2` inverts when paired with \\(r = \sqrt{x^2+y^2}\\).
- [ULP, faithful rounding, correct rounding](../foundations/ulp.md) for the
  ULP definitions used in the error analysis.
