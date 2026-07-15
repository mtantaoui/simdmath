# IEEE-754 in two slides

Every claim in this book — "ULP", "subnormal", "round-to-nearest-even" — is
ultimately a claim about the IEEE-754 binary floating-point formats. This
chapter is a compact reference for the parts that actually matter when
implementing `sin`, `exp`, `cbrt` and friends. We deliberately collapse the
2008/2019 standard's hundred-plus pages into a single chapter focused on
`binary32` (Rust's `f32`) and `binary64` (`f64`).

## The bit layout

Both formats share the same shape: a sign bit, a biased exponent, and a
significand (also called *mantissa*).

| Format     | Total | Sign | Exponent | Significand | Bias  | Precision \\(p\\) |
|------------|-------|------|----------|-------------|-------|----------------|
| `binary32` | 32    | 1    | 8        | 23          | 127   | 24             |
| `binary64` | 64    | 1    | 11       | 52          | 1023  | 53             |

Note that the *precision* \\(p\\) is one more than the number of stored
significand bits, because of the **implicit leading 1** in normal numbers.

## The encoding rule

Let \\(s \in \{0,1\}\\) be the sign, \\(E\\) the unsigned integer formed by the
exponent field, \\(T\\) the integer formed by the significand field, and
\\(\mathrm{bias} = 2^{w-1} - 1\\) where \\(w\\) is the exponent field width
from the table above (8 for `binary32`, 11 for `binary64`). Then:

- **Zero** (\\(E = 0,\ T = 0\\)): \\((-1)^s \cdot 0\\) — yes, \\(\pm 0\\) are distinct.
- **Subnormal** (\\(E = 0,\ T \neq 0\\)):
  \\[ x = (-1)^s \cdot 2^{1 - \mathrm{bias}} \cdot 0.T \\]
  with no implicit leading 1.
- **Normal** (\\(1 \le E \le 2^w - 2\\)):
  \\[ x = (-1)^s \cdot 2^{E - \mathrm{bias}} \cdot 1.T \\]
  with the **implicit** leading 1.
- **Infinity** (\\(E = 2^w - 1,\ T = 0\\)): \\((-1)^s \cdot \infty\\).
- **NaN** (\\(E = 2^w - 1,\ T \neq 0\\)): "Not a Number". The MSB of \\(T\\)
  distinguishes signalling (sNaN, MSB = 0) from quiet (qNaN, MSB = 1) on
  most architectures.

The reason for the bias is so that the encoding is *lexicographically
ordered*: comparing two non-negative IEEE-754 numbers as 32-bit (or 64-bit)
unsigned integers gives the correct numeric ordering. This trick underlies
the ULP-difference computation in [ULP, faithful rounding, correct rounding](./ulp.md).

## A worked example: \\(1.0\_{f32}\\)

The number \\(1.0\\) in `binary32` is encoded as `0x3F800000`. Let's decode it:

```text
bits  : 0 01111111 00000000000000000000000
fields: s   E=127     T=0
value : (-1)^0 · 2^(127 - 127) · 1.0 = 1.0
```

In Rust:

```rust
fn main() {
    assert_eq!(1.0_f32.to_bits(), 0x3F80_0000);
    assert_eq!(f32::from_bits(0x3F80_0000), 1.0_f32);

    // The next representable float above 1.0 is 1.0 + ulp(1.0):
    let next = f32::from_bits(0x3F80_0001);
    assert_eq!(next - 1.0_f32, f32::EPSILON);  // 2^-23
}
```

The same number in `binary64` is `0x3FF0_0000_0000_0000`, with bias 1023.

## Special exponent values, in pictures

```text
binary32 exponent field E (8 bits)
0          1 ··· 254          255
│          │                  │
└─ subnormal/zero             └─ ∞ / NaN
   (no implicit 1)            └─ normal numbers (1 ≤ E ≤ 254)
                                  with bias 127
```

A consequence of the implicit leading 1 is that the gap between a normal
number \\(x\\) and its successor is \\(2^{E - \mathrm{bias} - (p-1)}\\), doubling
every time you cross a power of two — see the next chapter for the formal
ULP definition.

## Rounding modes and the default

IEEE-754 specifies five rounding modes:

| Mode                                      | Symbol  | Direction                                       |
|-------------------------------------------|---------|-------------------------------------------------|
| Round to nearest, ties to even (default)  | RN-TE   | nearest representable; ties go to even mantissa |
| Round to nearest, ties away from zero     | RN-TA   | nearest; ties magnitude-up                      |
| Round toward \\(+\infty\\)                    | RU      | upward                                          |
| Round toward \\(-\infty\\)                    | RD      | downward                                        |
| Round toward 0                            | RZ      | truncate                                        |

`simdmath` assumes **RN-TE** throughout — it is the default in every modern
hardware FPU, in `glibc`, in Rust, and in `#[target_feature]`-enabled SIMD.
None of our error analyses are valid in any other mode.

## The five floating-point exceptions

IEEE-754 defines five sticky exception flags:

- **Invalid operation** — e.g. \\(\sqrt{-1}\\), \\(0/0\\), \\(\infty - \infty\\). Result: NaN.
- **Division by zero** — finite non-zero divided by zero. Result: \\(\pm\infty\\).
- **Overflow** — magnitude of infinitely-precise result exceeds the format's
  largest finite. Result: \\(\pm\infty\\) in RN-TE.
- **Underflow** — non-zero result with magnitude below the smallest normal.
  Result: subnormal or zero, with possible loss of precision.
- **Inexact** — the rounded result differs from the exact result. This flag
  is raised on **every** non-exact operation (including most multiplies);
  most code ignores it.

`simdmath` does not currently expose the flags through its public API. The
underlying intrinsics raise them as the hardware would, and they can be
inspected with the platform's `fenv.h` equivalents.

## Subnormals: the awkward case

Subnormals fill the "underflow gap" between \\(0\\) and the smallest normal
\\(2^{1 - \mathrm{bias}}\\). They preserve the property that
\\(x = y \iff x - y = 0\\) (no spurious zeros from cancellation), which is why
they exist. The cost is twofold:

1. They have **fewer significand bits** than normals (down to 1 bit for the
   smallest subnormal), so error analyses that quote \\(\varepsilon = 2^{-p}\\)
   must add a special clause for subnormal results.
2. On many CPUs they trigger a **microcode-assisted slow path**, costing
   100–1000× a normal op. Under x86 you can opt into "flush-to-zero" (FTZ)
   and "denormals-are-zero" (DAZ) via `MXCSR` to avoid the slowdown.

`simdmath` aims to be correct for subnormal *inputs* but does not promise
sub-ULP accuracy when the *output* is subnormal — see
[Why ≤ 1 ULP and not correctly rounded](./why_not_correct.md).

## How Rust exposes IEEE-754

The relevant pieces of the standard library:

```rust
// Constants
let _ = f32::EPSILON;        //  2^-23 = 1.1920929e-7
let _ = f32::MIN_POSITIVE;   //  smallest *normal* (not subnormal)
let _ = f32::INFINITY;
let _ = f32::NAN;
let _ = f64::EPSILON;        //  2^-52
let _ = f64::MIN_POSITIVE;

// Bit casts (always free, no rounding)
let bits: u32 = 1.0_f32.to_bits();
let x:    f32 = f32::from_bits(0x3f80_0000);

// Classification
let _ = (-0.0_f32).is_sign_negative();   // true
let _ = f32::NAN.is_nan();
let _ = 1.0e-40_f32.is_subnormal();      // 1e-40 < f32::MIN_POSITIVE
```

`to_bits` / `from_bits` are guaranteed to be no-ops at the machine level — they
only exist to express the *type* change to the borrow checker. Every kernel
in `simdmath` uses them to manipulate the exponent directly.

## A short historical note

The original 1985 standard (IEEE 754-1985) specified only `binary32` and
`binary64`. The 2008 revision (IEEE 754-2008) added `binary16` (used in
`simdmath`'s NEON `fp16` extension), `binary128`, and the *decimal* formats
`decimal32/64/128`. The 2019 minor revision tightened semantics around NaN
propagation and added `minimumNumber`/`maximumNumber`. Rust currently
exposes only `binary32` and `binary64` as stable types; `f16` and `f128` are
nightly-only as of writing.

## See also

- [ULP, faithful rounding, correct rounding](./ulp.md)
- [Why `≤ 1 ULP` and not `correctly rounded`](./why_not_correct.md)
- [Compensated arithmetic: two-sum and Dekker product](./compensated.md)
- IEEE Computer Society, *IEEE Standard for Floating-Point Arithmetic*,
  IEEE Std 754-2019.
- Goldberg, D. (1991), *What Every Computer Scientist Should Know About
  Floating-Point Arithmetic*, ACM Computing Surveys 23(1).
- Muller, J.-M. *et al.* (2018), *Handbook of Floating-Point Arithmetic*,
  2nd ed., Birkhäuser.
