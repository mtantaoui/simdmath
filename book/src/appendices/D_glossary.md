# Appendix D — Glossary

Terms used throughout this book. Cross-links lead to the chapter where the
term is treated in depth.

---

**AVX2** — Intel's 256-bit SIMD instruction-set extension introduced with
Haswell (2013). Provides 8-lane `f32` (`__m256`) and 4-lane `f64` (`__m256d`)
operations plus integer SIMD; FMA is a separate but commonly-bundled
extension. See [SIMD backends → AVX2](../backends/avx2.md).

**AVX-512** — A family of 512-bit SIMD extensions; the smallest useful subset
is **AVX-512F** (foundation), which provides 16-lane `f32` (`__m512`) and
8-lane `f64` (`__m512d`) plus *mask registers* `k0..k7`. Other relevant
sub-extensions are **AVX-512DQ** (extra `f64` ops), **AVX-512VL** (which
allows AVX-512 instructions on 128-/256-bit registers), and
**AVX-512BW** (byte/word integer ops). See [AVX-512](../backends/avx512.md).

**AVX-512F** — The foundation subset of AVX-512: register width and basic
ops. The minimum target_feature `simdmath` requires when AVX-512 is
selected.

**big-endian** — Byte ordering with the most-significant byte at the lowest
address. Irrelevant for math correctness on the platforms this crate targets
(x86_64 and aarch64 are both little-endian) but mentioned here because the
musl source files use the term when describing IEEE 754 layout.

**binade** — The interval \\([2^k,\\,2^{k+1})\\) for integer \\(k\\); one binade
contains \\(2^p\\) representable floats, where \\(p\\) is the mantissa width
(23 for f32, 52 for f64). One ULP doubles in absolute size at every binade
boundary, which is why ULP error is asymmetric across powers of two
(see [methodology](../precision/methodology.md)).

**Cody-Waite reduction** — Argument-reduction technique that splits an
irrational reference value \\(C\\) (e.g. \\(\pi/2\\) or \\(\ln 2\\)) into
\\(C = c_\text{hi} + c_\text{lo}\\) such that \\(k \cdot c_\text{hi}\\) is computed
exactly, with \\(k \cdot c_\text{lo}\\) as a small correction. Cheaper than
Payne-Hanek but limited to moderate \\(\lvert k \rvert\\). See
[argument reduction](../foundations/argument_reduction.md). Originally:
W. J. Cody and W. Waite, *Software Manual for the Elementary Functions*,
1980.

**correctly rounded** — A function \\(\hat{f}\\) is *correctly rounded* if for
every input \\(\hat{x}\\), \\(\hat{f}(\hat{x})\\) is the nearest representable float
to the true \\(f(\hat{x})\\) (with a fixed tie-break rule, usually round-to-even).
The strongest possible accuracy guarantee, equivalent to ≤ 0.5 ULP. Hardware
square root is correctly rounded; most transcendentals in this crate are not.
See [foundations/ulp.md](../foundations/ulp.md).

**Dekker product** — Algorithm by T. J. Dekker (1971) for computing the
exact product of two floats as a `(hi, lo)` pair, using `Veltkamp` splitting
plus six FMA-friendly multiplications. The basis of double-double arithmetic.
See [compensated arithmetic](../foundations/compensated.md).

**double-double** — Representation of a higher-precision number as the
unevaluated sum of two `f64` values \\(a_\text{hi} + a_\text{lo}\\) with
\\(\lvert a_\text{lo} \rvert \le \tfrac{1}{2}\\,\mathrm{ulp}(a_\text{hi})\\).
About 106 bits of precision, used for the `PIO2_HI/LO` and `LN2_HI/LO`
splits.

**faithful rounding** — A function \\(\hat{f}\\) is *faithfully rounded* if
\\(\hat{f}(\hat{x})\\) is *one of* the two representable floats adjacent to
\\(f(\hat{x})\\). Equivalent to ≤ 1 ULP. Weaker than correct rounding; achievable
without the Table Maker's Dilemma overhead.

**fdlibm** — *Freely Distributable Math Library*, written at Sun Microsystems
in 1993 (primary author: K.-C. Ng). The ancestor of essentially every modern
soft `libm`, including musl, FreeBSD's `msun`, and OpenBSD's libm. The
constants in [Appendix B](./B_constants.md) are taken verbatim from
fdlibm via musl. See <https://www.netlib.org/fdlibm/>.

**FMA** — *Fused Multiply-Add*: the operation \\(a \cdot b + c\\) computed with a
single rounding at the end. Available as `vfmadd*ps` / `vfmadd*pd` on AVX2
(with the FMA extension), as `vfmadd*` on AVX-512, and as `vfmaq_f32` /
`vfmaq_f64` on NEON. Reduces both error and latency.

**Horner's rule** — Evaluation of a polynomial as
\\(a_0 + x\\,(a_1 + x\\,(a_2 + \cdots))\\). Sequential, but minimizes both
operation count and rounding-error accumulation. See
[polynomial evaluation](../foundations/polynomial_evaluation.md).

**Estrin's scheme** — Alternative polynomial evaluation that exposes
instruction-level parallelism by computing \\(x^2,\\,x^4,\\,\ldots\\) first and
then combining odd/even halves. Higher latency-bound throughput than Horner
on superscalar CPUs.

**IEEE 754** — The 2008 / 2019 standard for binary floating-point arithmetic.
Defines representations, rounding modes, special values (NaN, ±∞), and the
correctly-rounded behaviour of basic operations (`+`, `−`, `×`, `÷`, `√`,
remainder, conversion). See [IEEE-754 in two slides](../foundations/ieee754.md).

**intrinsic** — A compiler builtin that maps to a single machine
instruction (or a tight sequence). Rust exposes intrinsics under
`std::arch::{x86_64,aarch64}` behind `unsafe`; `simdmath` calls them
directly in the per-backend kernel files.

**lane** — One of the parallel scalar values inside a SIMD register. An
8-lane f32 register contains eight f32 values processed in lockstep. Lane
indices are typically 0 = least-significant.

**libm** — The platform "math library" implementing C99 `<math.h>`. On Linux
it is `libm.so.6` from glibc (or musl); on macOS it is part of `libSystem`.
Rust's `f32::sin` etc. delegate to `libm` on most targets.

**libmvec** — Glibc's *vector libm*. Ships AVX2 / AVX-512 / NEON-SVE
versions of about 20 transcendentals with a documented 4-ULP contract.

**mantissa** — The significand part of an IEEE 754 number, after the
implicit leading 1 for normalized values. 23 bits for f32, 52 bits for f64.
Sometimes called the *fraction* in the IEEE 754 standard.

**mask register** — In AVX-512, a 16- to 64-bit predicate register
(`k0..k7`) used for predicated execution. AVX2 has no mask registers and
emulates predication with full-width vector masks plus blend operations.

**mdBook** — The Rust-native static-site generator that renders this book.
MathJax rendering is enabled via mdBook's `mathjax-support` option in
`book.toml`; bibtex-js is *not* enabled in the book (only in rustdoc), so
[Appendix E](./E_bibliography.md) is rendered as a numbered reference list.

**minimax polynomial** — The polynomial of given degree that *minimizes the
maximum* deviation from the target function over a closed interval (Chebyshev
or Remez fit). Differs from a Taylor polynomial, which minimizes deviation
*at a point*. The musl coefficients in [Appendix A](./A_coefficients.md) are
minimax-fit, which is why low-order coefficients deviate slightly from their
Taylor counterparts.

**musl libc** — A small, MIT-licensed C standard library. Its math
implementations are descendants of fdlibm and are the algorithmic source of
truth for `simdmath`. <https://musl.libc.org>.

**NaN** — *Not a Number*: the IEEE 754 sentinel value for undefined results
(e.g. `0/0`, `sqrt(-1)`, `log(-1)`). NaNs propagate through arithmetic and
compare unequal to everything, including themselves. Two flavours: *quiet*
(default) and *signalling*.

**NEON** — ARM's 128-bit SIMD extension, mandatory on aarch64 (`AArch64
Advanced SIMD`). 4-lane f32 and 2-lane f64. See [NEON](../backends/neon.md).

**no_std** — Rust crates that compile without the `std` crate, suitable for
embedded targets. `simdmath`'s kernels are no_std-compatible; the slice
wrappers in `math::VecMath` allocate via `alloc::Vec` and so require
`alloc`.

**normal / subnormal** — A *normal* IEEE 754 value has its implicit leading
1 in the mantissa (exponent field \\(\ne 0\\)); a *subnormal* (or *denormal*)
has exponent field 0 and represents very small numbers with reduced
precision. Subnormals incur a microcode penalty on most x86 CPUs and are
explicitly handled (`B2_*`, `X1P*_32`, `X1P*_64` constants) in `simdmath`'s
`cbrt` to avoid that penalty.

**oracle** — The reference implementation against which `simdmath` measures
its ULP error. For v0.1 the oracle is Rust `std`'s scalar math, which
delegates to the platform `libm`. See
[methodology → oracle policy](../precision/methodology.md#the-oracle-policy-for-v01).

**Padé approximation** — A rational approximation \\(p(x)/q(x)\\) chosen to
match the target function's Taylor expansion to as high an order as the
total degree allows. Used in `asin`/`acos` (degree 5/4 for f64, see
[Appendix A](./A_coefficients.md)).

**Payne-Hanek reduction** — Argument-reduction technique that uses a
precomputed extended-precision representation of \\(1/(\pi/2)\\) (typically
hundreds of bits) to reduce \\(\hat{x}\\) for *arbitrarily* large
\\(\lvert \hat{x} \rvert\\). Slower than Cody-Waite but exact. Originally:
M. Payne and R. Hanek, *Radian Reduction for Trigonometric Functions*,
SIGNUM Newsletter, 1983. Not yet implemented in `simdmath`.

**polynomial degree** — The highest exponent in a polynomial. Higher
degrees give tighter accuracy at the cost of latency and (mildly) more
rounding error. See [polynomial evaluation](../foundations/polynomial_evaluation.md).

**RUSTFLAGS** — Environment variable that passes additional flags to
`rustc`. `simdmath` uses it to set `-C target-feature=...` for selecting
AVX2 / AVX-512 / NEON at compile time. See
[Required CPU features](../getting_started/cpu_features.md).

**SIMD** — *Single Instruction, Multiple Data*: hardware execution of one
operation across multiple data lanes in lockstep. The umbrella term
covering AVX2, AVX-512, NEON, SVE, RVV, and so on.

**SSE** — Intel's earliest 128-bit SIMD ISA, predecessor of AVX. SSE2 is
mandatory on x86_64 and provides 4-lane f32 / 2-lane f64. `simdmath` does
not target SSE-only paths; the AVX2 backend assumes SSE/AVX baseline.

**SVE** — *Scalable Vector Extension*, ARM's 2016 vector ISA with a
runtime-determined vector length (128 to 2048 bits). Used on Fujitsu
A64FX, AWS Graviton 3 / 4, and Microsoft Cobalt 100. Not yet supported by
`simdmath`.

**target_feature** — Rust language attribute / `cfg` predicate for
conditionally compiling code that requires a specific CPU instruction-set
extension. Combined with `RUSTFLAGS="-C target-feature=..."` to select a
backend. See [compile-time dispatch](../backends/dispatch.md).

**Table Maker's Dilemma** — The problem of deciding whether the exact
real value of \\(f(\hat{x})\\) falls just above or just below a halfway point
between two representable floats — required for *correct* rounding but
not for faithful rounding. Solving it for a transcendental in IEEE 754
costs hundreds to thousands of bits of intermediate precision. Why
`simdmath` aims for ≤ 1 ULP rather than 0.5 ULP. See
[why-not-correct](../foundations/why_not_correct.md).

**Two-sum** — Algorithm (Knuth, *TAoCP* vol. 2, §4.2.2) for computing the
exact sum of two floats as a `(hi, lo)` pair. Together with Dekker's
product, the building block of compensated arithmetic. See
[compensated arithmetic](../foundations/compensated.md).

**ULP** — *Unit in the Last Place*: the gap between two adjacent
representable floats. Used as the unit for measuring rounding error. See
[foundations/ulp.md](../foundations/ulp.md) and
[methodology](../precision/methodology.md).

**vector lane** — Synonym for *lane*: one scalar slot inside a SIMD register.

**ZMM** — The 512-bit registers of AVX-512, named `zmm0..zmm31`. AVX-512F
adds 16 new registers on top of the 16 inherited from AVX2 (where the
256-bit halves are aliased to `ymm0..ymm15`).

## See also

- [Foundations](../foundations/ieee754.md) — long-form treatment of most of
  these terms.
- [Appendix E — Bibliography](./E_bibliography.md) — citations for the named
  algorithms (Cody-Waite, Dekker, Knuth, Payne-Hanek, Goldberg, Muller).
