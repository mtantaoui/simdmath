# Why `≤ 1 ULP` and not `correctly rounded`

`simdmath` promises **faithful rounding** — at most 1 ULP of error, often
under 0.6 ULP — across the entire domain of every supported function. It
deliberately stops short of **correctly rounded** transcendentals, which
the IEEE-754-2019 §9 recommendation calls the gold standard. This chapter
explains why the extra half-ULP is not worth chasing for a SIMD library.

## What "correctly rounded" actually costs

A correctly-rounded transcendental (call it \\(\hat f^\star\\)) must agree
with the *real* value \\(f\\) to within \\(0.5\\,\mathrm{ulp}\\) for every input.
The standard recipe is **Ziv's strategy** (Ziv, 1991):

1. Evaluate \\(f\\) in slightly-extended precision (say, double-double, ≈106
   bits).
2. If the result is *not* within ½ ULP of a midpoint, return it — done in
   one shot.
3. Otherwise, fall back to a *much* wider precision (multi-precision MPFR-
   style code, ≥ 200 bits) until the midpoint is unambiguously resolved.

The hard part is step 3. Because of the **table-maker's dilemma** (see
[ULP, faithful rounding, correct rounding](./ulp.md)), the worst-case
inputs require unbounded internal precision in principle and ≥ 110 bits in
practice for the common `binary64` transcendentals. CRlibm (de Dinechin
*et al.*, 2007) precomputes the worst cases offline so that 120 bits of
internal precision suffice; even so, the slow path runs ~5–10× slower
than the fast path.

For a SIMD kernel, that slow-path cost is worse than the scalar cost,
because of:

### Lock-step execution

A 256-bit AVX2 register processes eight `f32` lanes at once. If even
*one* lane hits the slow path, **all eight** lanes must execute the
multi-precision branch — there is no per-lane control flow on a single
SIMD instruction. So the average per-element cost is:

\\[
T_\text{avg} \\;\approx\\; T_\text{fast}
\\;+\\; \bigl(1 - (1-p)^L\bigr)\\,T_\text{slow}
\\]

where \\(p\\) is the per-input slow-path probability and \\(L\\) is the lane
count. With \\(p \approx 2^{-30}\\) and \\(L = 8\\), the lock-step penalty is
small — but for \\(L = 16\\) (AVX-512) and any \\(p \ge 2^{-25}\\) it dominates.

### Multi-precision tables

Ziv's slow path needs lookup tables of "hard cases" — typically tens of
megabytes for `binary64`. Streaming those tables from L3/RAM kills the
latency advantage that motivated SIMD in the first place.

### Branch divergence

`simdmath`'s polynomial cores are straight-line code: ten `vfmadd231`
instructions in a row, no branches, fully pipelined. A correctly-rounded
implementation has at minimum a "did we land near a midpoint?" branch per
function call, which limits the achievable IPC and disables compiler
auto-vectorisation of surrounding user code.

## What clients actually need

The transcendentals in `simdmath` are used predominantly in:

- **Real-time graphics** (lighting, BRDF evaluation): tolerates ≤ 4 ULP
  with no visible artefact.
- **Machine learning** (activations, loss functions): both training and
  inference are dominated by `bf16`/`f16` quantisation noise; ≤ 1 ULP
  `f32` math is far below the noise floor.
- **Game physics** (collision, ballistics): integration error grows
  linearly per step at hundreds of ULPs; the math kernel is not the
  bottleneck.
- **Signal processing** (FFTs, filters): cancellation in the algorithm
  costs more bits than any reasonable transcendental implementation.
- **Scientific simulation** (molecular dynamics, CFD): tolerances are set
  by discretisation error, typically 4–10 ULPs.

For any of these, paying a 5–10× throughput cost to go from 1 ULP to
0.5 ULP is not a sensible trade. Code that genuinely needs correctly
rounded transcendentals — interval arithmetic kernels, formal verification
of numerical algorithms — should reach for [CRlibm](http://crlibm.org/)
(scalar) or [Core-Math](https://core-math.gitlabpages.inria.fr/) and
accept the lower throughput.

## Comparison with other SIMD math libraries

| Library                        | Architecture       | Worst-case ULP (f32 / f64) | Correctly rounded? |
|--------------------------------|--------------------|----------------------------|--------------------|
| **`simdmath`** (this crate)    | AVX2/AVX-512/NEON  | ≤ 1 / ≤ 1                  | no                 |
| [SLEEF] *u10* family           | SSE2 → SVE         | ≤ 1.0 / ≤ 1.0              | no                 |
| [SLEEF] *u35* family           | SSE2 → SVE         | ≤ 3.5 / ≤ 3.5              | no                 |
| GCC `libmvec`                  | AVX2/AVX-512       | ≤ 4 / ≤ 4                  | no                 |
| Intel SVML (`-fveclib=SVML`)   | AVX2/AVX-512       | ≤ 4 / ≤ 4 (HA mode ≤ 1)    | no                 |
| Apple Accelerate `vForce`      | NEON               | ≤ 1 / ≤ 1                  | no                 |
| **CRlibm**                     | scalar only        | ≤ 0.5 / ≤ 0.5              | **yes**            |
| **Core-Math**                  | scalar only        | ≤ 0.5 / ≤ 0.5              | **yes**            |

[SLEEF]: https://sleef.org

The "u35" / "u10" naming follows SLEEF: *u10* means "≤ 1.0 ULP", *u35* means
"≤ 3.5 ULP" (a faster variant). `simdmath` sits at the *u10* tier, with
some functions (e.g. `sqrt`, the trivially-correctly-rounded one)
matching CRlibm.

## What we promise

For every function exposed by [`VecMath`](https://docs.rs/simdmath/latest/simdmath/math/trait.VecMath.html):

1. **≤ 1 ULP** worst-case error, measured by an exhaustive sweep on `f32`
   and a 10⁹-sample stratified Monte-Carlo on `f64`.
2. **Correct handling of special values** — \\(\pm 0\\), \\(\pm \infty\\), NaN,
   subnormals — matching the IEEE-754-2008 §9 recommendation, modulo the
   ULP relaxation.
3. **Bit-identical results across SIMD lanes**: if lane \\(i\\) and lane \\(j\\)
   receive the same input, they produce the same output, regardless of
   what their neighbours are computing.
4. **Bit-identical results across backends** for normal inputs: the
   AVX2/AVX-512/NEON kernels share the same algorithm and the same
   constants, so they agree to the last bit. (Subnormal results may
   differ by 0–1 ULP on platforms with FTZ enabled.)

## What we do *not* promise

1. **Correct rounding.** The result may be the wrong neighbour of the
   true value by up to 1 ULP. If you need bitwise-identical to CRlibm,
   `simdmath` is not the right tool.
2. **Constant-time execution.** All branches in `simdmath` are
   data-independent at the SIMD level (we use blends, not jumps), but the
   underlying CPU may still vary cycle counts based on operand magnitude
   (e.g. AVX-512 frequency throttling, subnormal slow paths).
3. **Cross-platform reproducibility under FTZ/DAZ.** If your program
   enables flush-to-zero we will respect it, which means subnormal
   inputs/outputs will be silently zeroed.
4. **Stability of internal constants.** The polynomial coefficients in
   `src/arch/consts/` may change between minor versions if we find a
   better approximation. The *contract* (≤ 1 ULP) does not change.

## See also

- [ULP, faithful rounding, correct rounding](./ulp.md)
- [Comparison with other SIMD math libraries](../appendices/C_comparison.md)
- [ULP measurement methodology](../precision/methodology.md)
- Ziv, A. (1991), *Fast evaluation of elementary mathematical functions
  with correctly rounded last bit*, ACM TOMS 17(3).
- de Dinechin, F., Lauter, C., Muller, J.-M. (2007), *Fast and correctly
  rounded logarithms in double-precision*, RAIRO Theor. Inform. Appl.
- Lefèvre, V., Muller, J.-M. (2001), *Worst cases for correct rounding of
  the elementary functions in double precision*, ARITH-15.
- Shibata, N. (2010+), [SLEEF reference manual](https://sleef.org/).
