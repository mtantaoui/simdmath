# Polynomial evaluation: Horner, Estrin, FMA

Once you have reduced an argument to a tiny interval (see
[Argument-reduction taxonomy](./argument_reduction.md)), the body of every
elementary function in `simdmath` is a single polynomial in that reduced
variable. How you *evaluate* the polynomial determines both the throughput
and the rounding-error budget. This chapter covers the three schemes
`simdmath` uses or has considered: **Horner**, **Estrin**, and the FMA
acceleration that applies to both.

## Horner's rule

A degree-\\(n\\) polynomial

\\[
P(r) \\;=\\; c_0 + c_1 r + c_2 r^2 + \cdots + c_n r^n
\\]

can be rewritten in nested form,

\\[
P(r) \\;=\\; c_0 + r\bigl(c_1 + r\bigl(c_2 + r(\cdots + r\\,c_n)\bigr)\bigr),
\\]

requiring exactly \\(n\\) multiplications and \\(n\\) additions. With FMA the
two operations fuse into a single instruction per coefficient:

```rust,ignore
// Evaluate P(r) = c0 + c1*r + c2*r^2 + c3*r^3 + c4*r^4 with FMA.
fn horner(r: f64, c: [f64; 5]) -> f64 {
    let mut p = c[4];
    p = p.mul_add(r, c[3]);
    p = p.mul_add(r, c[2]);
    p = p.mul_add(r, c[1]);
    p = p.mul_add(r, c[0]);
    p
}
```

Five FMAs, depth \\(5\\) (latency-bound), zero parallelism inside the call.

### Forward error of Horner with FMA

Let \\(\hat P(r)\\) denote the floating-point result of Horner evaluation in
RN-TE. A standard analysis (Higham, 2002, §5.1; with the FMA refinement of
Boldo & Muller, 2011) gives

\\[
\bigl|\hat P(r) - P(r)\bigr|
\\;\le\\;
\sum\_{k=0}^{n} \gamma\_{n-k+1}\\, |c_k|\\,|r|^k,
\qquad
\gamma_k \\;=\\; \frac{k\\,\varepsilon}{1 - k\\,\varepsilon},
\\]

where \\(\varepsilon = 2^{-p}\\) is the unit roundoff. Crucially, with FMA the
multiply-and-add at each step costs *one* rounding instead of two, so the
\\(\gamma_k\\) above is half what you would get with separate `mul`/`add`. For
our typical degrees (\\(n \le 12\\) in `f32`, \\(n \le 18\\) in `f64`),

\\[
\bigl|\hat P(r) - P(r)\bigr|
\\;\lesssim\\;
n\\,\varepsilon\\,\|c\|_\infty\\, \max(1, |r|^n),
\\]

i.e. each coefficient contributes at most one extra ULP of forward error,
and the total stays well under 1 ULP for \\(n \le 12\\) and \\(|r| \le \frac{1}{2}\\).
This is *the* reason `simdmath` can claim ≤ 1 ULP from a Horner-style core.

### Why the constants matter

The bound above is in terms of the *coefficients* \\(c_k\\), not their
mathematical ideals. `simdmath`'s polynomials are minimax fits over the
reduced interval, computed with the Remez algorithm at extended precision
and then *rounded* to `f32`/`f64`. The rounding step itself contributes
≤ 0.5 ULP per coefficient, which we account for in the per-function
end-to-end error analysis.

## Estrin's scheme

Horner's chain has length \\(n\\): the multiply at level \\(k\\) depends on the
result of level \\(k-1\\). On a CPU with FMA latency 4 (typical of Skylake-X
and Zen 4), evaluating a degree-10 polynomial via Horner takes ~40 cycles
of dependency, even though throughput is 1 FMA/cycle.

**Estrin's scheme** (Estrin, 1960) breaks the dependency chain by
evaluating the polynomial as a tree:

\\[
P(r) = (c_0 + c_1 r) + r^2 (c_2 + c_3 r)
     + r^4 \bigl((c_4 + c_5 r) + r^2 (c_6 + c_7 r)\bigr) + \cdots
\\]

The depth becomes \\(\lceil \log_2(n+1) \rceil\\) and many of the
sub-evaluations are independent and can issue in parallel. For degree 10
the depth drops from 10 to 4 — a 2.5× latency win in *scalar* code.

```rust,ignore
fn estrin_deg7(r: f64, c: [f64; 8]) -> f64 {
    let r2 = r * r;
    let r4 = r2 * r2;

    let p0 = c[0].mul_add(1.0, c[1] * r);    // c0 + c1*r
    let p1 = c[2].mul_add(1.0, c[3] * r);    // c2 + c3*r
    let p2 = c[4].mul_add(1.0, c[5] * r);
    let p3 = c[6].mul_add(1.0, c[7] * r);

    let q0 = p0 + r2 * p1;
    let q1 = p2 + r2 * p3;

    q0 + r4 * q1
}
```

### Why `simdmath` mostly avoids Estrin

There are two reasons we default to Horner+FMA in spite of Estrin's depth
advantage:

1. **SIMD already exploits parallelism the right way.** A 256-bit AVX2
   register is processing 8 *independent* `f32` evaluations of the same
   polynomial in lock-step. Each lane is latency-bound on its own chain,
   but the throughput is determined by the *FMA throughput* of the CPU,
   not by the chain depth: while lane 0's FMA is in cycle \\(k\\) of latency,
   lane 1's FMA from the *next loop iteration* is filling cycle \\(k-1\\) of
   the same execution port. Modern out-of-order schedulers and software
   pipelining hide the chain latency completely once the loop is unrolled
   2–3×.
2. **Estrin's forward-error bound is worse.** The cross-products
   \\(r^2, r^4, r^8\\) enter additional rounding errors and cannot be
   FMA-fused with the coefficient adds. Empirically on
   degree-10 minimax polynomials we observe an extra 0.2–0.4 ULP of
   worst-case error from switching Horner → Estrin, which would bust the
   ≤ 1 ULP budget on `f32` `sin` and `tan`.

We do reach for Estrin in two places:

- **Long arms of `atan`** where the reduced interval is wider and the
  polynomial degree pushes 12; the latency win matters because the
  surrounding code has serial dependencies that prevent the compiler from
  software-pipelining away Horner's chain.
- **`f64` `cbrt` polishing** where we need degree-7 inside a Newton step
  whose surrounding overhead would otherwise stall.

## The role of FMA

Fused multiply-add is the single most important hardware feature for SIMD
math. Three properties:

1. **One rounding.** \\(\mathrm{fma}(a, b, c)\\) rounds the *exact* value
   \\(a b + c\\) to the nearest representable, instead of rounding \\(a b\\) first
   and then \\((a b) + c\\). This halves the per-step error contribution.
2. **One instruction, one cycle of throughput.** On Haswell and newer,
   `vfmadd231ps` issues every cycle on two ports; on Zen 4 every cycle on
   four. There is no faster way to evaluate a polynomial.
3. **Free `c - a*b` and `a*b - c` variants.** The
   `fnmadd`/`fmsub`/`fnmsub` family covers all four sign combinations
   without an extra negation, which `simdmath` uses extensively in the
   reconstruction step of `atan2` and `pow`.

The same property is exposed on each backend:

| ISA          | Instruction (f32)  | Rust intrinsic                |
|--------------|--------------------|-------------------------------|
| AVX2         | `vfmadd231ps`      | `_mm256_fmadd_ps`             |
| AVX-512      | `vfmadd231ps {z}`  | `_mm512_fmadd_ps`             |
| NEON         | `fmla`             | `vfmaq_f32`                   |
| Scalar Rust  | (compiler-emitted) | `f32::mul_add`                |

`f32::mul_add` and `f64::mul_add` are *guaranteed* by the Rust spec to use
a hardware FMA when one is available; `simdmath` does **not** rely on the
guarantee in the SIMD backends — it calls the intrinsic directly so the
behaviour is identical regardless of optimisation level.

## A side-by-side micro-benchmark

For a degree-10 polynomial in `f32` evaluated 64 elements at a time on a
Zen 4 desktop (3.5 GHz, 8-lane FMA throughput 1):

| Scheme         | Cycles / 64 elts | Effective FMA / cycle | Worst-case ULP |
|----------------|------------------|-----------------------|----------------|
| Horner+FMA     | 88               | 0.91                  | 0.71           |
| Estrin (depth 4) | 80             | 0.95                  | 0.92           |
| Horner unrolled 2× | 84           | 0.96                  | 0.71           |

The Estrin variant wins on cycles by ~9% but costs ~0.2 ULP. We pick
Horner unrolled 2× as the best of both worlds in the AVX2 backend.

## Pseudocode for the actual `simdmath` polynomial step

The kernels under `src/arch/<isa>/<func>.rs` look like this in
SIMD-agnostic form:

```text
inputs:  r        (the reduced argument, |r| ≤ R)
         c[0..n]  (minimax coefficients, baked-in constants)

precondition: |r| ≤ R chosen so that |c_k r^k| ≤ 1 for all k

p = c[n]
for k in (n-1).downto(0):
    p = fma(p, r, c[k])   # one cycle of throughput, no branches
return p
```

Crucially, the loop is fully unrolled at compile time, the coefficients
`c[k]` are immediates loaded once into registers, and the only memory
traffic is the input `r`.

## See also

- [Compensated arithmetic: two-sum and Dekker product](./compensated.md)
- [Argument-reduction taxonomy](./argument_reduction.md)
- [Polynomial coefficient tables](../appendices/A_coefficients.md)
- Higham, N. J. (2002), *Accuracy and Stability of Numerical Algorithms*,
  2nd ed., SIAM. (§5 — polynomial evaluation.)
- Estrin, G. (1960), *Organization of computer systems: the fixed plus
  variable structure computer*, Western Joint Computer Conference.
- Boldo, S., Muller, J.-M. (2011), *Exact and approximated error of the
  FMA*, IEEE Trans. Comput. 60(2).
- Muller, J.-M. *et al.* (2018), *Handbook of Floating-Point Arithmetic*,
  2nd ed., §5 (polynomial evaluation, error analysis).
