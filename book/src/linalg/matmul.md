# Blocked matrix multiplication `matmul`

## Mathematical statement

Given \\(A \in \mathbb{R}^{m \times k}\\), \\(B \in \mathbb{R}^{k \times n}\\)
and \\(C \in \mathbb{R}^{m \times n}\\), compute the accumulating product

\\[
C \leftarrow C + A B,
\qquad
c_{ij} \leftarrow c_{ij} + \sum_{l=0}^{k-1} a_{il}\, b_{lj}.
\\]

The accumulating form (`C += A·B` rather than `C = A·B`) is the BLAS `GEMM`
convention with \\(\alpha = \beta = 1\\): callers who want a plain product
pass a zeroed `C`, and callers who chain products get the update for free.

## API and storage convention

```rust,ignore
use simdmath::linalg::{matmul, matmul_f64};

// C += A × B, all matrices column-major.
matmul(&a, &b, &mut c, m, n, k);       // f32
matmul_f64(&a, &b, &mut c, m, n, k);   // f64
```

All matrices are **column-major**: element \\((i, j)\\) of an \\(m\\)-row
matrix lives at index \\(j \cdot m + i\\). Slice lengths must match the
dimensions exactly (`a.len() == m*k`, `b.len() == k*n`, `c.len() == m*n`)
or the call panics; if any dimension is zero the call returns without
touching `C`.

Unlike the element-wise functions, `matmul` is available on **every**
target: AVX2, AVX-512 and NEON get hand-written SIMD microkernels, and all
other targets get a cache-blocked scalar implementation whose inner loops
are written for the auto-vectoriser. The backend is chosen at compile time
by the same `cfg` ladder as the rest of the crate
([Compile-time dispatch](../backends/dispatch.md)).

## Why the naive triple loop fails

The textbook implementation

```rust,ignore
for j in 0..n {
    for i in 0..m {
        for l in 0..k {
            c[j * m + i] += a[l * m + i] * b[j * k + l];
        }
    }
}
```

performs \\(2mnk\\) flops on \\(mk + kn + mn\\) values, so each element is
reused \\(\Theta(n)\\) times — but the loop order above streams the whole of
`A` through the cache once per column of `C`. As soon as `A` outgrows the
last-level cache, every reuse becomes a memory access and throughput
collapses to memory bandwidth. The fix, due to Goto [Goto2008] and
systematised by BLIS [VanZee2015], is to *block* the three loops so that
carefully sized sub-matrices stay resident in each level of the cache
hierarchy, and to *pack* those blocks into contiguous buffers so the inner
kernel reads them at unit stride.

## The blocked algorithm

`matmul` uses the BLIS five-loop structure. Three cache-blocking loops
partition the problem; two register-blocking loops sweep tiles of `C`:

```text
for jc in (0..n).step_by(nc):          // L3 blocking (N dimension)
    for pc in (0..k).step_by(kc):      // L1 blocking (K dimension)
        pack_b(B[pc.., jc..])          // → row-major NR-wide panels
        for ic in (0..m).step_by(mc):  // L2 blocking (M dimension)
            pack_a(A[ic.., pc..])      // → column-major MR-wide panels
            for jr in b_panels:        // register blocking
                for ir in a_panels:
                    kernel()           // C tile += A panel × B panel
```

- **`pack_b`** copies the \\(k_c \times n_c\\) block of `B` into panels of
  `NR` columns, stored row by row, so the kernel broadcasts consecutive
  elements from a contiguous buffer. It sits *outside* the `ic` loop: one
  packed `B` block is reused by every `A` block.
- **`pack_a`** copies the \\(m_c \times k_c\\) block of `A` into panels of
  `MR` rows, stored column by column, so the kernel loads one aligned
  `MR`-row column per \\(k\\) step.
- Partial panels at the fringes are **zero-padded**, so the microkernel
  always runs full-width FMAs — the padding contributes \\(0\\) to the
  accumulation. Rows and columns of the `C` tile beyond the fringe are
  masked on load/store instead.
- The packing buffers are allocated **once per call** (zeroed pages via
  `alloc_zeroed`, which is nearly free for block-sized allocations) and
  refilled for every block iteration.

## Choosing the block sizes analytically

Rather than hand-tuning \\(k_c, m_c, n_c\\) per machine, the crate ports the
analytical model of Low et al. [Low2016] (as popularised by the `gemm` and
`faer` crates):

- \\(k_c\\): one `MR × kc` A micropanel plus one `kc × NR` B micropanel must
  fit in L1 without evicting each other. The model works in units of cache
  *ways*: successive A micropanels should map onto every L1 set so that a
  new panel exactly replaces the previous one. A floor of \\(k_c \ge 512\\)
  amortises the per-kernel-call load/store of the `C` tile over a deep
  accumulation.
- \\(m_c\\): the packed \\(m_c \times k_c\\) A block must stay L2-resident
  while B micropanels stream through, keeping one way free.
- \\(n_c\\): the packed \\(k_c \times n_c\\) B block must stay L3-resident.

Each parameter is finally *balanced*: \\(k = 513\\) becomes two blocks of
\\(\approx 257\\) rather than \\(512 + 1\\), so the last block is never
degenerately small.

Cache geometry is detected once at runtime: from
`/sys/devices/system/cpu` on Linux; the AVX2 and AVX-512 backends
additionally fall back to the CPUID deterministic cache parameters (leaf
`0x4` on Intel, `0x8000001D` on AMD) on non-Linux systems; everywhere else
conservative defaults are used (32 KiB L1d, 256 KiB L2, 2 MiB L3).
L1/L2 sizes are divided by the number of SMT threads sharing them; L3 is
kept whole, since the packed B block is read-shared.

## The microkernel

The innermost kernel computes one `MR × NR` tile of `C` as a sequence of
\\(k_c\\) rank-1 updates, holding the whole tile in vector registers. Every
backend uses the same two-row-vector shape — the tile is `MR = 2 ×`
(vector width) rows by `NR` columns — scaled to its register file:

| Backend | f32 kernel | f64 kernel | Registers used (accumulators + A + B) |
|---------|:----------:|:----------:|:-------------------------------------:|
| AVX2    | 16×6       | 8×6        | 12 + 2 + 1 = 15 of 16 YMM             |
| AVX-512 | 32×14      | 16×14      | 28 + 2 + 1 = 31 of 32 ZMM             |
| NEON    | 8×12       | 4×12       | 24 + 2 + 1 = 27 of 32                 |
| scalar  | 8×6        | 4×6        | `[[T; MR]; NR]` array, auto-vectorised |

Per \\(k\\) step the SIMD kernels issue two aligned loads for the A column
and `NR` FMAs per row vector, with each B element **broadcast straight
from memory** — `vbroadcastss/sd` on x86 (runs on the load ports, leaving
the shuffle port free) and `ld1r` on NEON. Columns beyond `nr` at a fringe
multiply the B panel's zero padding into dummy accumulators that are never
stored, so the hot loop has no column guards.

The scalar kernel makes one deliberate departure: it accumulates with
`a * b + acc` rather than `f32::mul_add`. Without FMA hardware, `mul_add`
falls back to a correctly-rounded software `fma` (orders of magnitude
slower); a separate multiply/add lowers to plain vector instructions under
auto-vectorisation.

## The direct path for small matrices

Packing is a bandwidth investment that pays off only when blocks are
reused enough times. For matrices with \\(m, n < 128\\) (measured crossover
on AVX2), `matmul` skips packing entirely and runs the microkernel
directly on the column-major inputs: A columns are already contiguous, and
B elements are broadcast from their original locations. Only the K
dimension is chunked (depth 512) to bound the accumulation between
load/store of the `C` tile. The dispatch is automatic; both paths produce
identically blocked accumulation *within* a tile.

## Error analysis

`matmul` carries no per-element ULP guarantee — the error of a dot product
grows with \\(k\\), which is the caller's choice. The standard forward
bound [Higham2002] for a length-\\(k\\) dot product evaluated with FMAs is

\\[
|\hat{c}_{ij} - c_{ij}| \;\le\; \gamma_k \sum_{l} |a_{il}||b_{lj}|,
\qquad
\gamma_k = \frac{k\,\varepsilon}{1 - k\,\varepsilon},
\\]

i.e. the *relative* error is bounded by \\(\gamma_k\\) unless there is
heavy cancellation. Three practical consequences:

- FMA backends commit one rounding per multiply-add instead of two, so
  their constant is roughly half the scalar backend's.
- Blocking changes the **summation order** (each \\(k_c\\) chunk is
  accumulated into `C` separately, and lanes accumulate independently), so
  results may differ from the naive triple loop — and between backends —
  in the last bits. The crate's tests compare against a naive reference
  with small relative tolerances (\\(10^{-4}\\) for the deterministic f32
  sweeps, \\(10^{-12}\\) for f64) rather than bit-exactly.
- Zero-padding the panels is exact: padded lanes add \\(\pm 0\\).

## Where to look in the source

The implementation is structurally identical across backends; per backend
`<b>` ∈ {`avx2`, `avx512`, `neon`, `scalar`}:

| Topic | File |
|-------|------|
| Public API and backend dispatch | [`src/linalg.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/linalg.rs) |
| Driver, loop nest, `matmul_auto` dispatch | `src/arch/<b>/matmul/f32/mod.rs` (and `f64/`) |
| Microkernels | `src/arch/<b>/matmul/f32/kernels.rs` (and `f64/`) |
| Packing (generic over element type) | `src/arch/<b>/matmul/panels.rs` |
| Cache detection + analytical blocking model | `src/arch/<b>/matmul/cache.rs` |

## Benchmarks

Measured on the AVX2 reference machine (Core Ultra 7 155H, pinned to one
core, single-threaded), against `faer` 0.24 (`Par::Seq`) on square
matrices; one "element" of criterion throughput = one FLOP of
\\(2 n^3\\):

| \\(n\\) | f32 simdmath | f32 faer | f64 simdmath | f64 faer |
|------:|:-----------:|:--------:|:-----------:|:--------:|
| 64    | 86 GFLOPS   | 115      | 44          | 59       |
| 128   | 100         | 118      | 55          | 63       |
| 256   | 114         | 126      | 61          | 65       |
| 512   | 119         | 127      | 61          | 64       |
| 1024  | 118         | 109      | 61          | 62       |

simdmath reaches 75–95 % of faer's single-threaded throughput below
\\(n = 1024\\) and overtakes it (f32) or matches it (f64) at
\\(n = 1024\\). The numbers are a single interleaved run; run-to-run
thermal variance on this machine is around \\(\pm 8\,\%\\), and repeated
runs have measured the smaller sizes anywhere between 85 % and 100 % of
faer. Faer's remaining edge at small sizes comes from its hand-written
assembly kernels. AVX-512 and NEON numbers will be added once measured on
target hardware — their kernel shapes and the direct-path crossover are
analytically chosen starting points, not yet hardware-tuned.

## References

- [Goto2008] — the layered blocking/packing design.
- [VanZee2015] — the BLIS microkernel abstraction and five-loop structure.
- [Low2016] — the analytical cache model used for \\(k_c, m_c, n_c\\).
- [Higham2002] — forward error bounds for dot products and matrix
  multiplication.
