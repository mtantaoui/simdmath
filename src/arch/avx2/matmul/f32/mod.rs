//! BLIS-style blocked matrix multiplication (f32) with AVX2 microkernels.
//!
//! Implements cache-conscious matrix multiplication following the BLIS
//! (BLAS-like Library Instantiation Software) design: hierarchical blocking
//! for the L1/L2/L3 caches, packing of the operands into aligned panels, and
//! a 16×6 AVX2 FMA microkernel.
//!
//! ```text
//! for jc in (0..n).step_by(nc):          // L3 blocking (N dimension)
//!     for pc in (0..k).step_by(kc):      // L1 blocking (K dimension)
//!         pack_b(B[pc.., jc..])          // → row-major NR-wide panels
//!         for ic in (0..m).step_by(mc):  // L2 blocking (M dimension)
//!             pack_a(A[ic.., pc..])      // → column-major MR-wide panels
//!             for jr in b_panels:
//!                 for ir in a_panels:
//!                     kernel_16x6()      // C tile += A panel × B panel
//! ```
//!
//! Small matrices skip the packing entirely (see [`matmul_auto`]); the public
//! entry point is [`crate::linalg::matmul`].

pub(crate) mod kernels;

use std::cmp::min;

use crate::arch::avx2::matmul::cache::{KernelParams, kernel_params};
use crate::arch::avx2::matmul::f32::kernels::{kernel_16x6, kernel_16x6_direct};
use crate::arch::avx2::matmul::panels::{PackedBlock, at, pack_a, pack_b};

/// Microkernel row dimension: rows of `A`/`C` processed per kernel call
/// (two 8-lane YMM vectors).
pub(crate) const MR: usize = 16;

/// Microkernel column dimension: columns of `B`/`C` processed per kernel call.
pub(crate) const NR: usize = 6;

/// Depth of one k chunk in the direct (no-packing) path. Mirrors the packed
/// path's L1-derived `kc` upper bound.
const DIRECT_KC: usize = 512;

/// Computes `C += A × B` with the strategy and blocking parameters derived
/// from the machine's cache hierarchy (see [`crate::arch::avx2::matmul::cache`]).
///
/// Matrices whose panels fit in L2 skip packing entirely ([`matmul_direct`]);
/// larger inputs go through the packed, cache-blocked path ([`matmul`]).
///
/// Same contract as [`matmul`], minus the manual `mc`/`kc`/`nc` knobs.
pub fn matmul_auto(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // Below this square size the packing copy costs more than it saves;
    // measured crossover on AVX2 (see the `path_crossover` tuning test).
    const DIRECT_MAX_DIM: usize = 128;

    if m < DIRECT_MAX_DIM && n < DIRECT_MAX_DIM {
        matmul_direct(a, b, c, m, n, k);
    } else {
        let params = kernel_params(m, n, k, MR, NR, size_of::<f32>());
        matmul(a, b, c, m, n, k, params);
    }
}

/// Computes `C += A × B` directly on the column-major inputs, without
/// packing.
///
/// For matrices whose panels fit in L2, the microkernel can stream A and B
/// as-is: A columns are already contiguous and B elements are broadcast from
/// their original locations, so the packing copy would be pure overhead.
/// Only the K dimension is chunked (to bound the C-tile accumulation depth);
/// the M/N loops sweep microkernel tiles directly.
pub(crate) fn matmul_direct(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    if m == 0 || n == 0 || k == 0 {
        return; // Nothing to compute
    }

    assert_eq!(a.len(), m * k, "matrix A has incorrect dimensions");
    assert_eq!(b.len(), k * n, "matrix B has incorrect dimensions");
    assert_eq!(c.len(), m * n, "matrix C has incorrect dimensions");

    for pc in (0..k).step_by(DIRECT_KC) {
        let kc = min(DIRECT_KC, k - pc);

        for jc in (0..n).step_by(NR) {
            let nr = min(NR, n - jc);

            for ic in (0..m).step_by(MR) {
                let mr = min(MR, m - ic);

                // SAFETY: the mr×kc A block at (ic, pc), the kc×nr B block
                // at (pc, jc) and the mr×nr C tile at (ic, jc) all lie
                // inside their matrices (dimensions asserted above); the
                // kernel masks partial rows/columns.
                unsafe {
                    kernel_16x6_direct(
                        a.as_ptr().add(at(ic, pc, m)),
                        m,
                        b.as_ptr().add(at(pc, jc, k)),
                        k,
                        c.as_mut_ptr().add(at(ic, jc, m)),
                        mr,
                        nr,
                        kc,
                        m,
                    )
                }
            }
        }
    }
}

/// Computes `C += A × B` for column-major f32 matrices.
///
/// # Arguments
///
/// * `a` - Matrix A in column-major format (`m × k` elements)
/// * `b` - Matrix B in column-major format (`k × n` elements)
/// * `c` - Output matrix C in column-major format (`m × n` elements); the
///   product is **accumulated** onto its existing contents
/// * `m` - Number of rows in A and C
/// * `n` - Number of columns in B and C
/// * `k` - Number of columns in A and rows in B (shared dimension)
/// * `params` - `mc`/`kc`/`nc` blocking parameters (L2/L1/L3 block sizes;
///   see [`kernel_params`])
///
/// If any dimension is `0` the function returns without touching `c`.
///
/// # Panics
///
/// Panics if a slice length does not match its `m`/`n`/`k` dimensions, or if
/// a block size is `0`.
pub(crate) fn matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    params: KernelParams,
) {
    if m == 0 || n == 0 || k == 0 {
        return; // Nothing to compute
    }

    let KernelParams { kc, mc, nc } = params;

    assert_eq!(a.len(), m * k, "matrix A has incorrect dimensions");
    assert_eq!(b.len(), k * n, "matrix B has incorrect dimensions");
    assert_eq!(c.len(), m * n, "matrix C has incorrect dimensions");
    assert!(mc > 0 && kc > 0 && nc > 0, "block sizes must be non-zero");

    // Packing buffers are allocated once at the maximum block size and
    // refilled on every block iteration.
    let mut a_block = PackedBlock::<f32, MR>::with_capacity(min(mc, m), min(kc, k));
    let mut b_block = PackedBlock::<f32, NR>::with_capacity(min(nc, n), min(kc, k));

    // jc loop: process the N dimension in nc-wide blocks (L3 blocking)
    for jc in (0..n).step_by(nc) {
        let nc_actual = min(nc, n - jc);

        // pc loop: process the K dimension in kc-wide blocks. Placed outside
        // the ic loop so one packed B block is reused across all A blocks.
        for pc in (0..k).step_by(kc) {
            let kc_actual = min(kc, k - pc);

            pack_b(&mut b_block, b, nc_actual, kc_actual, k, pc, jc);

            // ic loop: process the M dimension in mc-wide blocks (L2 blocking)
            for ic in (0..m).step_by(mc) {
                let mc_actual = min(mc, m - ic);

                pack_a(&mut a_block, a, mc_actual, kc_actual, m, ic, pc);

                // jr/ir loops: sweep the packed panels with the MR×NR kernel
                let num_b_panels = nc_actual.div_ceil(NR);
                let num_a_panels = mc_actual.div_ceil(MR);

                for jr in 0..num_b_panels {
                    let nr = min(NR, nc_actual - jr * NR);

                    for ir in 0..num_a_panels {
                        let mr = min(MR, mc_actual - ir * MR);

                        // Tile top-left: C(ic + ir*MR, jc + jr*NR)
                        let c_start = at(ic + ir * MR, jc + jr * NR, m);

                        // SAFETY: the packed panels hold kc_actual slices and
                        // the mr×nr tile starting at c_start lies inside `c`,
                        // which matches the m×n dimensions (asserted above).
                        unsafe {
                            kernel_16x6(
                                a_block.panel(ir),
                                b_block.panel(jr),
                                c[c_start..].as_mut_ptr(),
                                mr,
                                nr,
                                kc_actual,
                                m,
                            )
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    /// Fixed blocking parameters for the tests, small enough that modest
    /// matrix sizes already exercise multiple blocks per loop.
    const TEST_PARAMS: KernelParams = KernelParams {
        kc: 256,
        mc: MR * 8,
        nc: NR * 8,
    };

    /// Naive O(mnk) reference: C += A × B with the standard triple loop.
    fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for j in 0..n {
            for i in 0..m {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[at(i, l, m)] * b[at(l, j, k)];
                }
                c[at(i, j, m)] += sum;
            }
        }
    }

    /// Test matrix with values `(row+1) + (col+1)*0.1` for easy verification.
    fn create_test_matrix(rows: usize, cols: usize) -> Vec<f32> {
        let mut matrix = vec![0.0; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                matrix[at(row, col, rows)] = (row + 1) as f32 + (col + 1) as f32 * 0.1;
            }
        }
        matrix
    }

    fn create_identity_matrix(size: usize) -> Vec<f32> {
        let mut matrix = vec![0.0; size * size];
        for i in 0..size {
            matrix[at(i, i, size)] = 1.0;
        }
        matrix
    }

    /// Runs the packed, direct and auto paths on the given inputs and
    /// asserts every element matches the naive reference within relative
    /// tolerance `tol`.
    fn check_against_naive(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, tol: f32) {
        let mut c_naive = vec![0.0; m * n];
        naive_matmul(a, b, &mut c_naive, m, n, k);

        let mut c_packed = vec![0.0; m * n];
        matmul(a, b, &mut c_packed, m, n, k, TEST_PARAMS);

        let mut c_direct = vec![0.0; m * n];
        matmul_direct(a, b, &mut c_direct, m, n, k);

        let mut c_auto = vec![0.0; m * n];
        matmul_auto(a, b, &mut c_auto, m, n, k);

        for (name, result) in [
            ("packed", &c_packed),
            ("direct", &c_direct),
            ("auto", &c_auto),
        ] {
            for i in 0..m * n {
                let diff = (result[i] - c_naive[i]).abs();
                let max_val = result[i].abs().max(c_naive[i].abs());
                let rel_err = if max_val > 1e-6 { diff / max_val } else { diff };

                assert!(
                    rel_err < tol,
                    "{name} {m}x{k} * {k}x{n} mismatch at {i}: got {}, naive={}, rel_err={rel_err}",
                    result[i],
                    c_naive[i],
                );
            }
        }
    }

    /// Fixed-seed RNG so randomized-test failures are reproducible. To debug
    /// a failure, the whole run can be replayed as-is; change the seed to
    /// explore other inputs.
    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(0x51D_5EED)
    }

    fn random_matrix(rng: &mut StdRng, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    /// Verifies `at()` implements column-major indexing.
    #[test]
    fn test_column_major_indexing() {
        // Column-major storage: [col0, col1] = [a,b,c, d,e,f]
        // Matrix layout:  | a d |
        //                 | b e |
        //                 | c f |
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = 3; // rows

        assert_eq!(matrix[at(0, 0, m)], 1.0); // top-left
        assert_eq!(matrix[at(1, 0, m)], 2.0); // middle-left
        assert_eq!(matrix[at(2, 0, m)], 3.0); // bottom-left
        assert_eq!(matrix[at(0, 1, m)], 4.0); // top-right
        assert_eq!(matrix[at(1, 1, m)], 5.0); // middle-right
        assert_eq!(matrix[at(2, 1, m)], 6.0); // bottom-right
    }

    #[test]
    fn test_matmul_identity() {
        let size = 4;
        let a = create_test_matrix(size, size);
        let identity = create_identity_matrix(size);
        let mut c = vec![0.0; size * size];

        matmul(&a, &identity, &mut c, size, size, size, TEST_PARAMS);

        for i in 0..size * size {
            assert!(
                (c[i] - a[i]).abs() < 1e-6,
                "A*I != A at index {i}: result={}, original={}",
                c[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_matmul_2x2_known_result() {
        // A = | 1 3 |   B = | 5 7 |   C = | 1*5+3*6  1*7+3*8 | = | 23 31 |
        //     | 2 4 |       | 6 8 |       | 2*5+4*6  2*7+4*8 |   | 34 46 |
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        matmul(&a, &b, &mut c, 2, 2, 2, TEST_PARAMS);

        // Expected C in column-major: col0=[23,34], col1=[31,46]
        let expected = [23.0, 34.0, 31.0, 46.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-6,
                "mismatch at {i}: got {}, expected {}",
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_matmul_empty() {
        // 0x0 matrices
        let mut c: Vec<f32> = vec![];
        matmul(&[], &[], &mut c, 0, 0, 0, TEST_PARAMS);
        assert!(c.is_empty());

        // One dimension zero: 2x1 * 1x0 → 2x0
        let a = vec![1.0, 2.0];
        let mut c: Vec<f32> = vec![];
        matmul(&a, &[], &mut c, 2, 0, 1, TEST_PARAMS);
        assert!(c.is_empty());
    }

    #[test]
    fn test_matmul_zeros() {
        let m = 4;
        let n = 4;
        let k = 4;
        let a = vec![0.0; m * k];
        let b = create_test_matrix(k, n);
        let mut c = vec![0.0; m * n];

        matmul(&a, &b, &mut c, m, n, k, TEST_PARAMS);

        for (i, &val) in c.iter().enumerate() {
            assert!(val.abs() < 1e-6, "result should be zero at {i}: {val}");
        }
    }

    /// C starts non-zero: verifies the product accumulates onto C.
    #[test]
    fn test_matmul_accumulation() {
        let m = 3;
        let n = 3;
        let k = 3;
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![1.0; m * n];
        let mut c_naive = vec![1.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k, TEST_PARAMS);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-4,
                "accumulation mismatch at {i}: optimized={}, naive={}",
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Very small and very large magnitudes cancelling to a modest result.
    #[test]
    fn test_matmul_special_values() {
        let a = vec![1e-10, 2e-10]; // 1x2
        let b = vec![3e10, 4e10]; // 2x1
        let mut c = vec![0.0; 1];

        matmul(&a, &b, &mut c, 1, 1, 2, TEST_PARAMS);

        // 1e-10 * 3e10 + 2e-10 * 4e10 = 3 + 8 = 11
        assert!((c[0] - 11.0).abs() < 1e-6, "expected 11.0, got {}", c[0]);
    }

    /// Deterministic sweep over sizes below, at, and above the kernel and
    /// blocking boundaries, checked against the naive reference.
    #[test]
    fn test_matmul_fixed_sizes_vs_naive() {
        // (m, n, k)
        let test_cases = [
            (1, 1, 1),
            (2, 2, 2),
            (3, 3, 3),
            (2, 4, 3), // non-square
            (3, 2, 4), // very small, all dims distinct
            (5, 7, 3),
            (7, 5, 9), // no dimension a multiple of 8
            (8, 8, 8), // exact kernel size
            (16, 16, 16),
            (7, 9, 8), // near kernel size
            (15, 17, 13),
            (64, 64, 64),    // n spans two nc blocks (nc = 48)
            (65, 63, 64),    // straddling the kernel tile boundaries
            (128, 128, 128), // m spans two mc blocks (mc = 128), n three nc
            (100, 75, 50),
            (1, 1, 300), // k spans two kc blocks (kc = 256)
        ];

        for &(m, n, k) in &test_cases {
            let a = create_test_matrix(m, k);
            let b = create_test_matrix(k, n);
            check_against_naive(&a, &b, m, n, k, 1e-4);
        }
    }

    /// Tall/thin and wide/short shapes that stress the blocking loops.
    #[test]
    fn test_matmul_extreme_aspect_ratios() {
        // 8×1 * 1×1 with a known expected result
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0];
        let mut c = vec![0.0; 8];
        matmul(&a, &b, &mut c, 8, 1, 1, TEST_PARAMS);
        let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        for i in 0..8 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-6,
                "8x1*1x1 mismatch at {i}: got {}, expected {}",
                c[i],
                expected[i]
            );
        }

        // Random data over assorted extreme shapes, vs the naive reference
        let shapes = [
            (1, 64, 32),
            (64, 1, 32),
            (128, 4, 8),
            (4, 128, 8),
            (8, 8, 256),
        ];
        let mut rng = seeded_rng();
        for &(m, n, k) in &shapes {
            let a = random_matrix(&mut rng, m * k);
            let b = random_matrix(&mut rng, k * n);
            check_against_naive(&a, &b, m, n, k, 1e-3);
        }
    }

    /// Rough GFLOPS comparison of the packed vs direct paths across sizes,
    /// for tuning the dispatch crossover. Run explicitly with:
    /// `cargo test --release path_crossover -- --ignored --nocapture`
    #[test]
    #[ignore = "manual tuning helper, not a correctness test"]
    fn path_crossover() {
        for &n in &[32, 48, 64, 96, 128, 192, 256, 384, 512] {
            let a = fill_seq(n * n);
            let b = fill_seq(n * n);
            let mut c = vec![0.0; n * n];
            let flops = 2.0 * (n as f64).powi(3);

            let mut time_it = |f: &mut dyn FnMut(&mut [f32])| {
                let iters = (5e8 / flops).max(1.0) as usize;
                // Warm-up
                f(&mut c);
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    f(&mut c);
                }
                flops * iters as f64 / start.elapsed().as_secs_f64() / 1e9
            };

            let packed = time_it(&mut |c| {
                let p = kernel_params(n, n, n, MR, NR, size_of::<f32>());
                matmul(&a, &b, c, n, n, n, p);
            });
            let direct = time_it(&mut |c| matmul_direct(&a, &b, c, n, n, n));

            println!("n={n:4}  packed {packed:6.1} GFLOPS   direct {direct:6.1} GFLOPS");
        }
    }

    fn fill_seq(len: usize) -> Vec<f32> {
        let mut rng = seeded_rng();
        random_matrix(&mut rng, len)
    }

    /// Random inputs over random sizes plus every size around the kernel
    /// boundary (multiples of 8 ± 1).
    #[test]
    fn test_matmul_random_vs_naive() {
        let mut rng = seeded_rng();

        for _ in 0..10 {
            let m = rng.random_range(1..=64);
            let n = rng.random_range(1..=64);
            let k = rng.random_range(1..=64);
            let a = random_matrix(&mut rng, m * k);
            let b = random_matrix(&mut rng, k * n);
            check_against_naive(&a, &b, m, n, k, 5e-3);
        }

        for size in [7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33] {
            let a = random_matrix(&mut rng, size * size);
            let b = random_matrix(&mut rng, size * size);
            check_against_naive(&a, &b, size, size, size, 2e-2);
        }
    }
}
