//! Benchmarks for blocked matrix multiplication vs `faer` (single-threaded).
//!
//! Throughput is reported in elements/s where one "element" is one FLOP
//! (2·m·n·k FLOPs per multiplication), so Gelem/s reads as GFLOPS.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

/// Square benchmark sizes (m = n = k).
const SIZES: &[usize] = &[64, 128, 256, 512, 1024];

fn fill(len: usize, seed: u32) -> Vec<f32> {
    // Cheap deterministic pseudo-random values in [-1, 1).
    let mut state = seed.wrapping_mul(0x9E3779B9) | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1 << 23) as f32 - 1.0
        })
        .collect()
}

fn fill_f64(len: usize, seed: u32) -> Vec<f64> {
    fill(len, seed).into_iter().map(f64::from).collect()
}

fn bench_matmul_f32(c: &mut Criterion) {
    for &n in SIZES {
        let mut g = c.benchmark_group(format!("f32/matmul/{n}"));
        g.throughput(Throughput::Elements((2 * n * n * n) as u64));
        g.sample_size(20);

        let a = fill(n * n, 1);
        let b = fill(n * n, 2);

        {
            let mut out = vec![0.0f32; n * n];
            g.bench_function("simdmath", |bencher| {
                bencher.iter(|| {
                    simdmath::linalg::matmul(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut out),
                        n,
                        n,
                        n,
                    )
                })
            });
        }

        {
            use faer::{Accum, Mat, Par, linalg::matmul::matmul};

            let a = Mat::from_fn(n, n, |i, j| a[j * n + i]);
            let b = Mat::from_fn(n, n, |i, j| b[j * n + i]);
            let mut out = Mat::<f32>::zeros(n, n);
            g.bench_function("faer", |bencher| {
                bencher.iter(|| {
                    matmul(
                        black_box(&mut out),
                        Accum::Add,
                        black_box(&a),
                        black_box(&b),
                        1.0f32,
                        Par::Seq,
                    )
                })
            });
        }

        g.finish();
    }
}

fn bench_matmul_f64(c: &mut Criterion) {
    for &n in SIZES {
        let mut g = c.benchmark_group(format!("f64/matmul/{n}"));
        g.throughput(Throughput::Elements((2 * n * n * n) as u64));
        g.sample_size(20);

        let a = fill_f64(n * n, 1);
        let b = fill_f64(n * n, 2);

        {
            let mut out = vec![0.0f64; n * n];
            g.bench_function("simdmath", |bencher| {
                bencher.iter(|| {
                    simdmath::linalg::matmul_f64(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut out),
                        n,
                        n,
                        n,
                    )
                })
            });
        }

        {
            use faer::{Accum, Mat, Par, linalg::matmul::matmul};

            let a = Mat::from_fn(n, n, |i, j| a[j * n + i]);
            let b = Mat::from_fn(n, n, |i, j| b[j * n + i]);
            let mut out = Mat::<f64>::zeros(n, n);
            g.bench_function("faer", |bencher| {
                bencher.iter(|| {
                    matmul(
                        black_box(&mut out),
                        Accum::Add,
                        black_box(&a),
                        black_box(&b),
                        1.0f64,
                        Par::Seq,
                    )
                })
            });
        }

        g.finish();
    }
}

criterion_group!(benches, bench_matmul_f32, bench_matmul_f64);
criterion_main!(benches);
