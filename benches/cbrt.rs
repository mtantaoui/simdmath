//! Benchmarks for cube root (cbrt).

#[path = "common.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simdmath::math::VecMath;

use common::*;

fn scalar_cbrt_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.cbrt()).collect()
}

fn scalar_cbrt_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.cbrt()).collect()
}

fn make_cbrt_input_f32(n: usize) -> Vec<f32> {
    // Values in [-1000, 1000] to cover a wide range
    (0..n)
        .map(|i| (i as f32 / n as f32) * 2000.0 - 1000.0)
        .collect()
}

fn make_cbrt_input_f64(n: usize) -> Vec<f64> {
    // Values in [-1000, 1000] to cover a wide range
    (0..n)
        .map(|i| (i as f64 / n as f64) * 2000.0 - 1000.0)
        .collect()
}

fn bench_cbrt_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/cbrt");
    for &n in SIZES_F32 {
        let a = make_cbrt_input_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.cbrt()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_cbrt_f32(black_box(&a))))
        });
    }
    g.finish();
}

fn bench_cbrt_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/cbrt");
    for &n in SIZES_F64 {
        let a = make_cbrt_input_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.cbrt()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_cbrt_f64(black_box(&a))))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_cbrt_f32, bench_cbrt_f64);
criterion_main!(benches);
