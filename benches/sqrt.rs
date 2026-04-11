//! Benchmarks for the square root function (sqrt).

#[path = "common.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simdmath::math::VecMath;

use common::*;

fn scalar_sqrt_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.sqrt()).collect()
}

fn scalar_sqrt_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.sqrt()).collect()
}

fn make_sqrt_inputs_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i + 1) as f32 * 0.5).collect()
}

fn make_sqrt_inputs_f64(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i + 1) as f64 * 0.5).collect()
}

fn bench_sqrt_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/sqrt");
    for &n in SIZES_F32 {
        let a = make_sqrt_inputs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.sqrt()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_sqrt_f32(black_box(&a))))
        });
    }
    g.finish();
}

fn bench_sqrt_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/sqrt");
    for &n in SIZES_F64 {
        let a = make_sqrt_inputs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.sqrt()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_sqrt_f64(black_box(&a))))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_sqrt_f32, bench_sqrt_f64);
criterion_main!(benches);
