//! Benchmarks for arc tangent (atan).

use criterion::{black_box, BenchmarkId, Criterion};
use simdmath::math::VecMath;

use super::common::*;

fn scalar_atan_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.atan()).collect()
}

fn scalar_atan_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.atan()).collect()
}

fn make_atan_input_f32(n: usize) -> Vec<f32> {
    // Values in [-10, 10] to cover all reduction ranges
    (0..n)
        .map(|i| (i as f32 / n as f32) * 20.0 - 10.0)
        .collect()
}

fn make_atan_input_f64(n: usize) -> Vec<f64> {
    // Values in [-10, 10] to cover all reduction ranges
    (0..n)
        .map(|i| (i as f64 / n as f64) * 20.0 - 10.0)
        .collect()
}

pub fn bench_atan_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/atan");
    for &n in SIZES_F32 {
        let a = make_atan_input_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.atan()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_atan_f32(black_box(&a))))
        });
    }
    g.finish();
}

pub fn bench_atan_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/atan");
    for &n in SIZES_F64 {
        let a = make_atan_input_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.atan()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_atan_f64(black_box(&a))))
        });
    }
    g.finish();
}
