//! Benchmarks for arc cosine (acos).

use criterion::{BenchmarkId, Criterion, black_box};
use simdmath::math::VecMath;

use super::common::*;

fn scalar_acos_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.acos()).collect()
}

fn scalar_acos_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.acos()).collect()
}

fn make_acos_input_f32(n: usize) -> Vec<f32> {
    // Values in [-1, 1] for valid acos domain
    (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect()
}

fn make_acos_input_f64(n: usize) -> Vec<f64> {
    // Values in [-1, 1] for valid acos domain
    (0..n).map(|i| (i as f64 / n as f64) * 2.0 - 1.0).collect()
}

pub fn bench_acos_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/acos");
    for &n in SIZES_F32 {
        let a = make_acos_input_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.acos()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_acos_f32(black_box(&a))))
        });
    }
    g.finish();
}

pub fn bench_acos_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/acos");
    for &n in SIZES_F64 {
        let a = make_acos_input_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.acos()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_acos_f64(black_box(&a))))
        });
    }
    g.finish();
}
