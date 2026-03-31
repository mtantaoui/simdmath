//! Benchmarks for element-wise division.

use criterion::{BenchmarkId, Criterion, black_box};
use simdmath::ops::vec::VecExt;

use super::common::*;

pub fn bench_div_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/div");
    for &n in SIZES_F32 {
        let (a, b) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.div(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_div(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

pub fn bench_div_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/div");
    for &n in SIZES_F64 {
        let (a, b) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.div(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_div(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}
