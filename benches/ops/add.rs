//! Benchmarks for element-wise addition.

use criterion::{BenchmarkId, Criterion, black_box};
use simdmath::ops::vec::VecExt;

use super::common::*;

pub fn bench_add_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/add");
    for &n in SIZES_F32 {
        let (a, b) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.add(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

pub fn bench_add_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/add");
    for &n in SIZES_F64 {
        let (a, b) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.add(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

pub fn bench_add_scalar_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/add_scalar");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.add_scalar(black_box(3.5f32))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_add_s(black_box(&a), black_box(3.5f32))))
        });
    }
    g.finish();
}

pub fn bench_add_scalar_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/add_scalar");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.add_scalar(black_box(3.5f64))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_add_s(black_box(&a), black_box(3.5f64))))
        });
    }
    g.finish();
}
