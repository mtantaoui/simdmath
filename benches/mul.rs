//! Benchmarks for element-wise multiplication.

#[path = "common.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simdmath::ops::vec::VecExt;

use common::*;

fn bench_mul_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/mul");
    for &n in SIZES_F32 {
        let (a, b) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.mul(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_mul(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_mul_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/mul");
    for &n in SIZES_F64 {
        let (a, b) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.mul(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_mul(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_mul_scalar_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/mul_scalar");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.mul_scalar(black_box(2.0f32))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_mul_s(black_box(&a), black_box(2.0f32))))
        });
    }
    g.finish();
}

fn bench_mul_scalar_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/mul_scalar");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.mul_scalar(black_box(2.0f64))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_mul_s(black_box(&a), black_box(2.0f64))))
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_mul_f32,
    bench_mul_f64,
    bench_mul_scalar_f32,
    bench_mul_scalar_f64
);
criterion_main!(benches);
