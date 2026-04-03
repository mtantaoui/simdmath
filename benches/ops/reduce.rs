//! Benchmarks for reduction operations (sum, product, min, max).

use criterion::{BenchmarkId, Criterion, black_box};
use simdmath::ops::vec::VecExt;

use super::common::*;

// ---------------------------------------------------------------------------
// Sum
// ---------------------------------------------------------------------------

pub fn bench_sum_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/sum");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.sum()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_sum::<f32>(black_box(&a))))
        });
    }
    g.finish();
}

pub fn bench_sum_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/sum");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.sum()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_sum::<f64>(black_box(&a))))
        });
    }
    g.finish();
}

// ---------------------------------------------------------------------------
// Product
// ---------------------------------------------------------------------------

pub fn bench_product_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/product");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.product()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_product::<f32>(black_box(&a))))
        });
    }
    g.finish();
}

pub fn bench_product_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/product");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.product()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_product::<f64>(black_box(&a))))
        });
    }
    g.finish();
}

// ---------------------------------------------------------------------------
// Min
// ---------------------------------------------------------------------------

pub fn bench_min_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/min");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.min()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_min(black_box(&a), f32::INFINITY)))
        });
    }
    g.finish();
}

pub fn bench_min_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/min");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.min()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_min(black_box(&a), f64::INFINITY)))
        });
    }
    g.finish();
}

// ---------------------------------------------------------------------------
// Max
// ---------------------------------------------------------------------------

pub fn bench_max_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/max");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.max()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_max(black_box(&a), f32::NEG_INFINITY)))
        });
    }
    g.finish();
}

pub fn bench_max_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/max");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.max()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_max(black_box(&a), f64::NEG_INFINITY)))
        });
    }
    g.finish();
}
