//! Benchmarks for reduction operations (sum, product, min, max).

#[path = "common.rs"]
mod common;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use simdmath::ops::vec::VecExt;

use common::*;

// ---------------------------------------------------------------------------
// Sum
// ---------------------------------------------------------------------------

fn bench_sum_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/sum/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, _) = make_vecs_f32(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.sum())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_sum::<f32>(black_box(&a)))));
        g.finish();
    }
}

fn bench_sum_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/sum/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, _) = make_vecs_f64(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.sum())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_sum::<f64>(black_box(&a)))));
        g.finish();
    }
}

// ---------------------------------------------------------------------------
// Product
// ---------------------------------------------------------------------------

fn bench_product_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/product/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, _) = make_vecs_f32(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.product())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_product::<f32>(black_box(&a)))));
        g.finish();
    }
}

fn bench_product_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/product/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, _) = make_vecs_f64(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.product())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_product::<f64>(black_box(&a)))));
        g.finish();
    }
}

// ---------------------------------------------------------------------------
// Min
// ---------------------------------------------------------------------------

fn bench_min_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/min/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, _) = make_vecs_f32(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.min())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_min(black_box(&a), f32::INFINITY))));
        g.finish();
    }
}

fn bench_min_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/min/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, _) = make_vecs_f64(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.min())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_min(black_box(&a), f64::INFINITY))));
        g.finish();
    }
}

// ---------------------------------------------------------------------------
// Max
// ---------------------------------------------------------------------------

fn bench_max_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/max/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, _) = make_vecs_f32(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.max())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_max(black_box(&a), f32::NEG_INFINITY))));
        g.finish();
    }
}

fn bench_max_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/max/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, _) = make_vecs_f64(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.max())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_max(black_box(&a), f64::NEG_INFINITY))));
        g.finish();
    }
}

criterion_group!(
    benches,
    bench_sum_f32,
    bench_sum_f64,
    bench_product_f32,
    bench_product_f64,
    bench_min_f32,
    bench_min_f64,
    bench_max_f32,
    bench_max_f64
);
criterion_main!(benches);
