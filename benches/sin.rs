//! Benchmarks for sine (sin).

#[path = "common.rs"]
mod common;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use simdmath::math::VecMath;

use common::*;

fn scalar_sin_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.sin()).collect()
}

fn scalar_sin_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.sin()).collect()
}

fn make_sin_input_f32(n: usize) -> Vec<f32> {
    // Values in [-10, 10] to cover several periods
    (0..n)
        .map(|i| (i as f32 / n as f32) * 20.0 - 10.0)
        .collect()
}

fn make_sin_input_f64(n: usize) -> Vec<f64> {
    // Values in [-10, 10] to cover several periods
    (0..n)
        .map(|i| (i as f64 / n as f64) * 20.0 - 10.0)
        .collect()
}

fn bench_sin_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/sin/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let a = make_sin_input_f32(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.sin())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_sin_f32(black_box(&a)))));
        g.finish();
    }
}

fn bench_sin_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/sin/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let a = make_sin_input_f64(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.sin())));
        g.bench_function("scalar", |b| b.iter(|| black_box(scalar_sin_f64(black_box(&a)))));
        g.finish();
    }
}

criterion_group!(benches, bench_sin_f32, bench_sin_f64);
criterion_main!(benches);
