//! Benchmarks for tangent (tan).

#[path = "common.rs"]
mod common;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use simdmath::math::VecMath;

use common::*;

fn scalar_tan_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.tan()).collect()
}

fn scalar_tan_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.tan()).collect()
}

fn make_tan_input_f32(n: usize) -> Vec<f32> {
    // Values in [-1.5, 1.5] to stay away from poles at ±π/2
    (0..n).map(|i| (i as f32 / n as f32) * 3.0 - 1.5).collect()
}

fn make_tan_input_f64(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 / n as f64) * 3.0 - 1.5).collect()
}

fn bench_tan_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/tan/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let a = make_tan_input_f32(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.tan())));
        g.bench_function("scalar", |b| {
            b.iter(|| black_box(scalar_tan_f32(black_box(&a))))
        });
        g.finish();
    }
}

fn bench_tan_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/tan/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let a = make_tan_input_f64(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.tan())));
        g.bench_function("scalar", |b| {
            b.iter(|| black_box(scalar_tan_f64(black_box(&a))))
        });
        g.finish();
    }
}

criterion_group!(benches, bench_tan_f32, bench_tan_f64);
criterion_main!(benches);
