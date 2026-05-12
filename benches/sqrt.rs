//! Benchmarks for the square root function (sqrt).

#[path = "common.rs"]
mod common;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
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
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/sqrt/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let a = make_sqrt_inputs_f32(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.sqrt())));
        g.bench_function("scalar", |b| {
            b.iter(|| black_box(scalar_sqrt_f32(black_box(&a))))
        });
        g.finish();
    }
}

fn bench_sqrt_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/sqrt/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let a = make_sqrt_inputs_f64(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.sqrt())));
        g.bench_function("scalar", |b| {
            b.iter(|| black_box(scalar_sqrt_f64(black_box(&a))))
        });
        g.finish();
    }
}

criterion_group!(benches, bench_sqrt_f32, bench_sqrt_f64);
criterion_main!(benches);
