//! Benchmarks for arc cosine (acos).

#[path = "common.rs"]
mod common;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use simdmath::prelude::*;

use common::*;

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

fn bench_acos_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/acos/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let a = make_acos_input_f32(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.acos())));
        g.bench_function("scalar", |b| {
            b.iter(|| black_box(scalar_acos_f32(black_box(&a))))
        });
        g.finish();
    }
}

fn bench_acos_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/acos/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let a = make_acos_input_f64(n);
        g.bench_function("simd", |b| b.iter(|| black_box(a.acos())));
        g.bench_function("scalar", |b| {
            b.iter(|| black_box(scalar_acos_f64(black_box(&a))))
        });
        g.finish();
    }
}

criterion_group!(benches, bench_acos_f32, bench_acos_f64);
criterion_main!(benches);
