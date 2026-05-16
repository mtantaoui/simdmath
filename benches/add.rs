//! Benchmarks for element-wise addition.

#[path = "common.rs"]
mod common;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use simdmath::prelude::*;

use common::*;

fn bench_add_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/add/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, b) = make_vecs_f32(n);
        g.bench_function("simd", |b_| b_.iter(|| black_box(a.add(black_box(&b)))));
        g.bench_function("scalar", |b_| {
            b_.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))))
        });
        g.finish();
    }
}

fn bench_add_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/add/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, b) = make_vecs_f64(n);
        g.bench_function("simd", |b_| b_.iter(|| black_box(a.add(black_box(&b)))));
        g.bench_function("scalar", |b_| {
            b_.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))))
        });
        g.finish();
    }
}

fn bench_add_scalar_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/add_scalar/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, _) = make_vecs_f32(n);
        g.bench_function("simd", |b| {
            b.iter(|| black_box(a.add_scalar(black_box(3.5f32))))
        });
        g.bench_function("scalar", |b| {
            b.iter(|| black_box(scalar_add_s(black_box(&a), black_box(3.5f32))))
        });
        g.finish();
    }
}

fn bench_add_scalar_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/add_scalar/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, _) = make_vecs_f64(n);
        g.bench_function("simd", |b| {
            b.iter(|| black_box(a.add_scalar(black_box(3.5f64))))
        });
        g.bench_function("scalar", |b| {
            b.iter(|| black_box(scalar_add_s(black_box(&a), black_box(3.5f64))))
        });
        g.finish();
    }
}

criterion_group!(
    benches,
    bench_add_f32,
    bench_add_f64,
    bench_add_scalar_f32,
    bench_add_scalar_f64
);
criterion_main!(benches);
