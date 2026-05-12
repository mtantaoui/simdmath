//! Benchmarks for element-wise remainder.

#[path = "common.rs"]
mod common;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use simdmath::ops::vec::VecExt;

use common::*;

fn bench_rem_f32(c: &mut Criterion) {
    for &n in SIZES_F32 {
        let mut g = c.benchmark_group(format!("f32/rem/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, b) = make_vecs_f32(n);
        g.bench_function("simd", |b_| b_.iter(|| black_box(a.rem(black_box(&b)))));
        g.bench_function("scalar", |b_| {
            b_.iter(|| black_box(scalar_rem(black_box(&a), black_box(&b))))
        });
        g.finish();
    }
}

fn bench_rem_f64(c: &mut Criterion) {
    for &n in SIZES_F64 {
        let mut g = c.benchmark_group(format!("f64/rem/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        let (a, b) = make_vecs_f64(n);
        g.bench_function("simd", |b_| b_.iter(|| black_box(a.rem(black_box(&b)))));
        g.bench_function("scalar", |b_| {
            b_.iter(|| black_box(scalar_rem(black_box(&a), black_box(&b))))
        });
        g.finish();
    }
}

criterion_group!(benches, bench_rem_f32, bench_rem_f64);
criterion_main!(benches);
