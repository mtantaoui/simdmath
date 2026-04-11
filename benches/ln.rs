//! Benchmarks for natural logarithm (ln).

#[path = "common.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simdmath::math::VecMath;

use common::*;

fn scalar_ln_f32(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| x.ln()).collect()
}

fn scalar_ln_f64(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| x.ln()).collect()
}

fn make_ln_input_f32(n: usize) -> Vec<f32> {
    // Positive values in (0.01, 1000] spanning subnormal-adjacent to large
    (0..n)
        .map(|i| 0.01 + (i as f32 / n as f32) * 999.99)
        .collect()
}

fn make_ln_input_f64(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.01 + (i as f64 / n as f64) * 999.99)
        .collect()
}

fn bench_ln_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/ln");
    for &n in SIZES_F32 {
        let a = make_ln_input_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.ln()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_ln_f32(black_box(&a))))
        });
    }
    g.finish();
}

fn bench_ln_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/ln");
    for &n in SIZES_F64 {
        let a = make_ln_input_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.ln()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_ln_f64(black_box(&a))))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_ln_f32, bench_ln_f64);
criterion_main!(benches);
