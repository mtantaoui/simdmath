//! Benchmarks for the power function (pow).

#[path = "common.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simdmath::math::VecMath;

use common::*;

fn scalar_pow_f32(bases: &[f32], exps: &[f32]) -> Vec<f32> {
    bases.iter().zip(exps).map(|(b, e)| b.powf(*e)).collect()
}

fn scalar_pow_f64(bases: &[f64], exps: &[f64]) -> Vec<f64> {
    bases.iter().zip(exps).map(|(b, e)| b.powf(*e)).collect()
}

/// Generate input vectors covering a variety of base/exponent combinations.
///
/// Bases range over positive values with varying magnitudes; exponents
/// include small integers, fractional values, and negative powers.
fn make_pow_inputs_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut bases = Vec::with_capacity(n);
    let mut exps = Vec::with_capacity(n);

    for i in 0..n {
        // Bases: positive values from 0.1 to ~10
        bases.push(0.1 + (i as f32 % 10.0));
        // Exponents: mix of small integers and fractions
        exps.push((i as f32 % 7.0) - 3.0); // range ~ [-3, 3]
    }

    (bases, exps)
}

fn make_pow_inputs_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut bases = Vec::with_capacity(n);
    let mut exps = Vec::with_capacity(n);

    for i in 0..n {
        bases.push(0.1 + (i as f64 % 10.0));
        exps.push((i as f64 % 7.0) - 3.0);
    }

    (bases, exps)
}

fn bench_pow_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/pow");
    for &n in SIZES_F32 {
        let (bases, exps) = make_pow_inputs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(bases.pow(black_box(&exps))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_pow_f32(black_box(&bases), black_box(&exps))))
        });
    }
    g.finish();
}

fn bench_pow_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/pow");
    for &n in SIZES_F64 {
        let (bases, exps) = make_pow_inputs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(bases.pow(black_box(&exps))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_pow_f64(black_box(&bases), black_box(&exps))))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_pow_f32, bench_pow_f64);
criterion_main!(benches);
