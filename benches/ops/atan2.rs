//! Benchmarks for two-argument arc tangent (atan2).

use criterion::{BenchmarkId, Criterion, black_box};
use simdmath::math::VecMath;

use super::common::*;

fn scalar_atan2_f32(y: &[f32], x: &[f32]) -> Vec<f32> {
    y.iter().zip(x).map(|(y, x)| y.atan2(*x)).collect()
}

fn scalar_atan2_f64(y: &[f64], x: &[f64]) -> Vec<f64> {
    y.iter().zip(x).map(|(y, x)| y.atan2(*x)).collect()
}

/// Generate input vectors covering all 4 quadrants.
///
/// Uses polar coordinates with varying angles and radii to ensure
/// comprehensive coverage of the atan2 domain.
fn make_atan2_inputs_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);

    for i in 0..n {
        // Angle from -π to π
        let theta = (i as f32 / n as f32) * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
        // Radius varies to test different magnitudes
        let r = 0.1 + (i as f32 % 10.0);

        x.push(r * theta.cos());
        y.push(r * theta.sin());
    }

    (y, x)
}

fn make_atan2_inputs_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);

    for i in 0..n {
        // Angle from -π to π
        let theta = (i as f64 / n as f64) * 2.0 * std::f64::consts::PI - std::f64::consts::PI;
        // Radius varies to test different magnitudes
        let r = 0.1 + (i as f64 % 10.0);

        x.push(r * theta.cos());
        y.push(r * theta.sin());
    }

    (y, x)
}

pub fn bench_atan2_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/atan2");
    for &n in SIZES_F32 {
        let (y, x) = make_atan2_inputs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(y.atan2(black_box(&x))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_atan2_f32(black_box(&y), black_box(&x))))
        });
    }
    g.finish();
}

pub fn bench_atan2_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/atan2");
    for &n in SIZES_F64 {
        let (y, x) = make_atan2_inputs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(y.atan2(black_box(&x))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_atan2_f64(black_box(&y), black_box(&x))))
        });
    }
    g.finish();
}
