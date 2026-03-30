//! Benchmarks comparing `VecExt<f32>` SIMD operations against equivalent
//! scalar (vanilla Rust iterator / loop) implementations.
//!
//! Run with:
//!   cargo bench
//!
//! HTML reports are written to `target/criterion/`.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simdmath::ops::vec::VecExt;

// ---------------------------------------------------------------------------
// Bench sizes
// ---------------------------------------------------------------------------

/// Sizes exercised by every benchmark group.
///
/// - 8   : exactly one SIMD lane (no tail)
/// - 64  : 8 full chunks (no tail)
/// - 256 : medium workload
/// - 1024: larger workload
const SIZES: &[usize] = &[8, 64, 256, 1024];

// ---------------------------------------------------------------------------
// Helpers — scalar baselines
// ---------------------------------------------------------------------------

fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn scalar_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn scalar_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

fn scalar_div(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x / y).collect()
}

fn scalar_rem(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x % y).collect()
}

fn scalar_add_s(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|x| x + s).collect()
}

fn scalar_mul_s(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|x| x * s).collect()
}

fn scalar_sum(a: &[f32]) -> f32 {
    a.iter().sum()
}

fn scalar_product(a: &[f32]) -> f32 {
    a.iter().product()
}

fn scalar_min(a: &[f32]) -> f32 {
    a.iter().copied().fold(f32::INFINITY, f32::min)
}

fn scalar_max(a: &[f32]) -> f32 {
    a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}





// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

fn make_vecs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.5 + 1.0).collect();
    (a, b)
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

fn bench_add(c: &mut Criterion) {
    let mut g = c.benchmark_group("add");
    for &n in SIZES {
        let (a, b) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.add(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_sub(c: &mut Criterion) {
    let mut g = c.benchmark_group("sub");
    for &n in SIZES {
        let (a, b) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.sub(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_sub(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_mul(c: &mut Criterion) {
    let mut g = c.benchmark_group("mul");
    for &n in SIZES {
        let (a, b) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.mul(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_mul(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_div(c: &mut Criterion) {
    let mut g = c.benchmark_group("div");
    for &n in SIZES {
        let (a, b) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.div(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_div(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_rem(c: &mut Criterion) {
    let mut g = c.benchmark_group("rem");
    for &n in SIZES {
        let (a, b) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.rem(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_rem(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_add_scalar(c: &mut Criterion) {
    let mut g = c.benchmark_group("add_scalar");
    for &n in SIZES {
        let (a, _) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.add_scalar(black_box(3.14))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_add_s(black_box(&a), black_box(3.14))))
        });
    }
    g.finish();
}

fn bench_mul_scalar(c: &mut Criterion) {
    let mut g = c.benchmark_group("mul_scalar");
    for &n in SIZES {
        let (a, _) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.mul_scalar(black_box(2.0))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_mul_s(black_box(&a), black_box(2.0))))
        });
    }
    g.finish();
}

fn bench_sum(c: &mut Criterion) {
    let mut g = c.benchmark_group("sum");
    for &n in SIZES {
        let (a, _) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.sum()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_sum(black_box(&a))))
        });
    }
    g.finish();
}

fn bench_product(c: &mut Criterion) {
    let mut g = c.benchmark_group("product");
    for &n in SIZES {
        let (a, _) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.product()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_product(black_box(&a))))
        });
    }
    g.finish();
}

fn bench_min(c: &mut Criterion) {
    let mut g = c.benchmark_group("min");
    for &n in SIZES {
        let (a, _) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.min()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_min(black_box(&a))))
        });
    }
    g.finish();
}

fn bench_max(c: &mut Criterion) {
    let mut g = c.benchmark_group("max");
    for &n in SIZES {
        let (a, _) = make_vecs(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.max()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_max(black_box(&a))))
        });
    }
    g.finish();
}





// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_add,
    bench_sub,
    bench_mul,
    bench_div,
    bench_rem,
    bench_add_scalar,
    bench_mul_scalar,
    bench_sum,
    bench_product,
    bench_min,
    bench_max,
);
criterion_main!(benches);
