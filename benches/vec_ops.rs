//! Benchmarks comparing `VecExt<f32>` and `VecExt<f64>` SIMD operations against
//! equivalent scalar (vanilla Rust iterator / loop) implementations.
//!
//! Run with:
//!   cargo bench
//!
//! HTML reports are written to `target/criterion/`.

use std::iter::{Product, Sum};
use std::ops::{Add, Div, Mul, Rem, Sub};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simdmath::ops::vec::VecExt;

// ---------------------------------------------------------------------------
// Bench sizes
// ---------------------------------------------------------------------------

/// Sizes for `f32` benchmarks (F32x8 — 8 lanes).
///
/// - 8   : exactly one SIMD register (no tail)
/// - 64  : 8 full chunks (no tail)
/// - 256 : medium workload
/// - 1024: larger workload
const SIZES_F32: &[usize] = &[8, 64, 256, 1024];

/// Sizes for `f64` benchmarks (F64x4 — 4 lanes).
///
/// - 4   : exactly one SIMD register (no tail)
/// - 64  : 16 full chunks (no tail)
/// - 256 : medium workload
/// - 1024: larger workload
const SIZES_F64: &[usize] = &[4, 64, 256, 1024];

// ---------------------------------------------------------------------------
// Helpers — generic scalar baselines
// ---------------------------------------------------------------------------

fn scalar_add<T: Copy + Add<Output = T>>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b).map(|(x, y)| *x + *y).collect()
}

fn scalar_sub<T: Copy + Sub<Output = T>>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b).map(|(x, y)| *x - *y).collect()
}

fn scalar_mul<T: Copy + Mul<Output = T>>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b).map(|(x, y)| *x * *y).collect()
}

fn scalar_div<T: Copy + Div<Output = T>>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b).map(|(x, y)| *x / *y).collect()
}

fn scalar_rem<T: Copy + Rem<Output = T>>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b).map(|(x, y)| *x % *y).collect()
}

fn scalar_add_s<T: Copy + Add<Output = T>>(a: &[T], s: T) -> Vec<T> {
    a.iter().map(|x| *x + s).collect()
}

fn scalar_mul_s<T: Copy + Mul<Output = T>>(a: &[T], s: T) -> Vec<T> {
    a.iter().map(|x| *x * s).collect()
}

fn scalar_sum<T: Copy + Sum>(a: &[T]) -> T {
    a.iter().copied().sum()
}

fn scalar_product<T: Copy + Product>(a: &[T]) -> T {
    a.iter().copied().product()
}

fn scalar_min<T: Copy + PartialOrd>(a: &[T], init: T) -> T {
    a.iter().copied().fold(init, |acc, x| if x < acc { x } else { acc })
}

fn scalar_max<T: Copy + PartialOrd>(a: &[T], init: T) -> T {
    a.iter().copied().fold(init, |acc, x| if x > acc { x } else { acc })
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

fn make_vecs_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.5 + 1.0).collect();
    (a, b)
}

fn make_vecs_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.5 + 1.0).collect();
    (a, b)
}

// ---------------------------------------------------------------------------
// Benchmark groups — f32
// ---------------------------------------------------------------------------

fn bench_add_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/add");
    for &n in SIZES_F32 {
        let (a, b) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.add(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_sub_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/sub");
    for &n in SIZES_F32 {
        let (a, b) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.sub(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_sub(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_mul_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/mul");
    for &n in SIZES_F32 {
        let (a, b) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.mul(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_mul(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_div_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/div");
    for &n in SIZES_F32 {
        let (a, b) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.div(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_div(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_rem_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/rem");
    for &n in SIZES_F32 {
        let (a, b) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.rem(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_rem(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_add_scalar_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/add_scalar");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.add_scalar(black_box(3.14f32))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_add_s(black_box(&a), black_box(3.14f32))))
        });
    }
    g.finish();
}

fn bench_mul_scalar_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/mul_scalar");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.mul_scalar(black_box(2.0f32))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_mul_s(black_box(&a), black_box(2.0f32))))
        });
    }
    g.finish();
}

fn bench_sum_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/sum");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.sum()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_sum::<f32>(black_box(&a))))
        });
    }
    g.finish();
}

fn bench_product_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/product");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.product()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_product::<f32>(black_box(&a))))
        });
    }
    g.finish();
}

fn bench_min_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/min");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.min()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_min(black_box(&a), f32::INFINITY)))
        });
    }
    g.finish();
}

fn bench_max_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32/max");
    for &n in SIZES_F32 {
        let (a, _) = make_vecs_f32(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.max()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_max(black_box(&a), f32::NEG_INFINITY)))
        });
    }
    g.finish();
}

// ---------------------------------------------------------------------------
// Benchmark groups — f64
// ---------------------------------------------------------------------------

fn bench_add_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/add");
    for &n in SIZES_F64 {
        let (a, b) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.add(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_add(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_sub_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/sub");
    for &n in SIZES_F64 {
        let (a, b) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.sub(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_sub(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_mul_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/mul");
    for &n in SIZES_F64 {
        let (a, b) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.mul(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_mul(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_div_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/div");
    for &n in SIZES_F64 {
        let (a, b) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.div(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_div(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_rem_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/rem");
    for &n in SIZES_F64 {
        let (a, b) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.rem(black_box(&b))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_rem(black_box(&a), black_box(&b))))
        });
    }
    g.finish();
}

fn bench_add_scalar_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/add_scalar");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.add_scalar(black_box(3.14f64))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_add_s(black_box(&a), black_box(3.14f64))))
        });
    }
    g.finish();
}

fn bench_mul_scalar_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/mul_scalar");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.mul_scalar(black_box(2.0f64))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_mul_s(black_box(&a), black_box(2.0f64))))
        });
    }
    g.finish();
}

fn bench_sum_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/sum");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.sum()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_sum::<f64>(black_box(&a))))
        });
    }
    g.finish();
}

fn bench_product_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/product");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.product()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_product::<f64>(black_box(&a))))
        });
    }
    g.finish();
}

fn bench_min_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/min");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.min()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_min(black_box(&a), f64::INFINITY)))
        });
    }
    g.finish();
}

fn bench_max_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("f64/max");
    for &n in SIZES_F64 {
        let (a, _) = make_vecs_f64(n);
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| black_box(a.max()))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| black_box(scalar_max(black_box(&a), f64::NEG_INFINITY)))
        });
    }
    g.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_add_f32,
    bench_sub_f32,
    bench_mul_f32,
    bench_div_f32,
    bench_rem_f32,
    bench_add_scalar_f32,
    bench_mul_scalar_f32,
    bench_sum_f32,
    bench_product_f32,
    bench_min_f32,
    bench_max_f32,
    bench_add_f64,
    bench_sub_f64,
    bench_mul_f64,
    bench_div_f64,
    bench_rem_f64,
    bench_add_scalar_f64,
    bench_mul_scalar_f64,
    bench_sum_f64,
    bench_product_f64,
    bench_min_f64,
    bench_max_f64,
);
criterion_main!(benches);
