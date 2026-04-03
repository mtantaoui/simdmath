//! Benchmarks comparing `VecExt<f32>` and `VecExt<f64>` SIMD operations against
//! equivalent scalar (vanilla Rust iterator / loop) implementations.
//!
//! Run with:
//!   cargo bench
//!
//! Run specific benchmark group:
//!   cargo bench -- "f32/add"
//!
//! HTML reports are written to `target/criterion/`.

mod ops;

use criterion::{criterion_group, criterion_main};

use ops::acos::{bench_acos_f32, bench_acos_f64};
use ops::add::{bench_add_f32, bench_add_f64, bench_add_scalar_f32, bench_add_scalar_f64};
use ops::asin::{bench_asin_f32, bench_asin_f64};
use ops::div::{bench_div_f32, bench_div_f64};
use ops::mul::{bench_mul_f32, bench_mul_f64, bench_mul_scalar_f32, bench_mul_scalar_f64};
use ops::reduce::{
    bench_max_f32, bench_max_f64, bench_min_f32, bench_min_f64, bench_product_f32,
    bench_product_f64, bench_sum_f32, bench_sum_f64,
};
use ops::rem::{bench_rem_f32, bench_rem_f64};
use ops::sub::{bench_sub_f32, bench_sub_f64};

criterion_group!(
    benches,
    // Transcendentals
    bench_acos_f32,
    bench_acos_f64,
    bench_asin_f32,
    bench_asin_f64,
    // Addition
    bench_add_f32,
    bench_add_f64,
    bench_add_scalar_f32,
    bench_add_scalar_f64,
    // Subtraction
    bench_sub_f32,
    bench_sub_f64,
    // Multiplication
    bench_mul_f32,
    bench_mul_f64,
    bench_mul_scalar_f32,
    bench_mul_scalar_f64,
    // Division
    bench_div_f32,
    bench_div_f64,
    // Remainder
    bench_rem_f32,
    bench_rem_f64,
    // Reductions
    bench_sum_f32,
    bench_sum_f64,
    bench_product_f32,
    bench_product_f64,
    bench_min_f32,
    bench_min_f64,
    bench_max_f32,
    bench_max_f64,
);
criterion_main!(benches);
