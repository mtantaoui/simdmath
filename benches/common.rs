//! Shared utilities for benchmarks.

#![allow(dead_code)]

use std::iter::{Product, Sum};
use std::ops::{Add, Div, Mul, Rem, Sub};

// ---------------------------------------------------------------------------
// Bench sizes
// ---------------------------------------------------------------------------

/// Sizes for `f32` benchmarks (F32x8 — 8 lanes).
pub const SIZES_F32: &[usize] = &[
    8, // , 64, 256, 1024
];

/// Sizes for `f64` benchmarks (F64x4 — 4 lanes).
pub const SIZES_F64: &[usize] = &[
    4, // , 64, 256, 1024
];

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

pub fn make_vecs_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.5 + 1.0).collect();
    (a, b)
}

pub fn make_vecs_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.5 + 1.0).collect();
    (a, b)
}

// ---------------------------------------------------------------------------
// Scalar baselines
// ---------------------------------------------------------------------------

pub fn scalar_add<T: Copy + Add<Output = T>>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b).map(|(x, y)| *x + *y).collect()
}

pub fn scalar_sub<T: Copy + Sub<Output = T>>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b).map(|(x, y)| *x - *y).collect()
}

pub fn scalar_mul<T: Copy + Mul<Output = T>>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b).map(|(x, y)| *x * *y).collect()
}

pub fn scalar_div<T: Copy + Div<Output = T>>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b).map(|(x, y)| *x / *y).collect()
}

pub fn scalar_rem<T: Copy + Rem<Output = T>>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b).map(|(x, y)| *x % *y).collect()
}

pub fn scalar_add_s<T: Copy + Add<Output = T>>(a: &[T], s: T) -> Vec<T> {
    a.iter().map(|x| *x + s).collect()
}

pub fn scalar_mul_s<T: Copy + Mul<Output = T>>(a: &[T], s: T) -> Vec<T> {
    a.iter().map(|x| *x * s).collect()
}

pub fn scalar_sum<T: Copy + Sum>(a: &[T]) -> T {
    a.iter().copied().sum()
}

pub fn scalar_product<T: Copy + Product>(a: &[T]) -> T {
    a.iter().copied().product()
}

pub fn scalar_min<T: Copy + PartialOrd>(a: &[T], init: T) -> T {
    a.iter()
        .copied()
        .fold(init, |acc, x| if x < acc { x } else { acc })
}

pub fn scalar_max<T: Copy + PartialOrd>(a: &[T], init: T) -> T {
    a.iter()
        .copied()
        .fold(init, |acc, x| if x > acc { x } else { acc })
}
