//! AVX2 implementation of [`VecExt`] for `Vec<f32>`.
//!
//! This module contains the architecture-specific helpers (using `F32x8` and AVX2
//! intrinsics) and the `impl VecExt for Vec<f32>` that wires them together. All
//! SIMD details are isolated here so that [`crate::ops::vec`] stays free of any
//! architecture-specific code.

use std::arch::x86_64::{_mm256_max_ps, _mm256_min_ps, _mm256_set1_ps, _mm256_setzero_ps};

use crate::arch::avx2::f32x8::{F32x8, LANE_COUNT};
use crate::ops::simd::{Load, Store};
use crate::ops::vec::VecExt;

// ---------------------------------------------------------------------------
// Internal helpers — one per SIMD loop shape
// ---------------------------------------------------------------------------

/// Applies `op` element-wise to `lhs` and `rhs`, writing results into a freshly
/// allocated `Vec`. Caller must guarantee `lhs.len() == rhs.len()`.
#[inline]
fn binary_op(lhs: &[f32], rhs: &[f32], op: impl Fn(F32x8, F32x8) -> F32x8) -> Vec<f32> {
    let n = lhs.len();
    let full_chunks = n / LANE_COUNT;
    let tail = n % LANE_COUNT;

    // SAFETY: capacity is n; every index in [0, n) will be written before
    // set_len is called, so no uninitialised bytes are ever exposed.
    let mut out: Vec<f32> = Vec::with_capacity(n);

    for i in 0..full_chunks {
        let offset = i * LANE_COUNT;
        // SAFETY: offset + LANE_COUNT <= n, so both pointers are in bounds.
        let a = unsafe { F32x8::load(lhs.as_ptr().add(offset), LANE_COUNT) };
        let b = unsafe { F32x8::load(rhs.as_ptr().add(offset), LANE_COUNT) };
        let result = op(a, b);
        // SAFETY: out has capacity n; store_at internally casts *const → *mut
        // and writes exactly LANE_COUNT elements at `offset`.
        unsafe { result.store_at(out.as_ptr().add(offset)) };
    }

    if tail > 0 {
        let offset = full_chunks * LANE_COUNT;
        // SAFETY: offset < n; tail < LANE_COUNT, meeting load_partial's precondition.
        let a = unsafe { F32x8::load_partial(lhs.as_ptr().add(offset), tail) };
        let b = unsafe { F32x8::load_partial(rhs.as_ptr().add(offset), tail) };
        let result = op(a, b);
        // SAFETY: store_at_partial writes only the first `tail` elements via a
        // sign-bit mask; out has capacity for them.
        unsafe { result.store_at_partial(out.as_mut_ptr().add(offset)) };
    }

    // SAFETY: full_chunks * LANE_COUNT + tail == n; every element in [0, n)
    // has been written by the loops above.
    unsafe { out.set_len(n) };
    out
}

/// Broadcasts `rhs` to all 8 lanes, then applies `op` element-wise to `lhs`.
#[inline]
fn scalar_op(lhs: &[f32], rhs: f32, op: impl Fn(F32x8, F32x8) -> F32x8) -> Vec<f32> {
    let n = lhs.len();
    let full_chunks = n / LANE_COUNT;
    let tail = n % LANE_COUNT;

    // SAFETY: _mm256_set1_ps is always safe for any finite or non-finite f32.
    let scalar_elements = unsafe { _mm256_set1_ps(rhs) };
    // Reuse one broadcast register for every full chunk.
    let scalar_full = F32x8 {
        size: LANE_COUNT,
        elements: scalar_elements,
    };

    // SAFETY: see binary_op for the set_len rationale.
    let mut out: Vec<f32> = Vec::with_capacity(n);

    for i in 0..full_chunks {
        let offset = i * LANE_COUNT;
        // SAFETY: offset + LANE_COUNT <= n.
        let a = unsafe { F32x8::load(lhs.as_ptr().add(offset), LANE_COUNT) };
        let result = op(a, scalar_full);
        // SAFETY: out has capacity for offset + LANE_COUNT.
        unsafe { result.store_at(out.as_ptr().add(offset)) };
    }

    if tail > 0 {
        let offset = full_chunks * LANE_COUNT;
        // SAFETY: offset < n; tail < LANE_COUNT.
        let a = unsafe { F32x8::load_partial(lhs.as_ptr().add(offset), tail) };
        // Match `size` so the F32x8 arithmetic debug-assert (sizes must equal) passes.
        let scalar_tail = F32x8 {
            size: tail,
            elements: scalar_elements,
        };
        let result = op(a, scalar_tail);
        // SAFETY: writes only the first `tail` elements.
        unsafe { result.store_at_partial(out.as_mut_ptr().add(offset)) };
    }

    // SAFETY: all n elements have been written.
    unsafe { out.set_len(n) };
    out
}

/// In-place element-wise `op`: `lhs[i] = op(lhs[i], rhs[i])`.
/// Caller must guarantee `lhs.len() == rhs.len()`.
#[inline]
fn binary_op_inplace(lhs: &mut [f32], rhs: &[f32], op: impl Fn(F32x8, F32x8) -> F32x8) {
    let n = lhs.len();
    let full_chunks = n / LANE_COUNT;
    let tail = n % LANE_COUNT;

    for i in 0..full_chunks {
        let offset = i * LANE_COUNT;
        // SAFETY: offset + LANE_COUNT <= n for both slices.
        let a = unsafe { F32x8::load(lhs.as_ptr().add(offset), LANE_COUNT) };
        let b = unsafe { F32x8::load(rhs.as_ptr().add(offset), LANE_COUNT) };
        let result = op(a, b);
        // SAFETY: `a` was already read into a register, so there is no aliasing
        // concern; store_at writes back to the same valid, mutable region.
        unsafe { result.store_at(lhs.as_ptr().add(offset)) };
    }

    if tail > 0 {
        let offset = full_chunks * LANE_COUNT;
        // SAFETY: offset < n; tail < LANE_COUNT.
        let a = unsafe { F32x8::load_partial(lhs.as_ptr().add(offset), tail) };
        let b = unsafe { F32x8::load_partial(rhs.as_ptr().add(offset), tail) };
        let result = op(a, b);
        // SAFETY: writes only the first `tail` elements.
        unsafe { result.store_at_partial(lhs.as_mut_ptr().add(offset)) };
    }
}

/// In-place scalar `op`: `lhs[i] = op(lhs[i], rhs)`.
#[inline]
fn scalar_op_inplace(lhs: &mut [f32], rhs: f32, op: impl Fn(F32x8, F32x8) -> F32x8) {
    let n = lhs.len();
    let full_chunks = n / LANE_COUNT;
    let tail = n % LANE_COUNT;

    // SAFETY: _mm256_set1_ps is always safe.
    let scalar_elements = unsafe { _mm256_set1_ps(rhs) };
    let scalar_full = F32x8 {
        size: LANE_COUNT,
        elements: scalar_elements,
    };

    for i in 0..full_chunks {
        let offset = i * LANE_COUNT;
        // SAFETY: offset + LANE_COUNT <= n.
        let a = unsafe { F32x8::load(lhs.as_ptr().add(offset), LANE_COUNT) };
        let result = op(a, scalar_full);
        // SAFETY: read-before-write; valid mutable region.
        unsafe { result.store_at(lhs.as_ptr().add(offset)) };
    }

    if tail > 0 {
        let offset = full_chunks * LANE_COUNT;
        // SAFETY: offset < n; tail < LANE_COUNT.
        let a = unsafe { F32x8::load_partial(lhs.as_ptr().add(offset), tail) };
        let scalar_tail = F32x8 {
            size: tail,
            elements: scalar_elements,
        };
        let result = op(a, scalar_tail);
        // SAFETY: writes only the first `tail` elements.
        unsafe { result.store_at_partial(lhs.as_mut_ptr().add(offset)) };
    }
}



impl VecExt<f32> for Vec<f32> {
    #[inline]
    fn add(&self, rhs: &Self) -> Vec<f32> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op(self, rhs, |a, b| a + b)
    }

    #[inline]
    fn sub(&self, rhs: &Self) -> Vec<f32> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op(self, rhs, |a, b| a - b)
    }

    #[inline]
    fn mul(&self, rhs: &Self) -> Vec<f32> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op(self, rhs, |a, b| a * b)
    }

    #[inline]
    fn div(&self, rhs: &Self) -> Vec<f32> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op(self, rhs, |a, b| a / b)
    }

    #[inline]
    fn rem(&self, rhs: &Self) -> Vec<f32> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op(self, rhs, |a, b| a % b)
    }

    #[inline]
    fn add_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op(self, rhs, |a, b| a + b)
    }

    #[inline]
    fn sub_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op(self, rhs, |a, b| a - b)
    }

    #[inline]
    fn mul_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op(self, rhs, |a, b| a * b)
    }

    #[inline]
    fn div_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op(self, rhs, |a, b| a / b)
    }

    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op_inplace(self, rhs, |a, b| a + b);
    }

    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op_inplace(self, rhs, |a, b| a - b);
    }

    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op_inplace(self, rhs, |a, b| a * b);
    }

    #[inline]
    fn div_assign(&mut self, rhs: &Self) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op_inplace(self, rhs, |a, b| a / b);
    }

    #[inline]
    fn add_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace(self, rhs, |a, b| a + b);
    }

    #[inline]
    fn sub_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace(self, rhs, |a, b| a - b);
    }

    #[inline]
    fn mul_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace(self, rhs, |a, b| a * b);
    }

    #[inline]
    fn div_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace(self, rhs, |a, b| a / b);
    }

    fn sum(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / LANE_COUNT;
        let tail = n % LANE_COUNT;

        // SAFETY: _mm256_setzero_ps is always safe.
        let mut acc = F32x8 {
            size: LANE_COUNT,
            elements: unsafe { _mm256_setzero_ps() },
        };

        for i in 0..full_chunks {
            let offset = i * LANE_COUNT;
            // SAFETY: offset + LANE_COUNT <= n.
            let chunk = unsafe { F32x8::load(self.as_ptr().add(offset), LANE_COUNT) };
            acc += chunk;
        }

        // Extract all 8 lanes from the accumulator register and reduce them.
        let mut arr = [0.0f32; LANE_COUNT];
        // SAFETY: `arr` is mut and valid for LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f32 = arr.iter().copied().sum();

        // Handle the tail with scalar arithmetic.
        if tail > 0 {
            let offset = full_chunks * LANE_COUNT;
            for j in 0..tail {
                result += self[offset + j];
            }
        }

        result
    }

    fn product(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / LANE_COUNT;
        let tail = n % LANE_COUNT;

        // SAFETY: _mm256_set1_ps is always safe.
        let mut acc = F32x8 {
            size: LANE_COUNT,
            elements: unsafe { _mm256_set1_ps(1.0) },
        };

        for i in 0..full_chunks {
            let offset = i * LANE_COUNT;
            // SAFETY: offset + LANE_COUNT <= n.
            let chunk = unsafe { F32x8::load(self.as_ptr().add(offset), LANE_COUNT) };
            acc *= chunk;
        }

        let mut arr = [0.0f32; LANE_COUNT];
        // SAFETY: `arr` is mut and valid for LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f32 = arr.iter().copied().product();

        if tail > 0 {
            let offset = full_chunks * LANE_COUNT;
            for j in 0..tail {
                result *= self[offset + j];
            }
        }

        result
    }

    fn min(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / LANE_COUNT;
        let tail = n % LANE_COUNT;

        // SAFETY: _mm256_set1_ps is always safe.
        let mut acc = F32x8 {
            size: LANE_COUNT,
            elements: unsafe { _mm256_set1_ps(f32::INFINITY) },
        };

        for i in 0..full_chunks {
            let offset = i * LANE_COUNT;
            // SAFETY: offset + LANE_COUNT <= n.
            let chunk = unsafe { F32x8::load(self.as_ptr().add(offset), LANE_COUNT) };
            // SAFETY: _mm256_min_ps is always safe for valid __m256 operands.
            acc = F32x8 {
                size: LANE_COUNT,
                elements: unsafe { _mm256_min_ps(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f32; LANE_COUNT];
        // SAFETY: `arr` is mut and valid for LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f32::INFINITY, f32::min);

        if tail > 0 {
            let offset = full_chunks * LANE_COUNT;
            for j in 0..tail {
                result = result.min(self[offset + j]);
            }
        }

        result
    }

    fn max(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / LANE_COUNT;
        let tail = n % LANE_COUNT;

        // SAFETY: _mm256_set1_ps is always safe.
        let mut acc = F32x8 {
            size: LANE_COUNT,
            elements: unsafe { _mm256_set1_ps(f32::NEG_INFINITY) },
        };

        for i in 0..full_chunks {
            let offset = i * LANE_COUNT;
            // SAFETY: offset + LANE_COUNT <= n.
            let chunk = unsafe { F32x8::load(self.as_ptr().add(offset), LANE_COUNT) };
            // SAFETY: _mm256_max_ps is always safe for valid __m256 operands.
            acc = F32x8 {
                size: LANE_COUNT,
                elements: unsafe { _mm256_max_ps(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f32; LANE_COUNT];
        // SAFETY: `arr` is mut and valid for LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if tail > 0 {
            let offset = full_chunks * LANE_COUNT;
            for j in 0..tail {
                result = result.max(self[offset + j]);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::ops::vec::VecExt;
    // Vec×Vec — length 11 exercises both the 8-lane full chunk and the 3-element tail.

    #[test]
    fn add_produces_correct_sum() {
        let a: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        let expected: Vec<f32> = (1..=11).map(|x| (x * 2) as f32).collect();
        assert_eq!(a.add(&b), expected);
    }

    #[test]
    fn sub_produces_correct_difference() {
        let a: Vec<f32> = (1..=11).map(|x| (x * 2) as f32).collect();
        let b: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        let expected: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        assert_eq!(a.sub(&b), expected);
    }

    #[test]
    fn mul_produces_correct_product() {
        let a: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        let b = vec![2.0f32; 11];
        let expected: Vec<f32> = (1..=11).map(|x| (x * 2) as f32).collect();
        assert_eq!(a.mul(&b), expected);
    }

    #[test]
    fn div_produces_correct_quotient() {
        let a: Vec<f32> = (1..=11).map(|x| (x * 2) as f32).collect();
        let b = vec![2.0f32; 11];
        let expected: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        assert_eq!(a.div(&b), expected);
    }

    #[test]
    fn rem_produces_correct_remainder() {
        // Includes negative values to verify truncation semantics: -7 % 3 == -1 (not 2).
        let a = vec![7.0f32, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, -7.0, -7.0, 7.0];
        let b = vec![3.0f32; 11];
        let expected = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
        assert_eq!(a.rem(&b), expected);
    }

    // Vec×scalar

    #[test]
    fn add_scalar_broadcasts_correctly() {
        let a: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        let expected: Vec<f32> = (11..=21).map(|x| x as f32).collect();
        assert_eq!(a.add_scalar(10.0), expected);
    }

    #[test]
    fn mul_scalar_broadcasts_correctly() {
        let a: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        let expected: Vec<f32> = (1..=11).map(|x| (x * 2) as f32).collect();
        assert_eq!(a.mul_scalar(2.0), expected);
    }

    // In-place

    #[test]
    fn add_assign_matches_add() {
        let a: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        let b = vec![1.0f32; 11];
        let expected = a.add(&b);
        let mut a_mut = a;
        a_mut.add_assign(&b);
        assert_eq!(a_mut, expected);
    }

    #[test]
    fn mul_scalar_assign_matches_mul_scalar() {
        let a: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        let expected = a.mul_scalar(3.0);
        let mut a_mut = a;
        a_mut.mul_scalar_assign(3.0);
        assert_eq!(a_mut, expected);
    }

    // Reductions

    #[test]
    fn sum_is_correct() {
        // 1 + 2 + … + 8 = 36
        let a: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        assert_eq!(a.sum(), 36.0);
    }

    #[test]
    fn product_is_correct() {
        // 1 * 2 * 3 * 4 = 24
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(a.product(), 24.0);
    }

    #[test]
    fn min_finds_minimum() {
        let a = vec![5.0f32, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 4.0, 6.0, 10.0, 11.0];
        assert_eq!(a.min(), 1.0);
    }

    #[test]
    fn max_finds_maximum() {
        let a = vec![5.0f32, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 4.0, 6.0, 10.0, 11.0];
        assert_eq!(a.max(), 11.0);
    }

    // Edge cases

    #[test]
    fn empty_vec_sum_is_zero() {
        let a: Vec<f32> = vec![];
        assert_eq!(a.sum(), 0.0);
    }

    #[test]
    fn empty_vec_product_is_one() {
        let a: Vec<f32> = vec![];
        assert_eq!(a.product(), 1.0);
    }

    #[test]
    fn single_element_add() {
        let a = vec![5.0f32];
        let b = vec![3.0f32];
        assert_eq!(a.add(&b), vec![8.0f32]);
    }

    #[test]
    #[should_panic]
    fn length_mismatch_panics() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0];
        let _ = a.add(&b);
    }
}
