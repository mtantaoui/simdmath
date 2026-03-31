//! AVX-512F implementation of [`VecExt`] for `Vec<f32>` and `Vec<f64>`.
//!
//! The generic loop helpers (`binary_op`, `scalar_op`, etc.) live in
//! [`crate::ops::vec`] and are reused here unchanged. Only the horizontal
//! reductions and the `impl VecExt<T>` blocks are architecture-specific.

use std::arch::x86_64::{_mm512_max_pd, _mm512_max_ps, _mm512_min_pd, _mm512_min_ps};

use crate::arch::avx512::{f32x16, f32x16::F32x16};
use crate::arch::avx512::{f64x8, f64x8::F64x8};
use crate::ops::simd::{Load, Store};
use crate::ops::vec::{binary_op, binary_op_inplace, scalar_op, scalar_op_inplace, VecExt};

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
        binary_op::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a + b)
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
        binary_op::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a - b)
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
        binary_op::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a * b)
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
        binary_op::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a / b)
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
        binary_op::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a % b)
    }

    #[inline]
    fn add_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a + b)
    }

    #[inline]
    fn sub_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a - b)
    }

    #[inline]
    fn mul_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a * b)
    }

    #[inline]
    fn div_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a / b)
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
        binary_op_inplace::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a + b);
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
        binary_op_inplace::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a - b);
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
        binary_op_inplace::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a * b);
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
        binary_op_inplace::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a / b);
    }

    #[inline]
    fn add_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a + b);
    }

    #[inline]
    fn sub_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a - b);
    }

    #[inline]
    fn mul_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a * b);
    }

    #[inline]
    fn div_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x16>(self, rhs, f32x16::LANE_COUNT, |a, b| a / b);
    }

    fn sum(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x16::LANE_COUNT;
        let tail = n % f32x16::LANE_COUNT;

        let mut acc = unsafe { F32x16::zero() };

        for i in 0..full_chunks {
            let offset = i * f32x16::LANE_COUNT;
            let chunk = unsafe { F32x16::load(self.as_ptr().add(offset), f32x16::LANE_COUNT) };
            acc += chunk;
        }

        let mut arr = [0.0f32; f32x16::LANE_COUNT];
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f32 = arr.iter().copied().sum();

        if tail > 0 {
            let offset = full_chunks * f32x16::LANE_COUNT;
            for j in 0..tail {
                result += self[offset + j];
            }
        }

        result
    }

    fn product(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x16::LANE_COUNT;
        let tail = n % f32x16::LANE_COUNT;

        let mut acc = unsafe { F32x16::broadcast(1.0) };

        for i in 0..full_chunks {
            let offset = i * f32x16::LANE_COUNT;
            let chunk = unsafe { F32x16::load(self.as_ptr().add(offset), f32x16::LANE_COUNT) };
            acc *= chunk;
        }

        let mut arr = [0.0f32; f32x16::LANE_COUNT];
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f32 = arr.iter().copied().product();

        if tail > 0 {
            let offset = full_chunks * f32x16::LANE_COUNT;
            for j in 0..tail {
                result *= self[offset + j];
            }
        }

        result
    }

    fn min(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x16::LANE_COUNT;
        let tail = n % f32x16::LANE_COUNT;

        let mut acc = unsafe { F32x16::broadcast(f32::INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f32x16::LANE_COUNT;
            let chunk = unsafe { F32x16::load(self.as_ptr().add(offset), f32x16::LANE_COUNT) };
            acc = F32x16 {
                size: f32x16::LANE_COUNT,
                elements: unsafe { _mm512_min_ps(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f32; f32x16::LANE_COUNT];
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f32::INFINITY, f32::min);

        if tail > 0 {
            let offset = full_chunks * f32x16::LANE_COUNT;
            for j in 0..tail {
                result = result.min(self[offset + j]);
            }
        }

        result
    }

    fn max(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x16::LANE_COUNT;
        let tail = n % f32x16::LANE_COUNT;

        let mut acc = unsafe { F32x16::broadcast(f32::NEG_INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f32x16::LANE_COUNT;
            let chunk = unsafe { F32x16::load(self.as_ptr().add(offset), f32x16::LANE_COUNT) };
            acc = F32x16 {
                size: f32x16::LANE_COUNT,
                elements: unsafe { _mm512_max_ps(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f32; f32x16::LANE_COUNT];
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if tail > 0 {
            let offset = full_chunks * f32x16::LANE_COUNT;
            for j in 0..tail {
                result = result.max(self[offset + j]);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vecs(n: usize) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.5 + 1.0).collect();
        (a, b)
    }

    // Use sizes that hit: exactly one register (16), no tail; and a tail case.
    const N_FULL: usize = 32;   // 2 full F32x16 chunks, no tail
    const N_TAIL: usize = 19;   // 1 full chunk + 3-lane tail

    #[test]
    fn add_produces_correct_result() {
        let (a, b) = make_vecs(N_FULL);
        let result = a.add(&b);
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn add_with_tail_produces_correct_result() {
        let (a, b) = make_vecs(N_TAIL);
        let result = a.add(&b);
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn sub_produces_correct_result() {
        let (a, b) = make_vecs(N_FULL);
        let result = a.sub(&b);
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn mul_produces_correct_result() {
        let (a, b) = make_vecs(N_FULL);
        let result = a.mul(&b);
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn div_produces_correct_result() {
        let (a, b) = make_vecs(N_FULL);
        let result = a.div(&b);
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x / y).collect();
        for (r, e) in result.iter().zip(&expected) {
            assert!((r - e).abs() < 1e-5, "expected {e}, got {r}");
        }
    }

    #[test]
    fn rem_produces_correct_result() {
        let a: Vec<f32> = vec![7.0f32; N_FULL];
        let b: Vec<f32> = vec![3.0f32; N_FULL];
        let result = a.rem(&b);
        assert!(result.iter().all(|&x| (x - 1.0f32).abs() < 1e-5));
    }

    #[test]
    fn add_scalar_produces_correct_result() {
        let (a, _) = make_vecs(N_FULL);
        let result = a.add_scalar(10.0);
        let expected: Vec<f32> = a.iter().map(|x| x + 10.0).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn mul_scalar_produces_correct_result() {
        let (a, _) = make_vecs(N_FULL);
        let result = a.mul_scalar(2.0);
        let expected: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn add_assign_modifies_in_place() {
        let (mut a, b) = make_vecs(N_FULL);
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        a.add_assign(&b);
        assert_eq!(a, expected);
    }

    #[test]
    fn mul_assign_modifies_in_place() {
        let (mut a, b) = make_vecs(N_FULL);
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        a.mul_assign(&b);
        assert_eq!(a, expected);
    }

    #[test]
    fn add_scalar_assign_modifies_in_place() {
        let (mut a, _) = make_vecs(N_FULL);
        let expected: Vec<f32> = a.iter().map(|x| x + 5.0).collect();
        a.add_scalar_assign(5.0);
        assert_eq!(a, expected);
    }

    #[test]
    fn sum_produces_correct_result() {
        let (a, _) = make_vecs(N_FULL);
        let result = a.sum();
        let expected: f32 = a.iter().sum();
        assert!((result - expected).abs() < 1e-2, "expected {expected}, got {result}");
    }

    #[test]
    fn sum_with_tail_produces_correct_result() {
        let (a, _) = make_vecs(N_TAIL);
        let result = a.sum();
        let expected: f32 = a.iter().sum();
        assert!((result - expected).abs() < 1e-2, "expected {expected}, got {result}");
    }

    #[test]
    fn product_produces_correct_result() {
        let a: Vec<f32> = vec![2.0f32; 16]; // 2^16 = 65536
        let result = a.product();
        assert!((result - 65536.0f32).abs() < 1.0);
    }

    #[test]
    fn min_produces_correct_result() {
        let (a, _) = make_vecs(N_FULL);
        let result = a.min();
        let expected = a.iter().copied().fold(f32::INFINITY, f32::min);
        assert_eq!(result, expected);
    }

    #[test]
    fn max_produces_correct_result() {
        let (a, _) = make_vecs(N_FULL);
        let result = a.max();
        let expected = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(result, expected);
    }

    #[test]
    fn min_with_tail_produces_correct_result() {
        let (a, _) = make_vecs(N_TAIL);
        let result = a.min();
        let expected = a.iter().copied().fold(f32::INFINITY, f32::min);
        assert_eq!(result, expected);
    }

    #[test]
    fn max_with_tail_produces_correct_result() {
        let (a, _) = make_vecs(N_TAIL);
        let result = a.max();
        let expected = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(result, expected);
    }
}

// ---------------------------------------------------------------------------
// VecExt<f64> for Vec<f64>
// ---------------------------------------------------------------------------

impl VecExt<f64> for Vec<f64> {
    #[inline]
    fn add(&self, rhs: &Self) -> Vec<f64> {
        assert_eq!(self.len(), rhs.len(), "length mismatch: lhs has {} elements, rhs has {} elements", self.len(), rhs.len());
        binary_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a + b)
    }

    #[inline]
    fn sub(&self, rhs: &Self) -> Vec<f64> {
        assert_eq!(self.len(), rhs.len(), "length mismatch: lhs has {} elements, rhs has {} elements", self.len(), rhs.len());
        binary_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a - b)
    }

    #[inline]
    fn mul(&self, rhs: &Self) -> Vec<f64> {
        assert_eq!(self.len(), rhs.len(), "length mismatch: lhs has {} elements, rhs has {} elements", self.len(), rhs.len());
        binary_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a * b)
    }

    #[inline]
    fn div(&self, rhs: &Self) -> Vec<f64> {
        assert_eq!(self.len(), rhs.len(), "length mismatch: lhs has {} elements, rhs has {} elements", self.len(), rhs.len());
        binary_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a / b)
    }

    #[inline]
    fn rem(&self, rhs: &Self) -> Vec<f64> {
        assert_eq!(self.len(), rhs.len(), "length mismatch: lhs has {} elements, rhs has {} elements", self.len(), rhs.len());
        binary_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a % b)
    }

    #[inline]
    fn add_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a + b)
    }

    #[inline]
    fn sub_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a - b)
    }

    #[inline]
    fn mul_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a * b)
    }

    #[inline]
    fn div_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a / b)
    }

    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len(), "length mismatch: lhs has {} elements, rhs has {} elements", self.len(), rhs.len());
        binary_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a + b);
    }

    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len(), "length mismatch: lhs has {} elements, rhs has {} elements", self.len(), rhs.len());
        binary_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a - b);
    }

    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len(), "length mismatch: lhs has {} elements, rhs has {} elements", self.len(), rhs.len());
        binary_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a * b);
    }

    #[inline]
    fn div_assign(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len(), "length mismatch: lhs has {} elements, rhs has {} elements", self.len(), rhs.len());
        binary_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a / b);
    }

    #[inline]
    fn add_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a + b);
    }

    #[inline]
    fn sub_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a - b);
    }

    #[inline]
    fn mul_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a * b);
    }

    #[inline]
    fn div_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a / b);
    }

    fn sum(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x8::LANE_COUNT;
        let tail = n % f64x8::LANE_COUNT;

        let mut acc = unsafe { F64x8::zero() };

        for i in 0..full_chunks {
            let offset = i * f64x8::LANE_COUNT;
            let chunk = unsafe { F64x8::load(self.as_ptr().add(offset), f64x8::LANE_COUNT) };
            acc += chunk;
        }

        let mut arr = [0.0f64; f64x8::LANE_COUNT];
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f64 = arr.iter().copied().sum();

        if tail > 0 {
            let offset = full_chunks * f64x8::LANE_COUNT;
            for j in 0..tail {
                result += self[offset + j];
            }
        }

        result
    }

    fn product(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x8::LANE_COUNT;
        let tail = n % f64x8::LANE_COUNT;

        let mut acc = unsafe { F64x8::broadcast(1.0) };

        for i in 0..full_chunks {
            let offset = i * f64x8::LANE_COUNT;
            let chunk = unsafe { F64x8::load(self.as_ptr().add(offset), f64x8::LANE_COUNT) };
            acc *= chunk;
        }

        let mut arr = [0.0f64; f64x8::LANE_COUNT];
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f64 = arr.iter().copied().product();

        if tail > 0 {
            let offset = full_chunks * f64x8::LANE_COUNT;
            for j in 0..tail {
                result *= self[offset + j];
            }
        }

        result
    }

    fn min(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x8::LANE_COUNT;
        let tail = n % f64x8::LANE_COUNT;

        let mut acc = unsafe { F64x8::broadcast(f64::INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f64x8::LANE_COUNT;
            let chunk = unsafe { F64x8::load(self.as_ptr().add(offset), f64x8::LANE_COUNT) };
            acc = F64x8 {
                size: f64x8::LANE_COUNT,
                elements: unsafe { _mm512_min_pd(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f64; f64x8::LANE_COUNT];
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f64::INFINITY, f64::min);

        if tail > 0 {
            let offset = full_chunks * f64x8::LANE_COUNT;
            for j in 0..tail {
                result = result.min(self[offset + j]);
            }
        }

        result
    }

    fn max(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x8::LANE_COUNT;
        let tail = n % f64x8::LANE_COUNT;

        let mut acc = unsafe { F64x8::broadcast(f64::NEG_INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f64x8::LANE_COUNT;
            let chunk = unsafe { F64x8::load(self.as_ptr().add(offset), f64x8::LANE_COUNT) };
            acc = F64x8 {
                size: f64x8::LANE_COUNT,
                elements: unsafe { _mm512_max_pd(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f64; f64x8::LANE_COUNT];
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if tail > 0 {
            let offset = full_chunks * f64x8::LANE_COUNT;
            for j in 0..tail {
                result = result.max(self[offset + j]);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests_f64 {
    use super::*;

    fn make_vecs(n: usize) -> (Vec<f64>, Vec<f64>) {
        let a: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.5 + 1.0).collect();
        (a, b)
    }

    const N_FULL: usize = 16;  // 2 full F64x8 chunks, no tail
    const N_TAIL: usize = 11;  // 1 full chunk + 3-lane tail

    #[test]
    fn add_produces_correct_result() {
        let (a, b) = make_vecs(N_FULL);
        let result = a.add(&b);
        let expected: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn add_with_tail_produces_correct_result() {
        let (a, b) = make_vecs(N_TAIL);
        let result = a.add(&b);
        let expected: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn sub_produces_correct_result() {
        let (a, b) = make_vecs(N_FULL);
        let result = a.sub(&b);
        let expected: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn mul_produces_correct_result() {
        let (a, b) = make_vecs(N_FULL);
        let result = a.mul(&b);
        let expected: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn div_produces_correct_result() {
        let (a, b) = make_vecs(N_FULL);
        let result = a.div(&b);
        let expected: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x / y).collect();
        for (r, e) in result.iter().zip(&expected) {
            assert!((r - e).abs() < 1e-10, "expected {e}, got {r}");
        }
    }

    #[test]
    fn rem_produces_correct_result() {
        let a: Vec<f64> = vec![7.0f64; N_FULL];
        let b: Vec<f64> = vec![3.0f64; N_FULL];
        let result = a.rem(&b);
        assert!(result.iter().all(|&x| (x - 1.0f64).abs() < 1e-10));
    }

    #[test]
    fn add_scalar_produces_correct_result() {
        let (a, _) = make_vecs(N_FULL);
        let result = a.add_scalar(10.0);
        let expected: Vec<f64> = a.iter().map(|x| x + 10.0).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn mul_scalar_produces_correct_result() {
        let (a, _) = make_vecs(N_FULL);
        let result = a.mul_scalar(2.0);
        let expected: Vec<f64> = a.iter().map(|x| x * 2.0).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn add_assign_modifies_in_place() {
        let (mut a, b) = make_vecs(N_FULL);
        let expected: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        a.add_assign(&b);
        assert_eq!(a, expected);
    }

    #[test]
    fn sum_produces_correct_result() {
        let (a, _) = make_vecs(N_FULL);
        let result = a.sum();
        let expected: f64 = a.iter().sum();
        assert!((result - expected).abs() < 1e-6, "expected {expected}, got {result}");
    }

    #[test]
    fn sum_with_tail_produces_correct_result() {
        let (a, _) = make_vecs(N_TAIL);
        let result = a.sum();
        let expected: f64 = a.iter().sum();
        assert!((result - expected).abs() < 1e-6, "expected {expected}, got {result}");
    }

    #[test]
    fn product_produces_correct_result() {
        let a: Vec<f64> = vec![2.0f64; 8]; // 2^8 = 256
        let result = a.product();
        assert!((result - 256.0f64).abs() < 1e-6);
    }

    #[test]
    fn min_produces_correct_result() {
        let (a, _) = make_vecs(N_FULL);
        let result = a.min();
        let expected = a.iter().copied().fold(f64::INFINITY, f64::min);
        assert_eq!(result, expected);
    }

    #[test]
    fn max_produces_correct_result() {
        let (a, _) = make_vecs(N_FULL);
        let result = a.max();
        let expected = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(result, expected);
    }

    #[test]
    fn min_with_tail_produces_correct_result() {
        let (a, _) = make_vecs(N_TAIL);
        let result = a.min();
        let expected = a.iter().copied().fold(f64::INFINITY, f64::min);
        assert_eq!(result, expected);
    }

    #[test]
    fn max_with_tail_produces_correct_result() {
        let (a, _) = make_vecs(N_TAIL);
        let result = a.max();
        let expected = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(result, expected);
    }
}
