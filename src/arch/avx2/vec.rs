//! AVX2 implementation of [`VecExt`] for `Vec<f32>` and `Vec<f64>`.
//!
//! The generic loop helpers (`binary_op`, `scalar_op`, etc.) live in
//! [`crate::ops::vec`] and are reused here unchanged. Only the horizontal
//! reductions and the `impl VecExt<T>` blocks are architecture-specific.

use std::arch::x86_64::{_mm256_max_pd, _mm256_max_ps, _mm256_min_pd, _mm256_min_ps};

use crate::arch::avx2::{f32x8, f32x8::F32x8};
use crate::arch::avx2::{f64x4, f64x4::F64x4};
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
        binary_op::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a + b)
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
        binary_op::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a - b)
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
        binary_op::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a * b)
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
        binary_op::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a / b)
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
        binary_op::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a % b)
    }

    #[inline]
    fn add_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a + b)
    }

    #[inline]
    fn sub_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a - b)
    }

    #[inline]
    fn mul_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a * b)
    }

    #[inline]
    fn div_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a / b)
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
        binary_op_inplace::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a + b);
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
        binary_op_inplace::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a - b);
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
        binary_op_inplace::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a * b);
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
        binary_op_inplace::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a / b);
    }

    #[inline]
    fn add_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a + b);
    }

    #[inline]
    fn sub_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a - b);
    }

    #[inline]
    fn mul_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a * b);
    }

    #[inline]
    fn div_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x8>(self, rhs, f32x8::LANE_COUNT, |a, b| a / b);
    }

    fn sum(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x8::LANE_COUNT;
        let tail = n % f32x8::LANE_COUNT;

        // SAFETY: zero() is always safe.
        let mut acc = unsafe { F32x8::zero() };

        for i in 0..full_chunks {
            let offset = i * f32x8::LANE_COUNT;
            // SAFETY: offset + f32x8::LANE_COUNT <= n.
            let chunk = unsafe { F32x8::load(self.as_ptr().add(offset), f32x8::LANE_COUNT) };
            acc += chunk;
        }

        let mut arr = [0.0f32; f32x8::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x8::LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f32 = arr.iter().copied().sum();

        if tail > 0 {
            let offset = full_chunks * f32x8::LANE_COUNT;
            for j in 0..tail {
                result += self[offset + j];
            }
        }

        result
    }

    fn product(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x8::LANE_COUNT;
        let tail = n % f32x8::LANE_COUNT;

        let mut acc = unsafe { F32x8::broadcast(1.0) };

        for i in 0..full_chunks {
            let offset = i * f32x8::LANE_COUNT;
            // SAFETY: offset + f32x8::LANE_COUNT <= n.
            let chunk = unsafe { F32x8::load(self.as_ptr().add(offset), f32x8::LANE_COUNT) };
            acc *= chunk;
        }

        let mut arr = [0.0f32; f32x8::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x8::LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f32 = arr.iter().copied().product();

        if tail > 0 {
            let offset = full_chunks * f32x8::LANE_COUNT;
            for j in 0..tail {
                result *= self[offset + j];
            }
        }

        result
    }

    fn min(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x8::LANE_COUNT;
        let tail = n % f32x8::LANE_COUNT;

        let mut acc = unsafe { F32x8::broadcast(f32::INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f32x8::LANE_COUNT;
            // SAFETY: offset + f32x8::LANE_COUNT <= n.
            let chunk = unsafe { F32x8::load(self.as_ptr().add(offset), f32x8::LANE_COUNT) };
            // SAFETY: _mm256_min_ps is always safe for valid __m256 operands.
            acc = F32x8 {
                size: f32x8::LANE_COUNT,
                elements: unsafe { _mm256_min_ps(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f32; f32x8::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x8::LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f32::INFINITY, f32::min);

        if tail > 0 {
            let offset = full_chunks * f32x8::LANE_COUNT;
            for j in 0..tail {
                result = result.min(self[offset + j]);
            }
        }

        result
    }

    fn max(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x8::LANE_COUNT;
        let tail = n % f32x8::LANE_COUNT;

        let mut acc = unsafe { F32x8::broadcast(f32::NEG_INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f32x8::LANE_COUNT;
            // SAFETY: offset + f32x8::LANE_COUNT <= n.
            let chunk = unsafe { F32x8::load(self.as_ptr().add(offset), f32x8::LANE_COUNT) };
            // SAFETY: _mm256_max_ps is always safe for valid __m256 operands.
            acc = F32x8 {
                size: f32x8::LANE_COUNT,
                elements: unsafe { _mm256_max_ps(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f32; f32x8::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x8::LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if tail > 0 {
            let offset = full_chunks * f32x8::LANE_COUNT;
            for j in 0..tail {
                result = result.max(self[offset + j]);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// VecExt<f64> for Vec<f64>
// ---------------------------------------------------------------------------

impl VecExt<f64> for Vec<f64> {
    #[inline]
    fn add(&self, rhs: &Self) -> Vec<f64> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a + b)
    }

    #[inline]
    fn sub(&self, rhs: &Self) -> Vec<f64> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a - b)
    }

    #[inline]
    fn mul(&self, rhs: &Self) -> Vec<f64> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a * b)
    }

    #[inline]
    fn div(&self, rhs: &Self) -> Vec<f64> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a / b)
    }

    #[inline]
    fn rem(&self, rhs: &Self) -> Vec<f64> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a % b)
    }

    #[inline]
    fn add_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a + b)
    }

    #[inline]
    fn sub_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a - b)
    }

    #[inline]
    fn mul_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a * b)
    }

    #[inline]
    fn div_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a / b)
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
        binary_op_inplace::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a + b);
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
        binary_op_inplace::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a - b);
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
        binary_op_inplace::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a * b);
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
        binary_op_inplace::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a / b);
    }

    #[inline]
    fn add_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a + b);
    }

    #[inline]
    fn sub_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a - b);
    }

    #[inline]
    fn mul_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a * b);
    }

    #[inline]
    fn div_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x4>(self, rhs, f64x4::LANE_COUNT, |a, b| a / b);
    }

    fn sum(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x4::LANE_COUNT;
        let tail = n % f64x4::LANE_COUNT;

        // SAFETY: zero() is always safe.
        let mut acc = unsafe { F64x4::zero() };

        for i in 0..full_chunks {
            let offset = i * f64x4::LANE_COUNT;
            // SAFETY: offset + f64x4::LANE_COUNT <= n.
            let chunk = unsafe { F64x4::load(self.as_ptr().add(offset), f64x4::LANE_COUNT) };
            acc += chunk;
        }

        let mut arr = [0.0f64; f64x4::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x4::LANE_COUNT f64 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f64 = arr.iter().copied().sum();

        if tail > 0 {
            let offset = full_chunks * f64x4::LANE_COUNT;
            for j in 0..tail {
                result += self[offset + j];
            }
        }

        result
    }

    fn product(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x4::LANE_COUNT;
        let tail = n % f64x4::LANE_COUNT;

        let mut acc = unsafe { F64x4::broadcast(1.0) };

        for i in 0..full_chunks {
            let offset = i * f64x4::LANE_COUNT;
            // SAFETY: offset + f64x4::LANE_COUNT <= n.
            let chunk = unsafe { F64x4::load(self.as_ptr().add(offset), f64x4::LANE_COUNT) };
            acc *= chunk;
        }

        let mut arr = [0.0f64; f64x4::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x4::LANE_COUNT f64 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f64 = arr.iter().copied().product();

        if tail > 0 {
            let offset = full_chunks * f64x4::LANE_COUNT;
            for j in 0..tail {
                result *= self[offset + j];
            }
        }

        result
    }

    fn min(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x4::LANE_COUNT;
        let tail = n % f64x4::LANE_COUNT;

        let mut acc = unsafe { F64x4::broadcast(f64::INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f64x4::LANE_COUNT;
            // SAFETY: offset + f64x4::LANE_COUNT <= n.
            let chunk = unsafe { F64x4::load(self.as_ptr().add(offset), f64x4::LANE_COUNT) };
            // SAFETY: _mm256_min_pd is always safe for valid __m256d operands.
            acc = F64x4 {
                size: f64x4::LANE_COUNT,
                elements: unsafe { _mm256_min_pd(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f64; f64x4::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x4::LANE_COUNT f64 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f64::INFINITY, f64::min);

        if tail > 0 {
            let offset = full_chunks * f64x4::LANE_COUNT;
            for j in 0..tail {
                result = result.min(self[offset + j]);
            }
        }

        result
    }

    fn max(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x4::LANE_COUNT;
        let tail = n % f64x4::LANE_COUNT;

        let mut acc = unsafe { F64x4::broadcast(f64::NEG_INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f64x4::LANE_COUNT;
            // SAFETY: offset + f64x4::LANE_COUNT <= n.
            let chunk = unsafe { F64x4::load(self.as_ptr().add(offset), f64x4::LANE_COUNT) };
            // SAFETY: _mm256_max_pd is always safe for valid __m256d operands.
            acc = F64x4 {
                size: f64x4::LANE_COUNT,
                elements: unsafe { _mm256_max_pd(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f64; f64x4::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x4::LANE_COUNT f64 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if tail > 0 {
            let offset = full_chunks * f64x4::LANE_COUNT;
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

// ---------------------------------------------------------------------------
// Unit tests — Vec<f64>
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests_f64 {
    use crate::ops::vec::VecExt;

    // Vec×Vec — length 7 exercises both 4-lane full chunks and a 3-element tail.

    #[test]
    fn add_produces_correct_sum() {
        let a: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let b: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let expected: Vec<f64> = (1..=7).map(|x| (x * 2) as f64).collect();
        assert_eq!(a.add(&b), expected);
    }

    #[test]
    fn sub_produces_correct_difference() {
        let a: Vec<f64> = (1..=7).map(|x| (x * 2) as f64).collect();
        let b: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let expected: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        assert_eq!(a.sub(&b), expected);
    }

    #[test]
    fn mul_produces_correct_product() {
        let a: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let b = vec![2.0f64; 7];
        let expected: Vec<f64> = (1..=7).map(|x| (x * 2) as f64).collect();
        assert_eq!(a.mul(&b), expected);
    }

    #[test]
    fn div_produces_correct_quotient() {
        let a: Vec<f64> = (1..=7).map(|x| (x * 2) as f64).collect();
        let b = vec![2.0f64; 7];
        let expected: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        assert_eq!(a.div(&b), expected);
    }

    #[test]
    fn rem_produces_correct_remainder() {
        // -7 % 3 == -1 (truncation semantics, same as Rust scalar `%`).
        let a = vec![7.0f64, 7.0, 7.0, 7.0, -7.0, -7.0, 7.0];
        let b = vec![3.0f64; 7];
        let expected = vec![1.0f64, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
        assert_eq!(a.rem(&b), expected);
    }

    // Vec×scalar

    #[test]
    fn add_scalar_broadcasts_correctly() {
        let a: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let expected: Vec<f64> = (11..=17).map(|x| x as f64).collect();
        assert_eq!(a.add_scalar(10.0), expected);
    }

    #[test]
    fn mul_scalar_broadcasts_correctly() {
        let a: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let expected: Vec<f64> = (1..=7).map(|x| (x * 2) as f64).collect();
        assert_eq!(a.mul_scalar(2.0), expected);
    }

    // In-place

    #[test]
    fn add_assign_matches_add() {
        let a: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let b = vec![1.0f64; 7];
        let expected = a.add(&b);
        let mut a_mut = a;
        a_mut.add_assign(&b);
        assert_eq!(a_mut, expected);
    }

    #[test]
    fn mul_scalar_assign_matches_mul_scalar() {
        let a: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let expected = a.mul_scalar(3.0);
        let mut a_mut = a;
        a_mut.mul_scalar_assign(3.0);
        assert_eq!(a_mut, expected);
    }

    // Reductions

    #[test]
    fn sum_is_correct() {
        // 1 + 2 + 3 + 4 = 10
        let a: Vec<f64> = (1..=4).map(|x| x as f64).collect();
        assert_eq!(a.sum(), 10.0);
    }

    #[test]
    fn product_is_correct() {
        // 1 * 2 * 3 * 4 = 24
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        assert_eq!(a.product(), 24.0);
    }

    #[test]
    fn min_finds_minimum() {
        let a = vec![5.0f64, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0];
        assert_eq!(a.min(), 1.0);
    }

    #[test]
    fn max_finds_maximum() {
        let a = vec![5.0f64, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0];
        assert_eq!(a.max(), 9.0);
    }

    // Edge cases

    #[test]
    fn empty_vec_sum_is_zero() {
        let a: Vec<f64> = vec![];
        assert_eq!(a.sum(), 0.0);
    }

    #[test]
    fn empty_vec_product_is_one() {
        let a: Vec<f64> = vec![];
        assert_eq!(a.product(), 1.0);
    }

    #[test]
    fn single_element_add() {
        let a = vec![5.0f64];
        let b = vec![3.0f64];
        assert_eq!(a.add(&b), vec![8.0f64]);
    }

    #[test]
    #[should_panic]
    fn length_mismatch_panics() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![1.0f64, 2.0];
        let _ = a.add(&b);
    }
}
