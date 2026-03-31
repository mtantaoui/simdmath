//! NEON implementation of [`VecExt`] for `Vec<f32>` and `Vec<f64>`.
//!
//! The generic loop helpers (`binary_op`, `scalar_op`, etc.) live in
//! [`crate::ops::vec`] and are reused here unchanged. Only the horizontal
//! reductions and the `impl VecExt<T>` blocks are architecture-specific.

use std::arch::aarch64::{vmaxq_f32, vmaxq_f64, vminq_f32, vminq_f64};

use crate::arch::neon::{f32x4, f32x4::F32x4};
use crate::arch::neon::{f64x2, f64x2::F64x2};
use crate::ops::simd::{Load, Store};
use crate::ops::vec::{VecExt, binary_op, binary_op_inplace, scalar_op, scalar_op_inplace};

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
        binary_op::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a + b)
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
        binary_op::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a - b)
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
        binary_op::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a * b)
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
        binary_op::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a / b)
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
        binary_op::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a % b)
    }

    #[inline]
    fn add_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a + b)
    }

    #[inline]
    fn sub_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a - b)
    }

    #[inline]
    fn mul_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a * b)
    }

    #[inline]
    fn div_scalar(&self, rhs: f32) -> Vec<f32> {
        scalar_op::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a / b)
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
        binary_op_inplace::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a + b);
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
        binary_op_inplace::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a - b);
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
        binary_op_inplace::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a * b);
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
        binary_op_inplace::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a / b);
    }

    #[inline]
    fn add_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a + b);
    }

    #[inline]
    fn sub_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a - b);
    }

    #[inline]
    fn mul_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a * b);
    }

    #[inline]
    fn div_scalar_assign(&mut self, rhs: f32) {
        scalar_op_inplace::<f32, F32x4>(self, rhs, f32x4::LANE_COUNT, |a, b| a / b);
    }

    fn sum(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x4::LANE_COUNT;
        let tail = n % f32x4::LANE_COUNT;

        // SAFETY: zero() is always safe.
        let mut acc = unsafe { F32x4::zero() };

        for i in 0..full_chunks {
            let offset = i * f32x4::LANE_COUNT;
            // SAFETY: offset + f32x4::LANE_COUNT <= n.
            let chunk = unsafe { F32x4::load(self.as_ptr().add(offset), f32x4::LANE_COUNT) };
            acc += chunk;
        }

        let mut arr = [0.0f32; f32x4::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x4::LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f32 = arr.iter().copied().sum();

        if tail > 0 {
            let offset = full_chunks * f32x4::LANE_COUNT;
            for j in 0..tail {
                result += self[offset + j];
            }
        }

        result
    }

    fn product(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x4::LANE_COUNT;
        let tail = n % f32x4::LANE_COUNT;

        let mut acc = unsafe { F32x4::broadcast(1.0) };

        for i in 0..full_chunks {
            let offset = i * f32x4::LANE_COUNT;
            // SAFETY: offset + f32x4::LANE_COUNT <= n.
            let chunk = unsafe { F32x4::load(self.as_ptr().add(offset), f32x4::LANE_COUNT) };
            acc *= chunk;
        }

        let mut arr = [0.0f32; f32x4::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x4::LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f32 = arr.iter().copied().product();

        if tail > 0 {
            let offset = full_chunks * f32x4::LANE_COUNT;
            for j in 0..tail {
                result *= self[offset + j];
            }
        }

        result
    }

    fn min(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x4::LANE_COUNT;
        let tail = n % f32x4::LANE_COUNT;

        let mut acc = unsafe { F32x4::broadcast(f32::INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f32x4::LANE_COUNT;
            // SAFETY: offset + f32x4::LANE_COUNT <= n.
            let chunk = unsafe { F32x4::load(self.as_ptr().add(offset), f32x4::LANE_COUNT) };
            // SAFETY: vminq_f32 is always safe for valid float32x4_t operands.
            acc = F32x4 {
                size: f32x4::LANE_COUNT,
                elements: unsafe { vminq_f32(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f32; f32x4::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x4::LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f32::INFINITY, f32::min);

        if tail > 0 {
            let offset = full_chunks * f32x4::LANE_COUNT;
            for j in 0..tail {
                result = result.min(self[offset + j]);
            }
        }

        result
    }

    fn max(&self) -> f32 {
        let n = self.len();
        let full_chunks = n / f32x4::LANE_COUNT;
        let tail = n % f32x4::LANE_COUNT;

        let mut acc = unsafe { F32x4::broadcast(f32::NEG_INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f32x4::LANE_COUNT;
            // SAFETY: offset + f32x4::LANE_COUNT <= n.
            let chunk = unsafe { F32x4::load(self.as_ptr().add(offset), f32x4::LANE_COUNT) };
            // SAFETY: vmaxq_f32 is always safe for valid float32x4_t operands.
            acc = F32x4 {
                size: f32x4::LANE_COUNT,
                elements: unsafe { vmaxq_f32(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f32; f32x4::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x4::LANE_COUNT f32 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if tail > 0 {
            let offset = full_chunks * f32x4::LANE_COUNT;
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
        binary_op::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a + b)
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
        binary_op::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a - b)
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
        binary_op::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a * b)
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
        binary_op::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a / b)
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
        binary_op::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a % b)
    }

    #[inline]
    fn add_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a + b)
    }

    #[inline]
    fn sub_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a - b)
    }

    #[inline]
    fn mul_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a * b)
    }

    #[inline]
    fn div_scalar(&self, rhs: f64) -> Vec<f64> {
        scalar_op::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a / b)
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
        binary_op_inplace::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a + b);
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
        binary_op_inplace::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a - b);
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
        binary_op_inplace::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a * b);
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
        binary_op_inplace::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a / b);
    }

    #[inline]
    fn add_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a + b);
    }

    #[inline]
    fn sub_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a - b);
    }

    #[inline]
    fn mul_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a * b);
    }

    #[inline]
    fn div_scalar_assign(&mut self, rhs: f64) {
        scalar_op_inplace::<f64, F64x2>(self, rhs, f64x2::LANE_COUNT, |a, b| a / b);
    }

    fn sum(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x2::LANE_COUNT;
        let tail = n % f64x2::LANE_COUNT;

        // SAFETY: zero() is always safe.
        let mut acc = unsafe { F64x2::zero() };

        for i in 0..full_chunks {
            let offset = i * f64x2::LANE_COUNT;
            // SAFETY: offset + f64x2::LANE_COUNT <= n.
            let chunk = unsafe { F64x2::load(self.as_ptr().add(offset), f64x2::LANE_COUNT) };
            acc += chunk;
        }

        let mut arr = [0.0f64; f64x2::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x2::LANE_COUNT f64 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f64 = arr.iter().copied().sum();

        if tail > 0 {
            let offset = full_chunks * f64x2::LANE_COUNT;
            for j in 0..tail {
                result += self[offset + j];
            }
        }

        result
    }

    fn product(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x2::LANE_COUNT;
        let tail = n % f64x2::LANE_COUNT;

        let mut acc = unsafe { F64x2::broadcast(1.0) };

        for i in 0..full_chunks {
            let offset = i * f64x2::LANE_COUNT;
            // SAFETY: offset + f64x2::LANE_COUNT <= n.
            let chunk = unsafe { F64x2::load(self.as_ptr().add(offset), f64x2::LANE_COUNT) };
            acc *= chunk;
        }

        let mut arr = [0.0f64; f64x2::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x2::LANE_COUNT f64 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result: f64 = arr.iter().copied().product();

        if tail > 0 {
            let offset = full_chunks * f64x2::LANE_COUNT;
            for j in 0..tail {
                result *= self[offset + j];
            }
        }

        result
    }

    fn min(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x2::LANE_COUNT;
        let tail = n % f64x2::LANE_COUNT;

        let mut acc = unsafe { F64x2::broadcast(f64::INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f64x2::LANE_COUNT;
            // SAFETY: offset + f64x2::LANE_COUNT <= n.
            let chunk = unsafe { F64x2::load(self.as_ptr().add(offset), f64x2::LANE_COUNT) };
            // SAFETY: vminq_f64 is always safe for valid float64x2_t operands.
            acc = F64x2 {
                size: f64x2::LANE_COUNT,
                elements: unsafe { vminq_f64(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f64; f64x2::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x2::LANE_COUNT f64 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f64::INFINITY, f64::min);

        if tail > 0 {
            let offset = full_chunks * f64x2::LANE_COUNT;
            for j in 0..tail {
                result = result.min(self[offset + j]);
            }
        }

        result
    }

    fn max(&self) -> f64 {
        let n = self.len();
        let full_chunks = n / f64x2::LANE_COUNT;
        let tail = n % f64x2::LANE_COUNT;

        let mut acc = unsafe { F64x2::broadcast(f64::NEG_INFINITY) };

        for i in 0..full_chunks {
            let offset = i * f64x2::LANE_COUNT;
            // SAFETY: offset + f64x2::LANE_COUNT <= n.
            let chunk = unsafe { F64x2::load(self.as_ptr().add(offset), f64x2::LANE_COUNT) };
            // SAFETY: vmaxq_f64 is always safe for valid float64x2_t operands.
            acc = F64x2 {
                size: f64x2::LANE_COUNT,
                elements: unsafe { vmaxq_f64(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f64; f64x2::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x2::LANE_COUNT f64 writes.
        unsafe { acc.store_unaligned_at(arr.as_mut_ptr()) };
        let mut result = arr.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if tail > 0 {
            let offset = full_chunks * f64x2::LANE_COUNT;
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
    use super::*;

    // ---- add ----------------------------------------------------------------

    #[test]
    fn add_f32_elementwise() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![4.0f32, 3.0, 2.0, 1.0];
        assert_eq!(a.add(&b), vec![5.0f32; 4]);
    }

    #[test]
    fn add_f64_elementwise() {
        let a = vec![1.0f64, 2.0];
        let b = vec![2.0f64, 1.0];
        assert_eq!(a.add(&b), vec![3.0f64; 2]);
    }

    // ---- sub ----------------------------------------------------------------

    #[test]
    fn sub_f32_elementwise() {
        let a = vec![5.0f32; 4];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(a.sub(&b), vec![4.0f32, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn sub_f64_elementwise() {
        let a = vec![5.0f64; 2];
        let b = vec![1.0f64, 2.0];
        assert_eq!(a.sub(&b), vec![4.0f64, 3.0]);
    }

    // ---- mul ----------------------------------------------------------------

    #[test]
    fn mul_f32_elementwise() {
        let a = vec![2.0f32; 4];
        let b = vec![3.0f32; 4];
        assert_eq!(a.mul(&b), vec![6.0f32; 4]);
    }

    #[test]
    fn mul_f64_elementwise() {
        let a = vec![2.0f64; 2];
        let b = vec![3.0f64; 2];
        assert_eq!(a.mul(&b), vec![6.0f64; 2]);
    }

    // ---- div ----------------------------------------------------------------

    #[test]
    fn div_f32_elementwise() {
        let a = vec![6.0f32; 4];
        let b = vec![2.0f32; 4];
        assert_eq!(a.div(&b), vec![3.0f32; 4]);
    }

    #[test]
    fn div_f64_elementwise() {
        let a = vec![6.0f64; 2];
        let b = vec![2.0f64; 2];
        assert_eq!(a.div(&b), vec![3.0f64; 2]);
    }

    // ---- scalar ops ---------------------------------------------------------

    #[test]
    fn add_scalar_f32() {
        let a = vec![1.0f32; 4];
        assert_eq!(a.add_scalar(2.0), vec![3.0f32; 4]);
    }

    #[test]
    fn mul_scalar_f64() {
        let a = vec![2.0f64; 2];
        assert_eq!(a.mul_scalar(3.0), vec![6.0f64; 2]);
    }

    // ---- reductions ---------------------------------------------------------

    #[test]
    fn sum_f32() {
        let a = vec![1.0f32; 8];
        assert_eq!(a.sum(), 8.0);
    }

    #[test]
    fn sum_f64() {
        let a = vec![1.0f64; 4];
        assert_eq!(a.sum(), 4.0);
    }

    #[test]
    fn product_f32() {
        let a = vec![2.0f32; 4];
        assert_eq!(a.product(), 16.0);
    }

    #[test]
    fn product_f64() {
        let a = vec![2.0f64; 2];
        assert_eq!(a.product(), 4.0);
    }

    #[test]
    fn min_f32() {
        let a = vec![3.0f32, 1.0, 4.0, 2.0];
        assert_eq!(a.min(), 1.0);
    }

    #[test]
    fn min_f64() {
        let a = vec![3.0f64, 1.0];
        assert_eq!(a.min(), 1.0);
    }

    #[test]
    fn max_f32() {
        let a = vec![3.0f32, 1.0, 4.0, 2.0];
        assert_eq!(a.max(), 4.0);
    }

    #[test]
    fn max_f64() {
        let a = vec![3.0f64, 1.0];
        assert_eq!(a.max(), 3.0);
    }

    // ---- tail handling ------------------------------------------------------

    #[test]
    fn add_f32_with_tail() {
        // 5 elements: 1 full F32x4 chunk + 1-lane tail
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(a.add(&b), vec![6.0f32; 5]);
    }

    #[test]
    fn sum_f32_with_tail() {
        // 7 elements: 1 full F32x4 chunk + 3-lane tail
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(a.sum(), 28.0);
    }
}
