//! AVX-512F implementation of [`SliceExt`] and [`VecExt`] for `f32` / `f64`.
//!
//! The generic loop helpers (`binary_op`, `scalar_op`, etc.) live in
//! [`crate::ops::vec`] and are reused here unchanged. Only the horizontal
//! reductions and the `impl SliceExt<T>` blocks are architecture-specific;
//! the `Vec<T>` impls are thin forwarders to the slice impls.

use std::arch::x86_64::{_mm512_max_pd, _mm512_max_ps, _mm512_min_pd, _mm512_min_ps};

use crate::arch::avx512::{f32x16, f32x16::F32x16};
use crate::arch::avx512::{f64x8, f64x8::F64x8};
use crate::ops::simd::{Load, Store};
use crate::ops::vec::{
    SliceExt, VecExt, binary_op, binary_op_inplace, scalar_op, scalar_op_inplace,
};

// ---------------------------------------------------------------------------
// SliceExt<f32> for [f32]
// ---------------------------------------------------------------------------

impl SliceExt<f32> for [f32] {
    #[inline]
    fn add(&self, rhs: &[f32]) -> Vec<f32> {
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
    fn sub(&self, rhs: &[f32]) -> Vec<f32> {
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
    fn mul(&self, rhs: &[f32]) -> Vec<f32> {
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
    fn div(&self, rhs: &[f32]) -> Vec<f32> {
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
    fn rem(&self, rhs: &[f32]) -> Vec<f32> {
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
    fn add_assign(&mut self, rhs: &[f32]) {
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
    fn sub_assign(&mut self, rhs: &[f32]) {
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
    fn mul_assign(&mut self, rhs: &[f32]) {
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
    fn div_assign(&mut self, rhs: &[f32]) {
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

        // SAFETY: zero() is always safe.
        let mut acc = unsafe { F32x16::zero() };

        for i in 0..full_chunks {
            let offset = i * f32x16::LANE_COUNT;
            // SAFETY: offset + f32x16::LANE_COUNT <= n.
            let chunk = unsafe { F32x16::load(self.as_ptr().add(offset), f32x16::LANE_COUNT) };
            acc += chunk;
        }

        let mut arr = [0.0f32; f32x16::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x16::LANE_COUNT f32 writes.
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
            // SAFETY: offset + f32x16::LANE_COUNT <= n.
            let chunk = unsafe { F32x16::load(self.as_ptr().add(offset), f32x16::LANE_COUNT) };
            acc *= chunk;
        }

        let mut arr = [0.0f32; f32x16::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x16::LANE_COUNT f32 writes.
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
            // SAFETY: offset + f32x16::LANE_COUNT <= n.
            let chunk = unsafe { F32x16::load(self.as_ptr().add(offset), f32x16::LANE_COUNT) };
            // SAFETY: _mm512_min_ps is always safe for valid __m512 operands.
            acc = F32x16 {
                size: f32x16::LANE_COUNT,
                elements: unsafe { _mm512_min_ps(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f32; f32x16::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x16::LANE_COUNT f32 writes.
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
            // SAFETY: offset + f32x16::LANE_COUNT <= n.
            let chunk = unsafe { F32x16::load(self.as_ptr().add(offset), f32x16::LANE_COUNT) };
            // SAFETY: _mm512_max_ps is always safe for valid __m512 operands.
            acc = F32x16 {
                size: f32x16::LANE_COUNT,
                elements: unsafe { _mm512_max_ps(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f32; f32x16::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f32x16::LANE_COUNT f32 writes.
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

// ---------------------------------------------------------------------------
// SliceExt<f64> for [f64]
// ---------------------------------------------------------------------------

impl SliceExt<f64> for [f64] {
    #[inline]
    fn add(&self, rhs: &[f64]) -> Vec<f64> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a + b)
    }

    #[inline]
    fn sub(&self, rhs: &[f64]) -> Vec<f64> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a - b)
    }

    #[inline]
    fn mul(&self, rhs: &[f64]) -> Vec<f64> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a * b)
    }

    #[inline]
    fn div(&self, rhs: &[f64]) -> Vec<f64> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a / b)
    }

    #[inline]
    fn rem(&self, rhs: &[f64]) -> Vec<f64> {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
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
    fn add_assign(&mut self, rhs: &[f64]) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a + b);
    }

    #[inline]
    fn sub_assign(&mut self, rhs: &[f64]) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a - b);
    }

    #[inline]
    fn mul_assign(&mut self, rhs: &[f64]) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
        binary_op_inplace::<f64, F64x8>(self, rhs, f64x8::LANE_COUNT, |a, b| a * b);
    }

    #[inline]
    fn div_assign(&mut self, rhs: &[f64]) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            self.len(),
            rhs.len()
        );
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

        // SAFETY: zero() is always safe.
        let mut acc = unsafe { F64x8::zero() };

        for i in 0..full_chunks {
            let offset = i * f64x8::LANE_COUNT;
            // SAFETY: offset + f64x8::LANE_COUNT <= n.
            let chunk = unsafe { F64x8::load(self.as_ptr().add(offset), f64x8::LANE_COUNT) };
            acc += chunk;
        }

        let mut arr = [0.0f64; f64x8::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x8::LANE_COUNT f64 writes.
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
            // SAFETY: offset + f64x8::LANE_COUNT <= n.
            let chunk = unsafe { F64x8::load(self.as_ptr().add(offset), f64x8::LANE_COUNT) };
            acc *= chunk;
        }

        let mut arr = [0.0f64; f64x8::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x8::LANE_COUNT f64 writes.
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
            // SAFETY: offset + f64x8::LANE_COUNT <= n.
            let chunk = unsafe { F64x8::load(self.as_ptr().add(offset), f64x8::LANE_COUNT) };
            // SAFETY: _mm512_min_pd is always safe for valid __m512d operands.
            acc = F64x8 {
                size: f64x8::LANE_COUNT,
                elements: unsafe { _mm512_min_pd(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f64; f64x8::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x8::LANE_COUNT f64 writes.
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
            // SAFETY: offset + f64x8::LANE_COUNT <= n.
            let chunk = unsafe { F64x8::load(self.as_ptr().add(offset), f64x8::LANE_COUNT) };
            // SAFETY: _mm512_max_pd is always safe for valid __m512d operands.
            acc = F64x8 {
                size: f64x8::LANE_COUNT,
                elements: unsafe { _mm512_max_pd(acc.elements, chunk.elements) },
            };
        }

        let mut arr = [0.0f64; f64x8::LANE_COUNT];
        // SAFETY: `arr` is mut and valid for f64x8::LANE_COUNT f64 writes.
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

// ---------------------------------------------------------------------------
// VecExt<T> for Vec<T> — thin delegators to the slice impls above.
// ---------------------------------------------------------------------------

macro_rules! impl_vecext_delegate {
    ($t:ty) => {
        impl VecExt<$t> for Vec<$t> {
            #[inline]
            fn add(&self, rhs: &Self) -> Vec<$t> {
                SliceExt::add(self.as_slice(), rhs.as_slice())
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Vec<$t> {
                SliceExt::sub(self.as_slice(), rhs.as_slice())
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Vec<$t> {
                SliceExt::mul(self.as_slice(), rhs.as_slice())
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Vec<$t> {
                SliceExt::div(self.as_slice(), rhs.as_slice())
            }
            #[inline]
            fn rem(&self, rhs: &Self) -> Vec<$t> {
                SliceExt::rem(self.as_slice(), rhs.as_slice())
            }
            #[inline]
            fn add_scalar(&self, rhs: $t) -> Vec<$t> {
                SliceExt::add_scalar(self.as_slice(), rhs)
            }
            #[inline]
            fn sub_scalar(&self, rhs: $t) -> Vec<$t> {
                SliceExt::sub_scalar(self.as_slice(), rhs)
            }
            #[inline]
            fn mul_scalar(&self, rhs: $t) -> Vec<$t> {
                SliceExt::mul_scalar(self.as_slice(), rhs)
            }
            #[inline]
            fn div_scalar(&self, rhs: $t) -> Vec<$t> {
                SliceExt::div_scalar(self.as_slice(), rhs)
            }
            #[inline]
            fn add_assign(&mut self, rhs: &Self) {
                SliceExt::add_assign(self.as_mut_slice(), rhs.as_slice())
            }
            #[inline]
            fn sub_assign(&mut self, rhs: &Self) {
                SliceExt::sub_assign(self.as_mut_slice(), rhs.as_slice())
            }
            #[inline]
            fn mul_assign(&mut self, rhs: &Self) {
                SliceExt::mul_assign(self.as_mut_slice(), rhs.as_slice())
            }
            #[inline]
            fn div_assign(&mut self, rhs: &Self) {
                SliceExt::div_assign(self.as_mut_slice(), rhs.as_slice())
            }
            #[inline]
            fn add_scalar_assign(&mut self, rhs: $t) {
                SliceExt::add_scalar_assign(self.as_mut_slice(), rhs)
            }
            #[inline]
            fn sub_scalar_assign(&mut self, rhs: $t) {
                SliceExt::sub_scalar_assign(self.as_mut_slice(), rhs)
            }
            #[inline]
            fn mul_scalar_assign(&mut self, rhs: $t) {
                SliceExt::mul_scalar_assign(self.as_mut_slice(), rhs)
            }
            #[inline]
            fn div_scalar_assign(&mut self, rhs: $t) {
                SliceExt::div_scalar_assign(self.as_mut_slice(), rhs)
            }
            #[inline]
            fn sum(&self) -> $t {
                SliceExt::sum(self.as_slice())
            }
            #[inline]
            fn product(&self) -> $t {
                SliceExt::product(self.as_slice())
            }
            #[inline]
            fn min(&self) -> $t {
                SliceExt::min(self.as_slice())
            }
            #[inline]
            fn max(&self) -> $t {
                SliceExt::max(self.as_slice())
            }
        }
    };
}

impl_vecext_delegate!(f32);
impl_vecext_delegate!(f64);

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::ops::vec::{SliceExt, VecExt};
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
    fn slice_add_smoke_f32() {
        let a: &[f32] = &[1.0, 2.0, 3.0];
        let b: &[f32] = &[4.0, 5.0, 6.0];
        assert_eq!(SliceExt::add(a, b), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn slice_mul_smoke_f64() {
        let a: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let b: &[f64] = &[2.0; 5];
        assert_eq!(SliceExt::mul(a, b), vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn slice_sum_smoke_f32() {
        let a: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        assert_eq!(SliceExt::sum(a), 66.0);
    }

    #[test]
    fn slice_add_assign_smoke_f32() {
        let mut a = vec![1.0f32, 2.0, 3.0];
        let b: &[f32] = &[10.0, 20.0, 30.0];
        SliceExt::add_assign(a.as_mut_slice(), b);
        assert_eq!(a, vec![11.0, 22.0, 33.0]);
    }
}
