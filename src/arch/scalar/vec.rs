//! Scalar implementation of [`SliceExt`] / [`VecExt`] — used when no SIMD ISA
//! is enabled.
//!
//! Each method is a straight iterator chain over the slice. The compiler may
//! auto-vectorise these loops, but no hand-tuned SIMD is performed.

use crate::ops::vec::{SliceExt, VecExt};

macro_rules! assert_same_len {
    ($lhs:expr, $rhs:expr) => {
        assert_eq!(
            $lhs.len(),
            $rhs.len(),
            "length mismatch: lhs has {} elements, rhs has {} elements",
            $lhs.len(),
            $rhs.len()
        )
    };
}

macro_rules! impl_vecext_scalar {
    ($t:ty, $zero:expr, $one:expr, $pos_inf:expr, $neg_inf:expr) => {
        impl SliceExt<$t> for [$t] {
            #[inline]
            fn add(&self, rhs: &[$t]) -> Vec<$t> {
                assert_same_len!(self, rhs);
                self.iter().zip(rhs).map(|(a, b)| *a + *b).collect()
            }
            #[inline]
            fn sub(&self, rhs: &[$t]) -> Vec<$t> {
                assert_same_len!(self, rhs);
                self.iter().zip(rhs).map(|(a, b)| *a - *b).collect()
            }
            #[inline]
            fn mul(&self, rhs: &[$t]) -> Vec<$t> {
                assert_same_len!(self, rhs);
                self.iter().zip(rhs).map(|(a, b)| *a * *b).collect()
            }
            #[inline]
            fn div(&self, rhs: &[$t]) -> Vec<$t> {
                assert_same_len!(self, rhs);
                self.iter().zip(rhs).map(|(a, b)| *a / *b).collect()
            }
            #[inline]
            fn rem(&self, rhs: &[$t]) -> Vec<$t> {
                assert_same_len!(self, rhs);
                self.iter().zip(rhs).map(|(a, b)| *a % *b).collect()
            }

            #[inline]
            fn add_scalar(&self, rhs: $t) -> Vec<$t> {
                self.iter().map(|a| *a + rhs).collect()
            }
            #[inline]
            fn sub_scalar(&self, rhs: $t) -> Vec<$t> {
                self.iter().map(|a| *a - rhs).collect()
            }
            #[inline]
            fn mul_scalar(&self, rhs: $t) -> Vec<$t> {
                self.iter().map(|a| *a * rhs).collect()
            }
            #[inline]
            fn div_scalar(&self, rhs: $t) -> Vec<$t> {
                self.iter().map(|a| *a / rhs).collect()
            }

            #[inline]
            fn add_assign(&mut self, rhs: &[$t]) {
                assert_same_len!(self, rhs);
                for (a, b) in self.iter_mut().zip(rhs) {
                    *a += *b;
                }
            }
            #[inline]
            fn sub_assign(&mut self, rhs: &[$t]) {
                assert_same_len!(self, rhs);
                for (a, b) in self.iter_mut().zip(rhs) {
                    *a -= *b;
                }
            }
            #[inline]
            fn mul_assign(&mut self, rhs: &[$t]) {
                assert_same_len!(self, rhs);
                for (a, b) in self.iter_mut().zip(rhs) {
                    *a *= *b;
                }
            }
            #[inline]
            fn div_assign(&mut self, rhs: &[$t]) {
                assert_same_len!(self, rhs);
                for (a, b) in self.iter_mut().zip(rhs) {
                    *a /= *b;
                }
            }

            #[inline]
            fn add_scalar_assign(&mut self, rhs: $t) {
                for a in self.iter_mut() {
                    *a += rhs;
                }
            }
            #[inline]
            fn sub_scalar_assign(&mut self, rhs: $t) {
                for a in self.iter_mut() {
                    *a -= rhs;
                }
            }
            #[inline]
            fn mul_scalar_assign(&mut self, rhs: $t) {
                for a in self.iter_mut() {
                    *a *= rhs;
                }
            }
            #[inline]
            fn div_scalar_assign(&mut self, rhs: $t) {
                for a in self.iter_mut() {
                    *a /= rhs;
                }
            }

            #[inline]
            fn sum(&self) -> $t {
                self.iter().copied().fold($zero, |acc, x| acc + x)
            }
            #[inline]
            fn product(&self) -> $t {
                self.iter().copied().fold($one, |acc, x| acc * x)
            }
            #[inline]
            fn min(&self) -> $t {
                self.iter()
                    .copied()
                    .fold($pos_inf, |acc, x| if x < acc { x } else { acc })
            }
            #[inline]
            fn max(&self) -> $t {
                self.iter()
                    .copied()
                    .fold($neg_inf, |acc, x| if x > acc { x } else { acc })
            }
        }

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

impl_vecext_scalar!(f32, 0.0_f32, 1.0_f32, f32::INFINITY, f32::NEG_INFINITY);
impl_vecext_scalar!(f64, 0.0_f64, 1.0_f64, f64::INFINITY, f64::NEG_INFINITY);
