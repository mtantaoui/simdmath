//! AVX-512F `[T]` and `Vec<T>` implementations of [`SliceMath`] / [`VecMath`].

use crate::arch::avx512::{f32x16, f32x16::F32x16};
use crate::arch::avx512::{f64x8, f64x8::F64x8};
use crate::math::{SliceMath, VecMath};
use crate::ops::vec::{binary_op, unary_op};

impl SliceMath<f32> for [f32] {
    /// Absolute value, processed 16 lanes at a time via AVX-512.
    #[inline]
    fn abs(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.abs())
    }
    /// Arc cosine, processed 16 lanes at a time via AVX-512.
    #[inline]
    fn acos(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.acos())
    }
    /// Arc sine, processed 16 lanes at a time via AVX-512.
    #[inline]
    fn asin(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.asin())
    }
    /// Arc tangent, processed 16 lanes at a time via AVX-512.
    #[inline]
    fn atan(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.atan())
    }
    /// Two-arg arc tangent per lane, AVX-512.
    #[inline]
    fn atan2(&self, other: &[f32]) -> Vec<f32> {
        binary_op::<f32, F32x16>(self, other, f32x16::LANE_COUNT, |y, x| y.atan2(&x))
    }
    /// Cube root, processed 16 lanes at a time via AVX-512.
    #[inline]
    fn cbrt(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.cbrt())
    }
    /// Cosine (radians), processed 16 lanes at a time via AVX-512.
    #[inline]
    fn cos(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.cos())
    }
    /// Exponential, processed 16 lanes at a time via AVX-512.
    #[inline]
    fn exp(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.exp())
    }
    /// Natural log, processed 16 lanes at a time via AVX-512.
    #[inline]
    fn ln(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.ln())
    }
    /// Sine (radians), processed 16 lanes at a time via AVX-512.
    #[inline]
    fn sin(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.sin())
    }
    /// Tangent (radians), processed 16 lanes at a time via AVX-512.
    #[inline]
    fn tan(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.tan())
    }
    /// `self^exp` per lane, processed 16 lanes at a time via AVX-512.
    #[inline]
    fn pow(&self, exp: &[f32]) -> Vec<f32> {
        binary_op::<f32, F32x16>(self, exp, f32x16::LANE_COUNT, |b, e| b.pow(&e))
    }
    /// Square root, processed 16 lanes at a time via AVX-512.
    #[inline]
    fn sqrt(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.sqrt())
    }
}

impl SliceMath<f64> for [f64] {
    /// Absolute value, processed 8 lanes at a time via AVX-512.
    #[inline]
    fn abs(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.abs())
    }
    /// Arc cosine, processed 8 lanes at a time via AVX-512.
    #[inline]
    fn acos(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.acos())
    }
    /// Arc sine, processed 8 lanes at a time via AVX-512.
    #[inline]
    fn asin(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.asin())
    }
    /// Arc tangent, processed 8 lanes at a time via AVX-512.
    #[inline]
    fn atan(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.atan())
    }
    /// Two-arg arc tangent per lane, AVX-512.
    #[inline]
    fn atan2(&self, other: &[f64]) -> Vec<f64> {
        binary_op::<f64, F64x8>(self, other, f64x8::LANE_COUNT, |y, x| y.atan2(&x))
    }
    /// Cube root, processed 8 lanes at a time via AVX-512.
    #[inline]
    fn cbrt(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.cbrt())
    }
    /// Cosine (radians), processed 8 lanes at a time via AVX-512.
    #[inline]
    fn cos(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.cos())
    }
    /// Exponential, processed 8 lanes at a time via AVX-512.
    #[inline]
    fn exp(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.exp())
    }
    /// Natural log, processed 8 lanes at a time via AVX-512.
    #[inline]
    fn ln(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.ln())
    }
    /// Sine (radians), processed 8 lanes at a time via AVX-512.
    #[inline]
    fn sin(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.sin())
    }
    /// Tangent (radians), processed 8 lanes at a time via AVX-512.
    #[inline]
    fn tan(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.tan())
    }
    /// `self^exp` per lane, processed 8 lanes at a time via AVX-512.
    #[inline]
    fn pow(&self, exp: &[f64]) -> Vec<f64> {
        binary_op::<f64, F64x8>(self, exp, f64x8::LANE_COUNT, |b, e| b.pow(&e))
    }
    /// Square root, processed 8 lanes at a time via AVX-512.
    #[inline]
    fn sqrt(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.sqrt())
    }
}

macro_rules! impl_vecmath_delegate {
    ($t:ty) => {
        impl VecMath<$t> for Vec<$t> {
            #[inline]
            fn abs(&self) -> Vec<$t> {
                SliceMath::abs(self.as_slice())
            }
            #[inline]
            fn acos(&self) -> Vec<$t> {
                SliceMath::acos(self.as_slice())
            }
            #[inline]
            fn asin(&self) -> Vec<$t> {
                SliceMath::asin(self.as_slice())
            }
            #[inline]
            fn atan(&self) -> Vec<$t> {
                SliceMath::atan(self.as_slice())
            }
            #[inline]
            fn atan2(&self, other: &Self) -> Vec<$t> {
                SliceMath::atan2(self.as_slice(), other.as_slice())
            }
            #[inline]
            fn cbrt(&self) -> Vec<$t> {
                SliceMath::cbrt(self.as_slice())
            }
            #[inline]
            fn cos(&self) -> Vec<$t> {
                SliceMath::cos(self.as_slice())
            }
            #[inline]
            fn exp(&self) -> Vec<$t> {
                SliceMath::exp(self.as_slice())
            }
            #[inline]
            fn ln(&self) -> Vec<$t> {
                SliceMath::ln(self.as_slice())
            }
            #[inline]
            fn sin(&self) -> Vec<$t> {
                SliceMath::sin(self.as_slice())
            }
            #[inline]
            fn tan(&self) -> Vec<$t> {
                SliceMath::tan(self.as_slice())
            }
            #[inline]
            fn pow(&self, exp: &Self) -> Vec<$t> {
                SliceMath::pow(self.as_slice(), exp.as_slice())
            }
            #[inline]
            fn sqrt(&self) -> Vec<$t> {
                SliceMath::sqrt(self.as_slice())
            }
        }
    };
}

impl_vecmath_delegate!(f32);
impl_vecmath_delegate!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_f32_positive_unchanged() {
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        assert_eq!(a.abs(), a);
    }

    #[test]
    fn abs_f32_negative_become_positive() {
        let a: Vec<f32> = (1..=32).map(|i| -(i as f32)).collect();
        let expected: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        assert_eq!(a.abs(), expected);
    }

    #[test]
    fn acos_f32_of_one_is_zero() {
        let a = vec![1.0f32; 16];
        assert!(a.acos().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn abs_f64_positive_unchanged() {
        let a: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        assert_eq!(a.abs(), a);
    }

    #[test]
    fn acos_f64_of_zero_is_pio2() {
        let a = vec![0.0f64; 8];
        let pio2 = std::f64::consts::FRAC_PI_2;
        assert!(a.acos().iter().all(|&x| (x - pio2).abs() < 1e-14));
    }

    #[test]
    fn slice_abs_smoke_f32() {
        let a: &[f32] = &[-1.0, 2.0, -3.0];
        assert_eq!(SliceMath::abs(a), vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn slice_sqrt_smoke_f64() {
        let a: &[f64] = &[1.0, 4.0, 9.0];
        assert_eq!(SliceMath::sqrt(a), vec![1.0, 2.0, 3.0]);
    }
}
