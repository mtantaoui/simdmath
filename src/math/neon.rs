//! NEON `[T]` and `Vec<T>` implementations of [`SliceMath`] / [`VecMath`].

use crate::arch::neon::{f32x4, f32x4::F32x4};
use crate::arch::neon::{f64x2, f64x2::F64x2};
use crate::math::{SliceMath, VecMath};
use crate::ops::vec::{binary_op, unary_op};

impl SliceMath<f32> for [f32] {
    /// Absolute value, processed 4 lanes at a time via NEON.
    #[inline]
    fn abs(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.abs())
    }
    /// Arc cosine, processed 4 lanes at a time via NEON.
    #[inline]
    fn acos(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.acos())
    }
    /// Arc sine, processed 4 lanes at a time via NEON.
    #[inline]
    fn asin(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.asin())
    }
    /// Arc tangent, processed 4 lanes at a time via NEON.
    #[inline]
    fn atan(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.atan())
    }
    /// Two-arg arc tangent per lane, NEON.
    #[inline]
    fn atan2(&self, other: &[f32]) -> Vec<f32> {
        binary_op::<f32, F32x4>(self, other, f32x4::LANE_COUNT, |y, x| y.atan2(&x))
    }
    /// Cube root, processed 4 lanes at a time via NEON.
    #[inline]
    fn cbrt(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.cbrt())
    }
    /// Cosine (radians), processed 4 lanes at a time via NEON.
    #[inline]
    fn cos(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.cos())
    }
    /// Exponential, processed 4 lanes at a time via NEON.
    #[inline]
    fn exp(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.exp())
    }
    /// Natural log, processed 4 lanes at a time via NEON.
    #[inline]
    fn ln(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.ln())
    }
    /// Sine (radians), processed 4 lanes at a time via NEON.
    #[inline]
    fn sin(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.sin())
    }
    /// Tangent (radians), processed 4 lanes at a time via NEON.
    #[inline]
    fn tan(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.tan())
    }
    /// `self^exp` per lane, processed 4 lanes at a time via NEON.
    #[inline]
    fn pow(&self, exp: &[f32]) -> Vec<f32> {
        binary_op::<f32, F32x4>(self, exp, f32x4::LANE_COUNT, |b, e| b.pow(&e))
    }
    /// Square root, processed 4 lanes at a time via NEON.
    #[inline]
    fn sqrt(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.sqrt())
    }
}

impl SliceMath<f64> for [f64] {
    /// Absolute value, processed 2 lanes at a time via NEON.
    #[inline]
    fn abs(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.abs())
    }
    /// Arc cosine, processed 2 lanes at a time via NEON.
    #[inline]
    fn acos(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.acos())
    }
    /// Arc sine, processed 2 lanes at a time via NEON.
    #[inline]
    fn asin(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.asin())
    }
    /// Arc tangent, processed 2 lanes at a time via NEON.
    #[inline]
    fn atan(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.atan())
    }
    /// Two-arg arc tangent per lane, NEON.
    #[inline]
    fn atan2(&self, other: &[f64]) -> Vec<f64> {
        binary_op::<f64, F64x2>(self, other, f64x2::LANE_COUNT, |y, x| y.atan2(&x))
    }
    /// Cube root, processed 2 lanes at a time via NEON.
    #[inline]
    fn cbrt(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.cbrt())
    }
    /// Cosine (radians), processed 2 lanes at a time via NEON.
    #[inline]
    fn cos(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.cos())
    }
    /// Exponential, processed 2 lanes at a time via NEON.
    #[inline]
    fn exp(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.exp())
    }
    /// Natural log, processed 2 lanes at a time via NEON.
    #[inline]
    fn ln(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.ln())
    }
    /// Sine (radians), processed 2 lanes at a time via NEON.
    #[inline]
    fn sin(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.sin())
    }
    /// Tangent (radians), processed 2 lanes at a time via NEON.
    #[inline]
    fn tan(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.tan())
    }
    /// `self^exp` per lane, processed 2 lanes at a time via NEON.
    #[inline]
    fn pow(&self, exp: &[f64]) -> Vec<f64> {
        binary_op::<f64, F64x2>(self, exp, f64x2::LANE_COUNT, |b, e| b.pow(&e))
    }
    /// Square root, processed 2 lanes at a time via NEON.
    #[inline]
    fn sqrt(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.sqrt())
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

    const TOL_F32: f32 = 5e-7;
    const TOL_F64: f64 = 1e-14;

    // ---- abs f32 -------------------------------------------------------------

    #[test]
    fn abs_f32_positive_unchanged() {
        let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        assert_eq!(a.abs(), a);
    }

    #[test]
    fn abs_f32_negative_become_positive() {
        let a: Vec<f32> = (1..=16).map(|i| -(i as f32)).collect();
        let expected: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        assert_eq!(a.abs(), expected);
    }

    #[test]
    fn abs_f32_with_tail() {
        let a = vec![-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0];
        assert_eq!(a.abs(), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn abs_f32_negative_zero_becomes_positive_zero() {
        let a = vec![-0.0f32; 4];
        for lane in a.abs() {
            assert_eq!(lane, 0.0f32);
            assert!(lane.is_sign_positive());
        }
    }

    #[test]
    fn abs_f32_empty() {
        assert_eq!(Vec::<f32>::new().abs(), vec![]);
    }

    #[test]
    fn acos_f32_of_one_is_zero() {
        let a = vec![1.0f32; 4];
        assert!(a.acos().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn acos_f32_of_neg_one_is_pi() {
        let a = vec![-1.0f32; 4];
        let pi = std::f32::consts::PI;
        assert!(a.acos().iter().all(|&x| (x - pi).abs() < TOL_F32));
    }

    #[test]
    fn acos_f32_of_zero_is_pio2() {
        let a = vec![0.0f32; 4];
        let pio2 = std::f32::consts::FRAC_PI_2;
        assert!(a.acos().iter().all(|&x| (x - pio2).abs() < TOL_F32));
    }

    #[test]
    fn acos_f32_of_half_is_pi_over_3() {
        let a = vec![0.5f32; 4];
        let expected = std::f32::consts::PI / 3.0;
        assert!(a.acos().iter().all(|&x| (x - expected).abs() < TOL_F32));
    }

    #[test]
    fn acos_f32_out_of_domain_is_nan() {
        let a = vec![1.5f32, -2.0, f32::INFINITY, f32::NAN];
        assert!(a.acos().iter().all(|x| x.is_nan()));
    }

    #[test]
    fn acos_f32_with_tail() {
        let inputs = vec![0.0f32, 0.5, -0.5, 0.9, -0.9, 1.0, -1.0];
        let result = inputs.acos();
        let expected: Vec<f32> = inputs.iter().map(|x| x.acos()).collect();
        for (r, e) in result.iter().zip(&expected) {
            if e.is_nan() {
                assert!(r.is_nan());
            } else {
                assert!((r - e).abs() < TOL_F32, "got {r}, expected {e}");
            }
        }
    }

    #[test]
    fn asin_f32_of_zero_is_zero() {
        let a = vec![0.0f32; 4];
        assert!(a.asin().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn asin_f32_of_one_is_pio2() {
        let a = vec![1.0f32; 4];
        let pio2 = std::f32::consts::FRAC_PI_2;
        assert!(a.asin().iter().all(|&x| (x - pio2).abs() < TOL_F32));
    }

    #[test]
    fn asin_f32_of_neg_one_is_neg_pio2() {
        let a = vec![-1.0f32; 4];
        let neg_pio2 = -std::f32::consts::FRAC_PI_2;
        assert!(a.asin().iter().all(|&x| (x - neg_pio2).abs() < TOL_F32));
    }

    #[test]
    fn asin_f32_out_of_domain_is_nan() {
        let a = vec![1.5f32, -2.0, f32::INFINITY, f32::NAN];
        assert!(a.asin().iter().all(|x| x.is_nan()));
    }

    #[test]
    fn abs_f64_positive_unchanged() {
        let a: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        assert_eq!(a.abs(), a);
    }

    #[test]
    fn abs_f64_negative_become_positive() {
        let a: Vec<f64> = (1..=8).map(|i| -(i as f64)).collect();
        let expected: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        assert_eq!(a.abs(), expected);
    }

    #[test]
    fn abs_f64_with_tail() {
        let a = vec![-1.0f64, 2.0, -3.0];
        assert_eq!(a.abs(), vec![1.0f64, 2.0, 3.0]);
    }

    #[test]
    fn abs_f64_empty() {
        assert_eq!(Vec::<f64>::new().abs(), vec![]);
    }

    #[test]
    fn acos_f64_of_zero_is_pio2() {
        let a = vec![0.0f64; 2];
        let pio2 = std::f64::consts::FRAC_PI_2;
        assert!(a.acos().iter().all(|&x| (x - pio2).abs() < TOL_F64));
    }

    #[test]
    fn acos_f64_of_one_is_zero() {
        let a = vec![1.0f64; 2];
        assert!(a.acos().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn asin_f64_of_zero_is_zero() {
        let a = vec![0.0f64; 2];
        assert!(a.asin().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn asin_f64_of_one_is_pio2() {
        let a = vec![1.0f64; 2];
        let pio2 = std::f64::consts::FRAC_PI_2;
        assert!(a.asin().iter().all(|&x| (x - pio2).abs() < TOL_F64));
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
