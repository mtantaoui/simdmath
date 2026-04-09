//! AVX-512F `Vec<T>` implementations of [`VecMath`].

use crate::arch::avx512::{f32x16, f32x16::F32x16};
use crate::arch::avx512::{f64x8, f64x8::F64x8};
use crate::math::VecMath;
use crate::ops::vec::{binary_op, unary_op};

impl VecMath<f32> for Vec<f32> {
    /// Absolute value of every element, processed 16 lanes at a time via AVX-512.
    #[inline]
    fn abs(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.abs())
    }

    /// Arc cosine of every element, processed 16 lanes at a time via AVX-512.
    ///
    /// Uses the three-range minimax rational approximation in
    /// [`crate::arch::avx512::acos`]. Lanes outside `[-1, 1]` produce `NaN`.
    #[inline]
    fn acos(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.acos())
    }

    /// Arc sine of every element, processed 16 lanes at a time via AVX-512.
    ///
    /// Uses the two-range minimax rational approximation in
    /// [`crate::arch::avx512::asin`]. Lanes outside `[-1, 1]` produce `NaN`.
    #[inline]
    fn asin(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.asin())
    }

    /// Arc tangent of every element, processed 16 lanes at a time via AVX-512.
    ///
    /// # Precision
    ///
    /// **≤ 3 ULP** error across the entire domain.
    #[inline]
    fn atan(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.atan())
    }

    /// Two-argument arc tangent: `atan2(self, other)` for every element,
    /// processed 16 lanes at a time via AVX-512.
    ///
    /// # Precision
    ///
    /// **≤ 3 ULP** error across the entire domain.
    #[inline]
    fn atan2(&self, other: &Self) -> Vec<f32> {
        binary_op::<f32, F32x16>(self, other, f32x16::LANE_COUNT, |y, x| y.atan2(&x))
    }

    /// Cube root of every element, processed 16 lanes at a time via AVX-512.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain.
    #[inline]
    fn cbrt(&self) -> Vec<f32> {
        unary_op::<f32, F32x16>(self, f32x16::LANE_COUNT, |v| v.cbrt())
    }
}

impl VecMath<f64> for Vec<f64> {
    /// Absolute value of every element, processed 8 lanes at a time via AVX-512.
    #[inline]
    fn abs(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.abs())
    }

    /// Arc cosine of every element, processed 8 lanes at a time via AVX-512.
    ///
    /// Uses the three-range minimax rational approximation in
    /// [`crate::arch::avx512::acos`]. Lanes outside `[-1, 1]` produce `NaN`.
    #[inline]
    fn acos(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.acos())
    }

    /// Arc sine of every element, processed 8 lanes at a time via AVX-512.
    ///
    /// Uses the two-range minimax rational approximation in
    /// [`crate::arch::avx512::asin`]. Lanes outside `[-1, 1]` produce `NaN`.
    #[inline]
    fn asin(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.asin())
    }

    /// Arc tangent of every element, processed 8 lanes at a time via AVX-512.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain (musl 4-range reduction).
    #[inline]
    fn atan(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.atan())
    }

    /// Two-argument arc tangent: `atan2(self, other)` for every element,
    /// processed 8 lanes at a time via AVX-512.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn atan2(&self, other: &Self) -> Vec<f64> {
        binary_op::<f64, F64x8>(self, other, f64x8::LANE_COUNT, |y, x| y.atan2(&x))
    }

    /// Cube root of every element, processed 8 lanes at a time via AVX-512.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain.
    #[inline]
    fn cbrt(&self) -> Vec<f64> {
        unary_op::<f64, F64x8>(self, f64x8::LANE_COUNT, |v| v.cbrt())
    }
}

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
}
