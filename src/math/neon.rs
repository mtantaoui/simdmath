//! NEON `Vec<T>` implementations of [`VecMath`].
//!
//! Each method uses [`unary_op`] or [`binary_op`] to partition the slice into
//! `F32x4` / `F64x2` chunks and applies the corresponding register-level
//! [`VecMath`] method. The tail (when `len % LANE_COUNT != 0`) is handled
//! automatically by `unary_op` / `binary_op` via a masked load/store.

use crate::arch::neon::{f32x4, f32x4::F32x4};
use crate::arch::neon::{f64x2, f64x2::F64x2};
use crate::math::VecMath;
use crate::ops::vec::{binary_op, unary_op};

impl VecMath<f32> for Vec<f32> {
    /// Absolute value of every element, processed 4 lanes at a time via NEON.
    #[inline]
    fn abs(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.abs())
    }

    /// Arc cosine of every element, processed 4 lanes at a time via NEON.
    ///
    /// Uses the three-range minimax rational approximation in
    /// [`crate::arch::neon::acos`]. Lanes outside `[-1, 1]` produce `NaN`.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the domain `[-1, 1]`.
    #[inline]
    fn acos(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.acos())
    }

    /// Arc sine of every element, processed 4 lanes at a time via NEON.
    ///
    /// Uses the two-range minimax rational approximation in
    /// [`crate::arch::neon::asin`]. Lanes outside `[-1, 1]` produce `NaN`.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the domain `[-1, 1]`.
    #[inline]
    fn asin(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.asin())
    }

    /// Arc tangent of every element, processed 4 lanes at a time via NEON.
    ///
    /// # Precision
    ///
    /// **≤ 3 ULP** error across the entire domain.
    #[inline]
    fn atan(&self) -> Vec<f32> {
        unary_op::<f32, F32x4>(self, f32x4::LANE_COUNT, |v| v.atan())
    }

    /// Two-argument arc tangent: `atan2(self, other)` for every element,
    /// processed 4 lanes at a time via NEON.
    ///
    /// # Precision
    ///
    /// **≤ 3 ULP** error across the entire domain.
    #[inline]
    fn atan2(&self, other: &Self) -> Vec<f32> {
        binary_op::<f32, F32x4>(self, other, f32x4::LANE_COUNT, |y, x| y.atan2(&x))
    }
}

impl VecMath<f64> for Vec<f64> {
    /// Absolute value of every element, processed 2 lanes at a time via NEON.
    #[inline]
    fn abs(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.abs())
    }

    /// Arc cosine of every element, processed 2 lanes at a time via NEON.
    ///
    /// Uses the three-range minimax rational approximation in
    /// [`crate::arch::neon::acos`]. Lanes outside `[-1, 1]` produce `NaN`.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the domain `[-1, 1]`.
    #[inline]
    fn acos(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.acos())
    }

    /// Arc sine of every element, processed 2 lanes at a time via NEON.
    ///
    /// Uses the two-range minimax rational approximation in
    /// [`crate::arch::neon::asin`]. Lanes outside `[-1, 1]` produce `NaN`.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the domain `[-1, 1]`.
    #[inline]
    fn asin(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.asin())
    }

    /// Arc tangent of every element, processed 2 lanes at a time via NEON.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain (musl 4-range reduction).
    #[inline]
    fn atan(&self) -> Vec<f64> {
        unary_op::<f64, F64x2>(self, f64x2::LANE_COUNT, |v| v.atan())
    }

    /// Two-argument arc tangent: `atan2(self, other)` for every element,
    /// processed 2 lanes at a time via NEON.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn atan2(&self, other: &Self) -> Vec<f64> {
        binary_op::<f64, F64x2>(self, other, f64x2::LANE_COUNT, |y, x| y.atan2(&x))
    }
}

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
        // 7 elements: 1 full F32x4 chunk + 3-lane tail
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

    // ---- acos f32 ------------------------------------------------------------

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
        // 7 elements spanning all three computational ranges
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

    // ---- asin f32 ------------------------------------------------------------

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

    // ---- abs f64 -------------------------------------------------------------

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
        // 3 elements: 1 full F64x2 chunk + 1-lane tail
        let a = vec![-1.0f64, 2.0, -3.0];
        assert_eq!(a.abs(), vec![1.0f64, 2.0, 3.0]);
    }

    #[test]
    fn abs_f64_empty() {
        assert_eq!(Vec::<f64>::new().abs(), vec![]);
    }

    // ---- acos f64 ------------------------------------------------------------

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

    // ---- asin f64 ------------------------------------------------------------

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
}
