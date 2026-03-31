//! AVX2 implementations of [`VecMath`] for the `F32x8` and `F64x4` register types.
//!
//! Each method wraps a single AVX2 intrinsic (or the `_mm256_acos_ps` rational
//! approximation) and returns a register of the same type, preserving `size`
//! so that the generic `unary_op` loop in [`crate::math`] can handle tails.

use std::arch::x86_64::*;

use crate::arch::avx2::abs::{_mm256_abs_pd, _mm256_abs_ps};
use crate::arch::avx2::acos::{_mm256_acos_pd, _mm256_acos_ps};
use crate::arch::avx2::f32x8::F32x8;
use crate::arch::avx2::f64x4::F64x4;
use crate::math::VecMath;

impl VecMath<f32> for F32x8 {
    /// Absolute value of every lane: clears the sign bit via `vandnps`.
    #[inline]
    fn abs(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_abs_ps(self.elements) },
        }
    }

    /// Arc cosine of every lane via the three-range minimax approximation.
    ///
    /// Lanes outside `[-1, 1]` or `NaN` inputs produce `NaN`.
    #[inline]
    fn acos(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_acos_ps(self.elements) },
        }
    }
}

impl VecMath<f64> for F64x4 {
    /// Absolute value of every lane: clears the sign bit via `vandnpd`.
    #[inline]
    fn abs(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_abs_pd(self.elements) },
        }
    }

    /// Arc cosine of every lane via the three-range minimax approximation.
    ///
    /// Lanes outside `[-1, 1]` or `NaN` inputs produce `NaN`.
    #[inline]
    fn acos(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_acos_pd(self.elements) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- VecMath<f32> for F32x8 ----------------------------------------------

    #[test]
    fn f32x8_abs_clears_sign() {
        unsafe {
            let v = F32x8 {
                size: 8,
                elements: _mm256_set_ps(-8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0),
            };
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), v.abs().elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn f32x8_acos_of_one_is_zero() {
        unsafe {
            let v = F32x8 {
                size: 8,
                elements: _mm256_set1_ps(1.0),
            };
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), v.acos().elements);
            assert!(out.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn f32x8_acos_of_neg_one_is_pi() {
        unsafe {
            let v = F32x8 {
                size: 8,
                elements: _mm256_set1_ps(-1.0),
            };
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), v.acos().elements);
            let pi = std::f32::consts::PI;
            assert!(out.iter().all(|&x| (x - pi).abs() < 5e-7));
        }
    }

    // ---- VecMath<f64> for F64x4 ----------------------------------------------

    #[test]
    fn f64x4_abs_clears_sign() {
        unsafe {
            let v = F64x4 {
                size: 4,
                elements: _mm256_set_pd(-4.0, 3.0, -2.0, 1.0),
            };
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), v.abs().elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn f64x4_acos_of_zero_is_pio2() {
        unsafe {
            let v = F64x4 {
                size: 4,
                elements: _mm256_set1_pd(0.0),
            };
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), v.acos().elements);
            let pio2 = std::f64::consts::FRAC_PI_2;
            assert!(out.iter().all(|&x| (x - pio2).abs() < 1e-15));
        }
    }
}
