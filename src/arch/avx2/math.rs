//! AVX2 implementations of [`VecMath`] for the `F32x8` and `F64x4` register types.
//!
//! Each method wraps a single AVX2 intrinsic (or the `_mm256_acos_ps` rational
//! approximation) and returns a register of the same type, preserving `size`
//! so that the generic `unary_op` loop in [`crate::math`] can handle tails.

#[cfg(target_arch = "x86")]
use std::arch::x86::{_mm256_sqrt_pd, _mm256_sqrt_ps};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm256_sqrt_pd, _mm256_sqrt_ps};

use crate::arch::avx2::abs::{_mm256_abs_pd, _mm256_abs_ps};
use crate::arch::avx2::acos::{_mm256_acos_pd, _mm256_acos_ps};
use crate::arch::avx2::asin::{_mm256_asin_pd, _mm256_asin_ps};
use crate::arch::avx2::atan::{_mm256_atan_pd, _mm256_atan_ps};
use crate::arch::avx2::atan2::{_mm256_atan2_pd, _mm256_atan2_ps};
use crate::arch::avx2::cbrt::{_mm256_cbrt_pd, _mm256_cbrt_ps};
use crate::arch::avx2::cos::{_mm256_cos_pd, _mm256_cos_ps};
use crate::arch::avx2::exp::{_mm256_exp_pd, _mm256_exp_ps};
use crate::arch::avx2::f32x8::F32x8;
use crate::arch::avx2::f64x4::F64x4;
use crate::arch::avx2::ln::{_mm256_ln_pd, _mm256_ln_ps};
use crate::arch::avx2::pow::{_mm256_pow_pd, _mm256_pow_ps};
use crate::arch::avx2::sin::{_mm256_sin_pd, _mm256_sin_ps};
use crate::arch::avx2::tan::{_mm256_tan_pd, _mm256_tan_ps};
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
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
    #[inline]
    fn acos(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_acos_ps(self.elements) },
        }
    }

    /// Arc sine of every lane via the two-range minimax approximation.
    ///
    /// Lanes outside `[-1, 1]` or `NaN` inputs produce `NaN`.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
    #[inline]
    fn asin(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_asin_ps(self.elements) },
        }
    }

    /// Arc tangent of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 3 ULP** error across the entire domain.
    #[inline]
    fn atan(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_atan_ps(self.elements) },
        }
    }

    /// Two-argument arc tangent: `atan2(self, other)` for every lane.
    ///
    /// # Precision
    ///
    /// **≤ 3 ULP** error across the entire domain.
    #[inline]
    fn atan2(&self, other: &F32x8) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_atan2_ps(self.elements, other.elements) },
        }
    }

    /// Cube root of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain.
    #[inline]
    fn cbrt(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_cbrt_ps(self.elements) },
        }
    }

    /// Cosine of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn cos(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_cos_ps(self.elements) },
        }
    }

    /// Exponential (`e^x`) of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn exp(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_exp_ps(self.elements) },
        }
    }

    /// Natural logarithm of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn ln(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_ln_ps(self.elements) },
        }
    }

    /// Sine of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 1.5 ULP** error across the entire domain.
    #[inline]
    fn sin(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_sin_ps(self.elements) },
        }
    }

    /// Tangent of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn tan(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_tan_ps(self.elements) },
        }
    }

    /// `self^exp` for every lane via compensated arithmetic.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn pow(&self, exp: &F32x8) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_pow_ps(self.elements, exp.elements) },
        }
    }

    /// Square root of every lane via `vsqrtps`.
    ///
    /// # Precision
    ///
    /// **≤ 0.5 ULP** — hardware correctly-rounded operation.
    #[inline]
    fn sqrt(&self) -> F32x8 {
        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_sqrt_ps(self.elements) },
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
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
    #[inline]
    fn acos(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_acos_pd(self.elements) },
        }
    }

    /// Arc sine of every lane via the two-range minimax approximation.
    ///
    /// Lanes outside `[-1, 1]` or `NaN` inputs produce `NaN`.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain `[-1, 1]`.
    #[inline]
    fn asin(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_asin_pd(self.elements) },
        }
    }

    /// Arc tangent of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain (musl 4-range reduction).
    #[inline]
    fn atan(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_atan_pd(self.elements) },
        }
    }

    /// Two-argument arc tangent: `atan2(self, other)` for every lane.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn atan2(&self, other: &F64x4) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_atan2_pd(self.elements, other.elements) },
        }
    }

    /// Cube root of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain.
    #[inline]
    fn cbrt(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_cbrt_pd(self.elements) },
        }
    }

    /// Cosine of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn cos(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_cos_pd(self.elements) },
        }
    }

    /// Exponential (`e^x`) of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn exp(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_exp_pd(self.elements) },
        }
    }

    /// Natural logarithm of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn ln(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_ln_pd(self.elements) },
        }
    }

    /// Sine of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 1.5 ULP** error across the entire domain.
    #[inline]
    fn sin(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_sin_pd(self.elements) },
        }
    }

    /// Tangent of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn tan(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_tan_pd(self.elements) },
        }
    }

    /// `self^exp` for every lane via compensated arithmetic.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn pow(&self, exp: &F64x4) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_pow_pd(self.elements, exp.elements) },
        }
    }

    /// Square root of every lane via `vsqrtpd`.
    ///
    /// # Precision
    ///
    /// **≤ 0.5 ULP** — hardware correctly-rounded operation.
    #[inline]
    fn sqrt(&self) -> F64x4 {
        F64x4 {
            size: self.size,
            elements: unsafe { _mm256_sqrt_pd(self.elements) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::x86_64::*;

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
