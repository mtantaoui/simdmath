//! NEON implementations of [`VecMath`] for the `F32x4` and `F64x2` register types.
//!
//! Each method wraps a single NEON intrinsic (or the `vacos_f32` / `vasin_f32`
//! rational approximation) and returns a register of the same type, preserving
//! `size` so that the generic `unary_op` loop in [`crate::math`] can handle tails.

use std::arch::aarch64::{vsqrtq_f32, vsqrtq_f64};

use crate::arch::neon::abs::{vabsq_f32_wrapper, vabsq_f64_wrapper};
use crate::arch::neon::acos::{vacos_f32, vacos_f64};
use crate::arch::neon::asin::{vasin_f32, vasin_f64};
use crate::arch::neon::atan::{vatan_f32, vatan_f64};
use crate::arch::neon::atan2::{vatan2_f32, vatan2_f64};
use crate::arch::neon::cbrt::{vcbrt_f32, vcbrt_f64};
use crate::arch::neon::cos::{vcos_f32, vcos_f64};
use crate::arch::neon::exp::{vexp_f32, vexp_f64};
use crate::arch::neon::f32x4::F32x4;
use crate::arch::neon::f64x2::F64x2;
use crate::arch::neon::ln::{vln_f32, vln_f64};
use crate::arch::neon::pow::{vpow_f32, vpow_f64};
use crate::arch::neon::sin::{vsin_f32, vsin_f64};
use crate::arch::neon::tan::{vtan_f32, vtan_f64};
use crate::math::VecMath;

impl VecMath<f32> for F32x4 {
    /// Absolute value of every lane: clears the sign bit via `vabsq_f32`.
    #[inline]
    fn abs(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vabsq_f32_wrapper(self.elements) },
        }
    }

    /// Arc cosine of every lane via the three-range minimax approximation.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the domain `[-1, 1]`.
    /// Lanes outside `[-1, 1]` or `NaN` inputs produce `NaN`.
    #[inline]
    fn acos(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vacos_f32(self.elements) },
        }
    }

    /// Arc sine of every lane via the two-range minimax approximation.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the domain `[-1, 1]`.
    /// Lanes outside `[-1, 1]` or `NaN` inputs produce `NaN`.
    #[inline]
    fn asin(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vasin_f32(self.elements) },
        }
    }

    /// Arc tangent of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 3 ULP** error across the entire domain.
    #[inline]
    fn atan(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vatan_f32(self.elements) },
        }
    }

    /// Two-argument arc tangent: `atan2(self, other)` for every lane.
    ///
    /// # Precision
    ///
    /// **≤ 3 ULP** error across the entire domain.
    #[inline]
    fn atan2(&self, other: &F32x4) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vatan2_f32(self.elements, other.elements) },
        }
    }

    /// Cube root of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain.
    #[inline]
    fn cbrt(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vcbrt_f32(self.elements) },
        }
    }

    /// Cosine of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn cos(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vcos_f32(self.elements) },
        }
    }

    /// Exponential (`e^x`) of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn exp(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vexp_f32(self.elements) },
        }
    }

    /// Natural logarithm of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn ln(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vln_f32(self.elements) },
        }
    }

    /// Sine of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn sin(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vsin_f32(self.elements) },
        }
    }

    /// Tangent of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn tan(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vtan_f32(self.elements) },
        }
    }

    /// `self^exp` for every lane via compensated arithmetic.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn pow(&self, exp: &F32x4) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vpow_f32(self.elements, exp.elements) },
        }
    }

    /// Square root of every lane via `vsqrtq_f32`.
    ///
    /// # Precision
    ///
    /// **≤ 0.5 ULP** — hardware correctly-rounded operation.
    #[inline]
    fn sqrt(&self) -> F32x4 {
        F32x4 {
            size: self.size,
            elements: unsafe { vsqrtq_f32(self.elements) },
        }
    }
}

impl VecMath<f64> for F64x2 {
    /// Absolute value of every lane: clears the sign bit via `vabsq_f64`.
    #[inline]
    fn abs(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vabsq_f64_wrapper(self.elements) },
        }
    }

    /// Arc cosine of every lane via the three-range minimax approximation.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the domain `[-1, 1]`.
    /// Lanes outside `[-1, 1]` or `NaN` inputs produce `NaN`.
    #[inline]
    fn acos(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vacos_f64(self.elements) },
        }
    }

    /// Arc sine of every lane via the two-range minimax approximation.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the domain `[-1, 1]`.
    /// Lanes outside `[-1, 1]` or `NaN` inputs produce `NaN`.
    #[inline]
    fn asin(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vasin_f64(self.elements) },
        }
    }

    /// Arc tangent of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain (musl 4-range reduction).
    #[inline]
    fn atan(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vatan_f64(self.elements) },
        }
    }

    /// Two-argument arc tangent: `atan2(self, other)` for every lane.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn atan2(&self, other: &F64x2) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vatan2_f64(self.elements, other.elements) },
        }
    }

    /// Cube root of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 1 ULP** error across the entire domain.
    #[inline]
    fn cbrt(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vcbrt_f64(self.elements) },
        }
    }

    /// Cosine of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn cos(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vcos_f64(self.elements) },
        }
    }

    /// Exponential (`e^x`) of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn exp(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vexp_f64(self.elements) },
        }
    }

    /// Natural logarithm of every lane.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn ln(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vln_f64(self.elements) },
        }
    }

    /// Sine of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn sin(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vsin_f64(self.elements) },
        }
    }

    /// Tangent of every lane (radians).
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn tan(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vtan_f64(self.elements) },
        }
    }

    /// `self^exp` for every lane via compensated arithmetic.
    ///
    /// # Precision
    ///
    /// **≤ 2 ULP** error across the entire domain.
    #[inline]
    fn pow(&self, exp: &F64x2) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vpow_f64(self.elements, exp.elements) },
        }
    }

    /// Square root of every lane via `vsqrtq_f64`.
    ///
    /// # Precision
    ///
    /// **≤ 0.5 ULP** — hardware correctly-rounded operation.
    #[inline]
    fn sqrt(&self) -> F64x2 {
        F64x2 {
            size: self.size,
            elements: unsafe { vsqrtq_f64(self.elements) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::aarch64::*;

    // ---- VecMath<f32> for F32x4 ----------------------------------------------

    #[test]
    fn f32x4_abs_clears_sign() {
        unsafe {
            let elems: [f32; 4] = [-1.0, 2.0, -3.0, 4.0];
            let v = F32x4 {
                size: 4,
                elements: vld1q_f32(elems.as_ptr()),
            };
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), v.abs().elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn f32x4_acos_of_one_is_zero() {
        unsafe {
            let v = F32x4 {
                size: 4,
                elements: vdupq_n_f32(1.0),
            };
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), v.acos().elements);
            assert!(out.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn f32x4_acos_of_neg_one_is_pi() {
        unsafe {
            let v = F32x4 {
                size: 4,
                elements: vdupq_n_f32(-1.0),
            };
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), v.acos().elements);
            let pi = std::f32::consts::PI;
            assert!(out.iter().all(|&x| (x - pi).abs() < 5e-7));
        }
    }

    #[test]
    fn f32x4_asin_of_zero_is_zero() {
        unsafe {
            let v = F32x4 {
                size: 4,
                elements: vdupq_n_f32(0.0),
            };
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), v.asin().elements);
            assert!(out.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn f32x4_asin_of_one_is_pio2() {
        unsafe {
            let v = F32x4 {
                size: 4,
                elements: vdupq_n_f32(1.0),
            };
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), v.asin().elements);
            let pio2 = std::f32::consts::FRAC_PI_2;
            assert!(out.iter().all(|&x| (x - pio2).abs() < 5e-7));
        }
    }

    // ---- VecMath<f64> for F64x2 ----------------------------------------------

    #[test]
    fn f64x2_abs_clears_sign() {
        unsafe {
            let elems: [f64; 2] = [-1.0, 2.0];
            let v = F64x2 {
                size: 2,
                elements: vld1q_f64(elems.as_ptr()),
            };
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), v.abs().elements);
            assert_eq!(out, [1.0, 2.0]);
        }
    }

    #[test]
    fn f64x2_acos_of_zero_is_pio2() {
        unsafe {
            let v = F64x2 {
                size: 2,
                elements: vdupq_n_f64(0.0),
            };
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), v.acos().elements);
            let pio2 = std::f64::consts::FRAC_PI_2;
            assert!(out.iter().all(|&x| (x - pio2).abs() < 1e-15));
        }
    }

    #[test]
    fn f64x2_asin_of_zero_is_zero() {
        unsafe {
            let v = F64x2 {
                size: 2,
                elements: vdupq_n_f64(0.0),
            };
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), v.asin().elements);
            assert!(out.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn f64x2_asin_of_one_is_pio2() {
        unsafe {
            let v = F64x2 {
                size: 2,
                elements: vdupq_n_f64(1.0),
            };
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), v.asin().elements);
            let pio2 = std::f64::consts::FRAC_PI_2;
            assert!(out.iter().all(|&x| (x - pio2).abs() < 1e-15));
        }
    }
}
