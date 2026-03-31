//! AVX-512F implementations of [`VecMath`] for the `F32x16` and `F64x8` register types.

use std::arch::x86_64::*;

use crate::arch::avx512::abs::{_mm512_abs_pd, _mm512_abs_ps};
use crate::arch::avx512::f32x16::F32x16;
use crate::arch::avx512::f64x8::F64x8;
use crate::math::VecMath;

impl VecMath<f32> for F32x16 {
    /// Absolute value of every lane via `vpandnd` (avx512f-only ANDNOT on integers).
    #[inline]
    fn abs(&self) -> F32x16 {
        F32x16 {
            size: self.size,
            elements: unsafe { _mm512_abs_ps(self.elements) },
        }
    }

    /// Arc cosine of every lane via scalar `f32::acos` per lane.
    ///
    /// A SIMD `_mm512_acos_ps` is not yet implemented.
    #[inline]
    fn acos(&self) -> F32x16 {
        let mut buf = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(buf.as_mut_ptr(), self.elements) };
        for x in &mut buf[..self.size] {
            *x = x.acos();
        }
        F32x16 {
            size: self.size,
            elements: unsafe { _mm512_loadu_ps(buf.as_ptr()) },
        }
    }
}

impl VecMath<f64> for F64x8 {
    /// Absolute value of every lane via `vpandnq` (avx512f-only ANDNOT on integers).
    #[inline]
    fn abs(&self) -> F64x8 {
        F64x8 {
            size: self.size,
            elements: unsafe { _mm512_abs_pd(self.elements) },
        }
    }

    /// Arc cosine of every lane via scalar `f64::acos` per lane.
    #[inline]
    fn acos(&self) -> F64x8 {
        let mut buf = [0.0f64; 8];
        unsafe { _mm512_storeu_pd(buf.as_mut_ptr(), self.elements) };
        for x in &mut buf[..self.size] {
            *x = x.acos();
        }
        F64x8 {
            size: self.size,
            elements: unsafe { _mm512_loadu_pd(buf.as_ptr()) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32x16_abs_clears_sign() {
        unsafe {
            let elems: [f32; 16] = core::array::from_fn(|i| if i % 2 == 0 { i as f32 + 1.0 } else { -(i as f32 + 1.0) });
            let v = F32x16 { size: 16, elements: _mm512_loadu_ps(elems.as_ptr()) };
            let mut out = [0.0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), v.abs().elements);
            assert!(out.iter().enumerate().all(|(i, &x)| x == i as f32 + 1.0));
        }
    }

    #[test]
    fn f32x16_acos_of_one_is_zero() {
        unsafe {
            let v = F32x16 { size: 16, elements: _mm512_set1_ps(1.0) };
            let mut out = [0.0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), v.acos().elements);
            assert!(out.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn f64x8_abs_clears_sign() {
        unsafe {
            let elems: [f64; 8] = core::array::from_fn(|i| if i % 2 == 0 { i as f64 + 1.0 } else { -(i as f64 + 1.0) });
            let v = F64x8 { size: 8, elements: _mm512_loadu_pd(elems.as_ptr()) };
            let mut out = [0.0f64; 8];
            _mm512_storeu_pd(out.as_mut_ptr(), v.abs().elements);
            assert!(out.iter().enumerate().all(|(i, &x)| x == i as f64 + 1.0));
        }
    }

    #[test]
    fn f64x8_acos_of_zero_is_pio2() {
        unsafe {
            let v = F64x8 { size: 8, elements: _mm512_set1_pd(0.0) };
            let mut out = [0.0f64; 8];
            _mm512_storeu_pd(out.as_mut_ptr(), v.acos().elements);
            let pio2 = std::f64::consts::FRAC_PI_2;
            assert!(out.iter().all(|&x| (x - pio2).abs() < 1e-15));
        }
    }
}
