//! Scalar implementation of [`SliceMath`] / [`VecMath`] for `f32` and `f64`.
//!
//! Used when no SIMD ISA is enabled. Each method delegates to the
//! corresponding `f32` / `f64` method from the standard library, which in
//! turn calls the platform `libm`. The compiler may auto-vectorise the loop
//! body, but no hand-tuned SIMD is performed.

use crate::math::{SliceMath, VecMath};

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

macro_rules! impl_vecmath_scalar {
    ($t:ty) => {
        impl SliceMath<$t> for [$t] {
            #[inline]
            fn abs(&self) -> Vec<$t> {
                self.iter().map(|x| x.abs()).collect()
            }
            #[inline]
            fn acos(&self) -> Vec<$t> {
                self.iter().map(|x| x.acos()).collect()
            }
            #[inline]
            fn asin(&self) -> Vec<$t> {
                self.iter().map(|x| x.asin()).collect()
            }
            #[inline]
            fn atan(&self) -> Vec<$t> {
                self.iter().map(|x| x.atan()).collect()
            }
            #[inline]
            fn atan2(&self, other: &[$t]) -> Vec<$t> {
                assert_same_len!(self, other);
                self.iter().zip(other).map(|(y, x)| y.atan2(*x)).collect()
            }
            #[inline]
            fn cbrt(&self) -> Vec<$t> {
                self.iter().map(|x| x.cbrt()).collect()
            }
            #[inline]
            fn cos(&self) -> Vec<$t> {
                self.iter().map(|x| x.cos()).collect()
            }
            #[inline]
            fn exp(&self) -> Vec<$t> {
                self.iter().map(|x| x.exp()).collect()
            }
            #[inline]
            fn ln(&self) -> Vec<$t> {
                self.iter().map(|x| x.ln()).collect()
            }
            #[inline]
            fn sin(&self) -> Vec<$t> {
                self.iter().map(|x| x.sin()).collect()
            }
            #[inline]
            fn tan(&self) -> Vec<$t> {
                self.iter().map(|x| x.tan()).collect()
            }
            #[inline]
            fn pow(&self, exp: &[$t]) -> Vec<$t> {
                assert_same_len!(self, exp);
                self.iter().zip(exp).map(|(b, e)| b.powf(*e)).collect()
            }
            #[inline]
            fn sqrt(&self) -> Vec<$t> {
                self.iter().map(|x| x.sqrt()).collect()
            }
        }

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

impl_vecmath_scalar!(f32);
impl_vecmath_scalar!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::vec::{SliceExt, VecExt};

    #[test]
    fn scalar_vecmath_smoke_f32() {
        let v = vec![0.0f32, 1.0, 2.0, 3.0];
        assert_eq!(v.abs(), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(v.sqrt()[1], 1.0);
        let sin = v.sin();
        assert!((sin[0]).abs() < 1e-6);
    }

    #[test]
    fn scalar_vecmath_smoke_f64() {
        let v = vec![0.0f64, 1.0, 4.0, 9.0];
        assert_eq!(v.sqrt(), vec![0.0, 1.0, 2.0, 3.0]);
        let exps = vec![2.0f64; 4];
        // `pow` is documented as ≤ 2 ULP; use approximate equality rather than
        // bit-exact, since the platform libm (e.g. Miri's) may compute
        // x^2 via exp(2·ln(x)) with sub-ULP rounding noise.
        let got = v.pow(&exps);
        let want = [0.0f64, 1.0, 16.0, 81.0];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= w.abs() * 1e-14, "got {g}, want {w}");
        }
    }

    #[test]
    fn scalar_vecext_smoke() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        assert_eq!(a.add(&b), vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(a.mul(&b), vec![10.0, 40.0, 90.0, 160.0]);
        assert_eq!(b.sub(&a), vec![9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn slice_abs_smoke_f32() {
        let a: &[f32] = &[-1.0, 2.0, -3.0];
        assert_eq!(SliceMath::abs(a), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn slice_pow_smoke_f64() {
        let bases: &[f64] = &[2.0, 3.0, 4.0];
        let exps: &[f64] = &[3.0, 2.0, 0.5];
        let r = SliceMath::pow(bases, exps);
        assert!((r[0] - 8.0).abs() < 1e-12);
        assert!((r[1] - 9.0).abs() < 1e-12);
        assert!((r[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn slice_add_smoke_f32() {
        let a: &[f32] = &[1.0, 2.0, 3.0];
        let b: &[f32] = &[4.0, 5.0, 6.0];
        assert_eq!(SliceExt::add(a, b), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn slice_sum_smoke_f64() {
        let a: &[f64] = &[1.0, 2.0, 3.0, 4.0];
        assert_eq!(SliceExt::sum(a), 10.0);
    }
}
