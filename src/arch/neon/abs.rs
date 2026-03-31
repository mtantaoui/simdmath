//! Portable absolute-value intrinsics for NEON `f32` and `f64` vectors.
//!
//! NEON provides the `vabs` instruction for floating-point absolute value,
//! which is directly exposed via `vabsq_f32` and `vabsq_f64`.

use std::arch::aarch64::*;

/// Computes the absolute value of each `f32` lane in a NEON `float32x4_t` register.
///
/// # Safety
/// `f` must be a valid `float32x4_t` value.
#[inline]
pub(crate) unsafe fn vabsq_f32_wrapper(f: float32x4_t) -> float32x4_t {
    unsafe { vabsq_f32(f) }
}

/// Computes the absolute value of each `f64` lane in a NEON `float64x2_t` register.
///
/// # Safety
/// `f` must be a valid `float64x2_t` value.
#[inline]
pub(crate) unsafe fn vabsq_f64_wrapper(f: float64x2_t) -> float64x2_t {
    unsafe { vabsq_f64(f) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_f32_positive_unchanged() {
        unsafe {
            let input = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = vabsq_f32_wrapper(input);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn abs_f32_negative_become_positive() {
        unsafe {
            let input = vld1q_f32([-1.0f32, -2.0, -3.0, -4.0].as_ptr());
            let result = vabsq_f32_wrapper(input);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn abs_f64_positive_unchanged() {
        unsafe {
            let input = vld1q_f64([1.0f64, 2.0].as_ptr());
            let result = vabsq_f64_wrapper(input);
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0]);
        }
    }

    #[test]
    fn abs_f64_negative_become_positive() {
        unsafe {
            let input = vld1q_f64([-1.0f64, -2.0].as_ptr());
            let result = vabsq_f64_wrapper(input);
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0]);
        }
    }
}
