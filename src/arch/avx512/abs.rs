//! Portable absolute-value intrinsics for AVX-512F `f32` and `f64` vectors.
//!
//! AVX-512F (like AVX2) has no dedicated `vabs` instruction for floating-point
//! registers. The technique is identical — clear the IEEE 754 sign bit using
//! `VANDNOT` applied to a sign-bit mask broadcast across all lanes:
//!
//! ```text
//! abs(x) = ~sign_mask & x          (ANDNOT clears the sign bit)
//! ```
//!
//! `_mm512_andnot_ps` / `_mm512_andnot_pd` are part of **avx512dq**, not
//! avx512f. To stay within the avx512f baseline this module instead casts
//! through `__m512i` and uses the integer ANDNOT:
//!
//! ```text
//! abs(x) = _mm512_castsi512_ps(
//!              _mm512_andnot_epi32(sign_mask_i, _mm512_castps_si512(x))
//!          )
//! ```
//!
//! Both functions compile to a single `vpandnd` / `vpandnq` instruction.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes the absolute value of each `f32` lane in an AVX-512 `__m512` register.
///
/// Casts to `__m512i`, clears the sign bit with `_mm512_andnot_epi32`, then
/// casts back. This avoids the avx512dq dependency of `_mm512_andnot_ps`.
///
/// # Safety
/// `f` must be a valid `__m512` value; no alignment or memory constraints.
#[inline]
pub(crate) unsafe fn _mm512_abs_ps(f: __m512) -> __m512 {
    // -0.0f32 == 0x8000_0000: broadcast the sign bit across all 16 lanes.
    let sign_mask = unsafe { _mm512_castps_si512(_mm512_set1_ps(-0.0)) };
    let f_i = unsafe { _mm512_castps_si512(f) };

    // ANDNOT: ~sign_mask & f_i — clears the sign bit in every lane.
    unsafe { _mm512_castsi512_ps(_mm512_andnot_epi32(sign_mask, f_i)) }
}

/// Computes the absolute value of each `f64` lane in an AVX-512 `__m512d` register.
///
/// Casts to `__m512i`, clears the sign bit with `_mm512_andnot_epi64`, then
/// casts back. This avoids the avx512dq dependency of `_mm512_andnot_pd`.
///
/// # Safety
/// `f` must be a valid `__m512d` value; no alignment or memory constraints.
#[inline]
pub(crate) unsafe fn _mm512_abs_pd(f: __m512d) -> __m512d {
    // -0.0f64 == 0x8000_0000_0000_0000: broadcast the sign bit across all 8 lanes.
    let sign_mask = unsafe { _mm512_castpd_si512(_mm512_set1_pd(-0.0)) };
    let f_i = unsafe { _mm512_castpd_si512(f) };

    // ANDNOT: ~sign_mask & f_i — clears the sign bit in every lane.
    unsafe { _mm512_castsi512_pd(_mm512_andnot_epi64(sign_mask, f_i)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- _mm512_abs_ps --------------------------------------------------------

    #[test]
    fn abs_ps_positive_values_are_unchanged() {
        unsafe {
            let src: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
            let input = _mm512_loadu_ps(src.as_ptr());
            let result = _mm512_abs_ps(input);
            let mut out = [0.0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), result);
            assert_eq!(out, src);
        }
    }

    #[test]
    fn abs_ps_negative_values_become_positive() {
        unsafe {
            let src: [f32; 16] = core::array::from_fn(|i| -((i + 1) as f32));
            let input = _mm512_loadu_ps(src.as_ptr());
            let result = _mm512_abs_ps(input);
            let mut out = [0.0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), result);
            let expected: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn abs_ps_mixed_signs() {
        unsafe {
            let src: [f32; 16] = core::array::from_fn(|i| {
                if i % 2 == 0 {
                    -((i + 1) as f32)
                } else {
                    (i + 1) as f32
                }
            });
            let input = _mm512_loadu_ps(src.as_ptr());
            let result = _mm512_abs_ps(input);
            let mut out = [0.0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), result);
            let expected: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn abs_ps_negative_zero_becomes_positive_zero() {
        unsafe {
            let input = _mm512_set1_ps(-0.0);
            let result = _mm512_abs_ps(input);
            let mut out = [f32::NAN; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), result);
            for lane in out {
                assert_eq!(lane, 0.0f32);
                assert!(lane.is_sign_positive(), "expected +0.0, got -0.0");
            }
        }
    }

    // ---- _mm512_abs_pd --------------------------------------------------------

    #[test]
    fn abs_pd_positive_values_are_unchanged() {
        unsafe {
            let src: [f64; 8] = core::array::from_fn(|i| (i + 1) as f64);
            let input = _mm512_loadu_pd(src.as_ptr());
            let result = _mm512_abs_pd(input);
            let mut out = [0.0f64; 8];
            _mm512_storeu_pd(out.as_mut_ptr(), result);
            assert_eq!(out, src);
        }
    }

    #[test]
    fn abs_pd_negative_values_become_positive() {
        unsafe {
            let src: [f64; 8] = core::array::from_fn(|i| -((i + 1) as f64));
            let input = _mm512_loadu_pd(src.as_ptr());
            let result = _mm512_abs_pd(input);
            let mut out = [0.0f64; 8];
            _mm512_storeu_pd(out.as_mut_ptr(), result);
            let expected: [f64; 8] = core::array::from_fn(|i| (i + 1) as f64);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn abs_pd_negative_zero_becomes_positive_zero() {
        unsafe {
            let input = _mm512_set1_pd(-0.0);
            let result = _mm512_abs_pd(input);
            let mut out = [f64::NAN; 8];
            _mm512_storeu_pd(out.as_mut_ptr(), result);
            for lane in out {
                assert_eq!(lane, 0.0f64);
                assert!(lane.is_sign_positive(), "expected +0.0, got -0.0");
            }
        }
    }
}
