//! Portable absolute-value intrinsics for AVX2 `f32` and `f64` vectors.
//!
//! AVX2 has no dedicated `vabs` instruction for floating-point registers.
//! The standard technique is to clear the IEEE 754 sign bit (bit 31 of each
//! `f32`, bit 63 of each `f64`) using `VANDNOT`:
//!
//! ```text
//! abs(x) = ~sign_mask & x          (ANDNOT clears the sign bit)
//! ```
//!
//! `-0.0` is the most readable way to spell the sign-bit mask in Rust:
//! its bit representation is `0x8000_0000` for `f32` and
//! `0x8000_0000_0000_0000` for `f64` — exactly one bit set.
//!
//! Both functions are `#[inline]` thin wrappers and compile to a single
//! `vandnps` / `vandnpd` instruction on AVX2 targets.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes the absolute value of each `f32` lane in an AVX2 `__m256` register.
///
/// Clears the sign bit of every lane using `_mm256_andnot_ps`:
/// `result = ~(-0.0) & f`, which leaves all other bits (exponent + mantissa)
/// unchanged and forces the sign bit to zero.
///
/// # Safety
/// `f` must be a valid `__m256` value; no alignment or memory constraints.
#[inline]
pub(crate) unsafe fn _mm256_abs_ps(f: __m256) -> __m256 {
    // -0.0 == 0x8000_0000: only the sign bit is set.
    let sign_mask = unsafe { _mm256_set1_ps(-0.0) };

    // ANDNOT: ~sign_mask & f — clears the sign bit in every lane.
    unsafe { _mm256_andnot_ps(sign_mask, f) }
}

/// Computes the absolute value of each `f64` lane in an AVX2 `__m256d` register.
///
/// Clears the sign bit of every lane using `_mm256_andnot_pd`:
/// `result = ~(-0.0) & f`, which leaves all other bits (exponent + mantissa)
/// unchanged and forces the sign bit to zero.
///
/// # Safety
/// `f` must be a valid `__m256d` value; no alignment or memory constraints.
#[inline]
pub(crate) unsafe fn _mm256_abs_pd(f: __m256d) -> __m256d {
    // -0.0 == 0x8000_0000_0000_0000: only the sign bit is set.
    let sign_mask = unsafe { _mm256_set1_pd(-0.0) };

    // ANDNOT: ~sign_mask & f — clears the sign bit in every lane.
    unsafe { _mm256_andnot_pd(sign_mask, f) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- _mm256_abs_ps --------------------------------------------------------

    #[test]
    fn abs_ps_positive_values_are_unchanged() {
        unsafe {
            let input = _mm256_set_ps(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
            let result = _mm256_abs_ps(input);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), result);
            // _mm256_set_ps fills lanes in reverse order (lane 0 = last arg).
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn abs_ps_negative_values_become_positive() {
        unsafe {
            let input = _mm256_set_ps(-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0);
            let result = _mm256_abs_ps(input);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn abs_ps_mixed_signs() {
        unsafe {
            let input = _mm256_set_ps(-4.0, 3.0, -2.0, 1.0, -4.0, 3.0, -2.0, 1.0);
            let result = _mm256_abs_ps(input);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn abs_ps_negative_zero_becomes_positive_zero() {
        unsafe {
            let input = _mm256_set1_ps(-0.0);
            let result = _mm256_abs_ps(input);
            let mut out = [f32::NAN; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), result);
            // -0.0 and +0.0 compare equal; verify the sign bit is cleared.
            for lane in out {
                assert_eq!(lane, 0.0f32);
                assert!(lane.is_sign_positive(), "expected +0.0, got -0.0");
            }
        }
    }

    // ---- _mm256_abs_pd --------------------------------------------------------

    #[test]
    fn abs_pd_positive_values_are_unchanged() {
        unsafe {
            let input = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);
            let result = _mm256_abs_pd(input);
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn abs_pd_negative_values_become_positive() {
        unsafe {
            let input = _mm256_set_pd(-4.0, -3.0, -2.0, -1.0);
            let result = _mm256_abs_pd(input);
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn abs_pd_mixed_signs() {
        unsafe {
            let input = _mm256_set_pd(-4.0, 3.0, -2.0, 1.0);
            let result = _mm256_abs_pd(input);
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn abs_pd_negative_zero_becomes_positive_zero() {
        unsafe {
            let input = _mm256_set1_pd(-0.0);
            let result = _mm256_abs_pd(input);
            let mut out = [f64::NAN; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), result);
            for lane in out {
                assert_eq!(lane, 0.0f64);
                assert!(lane.is_sign_positive(), "expected +0.0, got -0.0");
            }
        }
    }
}
