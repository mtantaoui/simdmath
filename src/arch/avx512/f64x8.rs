//! AVX-512F implementation of an 8-lane double-precision floating-point vector (`f64x8`).
//!
//! This module provides [`F64x8`], a wrapper around the AVX-512F `__m512d` register type,
//! which holds 8 × `f64` values processed in parallel. It implements the [`Align`],
//! [`Load`], and [`Store`] traits for aligned, unaligned, and partial (tail) load/store
//! patterns, as well as all standard arithmetic and bitwise operators.
//!
//! # Partial loads and stores
//! AVX-512 uses k-registers (`__mmask8`) for 8-lane double masking — a single `u8` where
//! each bit enables the corresponding lane. The tail mask is `(1u8 << size) - 1`.
//!
//! # Target architecture
//! Compiled only when `target_feature = "avx512f"` is active.

use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

use crate::ops::simd::{Align, Load, Store};

/// Number of `f64` lanes in a single [`F64x8`] register.
pub const LANE_COUNT: usize = 8;

/// An 8-lane double-precision floating-point SIMD vector backed by `__m512d`.
///
/// `size` tracks how many lanes are logically active; inactive lanes are zeroed
/// on load and masked on store so they never contaminate results.
#[derive(Copy, Clone, Debug)]
pub struct F64x8 {
    pub(crate) size: usize,
    pub(crate) elements: __m512d,
}

impl Align<f64> for F64x8 {
    #[inline]
    fn is_aligned(ptr: *const f64) -> bool {
        (ptr as usize).is_multiple_of(core::mem::align_of::<__m512d>())
    }
}

impl Load<f64> for F64x8 {
    type Output = Self;

    /// Loads exactly [`LANE_COUNT`] elements, choosing aligned or unaligned
    /// load automatically based on pointer alignment.
    #[inline]
    unsafe fn load(ptr: *const f64, size: usize) -> Self::Output {
        debug_assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        match F64x8::is_aligned(ptr) {
            true => unsafe { Self::load_aligned(ptr) },
            false => unsafe { Self::load_unaligned(ptr) },
        }
    }

    /// Loads [`LANE_COUNT`] elements from a 64-byte-aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and 64-byte aligned.
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self::Output {
        Self {
            elements: unsafe { _mm512_load_pd(ptr) },
            size: LANE_COUNT,
        }
    }

    /// Loads [`LANE_COUNT`] elements from an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null.
    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self::Output {
        Self {
            elements: unsafe { _mm512_loadu_pd(ptr) },
            size: LANE_COUNT,
        }
    }

    /// Loads `size` elements (where `size < LANE_COUNT`) using a `__mmask8`
    /// bitmask. Inactive lanes are zeroed by `_mm512_maskz_loadu_pd`.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for at least `size` reads.
    #[inline]
    unsafe fn load_partial(ptr: *const f64, size: usize) -> Self::Output {
        debug_assert!(size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mask: __mmask8 = (1u8 << size).wrapping_sub(1);
        Self {
            elements: unsafe { _mm512_maskz_loadu_pd(mask, ptr) },
            size,
        }
    }

    /// Broadcasts `val` to every lane of a new `F64x8` register (`_mm512_set1_pd`).
    #[inline]
    unsafe fn broadcast(val: f64) -> Self::Output {
        Self {
            size: LANE_COUNT,
            elements: unsafe { _mm512_set1_pd(val) },
        }
    }

    /// Returns a zeroed `F64x8` register (`_mm512_setzero_pd`).
    #[inline]
    unsafe fn zero() -> Self::Output {
        Self {
            size: LANE_COUNT,
            elements: unsafe { _mm512_setzero_pd() },
        }
    }
}

impl Store<f64> for F64x8 {
    type Output = Self;

    /// Stores all [`LANE_COUNT`] lanes to `ptr`, dispatching to aligned or unaligned.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for [`LANE_COUNT`] element writes.
    #[inline]
    unsafe fn store_at(&self, ptr: *const f64) {
        debug_assert!(self.size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mut_ptr = ptr as *mut f64;
        match F64x8::is_aligned(ptr) {
            true => unsafe { self.store_aligned_at(mut_ptr) },
            false => unsafe { self.store_unaligned_at(mut_ptr) },
        }
    }

    /// Non-temporal (streaming) store. Bypasses the cache.
    ///
    /// # Safety
    /// `ptr` must be non-null and 64-byte aligned.
    #[inline]
    unsafe fn stream_at(&self, ptr: *mut f64) {
        unsafe { _mm512_stream_pd(ptr, self.elements) }
    }

    /// Stores all [`LANE_COUNT`] lanes to a 64-byte-aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and 64-byte aligned.
    #[inline]
    unsafe fn store_aligned_at(&self, ptr: *mut f64) {
        unsafe { _mm512_store_pd(ptr, self.elements) }
    }

    /// Stores all [`LANE_COUNT`] lanes to an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null.
    #[inline]
    unsafe fn store_unaligned_at(&self, ptr: *mut f64) {
        unsafe { _mm512_storeu_pd(ptr, self.elements) }
    }

    /// Stores `self.size` lanes using a `__mmask8` bitmask. Inactive lanes
    /// are not written to memory.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for at least `self.size` writes.
    #[inline]
    unsafe fn store_at_partial(&self, ptr: *mut f64) {
        debug_assert!(self.size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mask: __mmask8 = (1u8 << self.size).wrapping_sub(1);
        unsafe { _mm512_mask_storeu_pd(ptr, mask, self.elements) }
    }
}

/// Element-wise addition (`_mm512_add_pd`).
impl Add for F64x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm512_add_pd(self.elements, rhs.elements) },
        }
    }
}

impl AddAssign for F64x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Element-wise subtraction (`_mm512_sub_pd`).
impl Sub for F64x8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm512_sub_pd(self.elements, rhs.elements) },
        }
    }
}

impl SubAssign for F64x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Element-wise multiplication (`_mm512_mul_pd`).
impl Mul for F64x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm512_mul_pd(self.elements, rhs.elements) },
        }
    }
}

impl MulAssign for F64x8 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Element-wise division (`_mm512_div_pd`).
impl Div for F64x8 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm512_div_pd(self.elements, rhs.elements) },
        }
    }
}

impl DivAssign for F64x8 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Element-wise floating-point remainder with truncation semantics, matching
/// Rust's `f64 %` operator. Computed via scalar extraction since
/// `_mm512_cvttpd_epi64` (truncating f64→i64) requires avx512dq.
impl Rem for F64x8 {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "mismatched lane counts: lhs has {} lanes, rhs has {} lanes",
            self.size,
            rhs.size
        );

        // rem = a - trunc(a / b) * b
        unsafe {
            let div = _mm512_div_pd(self.elements, rhs.elements);
            // Truncate toward zero: roundscale with _MM_FROUND_TO_ZERO (0x03)
            let trunc = _mm512_roundscale_pd(div, 0x03);
            let prod = _mm512_mul_pd(trunc, rhs.elements);
            Self {
                size: self.size,
                elements: _mm512_sub_pd(self.elements, prod),
            }
        }
    }
}

impl RemAssign for F64x8 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

/// Bitwise AND on raw IEEE 754 bit patterns. Casts through `__m512i` since
/// `_mm512_and_pd` requires avx512dq; only avx512f is needed here.
impl BitAnd for F64x8 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "mismatched lane counts: lhs has {} lanes, rhs has {} lanes",
            self.size,
            rhs.size
        );

        unsafe {
            let a_i = _mm512_castpd_si512(self.elements);
            let b_i = _mm512_castpd_si512(rhs.elements);
            F64x8 {
                size: self.size,
                elements: _mm512_castsi512_pd(_mm512_and_epi64(a_i, b_i)),
            }
        }
    }
}

impl BitAndAssign for F64x8 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

/// Bitwise OR on raw IEEE 754 bit patterns. Casts through `__m512i` since
/// `_mm512_or_pd` requires avx512dq; only avx512f is needed here.
impl BitOr for F64x8 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "mismatched lane counts: lhs has {} lanes, rhs has {} lanes",
            self.size,
            rhs.size
        );

        unsafe {
            let a_i = _mm512_castpd_si512(self.elements);
            let b_i = _mm512_castpd_si512(rhs.elements);
            F64x8 {
                size: self.size,
                elements: _mm512_castsi512_pd(_mm512_or_epi64(a_i, b_i)),
            }
        }
    }
}

impl BitOrAssign for F64x8 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(align(64))]
    struct Aligned([f64; 8]);

    // ---- Align ----------------------------------------------------------------

    #[test]
    fn is_aligned_returns_true_for_aligned_pointer() {
        let data = Aligned([0.0; 8]);
        assert!(F64x8::is_aligned(data.0.as_ptr()));
    }

    #[test]
    fn is_aligned_returns_false_for_unaligned_pointer() {
        let data = Aligned([0.0; 8]);
        let unaligned = unsafe { data.0.as_ptr().add(1) };
        assert!(!F64x8::is_aligned(unaligned));
    }

    // ---- Load -----------------------------------------------------------------

    #[test]
    fn broadcast_fills_all_lanes_with_value() {
        unsafe {
            let v = F64x8::broadcast(2.718);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; LANE_COUNT];
            _mm512_storeu_pd(out.as_mut_ptr(), v.elements);
            for lane in out {
                assert!((lane - 2.718f64).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn zero_produces_all_zero_lanes() {
        unsafe {
            let v = F64x8::zero();
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [1.0f64; LANE_COUNT];
            _mm512_storeu_pd(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [0.0f64; LANE_COUNT]);
        }
    }

    #[test]
    fn load_aligned_loads_all_lanes() {
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f64));
        unsafe {
            let v = F64x8::load_aligned(src.0.as_ptr());
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; LANE_COUNT];
            _mm512_storeu_pd(out.as_mut_ptr(), v.elements);
            assert_eq!(out, src.0);
        }
    }

    #[test]
    fn load_unaligned_loads_all_lanes() {
        let mut buf = [0.0f64; 9];
        for i in 0..8 {
            buf[i + 1] = (i + 1) as f64;
        }
        unsafe {
            let ptr = buf.as_ptr().add(1);
            let v = F64x8::load_unaligned(ptr);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; LANE_COUNT];
            _mm512_storeu_pd(out.as_mut_ptr(), v.elements);
            assert_eq!(out, core::array::from_fn(|i| (i + 1) as f64));
        }
    }

    #[test]
    fn load_partial_loads_n_active_lanes_and_zeros_the_rest() {
        let src: [f64; 8] = core::array::from_fn(|i| (i + 1) as f64);
        for size in 0..LANE_COUNT {
            unsafe {
                let v = F64x8::load_partial(src.as_ptr(), size);
                assert_eq!(v.size, size);
                let mut out = [0.0f64; LANE_COUNT];
                _mm512_storeu_pd(out.as_mut_ptr(), v.elements);
                for i in 0..size {
                    assert_eq!(out[i], src[i], "size={size}: lane {i} should match source");
                }
                for i in size..LANE_COUNT {
                    assert_eq!(out[i], 0.0, "size={size}: lane {i} should be zeroed");
                }
            }
        }
    }

    // ---- Store ----------------------------------------------------------------

    #[test]
    fn store_aligned_writes_all_lanes() {
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f64));
        let mut dst = Aligned([0.0; 8]);
        unsafe {
            let v = F64x8::load_aligned(src.0.as_ptr());
            v.store_aligned_at(dst.0.as_mut_ptr());
        }
        assert_eq!(dst.0, src.0);
    }

    #[test]
    fn store_unaligned_writes_all_lanes() {
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f64));
        let mut dst = [0.0f64; 9];
        unsafe {
            let v = F64x8::load_aligned(src.0.as_ptr());
            v.store_unaligned_at(dst.as_mut_ptr().add(1));
        }
        assert_eq!(&dst[1..], &src.0[..]);
    }

    #[test]
    fn stream_at_writes_all_lanes() {
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f64));
        let mut dst = Aligned([0.0; 8]);
        unsafe {
            let v = F64x8::load_aligned(src.0.as_ptr());
            v.stream_at(dst.0.as_mut_ptr());
            _mm_sfence();
        }
        assert_eq!(dst.0, src.0);
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
        let src: [f64; 8] = core::array::from_fn(|i| (i + 1) as f64);
        for size in 0..LANE_COUNT {
            let mut dst = [-1.0f64; LANE_COUNT];
            unsafe {
                let v = F64x8::load_partial(src.as_ptr(), size);
                v.store_at_partial(dst.as_mut_ptr());
            }
            for i in 0..size {
                assert_eq!(dst[i], src[i], "size={size}: lane {i} should be written");
            }
            for i in size..LANE_COUNT {
                assert_eq!(dst[i], -1.0, "size={size}: lane {i} should be untouched");
            }
        }
    }

    // ---- Arithmetic ops -------------------------------------------------------

    unsafe fn load_arr(arr: &[f64; LANE_COUNT]) -> F64x8 {
        unsafe { F64x8::load_unaligned(arr.as_ptr()) }
    }

    unsafe fn store_arr(v: F64x8) -> [f64; LANE_COUNT] {
        let mut out = [0.0f64; LANE_COUNT];
        unsafe { _mm512_storeu_pd(out.as_mut_ptr(), v.elements) };
        out
    }

    #[test]
    fn add_computes_element_wise_sum() {
        let a: [f64; LANE_COUNT] = core::array::from_fn(|i| (i + 1) as f64);
        let b: [f64; LANE_COUNT] = core::array::from_fn(|i| (LANE_COUNT - i) as f64);
        unsafe {
            let result = store_arr(load_arr(&a) + load_arr(&b));
            let expected: [f64; LANE_COUNT] = core::array::from_fn(|i| a[i] + b[i]);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn sub_computes_element_wise_difference() {
        let a: [f64; LANE_COUNT] = core::array::from_fn(|i| (i + 10) as f64);
        let b: [f64; LANE_COUNT] = core::array::from_fn(|i| (i + 1) as f64);
        unsafe {
            let result = store_arr(load_arr(&a) - load_arr(&b));
            let expected: [f64; LANE_COUNT] = core::array::from_fn(|i| a[i] - b[i]);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn mul_computes_element_wise_product() {
        let a: [f64; LANE_COUNT] = core::array::from_fn(|i| (i + 1) as f64);
        let b = [2.0f64; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) * load_arr(&b));
            let expected: [f64; LANE_COUNT] = core::array::from_fn(|i| a[i] * 2.0);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn div_computes_element_wise_quotient() {
        let a: [f64; LANE_COUNT] = core::array::from_fn(|i| ((i + 1) * 2) as f64);
        let b = [2.0f64; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) / load_arr(&b));
            let expected: [f64; LANE_COUNT] = core::array::from_fn(|i| a[i] / 2.0);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn rem_computes_element_wise_remainder_with_truncation_semantics() {
        let a = [7.0f64; LANE_COUNT];
        let b = [3.0f64; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) % load_arr(&b));
            assert_eq!(result, [1.0f64; LANE_COUNT]);
        }
    }

    #[test]
    fn rem_negative_dividend_gives_negative_result() {
        let a = [-7.0f64; LANE_COUNT];
        let b = [3.0f64; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) % load_arr(&b));
            assert_eq!(result, [-1.0f64; LANE_COUNT]);
        }
    }

    #[test]
    fn bitand_operates_on_raw_bit_patterns() {
        let a = [1.0f64; LANE_COUNT];
        let mask = [-0.0f64; LANE_COUNT]; // sign bit only
        unsafe {
            let result = store_arr(load_arr(&a) & load_arr(&mask));
            assert_eq!(result, [0.0f64; LANE_COUNT]);
        }
    }

    #[test]
    fn bitor_operates_on_raw_bit_patterns() {
        let a = [1.0f64; LANE_COUNT];
        let sign = [-0.0f64; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) | load_arr(&sign));
            assert_eq!(result, [-1.0f64; LANE_COUNT]);
        }
    }
}
