//! AVX-512F implementation of a 16-lane single-precision floating-point vector (`f32x16`).
//!
//! This module provides [`F32x16`], a wrapper around the AVX-512F `__m512` register type,
//! which holds 16 × `f32` values processed in parallel. It implements the [`Align`],
//! [`Load`], and [`Store`] traits for aligned, unaligned, and partial (tail) load/store
//! patterns, as well as all standard arithmetic and bitwise operators.
//!
//! # Partial loads and stores
//! AVX-512 uses k-registers (`__mmask16`) for lane masking — a single `u16` where each
//! bit enables the corresponding lane. This is simpler than the AVX2 sign-bit mask table:
//! the tail mask is computed inline as `(1u16 << size) - 1`.
//!
//! # Target architecture
//! This module is compiled only when `target_feature = "avx512f"` is active
//! (e.g. `RUSTFLAGS="-C target-feature=+avx512f"` or `RUSTFLAGS="-C target-cpu=native"`
//! on a Skylake-X or newer CPU).

use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

use crate::ops::simd::{Align, Load, Store};

/// Number of `f32` lanes in a single [`F32x16`] register.
pub const LANE_COUNT: usize = 16;

/// A 16-lane single-precision floating-point SIMD vector backed by `__m512`.
///
/// `size` tracks how many lanes are logically active; inactive lanes are zeroed
/// on load and masked on store so they never contaminate results.
#[derive(Copy, Clone, Debug)]
pub struct F32x16 {
    pub(crate) size: usize,
    pub(crate) elements: __m512,
}

impl Align<f32> for F32x16 {
    #[inline]
    fn is_aligned(ptr: *const f32) -> bool {
        (ptr as usize).is_multiple_of(core::mem::align_of::<__m512>())
    }
}

impl Load<f32> for F32x16 {
    type Output = Self;

    /// Loads exactly [`LANE_COUNT`] elements, choosing aligned or unaligned
    /// load automatically based on pointer alignment.
    #[inline]
    unsafe fn load(ptr: *const f32, size: usize) -> Self::Output {
        debug_assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        match F32x16::is_aligned(ptr) {
            true => unsafe { Self::load_aligned(ptr) },
            false => unsafe { Self::load_unaligned(ptr) },
        }
    }

    /// Loads [`LANE_COUNT`] elements from a 64-byte-aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and 64-byte aligned.
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Output {
        Self {
            elements: unsafe { _mm512_load_ps(ptr) },
            size: LANE_COUNT,
        }
    }

    /// Loads [`LANE_COUNT`] elements from an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null.
    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Output {
        Self {
            elements: unsafe { _mm512_loadu_ps(ptr) },
            size: LANE_COUNT,
        }
    }

    /// Loads `size` elements (where `size < LANE_COUNT`) using a `__mmask16`
    /// bitmask. Inactive lanes are zeroed by `_mm512_maskz_loadu_ps`.
    ///
    /// The mask is computed inline as `(1u16 << size) - 1`, setting exactly
    /// `size` low bits — simpler than the AVX2 sign-bit lookup table approach.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for at least `size` reads.
    #[inline]
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self::Output {
        debug_assert!(size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mask: __mmask16 = (1u16 << size).wrapping_sub(1);
        Self {
            elements: unsafe { _mm512_maskz_loadu_ps(mask, ptr) },
            size,
        }
    }

    /// Broadcasts `val` to every lane of a new `F32x16` register.
    ///
    /// Wraps `_mm512_set1_ps`. All 16 lanes receive `val`; `size` is set to
    /// [`LANE_COUNT`] since all lanes are active.
    #[inline]
    unsafe fn broadcast(val: f32) -> Self::Output {
        Self {
            size: LANE_COUNT,
            elements: unsafe { _mm512_set1_ps(val) },
        }
    }

    /// Returns a zeroed `F32x16` register (`_mm512_setzero_ps`).
    ///
    /// All 16 lanes are set to `0.0` and `size` is set to [`LANE_COUNT`].
    /// Used to initialise the accumulator in sum reductions.
    #[inline]
    unsafe fn zero() -> Self::Output {
        Self {
            size: LANE_COUNT,
            elements: unsafe { _mm512_setzero_ps() },
        }
    }
}

impl Store<f32> for F32x16 {
    type Output = Self;

    /// Stores all [`LANE_COUNT`] lanes to `ptr`, dispatching to the aligned or
    /// unaligned path based on pointer alignment.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for [`LANE_COUNT`] element writes.
    #[inline]
    unsafe fn store_at(&self, ptr: *const f32) {
        debug_assert!(self.size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mut_ptr = ptr as *mut f32;
        match F32x16::is_aligned(ptr) {
            true => unsafe { self.store_aligned_at(mut_ptr) },
            false => unsafe { self.store_unaligned_at(mut_ptr) },
        }
    }

    /// Writes all [`LANE_COUNT`] lanes to `ptr` using a non-temporal (streaming) store,
    /// bypassing the cache. Prefer this for large, write-once buffers.
    ///
    /// # Safety
    /// `ptr` must be non-null and 64-byte aligned.
    #[inline]
    unsafe fn stream_at(&self, ptr: *mut f32) {
        unsafe { _mm512_stream_ps(ptr, self.elements) }
    }

    /// Stores all [`LANE_COUNT`] lanes to a 64-byte-aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and 64-byte aligned.
    #[inline]
    unsafe fn store_aligned_at(&self, ptr: *mut f32) {
        unsafe { _mm512_store_ps(ptr, self.elements) }
    }

    /// Stores all [`LANE_COUNT`] lanes to an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null.
    #[inline]
    unsafe fn store_unaligned_at(&self, ptr: *mut f32) {
        unsafe { _mm512_storeu_ps(ptr, self.elements) }
    }

    /// Stores `self.size` lanes (where `self.size < LANE_COUNT`) using a `__mmask16`
    /// bitmask. Inactive lanes are not written to memory.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for at least `self.size` writes.
    #[inline]
    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        debug_assert!(self.size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mask: __mmask16 = (1u16 << self.size).wrapping_sub(1);
        unsafe { _mm512_mask_storeu_ps(ptr, mask, self.elements) }
    }
}

/// Element-wise addition of two [`F32x16`] vectors (`_mm512_add_ps`).
impl Add for F32x16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm512_add_ps(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise addition assignment (`self += rhs`).
impl AddAssign for F32x16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Element-wise subtraction of two [`F32x16`] vectors (`_mm512_sub_ps`).
impl Sub for F32x16 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm512_sub_ps(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise subtraction assignment (`self -= rhs`).
impl SubAssign for F32x16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Element-wise multiplication of two [`F32x16`] vectors (`_mm512_mul_ps`).
impl Mul for F32x16 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm512_mul_ps(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise multiplication assignment (`self *= rhs`).
impl MulAssign for F32x16 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Element-wise division of two [`F32x16`] vectors (`_mm512_div_ps`).
///
/// # Note
/// Inactive lanes are zeroed on load, so dividing two partial vectors will
/// produce `0.0 / 0.0 = NaN` in those lanes. This is harmless as long as
/// results are written back through [`store_at_partial`](Store::store_at_partial),
/// which masks out inactive lanes before writing.
impl Div for F32x16 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm512_div_ps(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise division assignment (`self /= rhs`).
impl DivAssign for F32x16 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Element-wise floating-point remainder using **truncation** semantics
/// (`self - trunc(self / rhs) * rhs`), matching Rust's scalar `f32 %` operator.
///
/// Truncation is performed via `_mm512_cvttps_epi32` (double-t converts with
/// truncation toward zero) followed by `_mm512_cvtepi32_ps`. This is correct
/// for quotients within the i32 range; for values beyond 2³¹ the remainder is
/// typically exact anyway.
impl Rem for F32x16 {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "mismatched lane counts: lhs has {} lanes, rhs has {} lanes",
            self.size,
            rhs.size
        );

        unsafe {
            let div = _mm512_div_ps(self.elements, rhs.elements);
            // _mm512_cvttps_epi32: convert with truncation toward zero (double-t)
            let trunc_i = _mm512_cvttps_epi32(div);
            let trunc = _mm512_cvtepi32_ps(trunc_i);
            let prod = _mm512_mul_ps(trunc, rhs.elements);
            Self {
                size: self.size,
                elements: _mm512_sub_ps(self.elements, prod),
            }
        }
    }
}

/// Element-wise remainder assignment (`self %= rhs`).
impl RemAssign for F32x16 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

/// Applies a bitwise AND between two [`F32x16`] vectors on the raw IEEE 754
/// bit patterns. Uses integer cast through `__m512i` since `_mm512_and_ps`
/// is in avx512dq; only avx512f is required here.
impl BitAnd for F32x16 {
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
            let a_i = _mm512_castps_si512(self.elements);
            let b_i = _mm512_castps_si512(rhs.elements);
            F32x16 {
                size: self.size,
                elements: _mm512_castsi512_ps(_mm512_and_epi32(a_i, b_i)),
            }
        }
    }
}

/// Element-wise bitwise AND assignment (`self &= rhs`).
impl BitAndAssign for F32x16 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

/// Applies a bitwise OR between two [`F32x16`] vectors on the raw IEEE 754
/// bit patterns. Uses integer cast through `__m512i` since `_mm512_or_ps`
/// is in avx512dq; only avx512f is required here.
impl BitOr for F32x16 {
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
            let a_i = _mm512_castps_si512(self.elements);
            let b_i = _mm512_castps_si512(rhs.elements);
            F32x16 {
                size: self.size,
                elements: _mm512_castsi512_ps(_mm512_or_epi32(a_i, b_i)),
            }
        }
    }
}

/// Element-wise bitwise OR assignment (`self |= rhs`).
impl BitOrAssign for F32x16 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64-byte-aligned buffer of 16 `f32` values, matching `__m512` alignment.
    #[repr(align(64))]
    struct Aligned([f32; 16]);

    /// A 64-byte-aligned buffer with one extra slot.
    /// `as_ptr().add(1)` is guaranteed to NOT be 64-byte aligned.
    #[repr(align(64))]
    struct AlignedBuf([f32; 17]);

    // ---- Align ----------------------------------------------------------------

    #[test]
    fn is_aligned_returns_true_for_aligned_pointer() {
        let data = Aligned([0.0; 16]);
        assert!(F32x16::is_aligned(data.0.as_ptr()));
    }

    #[test]
    fn is_aligned_returns_false_for_unaligned_pointer() {
        let data = Aligned([0.0; 16]);
        let unaligned = unsafe { data.0.as_ptr().add(1) };
        assert!(!F32x16::is_aligned(unaligned));
    }

    // ---- Load -----------------------------------------------------------------

    #[test]
    fn broadcast_fills_all_lanes_with_value() {
        unsafe {
            let v = F32x16::broadcast(3.14);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f32; LANE_COUNT];
            _mm512_storeu_ps(out.as_mut_ptr(), v.elements);
            for lane in out {
                assert!((lane - 3.14f32).abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn zero_produces_all_zero_lanes() {
        unsafe {
            let v = F32x16::zero();
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [1.0f32; LANE_COUNT];
            _mm512_storeu_ps(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [0.0f32; LANE_COUNT]);
        }
    }

    #[test]
    fn load_aligned_loads_all_lanes() {
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f32));
        unsafe {
            let v = F32x16::load_aligned(src.0.as_ptr());
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f32; LANE_COUNT];
            _mm512_storeu_ps(out.as_mut_ptr(), v.elements);
            assert_eq!(out, src.0);
        }
    }

    #[test]
    fn load_unaligned_loads_all_lanes() {
        let mut buf = [0.0f32; 17];
        for i in 0..16 {
            buf[i + 1] = (i + 1) as f32;
        }
        unsafe {
            let ptr = buf.as_ptr().add(1);
            let v = F32x16::load_unaligned(ptr);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f32; LANE_COUNT];
            _mm512_storeu_ps(out.as_mut_ptr(), v.elements);
            assert_eq!(out, core::array::from_fn(|i| (i + 1) as f32));
        }
    }

    #[test]
    fn load_dispatches_to_aligned_path() {
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f32));
        assert!(F32x16::is_aligned(src.0.as_ptr()));
        unsafe {
            let v = F32x16::load(src.0.as_ptr(), LANE_COUNT);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f32; LANE_COUNT];
            _mm512_storeu_ps(out.as_mut_ptr(), v.elements);
            assert_eq!(out, src.0);
        }
    }

    #[test]
    fn load_dispatches_to_unaligned_path() {
        let buf = AlignedBuf(core::array::from_fn(|i| i as f32));
        let ptr = unsafe { buf.0.as_ptr().add(1) };
        assert!(!F32x16::is_aligned(ptr));
        unsafe {
            let v = F32x16::load(ptr, LANE_COUNT);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f32; LANE_COUNT];
            _mm512_storeu_ps(out.as_mut_ptr(), v.elements);
            assert_eq!(out, core::array::from_fn(|i| (i + 1) as f32));
        }
    }

    #[test]
    fn load_partial_loads_n_active_lanes_and_zeros_the_rest() {
        let src: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
        for size in 0..LANE_COUNT {
            unsafe {
                let v = F32x16::load_partial(src.as_ptr(), size);
                assert_eq!(v.size, size);
                let mut out = [0.0f32; LANE_COUNT];
                _mm512_storeu_ps(out.as_mut_ptr(), v.elements);
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
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f32));
        let mut dst = Aligned([0.0; 16]);
        unsafe {
            let v = F32x16::load_aligned(src.0.as_ptr());
            v.store_aligned_at(dst.0.as_mut_ptr());
        }
        assert_eq!(dst.0, src.0);
    }

    #[test]
    fn store_unaligned_writes_all_lanes() {
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f32));
        let mut dst = [0.0f32; 17];
        unsafe {
            let v = F32x16::load_aligned(src.0.as_ptr());
            v.store_unaligned_at(dst.as_mut_ptr().add(1));
        }
        assert_eq!(&dst[1..], &src.0[..]);
    }

    #[test]
    fn stream_at_writes_all_lanes() {
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f32));
        let mut dst = Aligned([0.0; 16]);
        unsafe {
            let v = F32x16::load_aligned(src.0.as_ptr());
            v.stream_at(dst.0.as_mut_ptr());
            _mm_sfence();
        }
        assert_eq!(dst.0, src.0);
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
        let src: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
        for size in 0..LANE_COUNT {
            let mut dst = [-1.0f32; LANE_COUNT];
            unsafe {
                let v = F32x16::load_partial(src.as_ptr(), size);
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

    #[test]
    fn store_at_partial_writes_active_lanes_after_partial_load() {
        let src: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
        let mut dst = [-1.0f32; LANE_COUNT];
        unsafe {
            let v = F32x16::load_partial(src.as_ptr(), 5);
            v.store_at_partial(dst.as_mut_ptr());
        }
        assert_eq!(&dst[..5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(&dst[5..], &[-1.0; 11]);
    }

    #[test]
    fn store_at_dispatches_to_aligned_when_full_and_aligned() {
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f32));
        let dst = Aligned([0.0; 16]);
        assert!(F32x16::is_aligned(dst.0.as_ptr()));
        unsafe {
            let v = F32x16::load_aligned(src.0.as_ptr());
            v.store_at(dst.0.as_ptr());
        }
        assert_eq!(dst.0, src.0);
    }

    #[test]
    fn store_at_dispatches_to_unaligned_when_full_and_unaligned() {
        let src = Aligned(core::array::from_fn(|i| (i + 1) as f32));
        let mut dst = AlignedBuf([0.0; 17]);
        unsafe {
            let ptr = dst.0.as_mut_ptr().add(1);
            assert!(!F32x16::is_aligned(ptr));
            let v = F32x16::load_aligned(src.0.as_ptr());
            v.store_at(ptr as *const f32);
        }
        assert_eq!(&dst.0[1..], &src.0[..]);
    }

    // ---- Arithmetic ops -------------------------------------------------------

    unsafe fn load_arr(arr: &[f32; LANE_COUNT]) -> F32x16 {
        unsafe { F32x16::load_unaligned(arr.as_ptr()) }
    }

    unsafe fn store_arr(v: F32x16) -> [f32; LANE_COUNT] {
        let mut out = [0.0f32; LANE_COUNT];
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), v.elements) };
        out
    }

    #[test]
    fn add_computes_element_wise_sum() {
        let a: [f32; LANE_COUNT] = core::array::from_fn(|i| (i + 1) as f32);
        let b: [f32; LANE_COUNT] = core::array::from_fn(|i| (LANE_COUNT - i) as f32);
        unsafe {
            let va = load_arr(&a);
            let vb = load_arr(&b);
            let result = store_arr(va + vb);
            let expected: [f32; LANE_COUNT] = core::array::from_fn(|i| a[i] + b[i]);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn sub_computes_element_wise_difference() {
        let a: [f32; LANE_COUNT] = core::array::from_fn(|i| (i + 10) as f32);
        let b: [f32; LANE_COUNT] = core::array::from_fn(|i| (i + 1) as f32);
        unsafe {
            let result = store_arr(load_arr(&a) - load_arr(&b));
            let expected: [f32; LANE_COUNT] = core::array::from_fn(|i| a[i] - b[i]);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn mul_computes_element_wise_product() {
        let a: [f32; LANE_COUNT] = core::array::from_fn(|i| (i + 1) as f32);
        let b = [2.0f32; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) * load_arr(&b));
            let expected: [f32; LANE_COUNT] = core::array::from_fn(|i| a[i] * 2.0);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn div_computes_element_wise_quotient() {
        let a: [f32; LANE_COUNT] = core::array::from_fn(|i| ((i + 1) * 2) as f32);
        let b = [2.0f32; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) / load_arr(&b));
            let expected: [f32; LANE_COUNT] = core::array::from_fn(|i| a[i] / 2.0);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn rem_computes_element_wise_remainder_with_truncation_semantics() {
        let a = [7.0f32; LANE_COUNT];
        let b = [3.0f32; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) % load_arr(&b));
            assert_eq!(result, [1.0f32; LANE_COUNT]);
        }
    }

    #[test]
    fn rem_negative_dividend_gives_negative_result() {
        let a = [-7.0f32; LANE_COUNT];
        let b = [3.0f32; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) % load_arr(&b));
            assert_eq!(result, [-1.0f32; LANE_COUNT]);
        }
    }

    #[test]
    fn bitand_operates_on_raw_bit_patterns() {
        // AND with -0.0 (sign bit 1, rest 0) clears all bits except sign.
        let a = [1.0f32; LANE_COUNT];
        let mask = [-0.0f32; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) & load_arr(&mask));
            assert_eq!(result, [0.0f32; LANE_COUNT]);
        }
    }

    #[test]
    fn bitor_operates_on_raw_bit_patterns() {
        // OR with -0.0 sets the sign bit of every lane (forces negative).
        let a = [1.0f32; LANE_COUNT];
        let sign = [-0.0f32; LANE_COUNT];
        unsafe {
            let result = store_arr(load_arr(&a) | load_arr(&sign));
            assert_eq!(result, [-1.0f32; LANE_COUNT]);
        }
    }
}
