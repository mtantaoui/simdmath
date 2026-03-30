//! AVX2 implementation of an 8-lane single-precision floating-point vector (`f32x8`).
//!
//! This module provides [`F32x8`], a wrapper around the AVX2 `__m256` register type,
//! which holds 8 × `f32` values processed in parallel. It implements the [`Align`],
//! [`Load`], and [`Store`] traits for aligned, unaligned, and partial (tail) load/store
//! patterns, as well as all standard arithmetic and bitwise operators.
//!
//! # Target architecture
//! This module is compiled only on `x86` and `x86_64` targets. AVX2 support must be
//! enabled at compile time (e.g. `RUSTFLAGS="-C target-feature=+avx2"`).

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

use crate::ops::simd::{Align, Load, Store};

/// Number of `f32` lanes in a single [`F32x8`] register.
pub const LANE_COUNT: usize = 8;

/// Pre-computed sign-bit masks for partial (tail) loads and stores.
///
/// `MASK[n]` has the first `n` elements set to `-1` (0xFFFFFFFF, sign bit set)
/// and the remaining elements set to `0`. Pass to `_mm256_maskload_ps` /
/// `_mm256_maskstore_ps` to activate only the first `n` lanes.
pub static MASK: [[i32; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, 0, 0],
    [-1, -1, -1, -1, -1, -1, -1, 0],
];

/// An 8-lane single-precision floating-point SIMD vector backed by `__m256`.
///
/// `size` tracks how many lanes are logically active; inactive lanes are zeroed
/// on load and masked on store so they never contaminate results.
#[derive(Copy, Clone, Debug)]
pub struct F32x8 {
    pub(crate) size: usize,
    pub(crate) elements: __m256,
}

/// Stores the mask as a sequence of `bool` values into memory.
///
impl Align<f32> for F32x8 {
    #[inline]
    fn is_aligned(ptr: *const f32) -> bool {
        let ptr = ptr as usize;

        ptr.is_multiple_of(core::mem::align_of::<__m256>())
    }
}

impl Load<f32> for F32x8 {
    type Output = Self;

    /// Loads exactly [`LANE_COUNT`] elements, choosing aligned or unaligned
    /// load automatically based on pointer alignment.
    #[inline]
    unsafe fn load(ptr: *const f32, size: usize) -> Self::Output {
        debug_assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        match F32x8::is_aligned(ptr) {
            true => unsafe { Self::load_aligned(ptr) },
            false => unsafe { Self::load_unaligned(ptr) },
        }
    }

    /// Loads [`LANE_COUNT`] elements from a 32-byte-aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and 32-byte aligned.
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Output {
        Self {
            elements: unsafe { _mm256_load_ps(ptr) },
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
            elements: unsafe { _mm256_loadu_ps(ptr) },
            size: LANE_COUNT,
        }
    }

    /// Loads `size` elements (where `size < LANE_COUNT`) using a pre-computed
    /// sign-bit mask. Inactive lanes are zeroed by `_mm256_maskload_ps`.
    ///
    /// This is the standard pattern for processing the tail of a slice that
    /// does not fill a complete 8-wide register.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for at least `size` reads.
    #[inline]
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self::Output {
        debug_assert!(size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        // Index into the pre-computed mask table; inactive lanes read as zero.
        let mask = unsafe { _mm256_loadu_si256(MASK.get_unchecked(size).as_ptr().cast()) };

        Self {
            elements: unsafe { _mm256_maskload_ps(ptr, mask) },
            size,
        }
    }

    /// Broadcasts `val` to every lane of a new `F32x8` register.
    ///
    /// Wraps `_mm256_set1_ps`. All 8 lanes receive `val`; `size` is set to
    /// [`LANE_COUNT`] since all lanes are active.
    #[inline]
    unsafe fn broadcast(val: f32) -> Self::Output {
        Self {
            size: LANE_COUNT,
            elements: unsafe { _mm256_set1_ps(val) },
        }
    }

    /// Returns a zeroed `F32x8` register (`_mm256_setzero_ps`).
    ///
    /// All 8 lanes are set to `0.0` and `size` is set to [`LANE_COUNT`].
    /// Used to initialise the accumulator in sum reductions.
    #[inline]
    unsafe fn zero() -> Self::Output {
        Self {
            size: LANE_COUNT,
            elements: unsafe { _mm256_setzero_ps() },
        }
    }
}

impl Store<f32> for F32x8 {
    type Output = Self;

    /// Stores all [`LANE_COUNT`] lanes to `ptr`, dispatching to the aligned or
    /// unaligned path based on pointer alignment — mirroring [`Load::load`] exactly.
    ///
    /// For partial (tail) vectors, call [`store_at_partial`](Self::store_at_partial)
    /// directly, just as you call [`Load::load_partial`] for partial loads.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for [`LANE_COUNT`] element writes.
    /// - `self.size` must equal [`LANE_COUNT`].
    #[inline]
    unsafe fn store_at(&self, ptr: *const f32) {
        debug_assert!(self.size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mut_ptr = ptr as *mut f32;

        match F32x8::is_aligned(ptr) {
            true => unsafe { self.store_aligned_at(mut_ptr) },
            false => unsafe { self.store_unaligned_at(mut_ptr) },
        }
    }

    /// Writes all [`LANE_COUNT`] lanes to `ptr` using a non-temporal (streaming) store,
    /// bypassing the cache. Prefer this for large, write-once buffers.
    ///
    /// # Safety
    /// `ptr` must be non-null and 32-byte aligned.
    #[inline]
    unsafe fn stream_at(&self, ptr: *mut f32) {
        unsafe { _mm256_stream_ps(ptr, self.elements) }
    }

    /// Stores all [`LANE_COUNT`] lanes to a 32-byte-aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and 32-byte aligned.
    #[inline]
    unsafe fn store_aligned_at(&self, ptr: *mut f32) {
        unsafe { _mm256_store_ps(ptr, self.elements) }
    }

    /// Stores all [`LANE_COUNT`] lanes to an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null.
    #[inline]
    unsafe fn store_unaligned_at(&self, ptr: *mut f32) {
        unsafe { _mm256_storeu_ps(ptr, self.elements) }
    }

    /// Stores `self.size` lanes (where `self.size < LANE_COUNT`) using a pre-computed
    /// sign-bit mask. Inactive lanes are not written to memory.
    ///
    /// Mirrors [`load_partial`](Load::load_partial): reuses the same `MASK` table so
    /// only the logically active lanes are touched.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for at least `self.size` writes.
    #[inline]
    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        debug_assert!(self.size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        // Index into the pre-computed mask table; inactive lanes are not written.
        let mask = unsafe { _mm256_loadu_si256(MASK.get_unchecked(self.size).as_ptr().cast()) };

        unsafe { _mm256_maskstore_ps(ptr, mask, self.elements) };
    }
}

/// Element-wise addition of two [`F32x8`] vectors (`_mm256_add_ps`).
///
/// Inactive lanes (when `size < LANE_COUNT`) are still added, but since they
/// are zeroed on load, the result in those lanes is `0.0` and will not be
/// stored back via [`store_at_partial`](Store::store_at_partial).
impl Add for F32x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_add_ps(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise addition assignment (`self += rhs`).
impl AddAssign for F32x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Element-wise subtraction of two [`F32x8`] vectors (`_mm256_sub_ps`).
impl Sub for F32x8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_sub_ps(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise subtraction assignment (`self -= rhs`).
impl SubAssign for F32x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Element-wise multiplication of two [`F32x8`] vectors (`_mm256_mul_ps`).
impl Mul for F32x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_mul_ps(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise multiplication assignment (`self *= rhs`).
impl MulAssign for F32x8 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Element-wise division of two [`F32x8`] vectors (`_mm256_div_ps`).
///
/// # Note
/// Inactive lanes are zeroed on load, so dividing two partial vectors will
/// produce `0.0 / 0.0 = NaN` in those lanes. This is harmless as long as
/// results are written back through [`store_at_partial`](Store::store_at_partial),
/// which masks out inactive lanes before writing.
impl Div for F32x8 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_div_ps(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise division assignment (`self /= rhs`).
impl DivAssign for F32x8 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Element-wise floating-point remainder using **truncation** semantics
/// (`self - trunc(self / rhs) * rhs`), matching Rust's scalar `f32 %` operator.
///
/// This differs from floor-division remainder (Python's `%` / C's `fmod` when both
/// operands are negative): the result always has the same sign as the dividend.
/// For example, `-7.0 % 3.0` yields `-1.0`, not `2.0`.
///
/// # Examples
///
/// ```rust,ignore
/// // Positive dividend:
/// let a = F32x8::load_unaligned([7.0; 8].as_ptr());
/// let b = F32x8::load_unaligned([3.0; 8].as_ptr());
/// assert_eq!(store_arr(a % b), [1.0; 8]);
///
/// // Negative dividend — result is negative (truncation, not floor):
/// let neg = F32x8::load_unaligned([-7.0; 8].as_ptr());
/// assert_eq!(store_arr(neg % b), [-1.0; 8]); // -1.0, NOT 2.0
/// ```
impl Rem for F32x8 {
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
            let div = _mm256_div_ps(self.elements, rhs.elements);
            let trunc = _mm256_round_ps(div, _MM_FROUND_TRUNC);
            let prod = _mm256_mul_ps(trunc, rhs.elements);
            Self {
                size: self.size,
                elements: _mm256_sub_ps(self.elements, prod),
            }
        }
    }
}

impl RemAssign for F32x8 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

/// Applies a bit-wise AND between two [`F32x8`] vectors, operating on the raw
/// IEEE 754 bit patterns of each `f32` lane.
///
/// Useful for masking float data bit-for-bit — for example, clearing all bits
/// in lanes by AND-ing with `0.0`, or preserving only specific bit fields.
impl BitAnd for F32x8 {
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
            F32x8 {
                size: self.size,
                elements: _mm256_and_ps(self.elements, rhs.elements),
            }
        }
    }
}

impl BitAndAssign for F32x8 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

/// Applies a bit-wise OR between two [`F32x8`] vectors, operating on the raw
/// IEEE 754 bit patterns of each `f32` lane.
///
/// Useful for combining float bit patterns without going through scalar code —
/// for example, setting the sign bit of every lane to force magnitudes negative,
/// or merging two partially-masked results.
///
/// # Examples
///
/// ```rust,ignore
/// // Set the sign bit of every lane (force all values to be negative)
/// // by OR-ing with 0x80000000 in every lane (the IEEE 754 sign bit).
/// let x       = F32x8::load_unaligned([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].as_ptr());
/// let sign_mask = F32x8::load_unaligned([-0.0; 8].as_ptr()); // -0.0 = 0x80000000
/// let negated = x | sign_mask;
/// assert_eq!(store_arr(negated), [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]);
/// ```
impl BitOr for F32x8 {
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
            F32x8 {
                size: self.size,
                elements: _mm256_or_ps(self.elements, rhs.elements),
            }
        }
    }
}

impl BitOrAssign for F32x8 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 32-byte-aligned buffer of 8 `f32` values, matching `__m256` alignment.
    #[repr(align(32))]
    struct Aligned([f32; 8]);

    /// A 32-byte-aligned buffer with one extra slot.
    /// `as_ptr().add(1)` is guaranteed to NOT be 32-byte aligned (base+4 mod 32 ≠ 0),
    /// providing a reliable unaligned pointer for dispatch tests.
    #[repr(align(32))]
    struct AlignedBuf([f32; 9]);

    // ---- Align ----------------------------------------------------------------

    #[test]
    fn is_aligned_returns_true_for_aligned_pointer() {
        let data = Aligned([0.0; 8]);
        assert!(F32x8::is_aligned(data.0.as_ptr()));
    }

    #[test]
    fn is_aligned_returns_false_for_unaligned_pointer() {
        let data = Aligned([0.0; 8]);
        // Adding 4 bytes (one f32) breaks 32-byte alignment.
        let unaligned = unsafe { data.0.as_ptr().add(1) };
        assert!(!F32x8::is_aligned(unaligned));
    }

    // ---- Load -----------------------------------------------------------------

    #[test]
    fn broadcast_fills_all_lanes_with_value() {
        unsafe {
            let v = F32x8::broadcast(3.14);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f32; LANE_COUNT];
            _mm256_storeu_ps(out.as_mut_ptr(), v.elements);
            for lane in out {
                assert!((lane - 3.14f32).abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn zero_produces_all_zero_lanes() {
        unsafe {
            let v = F32x8::zero();
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [1.0f32; LANE_COUNT];
            _mm256_storeu_ps(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [0.0f32; LANE_COUNT]);
        }
    }

    #[test]
    fn load_aligned_loads_all_lanes() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        unsafe {
            let v = F32x8::load_aligned(src.0.as_ptr());
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn load_unaligned_loads_all_lanes() {
        // One extra element at the front so ptr.add(1) is guaranteed unaligned.
        let buf = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        unsafe {
            let ptr = buf.as_ptr().add(1);
            let v = F32x8::load_unaligned(ptr);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn load_dispatches_to_aligned_path() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(
            F32x8::is_aligned(src.0.as_ptr()),
            "prerequisite: pointer must be aligned"
        );
        unsafe {
            let v = F32x8::load(src.0.as_ptr(), LANE_COUNT);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn load_dispatches_to_unaligned_path() {
        // AlignedBuf is 32-byte aligned; add(1) shifts by 4 bytes, which cannot
        // be 32-byte aligned, so the unaligned path is guaranteed to be taken.
        let buf = AlignedBuf([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        unsafe {
            let ptr = buf.0.as_ptr().add(1);
            assert!(
                !F32x8::is_aligned(ptr),
                "prerequisite: pointer must be unaligned"
            );
            let v = F32x8::load(ptr, LANE_COUNT);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn load_partial_loads_n_active_lanes_and_zeros_the_rest() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for size in 0..LANE_COUNT {
            unsafe {
                let v = F32x8::load_partial(src.as_ptr(), size);
                assert_eq!(v.size, size);
                let mut out = [0.0f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), v.elements);
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
        let src = Aligned([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut dst = Aligned([0.0; 8]);
        unsafe {
            let v = F32x8::load_aligned(src.0.as_ptr());
            v.store_aligned_at(dst.0.as_mut_ptr());
        }
        assert_eq!(dst.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn store_unaligned_writes_all_lanes() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut dst = [0.0f32; 9];
        unsafe {
            let v = F32x8::load_aligned(src.0.as_ptr());
            // Write to dst[1..] to exercise an unaligned destination.
            v.store_unaligned_at(dst.as_mut_ptr().add(1));
        }
        assert_eq!(&dst[1..], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn stream_at_writes_all_lanes() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut dst = Aligned([0.0; 8]);
        unsafe {
            let v = F32x8::load_aligned(src.0.as_ptr());
            v.stream_at(dst.0.as_mut_ptr());
            // Fence ensures the non-temporal store is visible before the read below.
            _mm_sfence();
        }
        assert_eq!(dst.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for size in 0..LANE_COUNT {
            // Sentinel value confirms inactive lanes are not touched.
            let mut dst = [-1.0f32; 8];
            unsafe {
                let v = F32x8::load_partial(src.as_ptr(), size);
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
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = [-1.0f32; 8];
        unsafe {
            let v = F32x8::load_partial(src.as_ptr(), 3);
            v.store_at_partial(dst.as_mut_ptr());
        }
        assert_eq!(&dst[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&dst[3..], &[-1.0; 5]);
    }

    #[test]
    fn store_at_dispatches_to_aligned_when_full_and_aligned() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let dst = Aligned([0.0; 8]);
        assert!(
            F32x8::is_aligned(dst.0.as_ptr()),
            "prerequisite: destination must be aligned"
        );
        unsafe {
            let v = F32x8::load_aligned(src.0.as_ptr());
            v.store_at(dst.0.as_ptr());
        }
        assert_eq!(dst.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn store_at_dispatches_to_unaligned_when_full_and_unaligned() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut dst = AlignedBuf([0.0; 9]);
        unsafe {
            let ptr = dst.0.as_mut_ptr().add(1);
            assert!(
                !F32x8::is_aligned(ptr),
                "prerequisite: pointer must be unaligned"
            );
            let v = F32x8::load_aligned(src.0.as_ptr());
            v.store_at(ptr as *const f32);
        }
        assert_eq!(&dst.0[1..], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    // ---- Arithmetic ops -------------------------------------------------------

    /// Helper: load 8 values into an `F32x8` using an unaligned pointer.
    unsafe fn load_arr(arr: &[f32; 8]) -> F32x8 {
        unsafe { F32x8::load_unaligned(arr.as_ptr()) }
    }

    /// Helper: store an `F32x8` into a `[f32; 8]`.
    unsafe fn store_arr(v: F32x8) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v.elements) };
        out
    }

    #[test]
    fn add_computes_element_wise_sum() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        unsafe {
            let result = store_arr(load_arr(&a) + load_arr(&b));
            assert_eq!(result, [9.0; 8]);
        }
    }

    #[test]
    fn add_assign_matches_add() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        unsafe {
            let expected = store_arr(load_arr(&a) + load_arr(&b));
            let mut v = load_arr(&a);
            v += load_arr(&b);
            assert_eq!(store_arr(v), expected);
        }
    }

    #[test]
    fn sub_computes_element_wise_difference() {
        let a = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        unsafe {
            let result = store_arr(load_arr(&a) - load_arr(&b));
            assert_eq!(result, [7.0, 5.0, 3.0, 1.0, -1.0, -3.0, -5.0, -7.0]);
        }
    }

    #[test]
    fn sub_assign_matches_sub() {
        let a = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        unsafe {
            let expected = store_arr(load_arr(&a) - load_arr(&b));
            let mut v = load_arr(&a);
            v -= load_arr(&b);
            assert_eq!(store_arr(v), expected);
        }
    }

    #[test]
    fn mul_computes_element_wise_product() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32; 8];
        unsafe {
            let result = store_arr(load_arr(&a) * load_arr(&b));
            assert_eq!(result, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    #[test]
    fn mul_assign_matches_mul() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32; 8];
        unsafe {
            let expected = store_arr(load_arr(&a) * load_arr(&b));
            let mut v = load_arr(&a);
            v *= load_arr(&b);
            assert_eq!(store_arr(v), expected);
        }
    }

    #[test]
    fn div_computes_element_wise_quotient() {
        let a = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let b = [2.0f32; 8];
        unsafe {
            let result = store_arr(load_arr(&a) / load_arr(&b));
            assert_eq!(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn div_assign_matches_div() {
        let a = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let b = [2.0f32; 8];
        unsafe {
            let expected = store_arr(load_arr(&a) / load_arr(&b));
            let mut v = load_arr(&a);
            v /= load_arr(&b);
            assert_eq!(store_arr(v), expected);
        }
    }

    // ---- Bitwise ops (F32x8) --------------------------------------------------
    //
    // These operate on the raw IEEE 754 bit patterns of each f32 lane.
    // Tests use well-known bit-pattern identities:
    //   -0.0  == 0x80000000  (sign bit only)
    //   x & -0.0 == 0.0 for any positive x  (clears all but the sign bit, giving +0.0)
    //   x | -0.0 == -x for any positive x   (sets the sign bit, negating the value)

    #[test]
    fn bitand_assign_matches_bitand() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let zero_mask = [0.0f32; 8];
        unsafe {
            let expected = store_arr(load_arr(&a) & load_arr(&zero_mask));
            let mut v = load_arr(&a);
            v &= load_arr(&zero_mask);
            assert_eq!(store_arr(v), expected);
        }
    }

    #[test]
    fn bitor_assign_matches_bitor() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sign_mask = [-0.0f32; 8];
        unsafe {
            let expected = store_arr(load_arr(&a) | load_arr(&sign_mask));
            let mut v = load_arr(&a);
            v |= load_arr(&sign_mask);
            assert_eq!(store_arr(v), expected);
        }
    }

    #[test]
    fn rem_negative_value_is_negative() {
        // -7.0 % 3.0 == -1.0 (truncation semantics), NOT 2.0 (floor semantics).
        let a = [-7.0f32, -10.0, -15.0, -1.0, -100.0, -3.5, -9.0, -0.5];
        let b = [3.0f32, 3.0, 4.0, 1.0, 7.0, 2.0, 4.0, 0.3];
        unsafe {
            let result = store_arr(load_arr(&a) % load_arr(&b));
            for i in 0..8 {
                assert!(
                    result[i] <= 0.0,
                    "lane {i}: negative dividend must yield non-positive remainder, got {}",
                    result[i]
                );
                let expected = a[i] % b[i];
                assert!(
                    (result[i] - expected).abs() < 1e-5,
                    "lane {i}: expected {expected}, got {}",
                    result[i]
                );
            }
        }
    }
}
