//! AVX2 implementation of a 4-lane double-precision floating-point vector (`f64x4`).
//!
//! This module provides [`F64x4`], a wrapper around the AVX2 `__m256d` register type,
//! which holds 4 × `f64` values processed in parallel. It implements the [`Align`],
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

/// Number of `f64` lanes in a single [`F64x4`] register.
pub const LANE_COUNT: usize = 4;

/// Pre-computed sign-bit masks for partial (tail) loads and stores.
///
/// Pre-computed load/store masks for partial (tail) operations of 0–3 lanes.
///
/// `_mm256_maskload_pd` / `_mm256_maskstore_pd` inspect the **sign bit of each
/// 64-bit lane** (bit 63). Using `i64` — one entry per `f64` lane — is the
/// natural representation: `-1i64` = `0xFFFFFFFF_FFFFFFFF` (sign bit set = active),
/// `0i64` = inactive.
///
/// Only 4 rows are needed (0–3 active lanes). A full-register tail (size == LANE_COUNT)
/// is handled by `load` / `store_at`, not by the partial variants.
static MASK: [[i64; 4]; 4] = [
    [0, 0, 0, 0],    // 0 active lanes
    [-1, 0, 0, 0],   // 1 active lane
    [-1, -1, 0, 0],  // 2 active lanes
    [-1, -1, -1, 0], // 3 active lanes
];

/// A 4-lane double-precision floating-point SIMD vector backed by `__m256d`.
///
/// `size` tracks how many lanes are logically active; inactive lanes are zeroed
/// on load and masked on store so they never contaminate results.
#[derive(Copy, Clone, Debug)]
pub struct F64x4 {
    pub(crate) size: usize,
    pub(crate) elements: __m256d,
}

impl Align<f64> for F64x4 {
    #[inline]
    fn is_aligned(ptr: *const f64) -> bool {
        let ptr = ptr as usize;

        ptr.is_multiple_of(core::mem::align_of::<__m256d>())
    }
}

impl Load<f64> for F64x4 {
    type Output = Self;

    /// Loads exactly [`LANE_COUNT`] elements, choosing aligned or unaligned
    /// load automatically based on pointer alignment.
    #[inline]
    unsafe fn load(ptr: *const f64, size: usize) -> Self::Output {
        debug_assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        match F64x4::is_aligned(ptr) {
            true => unsafe { Self::load_aligned(ptr) },
            false => unsafe { Self::load_unaligned(ptr) },
        }
    }

    /// Loads [`LANE_COUNT`] elements from a 32-byte-aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and 32-byte aligned.
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self::Output {
        Self {
            elements: unsafe { _mm256_load_pd(ptr) },
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
            elements: unsafe { _mm256_loadu_pd(ptr) },
            size: LANE_COUNT,
        }
    }

    /// Loads `size` elements (where `size < LANE_COUNT`) using a pre-computed
    /// sign-bit mask. Inactive lanes are zeroed by `_mm256_maskload_pd`.
    ///
    /// This is the standard pattern for processing the tail of a slice that
    /// does not fill a complete 4-wide register.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for at least `size` reads.
    #[inline]
    unsafe fn load_partial(ptr: *const f64, size: usize) -> Self::Output {
        debug_assert!(size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        // Index into the pre-computed mask table; inactive lanes read as zero.
        let mask = unsafe { _mm256_loadu_si256(MASK.get_unchecked(size).as_ptr().cast()) };

        Self {
            elements: unsafe { _mm256_maskload_pd(ptr, mask) },
            size,
        }
    }

    /// Broadcasts `val` to every lane of a new `F64x4` register.
    ///
    /// Wraps `_mm256_set1_pd`. All 4 lanes receive `val`; `size` is set to
    /// [`LANE_COUNT`] since all lanes are active.
    #[inline]
    unsafe fn broadcast(val: f64) -> Self::Output {
        Self {
            size: LANE_COUNT,
            elements: unsafe { _mm256_set1_pd(val) },
        }
    }

    /// Returns a zeroed `F64x4` register (`_mm256_setzero_pd`).
    ///
    /// All 4 lanes are set to `0.0` and `size` is set to [`LANE_COUNT`].
    /// Used to initialise the accumulator in sum reductions.
    #[inline]
    unsafe fn zero() -> Self::Output {
        Self {
            size: LANE_COUNT,
            elements: unsafe { _mm256_setzero_pd() },
        }
    }
}

impl Store<f64> for F64x4 {
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
    unsafe fn store_at(&self, ptr: *const f64) {
        debug_assert!(self.size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mut_ptr = ptr as *mut f64;

        match F64x4::is_aligned(ptr) {
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
    unsafe fn stream_at(&self, ptr: *mut f64) {
        unsafe { _mm256_stream_pd(ptr, self.elements) }
    }

    /// Stores all [`LANE_COUNT`] lanes to a 32-byte-aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and 32-byte aligned.
    #[inline]
    unsafe fn store_aligned_at(&self, ptr: *mut f64) {
        unsafe { _mm256_store_pd(ptr, self.elements) }
    }

    /// Stores all [`LANE_COUNT`] lanes to an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null.
    #[inline]
    unsafe fn store_unaligned_at(&self, ptr: *mut f64) {
        unsafe { _mm256_storeu_pd(ptr, self.elements) }
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
    unsafe fn store_at_partial(&self, ptr: *mut f64) {
        debug_assert!(self.size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        // Index into the pre-computed mask table; inactive lanes are not written.
        let mask = unsafe { _mm256_loadu_si256(MASK.get_unchecked(self.size).as_ptr().cast()) };

        unsafe { _mm256_maskstore_pd(ptr, mask, self.elements) };
    }
}

/// Element-wise addition of two [`F64x4`] vectors (`_mm256_add_pd`).
///
/// Inactive lanes (when `size < LANE_COUNT`) are still added, but since they
/// are zeroed on load, the result in those lanes is `0.0` and will not be
/// stored back via [`store_at_partial`](Store::store_at_partial).
impl Add for F64x4 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_add_pd(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise addition assignment (`self += rhs`).
impl AddAssign for F64x4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Element-wise subtraction of two [`F64x4`] vectors (`_mm256_sub_pd`).
impl Sub for F64x4 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_sub_pd(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise subtraction assignment (`self -= rhs`).
impl SubAssign for F64x4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Element-wise multiplication of two [`F64x4`] vectors (`_mm256_mul_pd`).
impl Mul for F64x4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_mul_pd(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise multiplication assignment (`self *= rhs`).
impl MulAssign for F64x4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Element-wise division of two [`F64x4`] vectors (`_mm256_div_pd`).
///
/// # Note
/// Inactive lanes are zeroed on load, so dividing two partial vectors will
/// produce `0.0 / 0.0 = NaN` in those lanes. This is harmless as long as
/// results are written back through [`store_at_partial`](Store::store_at_partial),
/// which masks out inactive lanes before writing.
impl Div for F64x4 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { _mm256_div_pd(self.elements, rhs.elements) },
        }
    }
}

/// Element-wise division assignment (`self /= rhs`).
impl DivAssign for F64x4 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Element-wise floating-point remainder using **truncation** semantics
/// (`self - trunc(self / rhs) * rhs`), matching Rust's scalar `f64 %` operator.
///
/// This differs from floor-division remainder (Python's `%` / C's `fmod` when both
/// operands are negative): the result always has the same sign as the dividend.
/// For example, `-7.0 % 3.0` yields `-1.0`, not `2.0`.
///
/// # Examples
///
/// ```rust,ignore
/// // Positive dividend:
/// let a = F64x4::load_unaligned([7.0; 4].as_ptr());
/// let b = F64x4::load_unaligned([3.0; 4].as_ptr());
/// assert_eq!(store_arr(a % b), [1.0; 4]);
///
/// // Negative dividend — result is negative (truncation, not floor):
/// let neg = F64x4::load_unaligned([-7.0; 4].as_ptr());
/// assert_eq!(store_arr(neg % b), [-1.0; 4]); // -1.0, NOT 2.0
/// ```
impl Rem for F64x4 {
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
            let div = _mm256_div_pd(self.elements, rhs.elements);
            let trunc = _mm256_round_pd(div, _MM_FROUND_TRUNC);
            let prod = _mm256_mul_pd(trunc, rhs.elements);
            Self {
                size: self.size,
                elements: _mm256_sub_pd(self.elements, prod),
            }
        }
    }
}

impl RemAssign for F64x4 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

/// Applies a bit-wise AND between two [`F64x4`] vectors, operating on the raw
/// IEEE 754 bit patterns of each `f64` lane.
///
/// Useful for masking float data bit-for-bit — for example, clearing all bits
/// in lanes by AND-ing with `0.0`, or preserving only specific bit fields.
impl BitAnd for F64x4 {
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
            F64x4 {
                size: self.size,
                elements: _mm256_and_pd(self.elements, rhs.elements),
            }
        }
    }
}

impl BitAndAssign for F64x4 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

/// Applies a bit-wise OR between two [`F64x4`] vectors, operating on the raw
/// IEEE 754 bit patterns of each `f64` lane.
///
/// Useful for combining float bit patterns without going through scalar code —
/// for example, setting the sign bit of every lane to force magnitudes negative,
/// or merging two partially-masked results.
///
/// # Examples
///
/// ```rust,ignore
/// // Set the sign bit of every lane (force all values to be negative)
/// // by OR-ing with 0x8000000000000000 in every lane (the IEEE 754 sign bit).
/// let x         = F64x4::load_unaligned([1.0, 2.0, 3.0, 4.0].as_ptr());
/// let sign_mask = F64x4::load_unaligned([-0.0; 4].as_ptr()); // -0.0 = 0x8000000000000000
/// let negated   = x | sign_mask;
/// assert_eq!(store_arr(negated), [-1.0, -2.0, -3.0, -4.0]);
/// ```
impl BitOr for F64x4 {
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
            F64x4 {
                size: self.size,
                elements: _mm256_or_pd(self.elements, rhs.elements),
            }
        }
    }
}

impl BitOrAssign for F64x4 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 32-byte-aligned buffer of 4 `f64` values, matching `__m256d` alignment.
    #[repr(align(32))]
    struct Aligned([f64; 4]);

    /// A 32-byte-aligned buffer with one extra slot.
    /// `as_ptr().add(1)` is guaranteed to NOT be 32-byte aligned (base+8 mod 32 != 0),
    /// providing a reliable unaligned pointer for dispatch tests.
    #[repr(align(32))]
    struct AlignedBuf([f64; 5]);

    // ---- Align ----------------------------------------------------------------

    #[test]
    fn is_aligned_returns_true_for_aligned_pointer() {
        let data = Aligned([0.0; 4]);
        assert!(F64x4::is_aligned(data.0.as_ptr()));
    }

    #[test]
    fn is_aligned_returns_false_for_unaligned_pointer() {
        let data = Aligned([0.0; 4]);
        // Adding 8 bytes (one f64) breaks 32-byte alignment.
        let unaligned = unsafe { data.0.as_ptr().add(1) };
        assert!(!F64x4::is_aligned(unaligned));
    }

    // ---- Load -----------------------------------------------------------------

    #[test]
    fn load_aligned_loads_all_lanes() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0]);
        unsafe {
            let v = F64x4::load_aligned(src.0.as_ptr());
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn load_unaligned_loads_all_lanes() {
        // One extra element at the front so ptr.add(1) is guaranteed unaligned.
        let buf = [0.0f64, 1.0, 2.0, 3.0, 4.0];
        unsafe {
            let ptr = buf.as_ptr().add(1);
            let v = F64x4::load_unaligned(ptr);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn load_dispatches_to_aligned_path() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0]);
        assert!(
            F64x4::is_aligned(src.0.as_ptr()),
            "prerequisite: pointer must be aligned"
        );
        unsafe {
            let v = F64x4::load(src.0.as_ptr(), LANE_COUNT);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn load_dispatches_to_unaligned_path() {
        // AlignedBuf is 32-byte aligned; add(1) shifts by 8 bytes, which cannot
        // be 32-byte aligned, so the unaligned path is guaranteed to be taken.
        let buf = AlignedBuf([0.0, 1.0, 2.0, 3.0, 4.0]);
        unsafe {
            let ptr = buf.0.as_ptr().add(1);
            assert!(
                !F64x4::is_aligned(ptr),
                "prerequisite: pointer must be unaligned"
            );
            let v = F64x4::load(ptr, LANE_COUNT);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn load_partial_loads_n_active_lanes_and_zeros_the_rest() {
        let src = [1.0f64, 2.0, 3.0, 4.0];
        for size in 0..LANE_COUNT {
            unsafe {
                let v = F64x4::load_partial(src.as_ptr(), size);
                assert_eq!(v.size, size);
                let mut out = [0.0f64; 4];
                _mm256_storeu_pd(out.as_mut_ptr(), v.elements);
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
        let src = Aligned([1.0, 2.0, 3.0, 4.0]);
        let mut dst = Aligned([0.0; 4]);
        unsafe {
            let v = F64x4::load_aligned(src.0.as_ptr());
            v.store_aligned_at(dst.0.as_mut_ptr());
        }
        assert_eq!(dst.0, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn store_unaligned_writes_all_lanes() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0]);
        let mut dst = [0.0f64; 5];
        unsafe {
            let v = F64x4::load_aligned(src.0.as_ptr());
            // Write to dst[1..] to exercise an unaligned destination.
            v.store_unaligned_at(dst.as_mut_ptr().add(1));
        }
        assert_eq!(&dst[1..], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn broadcast_fills_all_lanes_with_value() {
        unsafe {
            let v = F64x4::broadcast(2.718);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; LANE_COUNT];
            _mm256_storeu_pd(out.as_mut_ptr(), v.elements);
            for lane in out {
                assert!((lane - 2.718f64).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn zero_produces_all_zero_lanes() {
        unsafe {
            let v = F64x4::zero();
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [1.0f64; LANE_COUNT];
            _mm256_storeu_pd(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [0.0f64; LANE_COUNT]);
        }
    }

    #[test]
    fn stream_at_writes_all_lanes() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0]);
        let mut dst = Aligned([0.0; 4]);
        unsafe {
            let v = F64x4::load_aligned(src.0.as_ptr());
            v.stream_at(dst.0.as_mut_ptr());
            // Fence ensures the non-temporal store is visible before the read below.
            _mm_sfence();
        }
        assert_eq!(dst.0, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
        let src = [1.0f64, 2.0, 3.0, 4.0];
        for size in 0..LANE_COUNT {
            // Sentinel value confirms inactive lanes are not touched.
            let mut dst = [-1.0f64; 4];
            unsafe {
                let v = F64x4::load_partial(src.as_ptr(), size);
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
        let src = [1.0f64, 2.0, 3.0, 4.0];
        let mut dst = [-1.0f64; 4];
        unsafe {
            let v = F64x4::load_partial(src.as_ptr(), 3);
            v.store_at_partial(dst.as_mut_ptr());
        }
        assert_eq!(&dst[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&dst[3..], &[-1.0; 1]);
    }

    #[test]
    fn store_at_dispatches_to_aligned_when_full_and_aligned() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0]);
        let dst = Aligned([0.0; 4]);
        assert!(
            F64x4::is_aligned(dst.0.as_ptr()),
            "prerequisite: destination must be aligned"
        );
        unsafe {
            let v = F64x4::load_aligned(src.0.as_ptr());
            v.store_at(dst.0.as_ptr());
        }
        assert_eq!(dst.0, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn store_at_dispatches_to_unaligned_when_full_and_unaligned() {
        let src = Aligned([1.0, 2.0, 3.0, 4.0]);
        let mut dst = AlignedBuf([0.0; 5]);
        unsafe {
            let ptr = dst.0.as_mut_ptr().add(1);
            assert!(
                !F64x4::is_aligned(ptr),
                "prerequisite: pointer must be unaligned"
            );
            let v = F64x4::load_aligned(src.0.as_ptr());
            v.store_at(ptr as *const f64);
        }
        assert_eq!(&dst.0[1..], &[1.0, 2.0, 3.0, 4.0]);
    }

    // ---- Arithmetic ops -------------------------------------------------------

    /// Helper: load 4 values into an `F64x4` using an unaligned pointer.
    unsafe fn load_arr(arr: &[f64; 4]) -> F64x4 {
        unsafe { F64x4::load_unaligned(arr.as_ptr()) }
    }

    /// Helper: store an `F64x4` into a `[f64; 4]`.
    unsafe fn store_arr(v: F64x4) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), v.elements) };
        out
    }

    #[test]
    fn add_computes_element_wise_sum() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [4.0f64, 3.0, 2.0, 1.0];
        unsafe {
            let result = store_arr(load_arr(&a) + load_arr(&b));
            assert_eq!(result, [5.0; 4]);
        }
    }

    #[test]
    fn add_assign_matches_add() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [4.0f64, 3.0, 2.0, 1.0];
        unsafe {
            let expected = store_arr(load_arr(&a) + load_arr(&b));
            let mut v = load_arr(&a);
            v += load_arr(&b);
            assert_eq!(store_arr(v), expected);
        }
    }

    #[test]
    fn sub_computes_element_wise_difference() {
        let a = [4.0f64, 3.0, 2.0, 1.0];
        let b = [1.0f64, 2.0, 3.0, 4.0];
        unsafe {
            let result = store_arr(load_arr(&a) - load_arr(&b));
            assert_eq!(result, [3.0, 1.0, -1.0, -3.0]);
        }
    }

    #[test]
    fn sub_assign_matches_sub() {
        let a = [4.0f64, 3.0, 2.0, 1.0];
        let b = [1.0f64, 2.0, 3.0, 4.0];
        unsafe {
            let expected = store_arr(load_arr(&a) - load_arr(&b));
            let mut v = load_arr(&a);
            v -= load_arr(&b);
            assert_eq!(store_arr(v), expected);
        }
    }

    #[test]
    fn mul_computes_element_wise_product() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [2.0f64; 4];
        unsafe {
            let result = store_arr(load_arr(&a) * load_arr(&b));
            assert_eq!(result, [2.0, 4.0, 6.0, 8.0]);
        }
    }

    #[test]
    fn mul_assign_matches_mul() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [2.0f64; 4];
        unsafe {
            let expected = store_arr(load_arr(&a) * load_arr(&b));
            let mut v = load_arr(&a);
            v *= load_arr(&b);
            assert_eq!(store_arr(v), expected);
        }
    }

    #[test]
    fn div_computes_element_wise_quotient() {
        let a = [2.0f64, 4.0, 6.0, 8.0];
        let b = [2.0f64; 4];
        unsafe {
            let result = store_arr(load_arr(&a) / load_arr(&b));
            assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn div_assign_matches_div() {
        let a = [2.0f64, 4.0, 6.0, 8.0];
        let b = [2.0f64; 4];
        unsafe {
            let expected = store_arr(load_arr(&a) / load_arr(&b));
            let mut v = load_arr(&a);
            v /= load_arr(&b);
            assert_eq!(store_arr(v), expected);
        }
    }

    // ---- Bitwise ops (F64x4) --------------------------------------------------
    //
    // These operate on the raw IEEE 754 bit patterns of each f64 lane.
    // Tests use well-known bit-pattern identities:
    //   -0.0  == 0x8000000000000000  (sign bit only)
    //   x & -0.0 == 0.0 for any positive x  (clears all but the sign bit, giving +0.0)
    //   x | -0.0 == -x for any positive x   (sets the sign bit, negating the value)

    #[test]
    fn bitand_assign_matches_bitand() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let zero_mask = [0.0f64; 4];
        unsafe {
            let expected = store_arr(load_arr(&a) & load_arr(&zero_mask));
            let mut v = load_arr(&a);
            v &= load_arr(&zero_mask);
            assert_eq!(store_arr(v), expected);
        }
    }

    #[test]
    fn bitor_assign_matches_bitor() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let sign_mask = [-0.0f64; 4];
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
        let a = [-7.0f64, -10.0, -15.0, -1.0];
        let b = [3.0f64, 3.0, 4.0, 1.0];
        unsafe {
            let result = store_arr(load_arr(&a) % load_arr(&b));
            for i in 0..4 {
                assert!(
                    result[i] <= 0.0,
                    "lane {i}: negative dividend must yield non-positive remainder, got {}",
                    result[i]
                );
                let expected = a[i] % b[i];
                assert!(
                    (result[i] - expected).abs() < 1e-10,
                    "lane {i}: expected {expected}, got {}",
                    result[i]
                );
            }
        }
    }
}
