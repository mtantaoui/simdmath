//! NEON implementation of a 2-lane double-precision floating-point vector (`f64x2`).
//!
//! This module provides [`F64x2`], a wrapper around the NEON `float64x2_t` register type,
//! which holds 2 × `f64` values processed in parallel. It implements the [`Align`],
//! [`Load`], and [`Store`] traits for aligned, unaligned, and partial (tail) load/store
//! patterns, as well as all standard arithmetic and bitwise operators.
//!
//! # Alignment
//!
//! NEON `float64x2_t` registers are **16-byte** (128-bit) aligned. This is smaller
//! than AVX2's 32-byte alignment for `__m256d`. Unlike x86 SIMD where aligned loads
//! (`_mm256_load_pd`) will fault on misaligned pointers, NEON's `vld1q_f64` /
//! `vst1q_f64` intrinsics handle both aligned and unaligned addresses transparently
//! with minimal performance penalty on modern ARM cores (Cortex-A53+, Apple M-series).
//!
//! For maximum performance on older Cortex-A cores, prefer aligned pointers where
//! possible. The [`Align::is_aligned`] method checks for 16-byte alignment.
//!
//! # Target architecture
//!
//! This module is compiled only on `aarch64` targets with NEON support (enabled
//! by default on all AArch64 CPUs).

use std::arch::aarch64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

use crate::ops::simd::{Align, Load, Store};

/// Number of `f64` lanes in a single [`F64x2`] register.
pub const LANE_COUNT: usize = 2;

/// A 2-lane double-precision floating-point SIMD vector backed by `float64x2_t`.
///
/// # Alignment
///
/// The underlying `float64x2_t` type requires **16-byte alignment**. Use
/// `#[repr(align(16))]` for stack-allocated buffers and `aligned_alloc` /
/// `posix_memalign` for heap allocations when passing pointers to
/// [`load_aligned`](Load::load_aligned) or [`store_aligned_at`](Store::store_aligned_at).
///
/// # Size tracking
///
/// `size` tracks how many lanes are logically active; inactive lanes are zeroed
/// on load and masked on store so they never contaminate results.
#[derive(Copy, Clone, Debug)]
pub struct F64x2 {
    pub(crate) size: usize,
    pub(crate) elements: float64x2_t,
}

/// Alignment checking for NEON `float64x2_t` (16-byte boundary).
///
/// # Alignment requirement
///
/// NEON `float64x2_t` is 16-byte (128-bit) aligned. While NEON load/store
/// intrinsics (`vld1q_f64`, `vst1q_f64`) support unaligned access without
/// faulting, aligned access may be faster on some microarchitectures.
///
/// ## Comparison with x86 SIMD
///
/// | Architecture | Register      | Lanes | Alignment |
/// |--------------|---------------|-------|-----------|
/// | AVX2         | `__m256d`     | 4     | 32 bytes  |
/// | AVX-512      | `__m512d`     | 8     | 64 bytes  |
/// | **NEON**     | `float64x2_t` | 2     | 16 bytes  |
impl Align<f64> for F64x2 {
    /// Returns `true` if `ptr` is 16-byte aligned (suitable for `float64x2_t`).
    ///
    /// This is used by [`Load::load`] to dispatch to the aligned path when
    /// possible, though on modern ARM cores the performance difference between
    /// aligned and unaligned loads is minimal.
    #[inline]
    fn is_aligned(ptr: *const f64) -> bool {
        let ptr = ptr as usize;
        // NEON float64x2_t requires 16-byte alignment
        ptr.is_multiple_of(core::mem::align_of::<float64x2_t>())
    }
}

impl Load<f64> for F64x2 {
    type Output = Self;

    /// Loads exactly [`LANE_COUNT`] elements from the given pointer.
    ///
    /// Unlike AVX2 where we dispatch to separate aligned/unaligned intrinsics,
    /// NEON's `vld1q_f64` handles both cases efficiently, so we always use
    /// the unaligned path.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for [`LANE_COUNT`] element reads.
    /// - `size` must equal [`LANE_COUNT`].
    #[inline]
    unsafe fn load(ptr: *const f64, size: usize) -> Self::Output {
        debug_assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        // NEON vld1q_f64 handles both aligned and unaligned loads efficiently
        unsafe { Self::load_unaligned(ptr) }
    }

    /// Loads [`LANE_COUNT`] elements from a 16-byte-aligned pointer.
    ///
    /// # Safety
    /// - `ptr` must be non-null and 16-byte aligned.
    ///
    /// # Note
    /// On NEON, this uses the same `vld1q_f64` intrinsic as [`load_unaligned`].
    /// The compiler may generate slightly better code if alignment is guaranteed.
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self::Output {
        Self {
            elements: unsafe { vld1q_f64(ptr) },
            size: LANE_COUNT,
        }
    }

    /// Loads [`LANE_COUNT`] elements from an unaligned pointer.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for [`LANE_COUNT`] element reads.
    ///
    /// # Performance
    /// On modern ARM cores (Cortex-A53+, Apple M-series), unaligned loads have
    /// negligible overhead compared to aligned loads.
    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self::Output {
        Self {
            elements: unsafe { vld1q_f64(ptr) },
            size: LANE_COUNT,
        }
    }

    /// Loads `size` elements (where `size < LANE_COUNT`) into the vector.
    /// Inactive lanes are zeroed.
    ///
    /// For `f64x2`, this means loading only 1 element (size=1) with the
    /// second lane zeroed.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for at least `size` reads.
    /// - `size` must be strictly less than [`LANE_COUNT`] (i.e., 0 or 1).
    ///
    /// # Implementation note
    /// Unlike AVX2's `_mm256_maskload_pd`, NEON lacks a native masked load
    /// intrinsic. We emulate this by loading into a temporary buffer and
    /// broadcasting to the register.
    #[inline]
    unsafe fn load_partial(ptr: *const f64, size: usize) -> Self::Output {
        debug_assert!(size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        unsafe {
            let mut arr = [0.0f64; LANE_COUNT];
            for (i, slot) in arr.iter_mut().enumerate().take(size) {
                *slot = *ptr.add(i);
            }
            Self {
                elements: vld1q_f64(arr.as_ptr()),
                size,
            }
        }
    }

    /// Broadcasts `val` to every lane of a new `F64x2` register.
    ///
    /// Wraps `vdupq_n_f64`. Both lanes receive `val`; `size` is set to
    /// [`LANE_COUNT`] since all lanes are active.
    #[inline]
    unsafe fn broadcast(val: f64) -> Self::Output {
        Self {
            size: LANE_COUNT,
            elements: unsafe { vdupq_n_f64(val) },
        }
    }

    /// Returns a zeroed `F64x2` register.
    ///
    /// Both lanes are set to `0.0` and `size` is set to [`LANE_COUNT`].
    /// Used to initialise the accumulator in sum reductions.
    #[inline]
    unsafe fn zero() -> Self::Output {
        Self {
            size: LANE_COUNT,
            elements: unsafe { vdupq_n_f64(0.0) },
        }
    }
}

impl Store<f64> for F64x2 {
    type Output = Self;

    /// Stores all [`LANE_COUNT`] lanes to `ptr`.
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
        unsafe { vst1q_f64(mut_ptr, self.elements) }
    }

    /// Writes all [`LANE_COUNT`] lanes to `ptr` using a non-temporal (streaming) store.
    ///
    /// # Safety
    /// - `ptr` must be non-null and 16-byte aligned.
    ///
    /// # NEON note
    /// ARM NEON does not have a direct equivalent to x86's non-temporal stores
    /// (`_mm256_stream_pd`). This implementation uses a regular store. For
    /// write-combining behaviour on ARM, consider using `__builtin_arm_dmb`
    /// memory barriers or architecture-specific cache hints.
    #[inline]
    unsafe fn stream_at(&self, ptr: *mut f64) {
        unsafe { vst1q_f64(ptr, self.elements) }
    }

    /// Stores all [`LANE_COUNT`] lanes to a 16-byte-aligned pointer.
    ///
    /// # Safety
    /// - `ptr` must be non-null and 16-byte aligned.
    #[inline]
    unsafe fn store_aligned_at(&self, ptr: *mut f64) {
        unsafe { vst1q_f64(ptr, self.elements) }
    }

    /// Stores all [`LANE_COUNT`] lanes to an unaligned pointer.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for [`LANE_COUNT`] element writes.
    #[inline]
    unsafe fn store_unaligned_at(&self, ptr: *mut f64) {
        unsafe { vst1q_f64(ptr, self.elements) }
    }

    /// Stores `self.size` lanes (where `self.size < LANE_COUNT`) to memory.
    /// Inactive lanes are not written.
    ///
    /// For `f64x2`, this means storing only 1 element (size=1).
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for at least `self.size` writes.
    /// - `self.size` must be strictly less than [`LANE_COUNT`] (i.e., 0 or 1).
    ///
    /// # Implementation note
    /// Unlike AVX2's `_mm256_maskstore_pd`, NEON lacks a native masked store
    /// intrinsic. We emulate this by storing to a temporary buffer and
    /// copying only the active lanes.
    #[inline]
    unsafe fn store_at_partial(&self, ptr: *mut f64) {
        debug_assert!(self.size < LANE_COUNT, "Size must be < {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        unsafe {
            let mut arr = [0.0f64; LANE_COUNT];
            vst1q_f64(arr.as_mut_ptr(), self.elements);
            for (i, &val) in arr.iter().enumerate().take(self.size) {
                *ptr.add(i) = val;
            }
        }
    }
}

// =============================================================================
// Arithmetic operators
// =============================================================================

/// Element-wise addition of two [`F64x2`] vectors (`vaddq_f64`).
///
/// Inactive lanes (when `size < LANE_COUNT`) are still added, but since they
/// are zeroed on load, the result in those lanes is `0.0` and will not be
/// stored back via [`store_at_partial`](Store::store_at_partial).
impl Add for F64x2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { vaddq_f64(self.elements, rhs.elements) },
        }
    }
}

impl AddAssign for F64x2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Element-wise subtraction of two [`F64x2`] vectors (`vsubq_f64`).
impl Sub for F64x2 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { vsubq_f64(self.elements, rhs.elements) },
        }
    }
}

impl SubAssign for F64x2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Element-wise multiplication of two [`F64x2`] vectors (`vmulq_f64`).
impl Mul for F64x2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { vmulq_f64(self.elements, rhs.elements) },
        }
    }
}

impl MulAssign for F64x2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Element-wise division of two [`F64x2`] vectors (`vdivq_f64`).
///
/// # Note
/// Inactive lanes are zeroed on load, so dividing two partial vectors will
/// produce `0.0 / 0.0 = NaN` in those lanes. This is harmless as long as
/// results are written back through [`store_at_partial`](Store::store_at_partial),
/// which masks out inactive lanes before writing.
impl Div for F64x2 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            size: self.size,
            elements: unsafe { vdivq_f64(self.elements, rhs.elements) },
        }
    }
}

impl DivAssign for F64x2 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Element-wise floating-point remainder using **truncation** semantics
/// (`self - trunc(self / rhs) * rhs`), matching Rust's scalar `f64 %` operator.
///
/// This differs from floor-division remainder (Python's `%` / C's `fmod` when
/// both operands are negative): the result always has the same sign as the
/// dividend. For example, `-7.0 % 3.0` yields `-1.0`, not `2.0`.
///
/// # NEON note
/// Uses `vrndq_f64` (round towards zero) for truncation, available on ARMv8+.
impl Rem for F64x2 {
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
            let div = vdivq_f64(self.elements, rhs.elements);
            // vrndq_f64 rounds towards zero (truncation)
            let trunc = vrndq_f64(div);
            let prod = vmulq_f64(trunc, rhs.elements);
            Self {
                size: self.size,
                elements: vsubq_f64(self.elements, prod),
            }
        }
    }
}

impl RemAssign for F64x2 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// =============================================================================
// Bitwise operators
// =============================================================================

/// Applies a bit-wise AND between two [`F64x2`] vectors, operating on the raw
/// IEEE 754 bit patterns of each `f64` lane.
///
/// Useful for masking float data bit-for-bit — for example, clearing all bits
/// in lanes by AND-ing with `0.0`, or extracting the sign/exponent/mantissa
/// fields.
///
/// # NEON implementation
/// Reinterprets `float64x2_t` → `uint64x2_t`, applies `vandq_u64`, then
/// reinterprets back. This is a zero-cost operation at runtime (just type
/// punning).
impl BitAnd for F64x2 {
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
            F64x2 {
                size: self.size,
                elements: vreinterpretq_f64_u64(vandq_u64(
                    vreinterpretq_u64_f64(self.elements),
                    vreinterpretq_u64_f64(rhs.elements),
                )),
            }
        }
    }
}

impl BitAndAssign for F64x2 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

/// Applies a bit-wise OR between two [`F64x2`] vectors, operating on the raw
/// IEEE 754 bit patterns of each `f64` lane.
///
/// Useful for combining float bit patterns without going through scalar code —
/// for example, setting the sign bit of every lane to force magnitudes
/// negative, or merging two partially-masked results.
///
/// # NEON implementation
/// Reinterprets `float64x2_t` → `uint64x2_t`, applies `vorrq_u64`, then
/// reinterprets back. This is a zero-cost operation at runtime.
impl BitOr for F64x2 {
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
            F64x2 {
                size: self.size,
                elements: vreinterpretq_f64_u64(vorrq_u64(
                    vreinterpretq_u64_f64(self.elements),
                    vreinterpretq_u64_f64(rhs.elements),
                )),
            }
        }
    }
}

impl BitOrAssign for F64x2 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

// =============================================================================
// Tests (only run on aarch64)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16-byte-aligned buffer of 2 `f64` values, matching `float64x2_t` alignment.
    #[repr(align(16))]
    struct Aligned([f64; 2]);

    /// A 16-byte-aligned buffer with one extra slot.
    /// `as_ptr().add(1)` is guaranteed to NOT be 16-byte aligned (base+8 mod 16 ≠ 0),
    /// providing a reliable unaligned pointer for dispatch tests.
    #[repr(align(16))]
    struct AlignedBuf([f64; 3]);

    // ---- Align ----------------------------------------------------------------

    #[test]
    fn is_aligned_returns_true_for_aligned_pointer() {
        let data = Aligned([0.0; 2]);
        assert!(F64x2::is_aligned(data.0.as_ptr()));
    }

    #[test]
    fn is_aligned_returns_false_for_unaligned_pointer() {
        let data = Aligned([0.0; 2]);
        // Adding 8 bytes (one f64) breaks 16-byte alignment.
        let unaligned = unsafe { data.0.as_ptr().add(1) };
        assert!(!F64x2::is_aligned(unaligned));
    }

    // ---- Load -----------------------------------------------------------------

    #[test]
    fn broadcast_fills_all_lanes_with_value() {
        unsafe {
            let v = F64x2::broadcast(3.5);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; LANE_COUNT];
            vst1q_f64(out.as_mut_ptr(), v.elements);
            for lane in out {
                assert!((lane - 3.5f64).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn zero_produces_all_zero_lanes() {
        unsafe {
            let v = F64x2::zero();
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [1.0f64; LANE_COUNT];
            vst1q_f64(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [0.0f64; LANE_COUNT]);
        }
    }

    #[test]
    fn load_aligned_loads_all_lanes() {
        let src = Aligned([1.0, 2.0]);
        unsafe {
            let v = F64x2::load_aligned(src.0.as_ptr());
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0]);
        }
    }

    #[test]
    fn load_unaligned_loads_all_lanes() {
        // One extra element at the front so ptr.add(1) is guaranteed unaligned.
        let buf = AlignedBuf([0.0f64, 1.0, 2.0]);
        unsafe {
            let ptr = buf.0.as_ptr().add(1);
            let v = F64x2::load_unaligned(ptr);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0]);
        }
    }

    #[test]
    fn load_dispatches_correctly() {
        let src = Aligned([1.0, 2.0]);
        assert!(
            F64x2::is_aligned(src.0.as_ptr()),
            "prerequisite: pointer must be aligned"
        );
        unsafe {
            let v = F64x2::load(src.0.as_ptr(), LANE_COUNT);
            assert_eq!(v.size, LANE_COUNT);
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), v.elements);
            assert_eq!(out, [1.0, 2.0]);
        }
    }

    #[test]
    fn load_partial_zeroes_inactive_lanes() {
        let src = [1.0f64, 2.0];
        // For f64x2, size can only be 0 or 1 for partial
        unsafe {
            let v = F64x2::load_partial(src.as_ptr(), 1);
            assert_eq!(v.size, 1);
            let mut out = [0.0f64; LANE_COUNT];
            vst1q_f64(out.as_mut_ptr(), v.elements);
            // First lane should match source.
            assert_eq!(out[0], src[0], "lane 0 mismatch");
            // Second lane should be zero.
            assert_eq!(out[1], 0.0, "lane 1 not zeroed");
        }
    }

    // ---- Store ----------------------------------------------------------------

    #[test]
    fn store_at_writes_all_lanes() {
        unsafe {
            let v = F64x2 {
                size: LANE_COUNT,
                elements: vld1q_f64([1.0f64, 2.0].as_ptr()),
            };
            let dst = [0.0f64; LANE_COUNT];
            v.store_at(dst.as_ptr());
            assert_eq!(dst, [1.0, 2.0]);
        }
    }

    #[test]
    fn store_at_partial_writes_only_active_lanes() {
        unsafe {
            let v = F64x2 {
                size: 1,
                elements: vld1q_f64([1.0f64, 2.0].as_ptr()),
            };
            let mut dst = [0.0f64; LANE_COUNT];
            v.store_at_partial(dst.as_mut_ptr());
            // First lane should be written.
            assert_eq!(dst[0], 1.0, "lane 0 mismatch");
            // Second lane should remain zero.
            assert_eq!(dst[1], 0.0, "lane 1 written unexpectedly");
        }
    }

    // ---- Arithmetic ops -------------------------------------------------------

    #[test]
    fn add_produces_element_wise_sum() {
        unsafe {
            let a = F64x2::load_unaligned([1.0f64, 2.0].as_ptr());
            let b = F64x2::load_unaligned([10.0f64, 20.0].as_ptr());
            let c = a + b;
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), c.elements);
            assert_eq!(out, [11.0, 22.0]);
        }
    }

    #[test]
    fn sub_produces_element_wise_difference() {
        unsafe {
            let a = F64x2::load_unaligned([10.0f64, 20.0].as_ptr());
            let b = F64x2::load_unaligned([1.0f64, 2.0].as_ptr());
            let c = a - b;
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), c.elements);
            assert_eq!(out, [9.0, 18.0]);
        }
    }

    #[test]
    fn mul_produces_element_wise_product() {
        unsafe {
            let a = F64x2::load_unaligned([2.0f64, 3.0].as_ptr());
            let b = F64x2::load_unaligned([4.0f64, 5.0].as_ptr());
            let c = a * b;
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), c.elements);
            assert_eq!(out, [8.0, 15.0]);
        }
    }

    #[test]
    fn div_produces_element_wise_quotient() {
        unsafe {
            let a = F64x2::load_unaligned([10.0f64, 20.0].as_ptr());
            let b = F64x2::load_unaligned([2.0f64, 4.0].as_ptr());
            let c = a / b;
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), c.elements);
            assert_eq!(out, [5.0, 5.0]);
        }
    }

    #[test]
    fn rem_produces_element_wise_remainder() {
        unsafe {
            let a = F64x2::load_unaligned([7.0f64, 8.0].as_ptr());
            let b = F64x2::load_unaligned([3.0f64, 3.0].as_ptr());
            let c = a % b;
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), c.elements);
            assert_eq!(out, [1.0, 2.0]);
        }
    }

    #[test]
    fn rem_negative_value_is_negative() {
        unsafe {
            let a = F64x2::load_unaligned([-7.0f64, -8.0].as_ptr());
            let b = F64x2::load_unaligned([3.0f64, 3.0].as_ptr());
            let c = a % b;
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), c.elements);
            // Truncation semantics: result has same sign as dividend
            assert_eq!(out, [-1.0, -2.0]);
        }
    }

    // ---- Bitwise ops ----------------------------------------------------------

    #[test]
    fn bitand_clears_lanes_via_zero_mask() {
        unsafe {
            let a = F64x2::load_unaligned([1.0f64, 2.0].as_ptr());
            let mask = F64x2::load_unaligned([0.0f64; 2].as_ptr());
            let result = a & mask;
            let mut out = [1.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), result.elements);
            assert_eq!(out, [0.0; 2]);
        }
    }

    #[test]
    fn bitor_sets_sign_bit() {
        unsafe {
            let a = F64x2::load_unaligned([1.0f64, 2.0].as_ptr());
            let sign_mask = F64x2::load_unaligned([-0.0f64; 2].as_ptr());
            let result = a | sign_mask;
            let mut out = [0.0f64; 2];
            vst1q_f64(out.as_mut_ptr(), result.elements);
            assert_eq!(out, [-1.0, -2.0]);
        }
    }
}
