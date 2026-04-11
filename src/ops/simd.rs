//! Core SIMD operation traits.
//!
//! This module defines the abstract contracts that every architecture-specific
//! SIMD vector type must satisfy. Each trait is generic over the scalar element
//! type `T` (e.g. `f32`, `f64`) and is implemented per-backend in `arch/`.
//!
//! # Trait overview
//!
//! | Trait          | Responsibility                                              |
//! |----------------|-------------------------------------------------------------|
//! | [`Align<T>`]   | Pointer alignment checks                                    |
//! | [`Load<T>`]    | Loading data from memory into a SIMD register               |
//! | [`Store<T>`]   | Writing a SIMD register back to memory                      |
//! | [`Math`]       | Element-wise and fully-parallel mathematical operations      |
//!
//! # Sequential vs parallel (`par_`) methods
//!
//! [`Math`] exposes two tiers of every operation:
//!
//! - **Sequential** (`fn name`) — applies the scalar operation lane-by-lane.
//!   Useful when correctness (e.g. faithfully-rounded `libm` results) matters
//!   more than throughput.
//! - **Parallel** (`fn par_name`) — applies a SIMD-native implementation across
//!   all lanes simultaneously. Where a hardware instruction exists (e.g.
//!   `_mm256_sqrt_ps`), it is used directly. For transcendentals without a
//!   hardware counterpart (sin, cos, exp, …) a polynomial approximation is
//!   applied to all lanes in parallel.
//!
//! # Safety
//!
//! [`Load`] and [`Store`] methods are `unsafe` because they dereference raw
//! pointers. Callers must ensure pointers are non-null, sufficiently aligned
//! (where required), and valid for the indicated number of element reads/writes.

/// Pointer alignment checking for a SIMD vector type.
///
/// Implementors report whether a raw pointer meets the alignment requirement
/// of their underlying register type (e.g. 32-byte alignment for `__m256`).
pub(crate) trait Align<T> {
    /// Returns `true` if `ptr` meets the alignment requirement of the SIMD
    /// register type.
    ///
    /// This is used by [`Load::load`] to dispatch to the faster aligned load
    /// path when possible, avoiding the penalty of an unaligned load on some
    /// microarchitectures.
    fn is_aligned(ptr: *const T) -> bool;
}

/// Loading data from memory into a SIMD register.
///
/// Three load strategies are provided to cover the common cases seen when
/// processing a slice whose length may not be a multiple of the lane count:
///
/// - **Full aligned** — fastest path; requires 32-byte alignment.
/// - **Full unaligned** — general path; no alignment constraint.
/// - **Partial** — tail of a slice smaller than one full register; inactive
///   lanes are zeroed so subsequent arithmetic is well-defined.
///
/// [`load`](Self::load) automatically selects aligned or unaligned based on
/// the pointer, and is the preferred entry point for full-register loads.
pub(crate) trait Load<T> {
    /// The concrete SIMD vector type produced by a load.
    type Output;

    /// Loads exactly `size` elements, dispatching to the aligned or unaligned
    /// path based on pointer alignment.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for `size` element reads.
    /// - `size` must equal the lane count of the implementing type.
    unsafe fn load(ptr: *const T, size: usize) -> Self::Output;

    /// Loads a full register from a pointer that is guaranteed to be aligned
    /// to the register's natural boundary (e.g. 32 bytes for AVX2).
    ///
    /// Prefer [`load`](Self::load) unless you have already verified alignment,
    /// as an unaligned pointer will cause a general-protection fault.
    ///
    /// # Safety
    /// - `ptr` must be non-null and aligned to the register boundary.
    unsafe fn load_aligned(ptr: *const T) -> Self::Output;

    /// Loads a full register from a pointer with no alignment constraint.
    ///
    /// Slightly slower than [`load_aligned`](Self::load_aligned) on older
    /// microarchitectures; on modern CPUs (Haswell+) the penalty is negligible.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for a full register-width read.
    unsafe fn load_unaligned(ptr: *const T) -> Self::Output;

    /// Loads the first `size` elements (where `size < lane_count`) from `ptr`.
    ///
    /// Inactive lanes (from `size` up to the lane count) are zeroed, making
    /// this safe to use in arithmetic without producing spurious results.
    /// This is the standard pattern for processing the tail of a slice that
    /// does not fill a complete register.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for at least `size` element reads.
    /// - `size` must be strictly less than the lane count of the implementing type.
    unsafe fn load_partial(ptr: *const T, size: usize) -> Self::Output;

    /// Broadcasts `val` to every lane, with `size = LANE_COUNT` active lanes.
    /// Wraps `_mm256_set1_ps`, `_mm256_set1_pd`, etc.
    ///
    /// # Safety
    /// No preconditions beyond a valid `val`.
    unsafe fn broadcast(val: T) -> Self::Output;

    /// Returns a zeroed register with all lanes set to `0` and `size = LANE_COUNT`.
    /// Wraps `_mm256_setzero_ps`, `_mm256_setzero_pd`, etc.
    ///
    /// Used to initialise accumulators for sum reductions.
    unsafe fn zero() -> Self::Output;
}

/// Writing a SIMD register back to memory.
///
/// Mirrors [`Load`] with the same three strategies (aligned, unaligned,
/// partial) plus a non-temporal streaming store for write-once buffers.
///
/// [`store_at`](Self::store_at) and [`store_at_partial`](Self::store_at_partial)
/// are the two entry points, mirroring [`Load::load`] and [`Load::load_partial`]
/// respectively. Callers are responsible for choosing the right variant —
/// `store_at` for full registers, `store_at_partial` for tails.
pub(crate) trait Store<T> {
    /// The concrete SIMD vector type (typically `Self`).
    type Output;

    /// Stores all [`LANE_COUNT`] lanes to `ptr`, dispatching to the aligned or
    /// unaligned path based on pointer alignment.
    ///
    /// Mirrors [`Load::load`]: handles the full-register case only. For tail
    /// vectors (fewer than `LANE_COUNT` elements), call
    /// [`store_at_partial`](Self::store_at_partial) directly.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for a full register-width write.
    /// - `self.size` must equal the lane count of the implementing type.
    unsafe fn store_at(&self, ptr: *const T);

    /// Writes all lanes to `ptr` using a non-temporal (streaming) store,
    /// bypassing the CPU cache entirely.
    ///
    /// Streaming stores are beneficial when writing large output buffers that
    /// will not be read again soon, as they avoid polluting the cache.
    /// Follow with an `_mm_sfence()` before any subsequent reads to ensure
    /// store visibility.
    ///
    /// # Safety
    /// - `ptr` must be non-null and aligned to the register boundary.
    #[allow(dead_code)]
    unsafe fn stream_at(&self, ptr: *mut T);

    /// Writes all lanes to a pointer that is guaranteed to be register-aligned.
    ///
    /// # Safety
    /// - `ptr` must be non-null and aligned to the register boundary.
    unsafe fn store_aligned_at(&self, ptr: *mut T);

    /// Writes all lanes to a pointer with no alignment constraint.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for a full register-width write.
    unsafe fn store_unaligned_at(&self, ptr: *mut T);

    /// Writes only the first `self.size` lanes to `ptr` using a sign-bit mask.
    /// Inactive lanes are not written, leaving the destination memory unchanged.
    ///
    /// This is the counterpart to [`Load::load_partial`] and is used to write
    /// the tail of a slice that does not fill a complete register.
    ///
    /// # Safety
    /// - `ptr` must be non-null and valid for at least `self.size` element writes.
    /// - `self.size` must be strictly less than the lane count.
    unsafe fn store_at_partial(&self, ptr: *mut T);
}
