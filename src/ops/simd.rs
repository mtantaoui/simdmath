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

/// Mathematical operations over a SIMD vector, in both sequential and
/// fully-parallel forms.
///
/// Every method has two variants:
///
/// - **`fn name`** — sequential: applies the scalar operation to each lane
///   individually. Results match the platform's `libm` implementation.
/// - **`fn par_name`** — parallel: applies a SIMD-native implementation across
///   all lanes simultaneously for maximum throughput.
///
/// # Implementing types
///
/// Implementors are architecture-specific SIMD vector structs (e.g. `F32x8`
/// for AVX2). The associated [`Output`](Self::Output) type is typically `Self`.
pub(crate) trait Math {
    /// The type returned by every math operation (typically `Self`).
    type Output;

    // -------------------------------------------------------------------------
    // Sequential methods — scalar precision, applied lane-by-lane
    // -------------------------------------------------------------------------

    /// Absolute value: `|self[i]|` for each lane.
    fn abs(&self) -> Self::Output;

    /// Arccosine (inverse cosine) in radians: `acos(self[i])`.
    ///
    /// Domain: `[-1.0, 1.0]`. Returns `NaN` outside the domain.
    fn acos(&self) -> Self::Output;

    /// Arcsine (inverse sine) in radians: `asin(self[i])`.
    ///
    /// Domain: `[-1.0, 1.0]`. Returns `NaN` outside the domain.
    fn asin(&self) -> Self::Output;

    /// Arctangent in radians: `atan(self[i])`.
    fn atan(&self) -> Self::Output;

    /// Two-argument arctangent: `atan2(self[i], other[i])`.
    ///
    /// Returns the angle (in radians) of the vector `(other[i], self[i])`,
    /// in the range `(-π, π]`. Handles the sign of both arguments correctly,
    /// unlike `atan(self / other)`.
    fn atan2(&self, other: Self) -> Self::Output;

    /// Cube root: `self[i]^(1/3)`.
    fn cbrt(&self) -> Self::Output;

    /// Floor: largest integer ≤ `self[i]` for each lane.
    fn floor(&self) -> Self::Output;

    /// Natural exponential: `e^self[i]` for each lane.
    fn exp(&self) -> Self::Output;

    /// Natural logarithm: `ln(self[i])`.
    ///
    /// Domain: `(0.0, +∞)`. Returns `NaN` for negative values, `-∞` for `0.0`.
    fn ln(&self) -> Self::Output;

    /// Sine in radians: `sin(self[i])`.
    fn sin(&self) -> Self::Output;

    /// Cosine in radians: `cos(self[i])`.
    fn cos(&self) -> Self::Output;

    /// Tangent in radians: `tan(self[i])`.
    fn tan(&self) -> Self::Output;

    /// Square root: `√self[i]` for each lane.
    ///
    /// Returns `NaN` for negative values.
    fn sqrt(&self) -> Self::Output;

    /// Ceiling: smallest integer ≥ `self[i]` for each lane.
    fn ceil(&self) -> Self::Output;

    /// Power: `self[i]^other[i]` for each lane.
    fn pow(&self, other: Self) -> Self::Output;

    /// Fused multiply-add: `(self[i] * multiplier[i]) + multiplicand[i]`.
    ///
    /// A single fused operation with only one rounding step, giving higher
    /// accuracy than a separate multiply followed by add.
    fn fma(&self, multiplier: Self, multiplicand: Self) -> Self::Output;

    /// 2-argument hypotenuse: `√(self[i]² + other[i]²)` for each lane.
    fn hypot(&self, other: Self) -> Self::Output;

    /// 3-argument hypotenuse: `√(self[i]² + other1[i]² + other2[i]²)`.
    fn hypot3(&self, other1: Self, other2: Self) -> Self::Output;

    /// 4-argument hypotenuse: `√(self[i]² + other1[i]² + other2[i]² + other3[i]²)`.
    fn hypot4(&self, other1: Self, other2: Self, other3: Self) -> Self::Output;

    // -------------------------------------------------------------------------
    // Parallel SIMD methods — all lanes computed simultaneously
    // -------------------------------------------------------------------------

    /// Parallel absolute value across all lanes simultaneously.
    fn par_abs(&self) -> Self::Output;

    /// Parallel arccosine across all lanes simultaneously.
    fn par_acos(&self) -> Self::Output;

    /// Parallel arcsine across all lanes simultaneously.
    fn par_asin(&self) -> Self::Output;

    /// Parallel arctangent across all lanes simultaneously.
    fn par_atan(&self) -> Self::Output;

    /// Parallel two-argument arctangent across all lanes simultaneously.
    fn par_atan2(&self, other: Self) -> Self::Output;

    /// Parallel cube root across all lanes simultaneously.
    fn par_cbrt(&self) -> Self::Output;

    /// Parallel ceiling across all lanes simultaneously.
    fn par_ceil(&self) -> Self::Output;

    /// Parallel cosine across all lanes simultaneously.
    fn par_cos(&self) -> Self::Output;

    /// Parallel natural exponential across all lanes simultaneously.
    fn par_exp(&self) -> Self::Output;

    /// Parallel floor across all lanes simultaneously.
    fn par_floor(&self) -> Self::Output;

    /// Parallel natural logarithm across all lanes simultaneously.
    fn par_ln(&self) -> Self::Output;

    /// Parallel sine across all lanes simultaneously.
    fn par_sin(&self) -> Self::Output;

    /// Parallel square root across all lanes simultaneously.
    ///
    /// On AVX2 this maps directly to `_mm256_sqrt_ps` / `_mm256_sqrt_pd`,
    /// a single hardware instruction with full IEEE 754 precision.
    fn par_sqrt(&self) -> Self::Output;

    /// Parallel tangent across all lanes simultaneously.
    fn par_tan(&self) -> Self::Output;

    /// Parallel 2-argument hypotenuse across all lanes simultaneously.
    fn par_hypot(&self, other: Self) -> Self::Output;

    /// Parallel 3-argument hypotenuse across all lanes simultaneously.
    fn par_hypot3(&self, other1: Self, other2: Self) -> Self::Output;

    /// Parallel 4-argument hypotenuse across all lanes simultaneously.
    fn par_hypot4(&self, other1: Self, other2: Self, other3: Self) -> Self::Output;

    /// Parallel power across all lanes simultaneously.
    fn par_pow(&self, other: Self) -> Self::Output;

    /// Parallel fused multiply-add across all lanes simultaneously.
    ///
    /// On AVX2 + FMA this maps directly to `_mm256_fmadd_ps` / `_mm256_fmadd_pd`.
    fn par_fma(&self, multiplier: Self, multiplicand: Self) -> Self::Output;
}
