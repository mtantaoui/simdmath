//! Packed panel storage shared by the blocked matrix multiplications.
//!
//! `pack_a` / `pack_b` copy cache-block-sized regions of the column-major
//! input matrices into contiguous, cache-line-aligned buffers laid out for the
//! microkernels ([`f32`](crate::arch::neon::matmul::f32) uses 8×12 panels,
//! [`f64`](crate::arch::neon::matmul::f64) 4×12):
//!
//! - **A panels** hold `MR` rows × `kc` columns, stored column by column, so
//!   the kernel loads two aligned vectors per `k` step.
//! - **B panels** hold `kc` rows × `NR` columns, stored row by row, so the
//!   kernel broadcasts individual elements straight from the panel.
//!
//! Partial panels (when the block size is not a multiple of `MR`/`NR`) are
//! zero-padded, which lets the microkernels always run full-width FMAs:
//! padded lanes contribute `0` to the accumulation.

use std::alloc::{self, Layout};
use std::cmp::min;

/// Panels are cache-line aligned. NEON loads have no alignment requirement,
/// but line-aligned panels avoid split-line accesses in the kernel.
const PANEL_ALIGNMENT: usize = 64;

/// Element types whose all-zero-bytes representation is a valid value, so a
/// zeroed allocation may be handed out as initialised `&mut [T]`.
///
/// # Safety
/// Implementors must guarantee that the all-zero bit pattern is a valid `T`
/// (true for the IEEE 754 floats: all-zero bits = `+0.0`).
pub(crate) unsafe trait Zeroable: Copy + Default {}

// SAFETY: all-zero bits are +0.0 for both IEEE 754 float widths.
unsafe impl Zeroable for f32 {}
unsafe impl Zeroable for f64 {}

/// 1D index of element `(i, j)` in a column-major matrix with leading
/// dimension `ld` (the number of rows).
#[inline(always)]
pub(crate) fn at(i: usize, j: usize, ld: usize) -> usize {
    j * ld + i
}

/// A contiguous, cache-line-aligned buffer of packed panels, allocated once per
/// `matmul` call and refilled by `pack_a`/`pack_b` for every block iteration.
///
/// Each of the `num_panels` panels holds `kc` slices of `R` elements, where
/// `R` is `MR` for A blocks (a slice = one column of an `MR`-row panel) and
/// `NR` for B blocks (a slice = one row of an `NR`-column panel).
///
/// The buffer is allocated **zeroed** (via `alloc_zeroed`, which for these
/// block sizes hands out copy-on-write zero pages instead of running a
/// memset), so every element is always initialised; the pack functions
/// overwrite the panels they configure, values plus explicit zero padding,
/// so the kernel never observes stale data from a previous block.
pub(crate) struct PackedBlock<T, const R: usize> {
    data_ptr: *mut T,
    /// Allocated capacity in elements.
    capacity: usize,
    num_panels: usize,
    /// Depth (shared K-dimension length) of every panel.
    kc: usize,
    layout: Layout,
}

impl<T: Zeroable, const R: usize> PackedBlock<T, R> {
    /// Allocates zeroed capacity for packing up to `len` rows/columns of the
    /// source matrix at depth up to `kc` (`ceil(len / R)` panels of `kc × R`
    /// elements each).
    pub(crate) fn with_capacity(len: usize, kc: usize) -> Self {
        // A zero-size allocation is undefined behaviour; `matmul` returns
        // early on empty inputs so every block it packs is non-empty.
        assert!(len > 0 && kc > 0, "PackedBlock dimensions must be non-zero");

        let num_panels = len.div_ceil(R);
        let capacity = num_panels * kc * R;
        let layout = Layout::array::<T>(capacity)
            .unwrap()
            .align_to(PANEL_ALIGNMENT)
            .unwrap();

        let data_ptr = unsafe {
            let raw_ptr = alloc::alloc_zeroed(layout);
            if raw_ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            raw_ptr.cast::<T>()
        };

        PackedBlock {
            data_ptr,
            capacity,
            num_panels,
            kc,
            layout,
        }
    }

    /// Reconfigures the panel geometry for the next pack (must fit within
    /// the allocated capacity).
    fn reset(&mut self, len: usize, kc: usize) {
        let num_panels = len.div_ceil(R);
        debug_assert!(
            num_panels * kc * R <= self.capacity,
            "packed block overflow: {len}x{kc} panels of {R} exceed capacity {}",
            self.capacity
        );
        self.num_panels = num_panels;
        self.kc = kc;
    }

    /// Pointer to the start of a panel (`kc * R` elements), for the kernel.
    #[inline(always)]
    pub(crate) fn panel(&self, panel_idx: usize) -> *const T {
        debug_assert!(panel_idx < self.num_panels);
        unsafe { self.data_ptr.add(panel_idx * self.kc * R) }
    }

    /// The `k`-th `R`-element slice of a panel, for packing.
    #[inline(always)]
    fn slice_mut(&mut self, panel_idx: usize, k: usize) -> &mut [T; R] {
        debug_assert!(panel_idx < self.num_panels && k < self.kc);
        // SAFETY: the slice lies within the allocation (`reset` checked that
        // the panels fit) and the zeroed allocation keeps every element
        // initialised (`T: Zeroable`), so handing out a reference is sound.
        unsafe {
            &mut *self
                .data_ptr
                .add((panel_idx * self.kc + k) * R)
                .cast::<[T; R]>()
        }
    }
}

impl<T, const R: usize> Drop for PackedBlock<T, R> {
    fn drop(&mut self) {
        unsafe { alloc::dealloc(self.data_ptr.cast::<u8>(), self.layout) };
    }
}

/// Packs the `mc × kc` block `A(ic.., pc..)` of column-major `a` (leading
/// dimension `m`) into `MR`-row panels inside `block`.
///
/// Within each panel, slice `k` holds column `pc + k` of that panel's `MR`
/// rows, so the microkernel reads consecutive aligned vectors as `k` advances.
/// Rows beyond `mc` in the last panel are zero-padded.
#[inline(always)]
pub(crate) fn pack_a<T: Zeroable, const MR: usize>(
    block: &mut PackedBlock<T, MR>,
    a: &[T],
    mc: usize,
    kc: usize,
    m: usize,
    ic: usize,
    pc: usize,
) {
    block.reset(mc, kc);

    for (panel_idx, i_panel_start) in (0..mc).step_by(MR).enumerate() {
        let mr_in_panel = min(MR, mc - i_panel_start);

        for p_col in 0..kc {
            let src_start = at(ic + i_panel_start, pc + p_col, m);
            let dest = block.slice_mut(panel_idx, p_col);

            if mr_in_panel == MR {
                // Full panel: a fixed-size MR-element copy the compiler
                // lowers to a couple of vector moves.
                dest.copy_from_slice(&a[src_start..src_start + MR]);
            } else {
                // Partial panel: copy the active rows, zero-pad the rest
                // (the buffer is reused, so padding must be written).
                dest[..mr_in_panel].copy_from_slice(&a[src_start..src_start + mr_in_panel]);
                dest[mr_in_panel..].fill(T::default());
            }
        }
    }
}

/// Packs the `kc × nc` block `B(pc.., jc..)` of column-major `b` (leading
/// dimension `k`) into `NR`-column panels inside `block`.
///
/// Within each panel, slice `p` holds row `pc + p` of that panel's `NR`
/// columns in row-major order, so the microkernel can broadcast individual
/// `B` elements straight from the panel. Columns beyond `nc` in the last
/// panel are zero-padded.
#[inline(always)]
pub(crate) fn pack_b<T: Zeroable, const NR: usize>(
    block: &mut PackedBlock<T, NR>,
    b: &[T],
    nc: usize,
    kc: usize,
    k: usize,
    pc: usize,
    jc: usize,
) {
    block.reset(nc, kc);

    for (panel_idx, j_panel_start) in (0..nc).step_by(NR).enumerate() {
        let nr_in_panel = min(NR, nc - j_panel_start);

        for p_row in 0..kc {
            let dest = block.slice_mut(panel_idx, p_row);
            let src_row = pc + p_row;

            // Transpose one B row into the panel: the source elements are
            // strided (one per column), so this stays a scalar gather.
            for (j, d) in dest[..nr_in_panel].iter_mut().enumerate() {
                *d = b[at(src_row, jc + j_panel_start + j, k)];
            }
            // Zero-pad (the buffer is reused, so padding must be written).
            dest[nr_in_panel..].fill(T::default());
        }
    }
}
