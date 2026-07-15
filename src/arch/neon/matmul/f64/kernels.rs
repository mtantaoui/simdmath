//! NEON microkernels (f64) for the BLIS-style matrix multiplication.
//!
//! The 4×12 microkernel is the computational core of
//! [`matmul`](crate::arch::neon::matmul::f64::matmul): it multiplies one
//! packed `MR × kc` A panel with one packed `kc × NR` B panel, accumulating
//! into an up-to-4×12 tile of the column-major output matrix `C`.
//!
//! The 4×12 shape follows the same two-row-vector design as the x86
//! backends, scaled to the 32 NEON registers: 24 accumulators (2 row
//! vectors × 12 columns), 2 registers for the A column, and 1 for the
//! broadcast B element — 27 of the 32 architectural vector registers,
//! leaving headroom for address arithmetic. B elements are broadcast
//! straight from memory (`ld1r`, the load-replicate instruction).

use std::arch::aarch64::*;

use std::cmp::min;

use crate::arch::neon::f64x2::{F64x2, LANE_COUNT};
use crate::arch::neon::matmul::f64::{MR, NR};
use crate::ops::simd::{Load, Store};

/// Loads `len` elements of a `C` column, zero-filling the inactive lanes.
///
/// # Safety
/// `ptr` must be valid for `len` reads (`1 ≤ len ≤ 2`).
#[inline(always)]
unsafe fn load_c(ptr: *const f64, len: usize) -> float64x2_t {
    unsafe {
        if len == LANE_COUNT {
            F64x2::load(ptr, LANE_COUNT).elements
        } else {
            F64x2::load_partial(ptr, len).elements
        }
    }
}

/// Stores `len` lanes of a `C` column back to memory, masking the rest.
///
/// # Safety
/// `ptr` must be valid for `len` writes (`1 ≤ len ≤ 2`).
#[inline(always)]
unsafe fn store_c(v: float64x2_t, ptr: *mut f64, len: usize) {
    unsafe {
        let v = F64x2 {
            size: len,
            elements: v,
        };
        if len == LANE_COUNT {
            v.store_at(ptr)
        } else {
            v.store_at_partial(ptr)
        }
    }
}

/// The 4×12 accumulator tile: one pair of vector registers per output
/// column, covering rows `0..2` (`lo`) and `2..4` (`hi`).
///
/// Boundary tiles never touch memory outside `C`:
/// - rows beyond `mr` are masked on load/store ([`load_c`]/[`store_c`]);
/// - columns beyond `nr` live only in registers as zeroed dummies that are
///   loaded from nowhere and never stored.
struct CTile {
    lo: [float64x2_t; NR],
    hi: [float64x2_t; NR],
    /// Active rows in the low/high vectors (`mr_lo ≤ 2`, `mr_hi ≤ 2`).
    mr_lo: usize,
    mr_hi: usize,
    /// Active columns (`1 ≤ nr ≤ NR`).
    nr: usize,
}

impl CTile {
    /// Loads the `mr × nr` tile whose top-left element is `c` (column-major,
    /// leading dimension `m`).
    ///
    /// # Safety
    /// `c` must be valid for reads of the `mr × nr` tile, i.e. up to offset
    /// `(nr - 1) * m + mr`.
    #[inline(always)]
    unsafe fn load(c: *const f64, m: usize, mr: usize, nr: usize) -> Self {
        unsafe {
            let mr_lo = min(mr, LANE_COUNT);
            let mr_hi = mr.saturating_sub(LANE_COUNT);
            let zero = vdupq_n_f64(0.0);

            let mut lo = [zero; NR];
            let mut hi = [zero; NR];
            // Constant trip count with an inner guard (rather than `0..nr`)
            // so the loop fully unrolls and the arrays stay in registers.
            for j in 0..NR {
                if j < nr {
                    let col = c.add(j * m);
                    lo[j] = load_c(col, mr_lo);
                    if mr_hi > 0 {
                        hi[j] = load_c(col.add(LANE_COUNT), mr_hi);
                    }
                }
            }

            CTile {
                lo,
                hi,
                mr_lo,
                mr_hi,
                nr,
            }
        }
    }

    /// One outer-product step: `tile[.., j] += a * b` for every column `j`,
    /// with `b(j)` supplying that column's broadcast B element.
    ///
    /// Always runs all `NR` columns (inactive ones accumulate garbage in
    /// their dummy registers, which [`Self::store`] discards) so the hot
    /// loop stays branch-free.
    ///
    /// NEON's FMA takes the accumulator first: `vfmaq(c, a, b) = a*b + c`.
    #[inline(always)]
    fn fmadd(&mut self, a_lo: float64x2_t, a_hi: float64x2_t, b: impl Fn(usize) -> float64x2_t) {
        for j in 0..NR {
            let bj = b(j);
            self.lo[j] = unsafe { vfmaq_f64(self.lo[j], a_lo, bj) };
            self.hi[j] = unsafe { vfmaq_f64(self.hi[j], a_hi, bj) };
        }
    }

    /// Stores the active `mr × nr` part of the tile back to `c`.
    ///
    /// # Safety
    /// `c` must be valid for writes of the `mr × nr` tile (see
    /// [`Self::load`]).
    #[inline(always)]
    unsafe fn store(self, c: *mut f64, m: usize) {
        unsafe {
            // Constant trip count for the same reason as in `load`.
            for j in 0..NR {
                if j < self.nr {
                    let col = c.add(j * m);
                    store_c(self.lo[j], col, self.mr_lo);
                    if self.mr_hi > 0 {
                        store_c(self.hi[j], col.add(LANE_COUNT), self.mr_hi);
                    }
                }
            }
        }
    }
}

/// NEON microkernel computing `C(0..mr, 0..nr) += A_panel × B_panel`.
///
/// Implements the outer-product formulation: for each of the `kc` steps, the
/// A column (4 rows in two vector registers) is multiplied by each of the 12
/// B row elements, broadcast one at a time straight from the packed panel.
/// Columns beyond `nr` multiply the B panel's zero padding into dummy
/// accumulators, so the loop needs no column guards.
///
/// # Arguments
/// * `a_panel` - Packed A panel (`kc` columns of `MR` elements)
/// * `b_panel` - Packed B panel (`kc` rows of `NR` elements)
/// * `c_micropanel` - Top-left element of the output tile in column-major `C`
/// * `mr` - Active rows in the tile (`1 ≤ mr ≤ MR`)
/// * `nr` - Active columns in the tile (`1 ≤ nr ≤ NR`)
/// * `kc` - Panel depth (number of accumulation steps)
/// * `m` - Leading dimension (row count) of `C`
///
/// # Safety
/// - `a_panel` must point to a packed panel of at least `kc * MR` elements;
///   `b_panel` to a packed panel of at least `kc * NR`.
/// - `c_micropanel` must be valid for reads/writes of the `mr × nr` tile,
///   i.e. up to offset `(nr - 1) * m + mr`.
#[inline(always)]
pub(crate) unsafe fn kernel_4x12(
    a_panel: *const f64,
    b_panel: *const f64,
    c_micropanel: *mut f64,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    unsafe {
        let mut tile = CTile::load(c_micropanel, m, mr, nr);

        for k in 0..kc {
            let a_lo = vld1q_f64(a_panel.add(k * MR));
            let a_hi = vld1q_f64(a_panel.add(k * MR + LANE_COUNT));
            let b_row = b_panel.add(k * NR);

            tile.fmadd(a_lo, a_hi, |j| vld1q_dup_f64(b_row.add(j)));
        }

        tile.store(c_micropanel, m);
    }
}

/// NEON microkernel computing `C(0..mr, 0..nr) += A × B` directly on the
/// **unpacked** column-major inputs.
///
/// Used for matrices small enough that the working set fits in L2, where
/// packing costs more than it saves: A columns are already contiguous
/// (loads with stride `lda` between k steps) and B elements are broadcast
/// straight from their column-major locations.
///
/// The full 4×12 tile runs an unguarded hot loop; boundary tiles fall back
/// to a guarded loop with masked A loads (edge tiles are a small fraction of
/// the work).
///
/// # Arguments
/// * `a` - First element of the A block: `A(ic, pc)` (column-major, leading
///   dimension `lda`)
/// * `b` - First element of the B block: `B(pc, jc)` (column-major, leading
///   dimension `ldb`)
/// * `c_micropanel` - Top-left element of the output tile in column-major `C`
/// * `mr` - Active rows in the tile (`1 ≤ mr ≤ MR`)
/// * `nr` - Active columns in the tile (`1 ≤ nr ≤ NR`)
/// * `kc` - Depth of this k chunk
/// * `m` - Leading dimension (row count) of `C`
///
/// # Safety
/// - `a` must be valid for reads of `mr` rows × `kc` columns at stride `lda`;
///   `b` for `kc` rows × `nr` columns at stride `ldb`.
/// - `c_micropanel` must be valid for reads/writes of the `mr × nr` tile.
// Pointers, strides and tile geometry are individually meaningful; mirrors
// the packed kernel's signature.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn kernel_4x12_direct(
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    c_micropanel: *mut f64,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    unsafe {
        let mut tile = CTile::load(c_micropanel, m, mr, nr);

        if mr == MR && nr == NR {
            // Full tile: unguarded hot loop.
            for k in 0..kc {
                let a_col = a.add(k * lda);
                let a_lo = vld1q_f64(a_col);
                let a_hi = vld1q_f64(a_col.add(LANE_COUNT));

                tile.fmadd(a_lo, a_hi, |j| vld1q_dup_f64(b.add(j * ldb + k)));
            }
        } else {
            // Boundary tile: masked A loads, guarded B reads. The guards
            // resolve identically on every iteration, so they predict
            // perfectly.
            let zero = vdupq_n_f64(0.0);
            for k in 0..kc {
                let a_col = a.add(k * lda);
                let a_lo = load_c(a_col, tile.mr_lo);
                let a_hi = if tile.mr_hi > 0 {
                    load_c(a_col.add(LANE_COUNT), tile.mr_hi)
                } else {
                    zero
                };

                tile.fmadd(a_lo, a_hi, |j| {
                    if nr > j {
                        vld1q_dup_f64(b.add(j * ldb + k))
                    } else {
                        zero
                    }
                });
            }
        }

        tile.store(c_micropanel, m);
    }
}
