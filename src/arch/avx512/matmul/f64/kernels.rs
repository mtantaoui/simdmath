//! AVX-512 microkernels (f64) for the BLIS-style matrix multiplication.
//!
//! The 16×14 microkernel is the computational core of
//! [`matmul`](crate::arch::avx512::matmul::f64::matmul): it multiplies one
//! packed `MR × kc` A panel with one packed `kc × NR` B panel, accumulating
//! into an up-to-16×14 tile of the column-major output matrix `C`.
//!
//! The 16×14 shape is the f64 twin of the f32 kernel's 32×14: with 8-lane
//! `__m512d` registers it uses the identical register budget — 28 ZMM
//! accumulators (2 row vectors × 14 columns), 2 registers for the A column,
//! and 1 for the broadcast B element — 31 of the 32 architectural ZMM
//! registers. B elements are broadcast straight from memory
//! (`vbroadcastsd`), which runs on the load ports and leaves the shuffle
//! port free.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::cmp::min;

use crate::arch::avx512::f64x8::{F64x8, LANE_COUNT};
use crate::arch::avx512::matmul::f64::{MR, NR};
use crate::ops::simd::{Load, Store};

/// Loads `len` elements of a `C` column, zero-filling the inactive lanes.
///
/// # Safety
/// `ptr` must be valid for `len` reads (`1 ≤ len ≤ 8`).
#[inline(always)]
unsafe fn load_c(ptr: *const f64, len: usize) -> __m512d {
    unsafe {
        if len == LANE_COUNT {
            F64x8::load(ptr, LANE_COUNT).elements
        } else {
            F64x8::load_partial(ptr, len).elements
        }
    }
}

/// Stores `len` lanes of a `C` column back to memory, masking the rest.
///
/// # Safety
/// `ptr` must be valid for `len` writes (`1 ≤ len ≤ 8`).
#[inline(always)]
unsafe fn store_c(v: __m512d, ptr: *mut f64, len: usize) {
    unsafe {
        let v = F64x8 {
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

/// The 16×14 accumulator tile: one pair of ZMM vectors per output column,
/// covering rows `0..8` (`lo`) and `8..16` (`hi`).
///
/// Boundary tiles never touch memory outside `C`:
/// - rows beyond `mr` are masked on load/store ([`load_c`]/[`store_c`]);
/// - columns beyond `nr` live only in registers as zeroed dummies that are
///   loaded from nowhere and never stored.
struct CTile {
    lo: [__m512d; NR],
    hi: [__m512d; NR],
    /// Active rows in the low/high vectors (`mr_lo ≤ 8`, `mr_hi ≤ 8`).
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
            let zero = _mm512_setzero_pd();

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
    #[inline(always)]
    fn fmadd(&mut self, a_lo: __m512d, a_hi: __m512d, b: impl Fn(usize) -> __m512d) {
        for j in 0..NR {
            let bj = b(j);
            self.lo[j] = unsafe { _mm512_fmadd_pd(a_lo, bj, self.lo[j]) };
            self.hi[j] = unsafe { _mm512_fmadd_pd(a_hi, bj, self.hi[j]) };
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

/// AVX-512 microkernel computing `C(0..mr, 0..nr) += A_panel × B_panel`.
///
/// Implements the outer-product formulation: for each of the `kc` steps, the
/// A column (16 rows in two ZMM vectors) is multiplied by each of the 14 B
/// row elements, broadcast one at a time straight from the packed panel.
/// Columns beyond `nr` multiply the B panel's zero padding into dummy
/// accumulators, so the loop needs no column guards.
///
/// # Arguments
/// * `a_panel` - Packed A panel (`kc` aligned columns of `MR` elements)
/// * `b_panel` - Packed B panel (`kc` rows of `NR` elements)
/// * `c_micropanel` - Top-left element of the output tile in column-major `C`
/// * `mr` - Active rows in the tile (`1 ≤ mr ≤ MR`)
/// * `nr` - Active columns in the tile (`1 ≤ nr ≤ NR`)
/// * `kc` - Panel depth (number of accumulation steps)
/// * `m` - Leading dimension (row count) of `C`
///
/// # Safety
/// - `a_panel` must point to a 64-byte-aligned packed panel of at least
///   `kc * MR` elements; `b_panel` to a packed panel of at least `kc * NR`.
/// - `c_micropanel` must be valid for reads/writes of the `mr × nr` tile,
///   i.e. up to offset `(nr - 1) * m + mr`.
#[inline(always)]
pub(crate) unsafe fn kernel_16x14(
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
            let a_lo = _mm512_load_pd(a_panel.add(k * MR));
            let a_hi = _mm512_load_pd(a_panel.add(k * MR + LANE_COUNT));
            let b_row = b_panel.add(k * NR);

            tile.fmadd(a_lo, a_hi, |j| _mm512_set1_pd(*b_row.add(j)));
        }

        tile.store(c_micropanel, m);
    }
}

/// AVX-512 microkernel computing `C(0..mr, 0..nr) += A × B` directly on the
/// **unpacked** column-major inputs.
///
/// Used for matrices small enough that the working set fits in L2, where
/// packing costs more than it saves: A columns are already contiguous
/// (unaligned loads with stride `lda` between k steps) and B elements are
/// broadcast straight from their column-major locations.
///
/// The full 16×14 tile runs an unguarded hot loop; boundary tiles fall back
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
pub(crate) unsafe fn kernel_16x14_direct(
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
                let a_lo = _mm512_loadu_pd(a_col);
                let a_hi = _mm512_loadu_pd(a_col.add(LANE_COUNT));

                tile.fmadd(a_lo, a_hi, |j| _mm512_set1_pd(*b.add(j * ldb + k)));
            }
        } else {
            // Boundary tile: masked A loads, guarded B reads. The guards
            // resolve identically on every iteration, so they predict
            // perfectly.
            let zero = _mm512_setzero_pd();
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
                        _mm512_set1_pd(*b.add(j * ldb + k))
                    } else {
                        zero
                    }
                });
            }
        }

        tile.store(c_micropanel, m);
    }
}
