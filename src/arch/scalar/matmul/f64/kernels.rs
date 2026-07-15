//! Scalar microkernels (f64) for the BLIS-style matrix multiplication.
//!
//! The 4×6 microkernel is the computational core of
//! [`matmul`](crate::arch::scalar::matmul::f64::matmul): it multiplies one
//! packed `MR × kc` A panel with one packed `kc × NR` B panel, accumulating
//! into an up-to-4×6 tile of the column-major output matrix `C`.
//!
//! The 4×6 shape is the AVX2 backend's 8×6 halved to the width the
//! auto-vectoriser can rely on everywhere (SSE2 on baseline `x86_64`): the
//! accumulator tile is a plain `[[f64; 4]; 6]` array updated in
//! constant-trip loops, which the compiler keeps in vector registers on
//! targets that have them. Products use `a * b + acc` rather than
//! `f64::mul_add`: without FMA hardware, `mul_add` falls back to a slow
//! correctly-rounded software `fma`, whereas separate multiply/add lowers
//! to plain vector instructions.

use crate::arch::scalar::matmul::f64::{MR, NR};

/// Scalar microkernel computing `C(0..mr, 0..nr) += A_panel × B_panel`.
///
/// Implements the outer-product formulation: for each of the `kc` steps, the
/// A column (`MR` contiguous elements) is multiplied by each of the `NR` B
/// row elements. Columns beyond `nr` multiply the B panel's zero padding
/// into dummy accumulator rows that are never stored, so the hot loop needs
/// no column guards.
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
pub(crate) unsafe fn kernel_4x6(
    a_panel: *const f64,
    b_panel: *const f64,
    c_micropanel: *mut f64,
    mr: usize,
    nr: usize,
    kc: usize,
    m: usize,
) {
    unsafe {
        // Load the active part of the C tile; inactive rows/columns stay
        // zero and are never stored back.
        let mut acc = [[0.0f64; MR]; NR];
        for (j, col) in acc.iter_mut().enumerate().take(nr) {
            for (i, elem) in col.iter_mut().enumerate().take(mr) {
                *elem = *c_micropanel.add(j * m + i);
            }
        }

        for k in 0..kc {
            let a_col = a_panel.add(k * MR);
            let b_row = b_panel.add(k * NR);

            // Constant trip counts so the loops fully unroll and the
            // accumulator array stays in registers.
            for (j, col) in acc.iter_mut().enumerate() {
                let b = *b_row.add(j);
                for (i, elem) in col.iter_mut().enumerate() {
                    *elem += *a_col.add(i) * b;
                }
            }
        }

        for (j, col) in acc.iter().enumerate().take(nr) {
            for (i, elem) in col.iter().enumerate().take(mr) {
                *c_micropanel.add(j * m + i) = *elem;
            }
        }
    }
}

/// Scalar microkernel computing `C(0..mr, 0..nr) += A × B` directly on the
/// **unpacked** column-major inputs.
///
/// Used for matrices small enough that the working set fits in L2, where
/// packing costs more than it saves: A columns are already contiguous and B
/// elements are read straight from their column-major locations.
///
/// The full 4×6 tile runs an unguarded hot loop; boundary tiles fall back
/// to a guarded loop that zero-pads the A column (edge tiles are a small
/// fraction of the work).
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
pub(crate) unsafe fn kernel_4x6_direct(
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
        let mut acc = [[0.0f64; MR]; NR];
        for (j, col) in acc.iter_mut().enumerate().take(nr) {
            for (i, elem) in col.iter_mut().enumerate().take(mr) {
                *elem = *c_micropanel.add(j * m + i);
            }
        }

        if mr == MR && nr == NR {
            // Full tile: unguarded hot loop.
            for k in 0..kc {
                let a_col = a.add(k * lda);
                for (j, col) in acc.iter_mut().enumerate() {
                    let bj = *b.add(j * ldb + k);
                    for (i, elem) in col.iter_mut().enumerate() {
                        *elem += *a_col.add(i) * bj;
                    }
                }
            }
        } else {
            // Boundary tile: zero-pad the A column, guard the B reads. The
            // guards resolve identically on every iteration, so they
            // predict perfectly.
            for k in 0..kc {
                let a_col = a.add(k * lda);
                let mut a_pad = [0.0f64; MR];
                for (i, elem) in a_pad.iter_mut().enumerate().take(mr) {
                    *elem = *a_col.add(i);
                }

                for (j, col) in acc.iter_mut().enumerate() {
                    let bj = if j < nr { *b.add(j * ldb + k) } else { 0.0 };
                    for (i, elem) in col.iter_mut().enumerate() {
                        *elem += a_pad[i] * bj;
                    }
                }
            }
        }

        for (j, col) in acc.iter().enumerate().take(nr) {
            for (i, elem) in col.iter().enumerate().take(mr) {
                *c_micropanel.add(j * m + i) = *elem;
            }
        }
    }
}
