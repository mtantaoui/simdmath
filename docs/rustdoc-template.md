# `simdmath` rustdoc template

This document is the **single source of truth** for the structure of
per-function rustdoc comments in `simdmath`. Every function on the
[`VecMath`](crate::math::VecMath) and [`VecExt`](crate::ops::vec::VecExt)
traits, and every `_mm{256,512}_<func>_{ps,pd}` / `v<func>_f{32,64}` wrapper
under `src/arch/`, follows this template.

The template is intentionally short. The deep derivations live in the
[mathematical reference book](../book/) — rustdoc points at the relevant
book chapter via a final cross-reference, but never duplicates the proof.

## Module-level (`//!`) block

For each per-function module under `src/arch/<backend>/<func>.rs`:

```rust
//! <BACKEND> SIMD implementation of `<func>(x[, y, ...])` for `f32` and `f64` vectors.
//!
//! # Algorithm
//!
//! 1. <Reduction step>
//! 2. <Approximation step>
//! 3. <Reconstruction / sign handling>
//!
//! ## f32
//! <variant-specific notes>
//!
//! ## f64
//! <variant-specific notes>
//!
//! # Special values
//!
//! - `<func>(NaN) = NaN`
//! - <other IEEE 754 special cases>
//!
//! # References
//!
//! - musl libc: `<file>.c`
//! - simdmath book: [`<chapter-slug>`](../../../book/src/functions/<func>.md)
```

## Trait method (`///`) block

For each method on `VecMath` / `VecExt`:

```rust
/// <One-sentence summary in the indicative present tense.>
///
/// <Optional 1-3 sentence description: input semantics, broadcasting,
/// allocation behaviour. Avoid restating what the type signature says.>
///
/// # Panics
///
/// <Only if applicable. Always present for binary methods that require
/// equal lengths.>
///
/// # Precision
///
/// **≤ N ULP** error across <domain description>.
///
/// The single source of truth for these numbers is the table at the top of
/// [`crate`]. Changes here must be reflected there.
///
/// # Special values
///
/// - <bullet list of IEEE 754 special-value behaviour>
///
/// # Examples
///
/// ```rust,ignore
/// use simdmath::prelude::*;
/// let v = vec![1.0_f32, 2.0, 3.0];
/// let r = v.<func>();
/// assert_eq!(r[0], <expected>);
/// ```
///
/// # See also
///
/// - Book chapter: [<func>](../../book/src/functions/<func>.md)
```

## Required sections

Every public method **must** have, in order:

1. One-sentence summary (mandatory).
2. `# Panics` (only if it can panic).
3. `# Precision` (mandatory; must cite the same ULP number as
   [`crate`]'s accuracy table).
4. `# Special values` (mandatory for transcendentals; optional for `abs`).
5. `# Examples` (recommended).

## Conventions

- **ULP claims** are written as `**≤ N ULP**` (markdown bold, ≤ U+2264).
  Never use `<= N ULP` or `≤N ULP`.
- **KaTeX math** is allowed inline (`$..$`) or display (`$$..$$`); rendering
  is enabled by `--html-in-header docs/docs-header.html`.
- **Special-value bullets** use the form `<func>(<input>) = <output>`.
- **Cross-references**: link to other items via `[`Type::method`]`. Link to
  book chapters with relative `../../book/src/...` paths so docs.rs and
  GitHub both resolve them.
- **`# Safety`** is required on every `unsafe fn` (intrinsic wrappers).

## Anti-patterns

- ❌ Restating the type signature ("`x: f32` is the input").
- ❌ Listing musl coefficients in rustdoc (put them in `arch/consts/`).
- ❌ Embedding multi-page derivations — link to the book instead.
- ❌ Mismatched ULP numbers between rustdoc and the
  [`crate`]-level accuracy table.
