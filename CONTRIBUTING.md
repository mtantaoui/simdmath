# Contributing to simdmath

Thanks for considering a contribution! This document covers the project
conventions; the agent-facing version is in [`AGENTS.md`](./AGENTS.md).

## Development setup

```sh
git clone https://github.com/mtantaoui/simdmath
cd simdmath

# Tests on x86_64 require AVX2:
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test

# On aarch64, NEON is enabled by default:
cargo test
```

## Code conventions

- **Precision target**: every transcendental function must achieve a
  documented worst-case ULP bound and have a sweep test that exercises it.
  See [`docs/ulp-methodology.md`](./docs/ulp-methodology.md).
- **Documentation**: every public function must include the seven sections
  defined in the [rustdoc template](#rustdoc-template) below.
- **Constants**: math constants must be ported verbatim from musl libc /
  fdlibm, with a comment citing the source file and line. Do not substitute
  `std::f32::consts` or `std::f64::consts`.
- **Style**: `cargo fmt` is enforced in CI. Clippy with `-D warnings` is
  also enforced.
- **NEON quirks**: see [`AGENTS.md`](./AGENTS.md) for the catalogue of
  `vbslq` ordering, `vmvnq_u64` emulation, and FMA accumulator-position
  pitfalls.

## Rustdoc template

Every public function gets a doc comment with these sections, in order:

```rust,ignore
/// Brief one-line summary.
///
/// # Mathematical definition
/// $$ f(x) = \dots $$
///
/// # Special values
/// (table of IEEE-754 special inputs)
///
/// # Algorithm
/// (1–2 paragraphs, key equation, reference to book chapter)
///
/// # Error analysis
/// **≤ N ULP** across the entire domain.
/// (one sentence on how the bound is obtained)
///
/// # Performance notes
/// (lanes per instruction, FMA usage)
///
/// # References
/// - musl `src/math/<file>.c`
/// - (papers, book chapter)
///
/// # Examples
/// ```rust,ignore
/// use simdmath::math::VecMath;
/// let v = vec![0.0f32, 1.0, 2.0];
/// let r = v.sin();
/// ```
```

## Mathematical-documentation style

When writing book chapters or rustdoc with KaTeX math, follow the style
guide in [`book/src/STYLE.md`](./book/src/STYLE.md).

## Pull requests

1. Open an issue first for non-trivial changes.
2. One topic per PR.
3. Include or update tests; ULP-affecting changes must include a sweep test.
4. Update the relevant book chapter and the changelog.
5. Run `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`
   locally before pushing.

## License

By contributing you agree that your contributions will be licensed under the
MIT License.
