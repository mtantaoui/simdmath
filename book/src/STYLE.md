# Mathematical-documentation style guide

This file is contributor-facing — it does not appear in `SUMMARY.md` and is
not rendered as a book chapter.

## Notation

| Concept                          | Notation                              |
|----------------------------------|---------------------------------------|
| Real-valued ideal function       | \\(f\\)                                   |
| Finite-precision approximation   | \\(\hat{f}\\)                             |
| Real-valued inputs               | \\(x, y, r, k\\) (lowercase italic)       |
| Floating-point inputs            | \\(\hat{x}, \hat{y}\\)                    |
| Unit roundoff                    | \\(\varepsilon\\)                         |
| ULP function                     | \\(\mathrm{ulp}(\cdot)\\)                 |
| Floor / ceiling                  | \\(\lfloor \cdot \rfloor, \lceil \cdot \rceil\\) |
| High / low parts of a hi-lo split | \\(a_\text{hi}, a_\text{lo}\\)           |
| Polynomial in \\(r\\) of degree \\(n\\)  | \\(P_n(r)\\)                              |
| Reference oracle                 | \\(f^\star\\)                             |

## Math delimiters

- Inline math: `\\( x + y \\)`.
- Display math: `\\[ x + y \\]` on its own paragraph.
- mdBook is configured with `mathjax-support = true`; MathJax recognises
  **only** the `\\(..\\)` and `\\[..\\]` delimiters. The `$..$` / `$$..$$`
  KaTeX-style delimiters do **not** render and will appear as literal
  dollar signs in the output.
- This convention matches the companion `Integrate` book.
- Do **not** use `\begin{equation}` / `\begin{align}` — MathJax's
  auto-render does not recognise them inside markdown; use
  `\begin{aligned}` *inside* a display math block instead.

## Section ordering for function chapters

The 12-section template is mandatory. Use the same `##` headings:

1. Mathematical statement
2. Special values
3. Naive algorithm and why it fails
4. Argument reduction
5. Polynomial approximation
6. Reconstruction
7. End-to-end error analysis
8. SIMD considerations
9. Per-backend code walk-through
10. Empirical ULP results
11. Benchmarks
12. References

## Citations

References appear at the end of each chapter. Use BibTeX keys consistent
with `book/src/appendices/E_bibliography.md`. Inline citations use
`[Muller2016]` not footnotes.

## Figure naming

`book/src/images/<chapter>_<topic>.svg` — e.g. `atan2_quadrants.svg`,
`exp_overflow_cliff.svg`. Provide both SVG (preferred) and PNG fallback.

## Code blocks

- Rust code blocks that should be compiled by `mdbook test`: open the
  fence with ```` ```rust,ignore ````.
- Pseudocode: open the fence with ```` ```text ````.

## Cross-linking

- From book to rustdoc: ``[`F32x8`](https://docs.rs/simdmath/latest/simdmath/...)``
- From book to source: relative GitHub link
  `[arch/avx2/exp.rs](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/exp.rs)`.
- From rustdoc to book: include a sentence
  `// See the [book chapter on exp](https://mtantaoui.github.io/simdmath/functions/exp.html) for the full derivation.`
