# Appendix E — Bibliography

The references cited throughout this book and through the rustdoc of
`simdmath`. Inline citations use the BibTeX-style key shown after each
entry, e.g. `[Goldberg1991]` in prose maps to entry [1] below.

The entries are also provided as machine-readable BibTeX inside
`<pre class="bibtex">` blocks at the bottom of this page. The
[`docs/docs-header.html`](https://github.com/mtantaoui/simdmath/blob/main/docs/docs-header.html)
header used by **rustdoc** loads `bibtex-js` and renders these blocks
automatically; **mdBook does not** load `bibtex-js`, so the numbered list
above is the authoritative human-readable form. The BibTeX is kept here so
that downstream tools (Zotero, BibLaTeX, the rustdoc build) can ingest the
same source.

## Numbered references

[1] Goldberg, D. *What Every Computer Scientist Should Know About Floating-
Point Arithmetic*. ACM Computing Surveys, 23(1):5–48, March 1991.
DOI:10.1145/103162.103163. <https://dl.acm.org/doi/10.1145/103162.103163>.
Key: `Goldberg1991`.

[2] Muller, J.-M., Brunie, N., de Dinechin, F., Jeannerod, C.-P., Joldes, M.,
Lefèvre, V., Melquiond, G., Revol, N., Torres, S. *Handbook of Floating-Point
Arithmetic*. 2nd edition, Birkhäuser, 2018. ISBN 978-3-319-76525-9.
DOI:10.1007/978-3-319-76526-6. Key: `Muller2018`.

[3] Muller, J.-M. *Elementary Functions: Algorithms and Implementation*.
3rd edition, Birkhäuser, 2016. ISBN 978-1-4899-7981-0.
DOI:10.1007/978-1-4899-7983-4. Key: `Muller2016`.

[4] Knuth, D. E. *The Art of Computer Programming, Volume 2:
Seminumerical Algorithms*. 3rd edition, Addison-Wesley, 1997.
ISBN 0-201-89684-2. (TwoSum and accurate summation algorithms appear in
§4.2.2.) Key: `Knuth1997`.

[5] Dekker, T. J. *A Floating-Point Technique for Extending the Available
Precision*. Numerische Mathematik, 18(3):224–242, 1971.
DOI:10.1007/BF01397083. Key: `Dekker1971`.

[6] Cody, W. J. and Waite, W. *Software Manual for the Elementary
Functions*. Prentice-Hall, 1980. ISBN 0-13-822064-6. (Origin of the
two-part splitting of \\(\pi/2\\) and \\(\ln 2\\) used in `simdmath`.)
Key: `CodyWaite1980`.

[7] Payne, M. H. and Hanek, R. N. *Radian Reduction for Trigonometric
Functions*. ACM SIGNUM Newsletter, 18(1):19–24, January 1983.
DOI:10.1145/1057600.1057602. Key: `PayneHanek1983`.

[8] IEEE Computer Society. *IEEE Standard for Floating-Point Arithmetic*.
IEEE Std 754-2008, 29 August 2008. DOI:10.1109/IEEESTD.2008.4610935.
Key: `IEEE754_2008`.

[9] IEEE Computer Society. *IEEE Standard for Floating-Point Arithmetic*.
IEEE Std 754-2019 (revision of 754-2008), 22 July 2019.
DOI:10.1109/IEEESTD.2019.8766229. Key: `IEEE754_2019`.

[10] Kahan, W. *Pracniques: Further Remarks on Reducing Truncation
Errors*. Communications of the ACM, 8(1):40, January 1965.
DOI:10.1145/363707.363723. (Compensated summation.)
Key: `Kahan1965`.

[11] Sun Microsystems. *fdlibm — Freely Distributable Math Library*,
version 5.3, 1993–2008. K.-C. Ng et al. <https://www.netlib.org/fdlibm/>.
The algorithmic ancestor of musl libm and therefore of every kernel in
`simdmath`. Key: `fdlibm1993`.

[12] musl libc project. *Source code: src/math/*. <https://musl.libc.org/>
(file-by-file source mirror at <https://git.musl-libc.org/cgit/musl/tree/src/math>).
Specifically: `sinf.c`, `sin.c`, `cosf.c`, `cos.c`, `tanf.c`, `tan.c`,
`asinf.c`, `asin.c`, `acosf.c`, `acos.c`, `atanf.c`, `atan.c`,
`atan2f.c`, `atan2.c`, `expf.c`, `exp.c`, `logf.c`, `log.c`, `cbrtf.c`,
`cbrt.c`. Key: `musl`.

[13] Shibata, N. and Petrogalli, F. *SLEEF: A Portable Vectorized Library
of C Standard Mathematical Functions*. IEEE Transactions on Parallel and
Distributed Systems, 31(6):1316–1327, June 2020.
DOI:10.1109/TPDS.2019.2960333. Key: `SLEEF2020`.

[14] Intel Corporation. *Intel® C++ Compiler Classic Developer Guide and
Reference: Short Vector Math Library Operations (SVML)*. Intel oneAPI
documentation, 2021. <https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/short-vector-math-library-operations-svml.html>.
Key: `IntelSVML`.

[15] ARM Ltd. *Optimized Routines (mathlib subdirectory)*.
<https://github.com/ARM-software/optimized-routines>. Key: `ArmMathlib`.

[16] ISO/IEC. *Programming languages — C*. ISO/IEC 9899:1999 (C99),
§7.12 *Mathematics `<math.h>`*. (Specifies special-value behaviour for
elementary functions, which `simdmath` follows.) Key: `C99`.

[17] Markstein, P. *IA-64 and Elementary Functions: Speed and Precision*.
Prentice-Hall, 2000. ISBN 0-13-018348-2. (Reference for FMA-based
polynomial evaluation and Markstein-style argument reduction.)
Key: `Markstein2000`.

[18] Higham, N. J. *Accuracy and Stability of Numerical Algorithms*.
2nd edition, SIAM, 2002. ISBN 0-89871-521-0.
DOI:10.1137/1.9780898718027. (Backward-error analysis used implicitly
in the [methodology chapter](../precision/methodology.md).)
Key: `Higham2002`.

[19] Sterbenz, P. H. *Floating-Point Computation*. Prentice-Hall, 1974.
(Sterbenz lemma: \\(a/2 \le b \le 2a \implies a - b\\) is exact.)
Key: `Sterbenz1974`.

[20] Veltkamp, G. W. *ALGOL Procedures voor het Berekenen van een
Inwendig Product in Dubbele Precisie*. RC-Informatie 22, Technische
Hogeschool Eindhoven, 1968. (Veltkamp splitting, the precursor of
Dekker's product.) Key: `Veltkamp1968`.

## BibTeX source

The same entries in BibTeX form, for ingestion by reference managers and
the rustdoc `bibtex-js` integration:

<pre class="bibtex">
@article{Goldberg1991,
  author  = {Goldberg, David},
  title   = {What Every Computer Scientist Should Know About Floating-Point Arithmetic},
  journal = {ACM Computing Surveys},
  volume  = {23}, number = {1}, pages = {5--48},
  year    = {1991}, month = mar,
  doi     = {10.1145/103162.103163}
}

@book{Muller2018,
  author    = {Muller, Jean-Michel and Brunie, Nicolas and de Dinechin, Florent
               and Jeannerod, Claude-Pierre and Joldes, Mioara and Lef{\`e}vre, Vincent
               and Melquiond, Guillaume and Revol, Nathalie and Torres, Serge},
  title     = {Handbook of Floating-Point Arithmetic},
  edition   = {2nd},
  publisher = {Birkh{\"a}user},
  year      = {2018},
  isbn      = {978-3-319-76525-9},
  doi       = {10.1007/978-3-319-76526-6}
}

@book{Muller2016,
  author    = {Muller, Jean-Michel},
  title     = {Elementary Functions: Algorithms and Implementation},
  edition   = {3rd},
  publisher = {Birkh{\"a}user},
  year      = {2016},
  isbn      = {978-1-4899-7981-0},
  doi       = {10.1007/978-1-4899-7983-4}
}

@book{Knuth1997,
  author    = {Knuth, Donald E.},
  title     = {The Art of Computer Programming, Volume 2: Seminumerical Algorithms},
  edition   = {3rd},
  publisher = {Addison-Wesley},
  year      = {1997},
  isbn      = {0-201-89684-2}
}

@article{Dekker1971,
  author  = {Dekker, Theodorus J.},
  title   = {A Floating-Point Technique for Extending the Available Precision},
  journal = {Numerische Mathematik},
  volume  = {18}, number = {3}, pages = {224--242},
  year    = {1971},
  doi     = {10.1007/BF01397083}
}

@book{CodyWaite1980,
  author    = {Cody, William J. and Waite, William},
  title     = {Software Manual for the Elementary Functions},
  publisher = {Prentice-Hall},
  year      = {1980},
  isbn      = {0-13-822064-6}
}

@article{PayneHanek1983,
  author  = {Payne, Mary H. and Hanek, Robert N.},
  title   = {Radian Reduction for Trigonometric Functions},
  journal = {ACM SIGNUM Newsletter},
  volume  = {18}, number = {1}, pages = {19--24},
  year    = {1983}, month = jan,
  doi     = {10.1145/1057600.1057602}
}

@misc{IEEE754_2008,
  title        = {{IEEE} Standard for Floating-Point Arithmetic},
  howpublished = {IEEE Std 754-2008},
  year         = {2008},
  doi          = {10.1109/IEEESTD.2008.4610935}
}

@misc{IEEE754_2019,
  title        = {{IEEE} Standard for Floating-Point Arithmetic},
  howpublished = {IEEE Std 754-2019 (revision of 754-2008)},
  year         = {2019},
  doi          = {10.1109/IEEESTD.2019.8766229}
}

@article{Kahan1965,
  author  = {Kahan, William},
  title   = {Pracniques: Further Remarks on Reducing Truncation Errors},
  journal = {Communications of the ACM},
  volume  = {8}, number = {1}, pages = {40},
  year    = {1965}, month = jan,
  doi     = {10.1145/363707.363723}
}

@misc{fdlibm1993,
  author       = {Ng, K.-C. and {Sun Microsystems}},
  title        = {fdlibm --- Freely Distributable Math Library},
  howpublished = {Version 5.3},
  year         = {1993--2008},
  url          = {https://www.netlib.org/fdlibm/}
}

@misc{musl,
  title        = {musl libc --- math sources},
  author       = {{musl libc project}},
  howpublished = {\url{https://git.musl-libc.org/cgit/musl/tree/src/math}}
}

@article{SLEEF2020,
  author  = {Shibata, Naoki and Petrogalli, Francesco},
  title   = {{SLEEF}: A Portable Vectorized Library of {C} Standard Mathematical Functions},
  journal = {IEEE Transactions on Parallel and Distributed Systems},
  volume  = {31}, number = {6}, pages = {1316--1327},
  year    = {2020}, month = jun,
  doi     = {10.1109/TPDS.2019.2960333}
}

@misc{IntelSVML,
  author       = {{Intel Corporation}},
  title        = {{Intel}\textregistered{} {C++} Compiler Classic Developer Guide and Reference: Short Vector Math Library Operations ({SVML})},
  year         = {2021},
  url          = {https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/short-vector-math-library-operations-svml.html}
}

@misc{ArmMathlib,
  author       = {{ARM Ltd.}},
  title        = {Optimized Routines (mathlib subdirectory)},
  url          = {https://github.com/ARM-software/optimized-routines}
}

@techreport{C99,
  author      = {{ISO/IEC}},
  title       = {Programming languages --- {C}},
  institution = {ISO/IEC},
  number      = {9899:1999},
  year        = {1999}
}

@book{Markstein2000,
  author    = {Markstein, Peter},
  title     = {{IA-64} and Elementary Functions: Speed and Precision},
  publisher = {Prentice-Hall},
  year      = {2000},
  isbn      = {0-13-018348-2}
}

@book{Higham2002,
  author    = {Higham, Nicholas J.},
  title     = {Accuracy and Stability of Numerical Algorithms},
  edition   = {2nd},
  publisher = {SIAM},
  year      = {2002},
  isbn      = {0-89871-521-0},
  doi       = {10.1137/1.9780898718027}
}

@book{Sterbenz1974,
  author    = {Sterbenz, Pat H.},
  title     = {Floating-Point Computation},
  publisher = {Prentice-Hall},
  year      = {1974}
}

@techreport{Veltkamp1968,
  author      = {Veltkamp, G. W.},
  title       = {{ALGOL} Procedures voor het Berekenen van een Inwendig Product in Dubbele Precisie},
  institution = {Technische Hogeschool Eindhoven},
  number      = {RC-Informatie 22},
  year        = {1968}
}
</pre>
