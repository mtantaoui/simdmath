# Reproducible figures for the simdmath book

This directory contains the **scripts** that generate every figure embedded in
the book. The rendered PNG/SVG outputs live alongside the chapters in
`book/src/images/` and are committed to the repository so that readers (and
docs.rs) can view the book without running a build step.

## Why version the scripts?

Every figure must be reproducible from a deterministic, version-pinned recipe.
If a reader (or future maintainer) wants to know how the "sin worst-case ULP
heatmap" was made, the answer is in this directory.

## Layout

```
figures/
├── README.md                # this file
├── requirements.txt         # pinned Python dependencies
├── lib/
│   └── ulp.py               # shared ULP helpers
└── plots/
    └── ulp_sweep.py         # generates per-function ULP plots
```

## Running

```sh
cd figures
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Regenerate every figure into ../book/src/images/
python plots/ulp_sweep.py sin <csv-path> ../book/src/images/sin_ulp.svg
```

## Conventions

- **Format**: SVG for vector plots, PNG only when raster is unavoidable.
- **Style**: matplotlib's `seaborn-v0_8-whitegrid` for line plots, `viridis`
  for heatmaps. Always include axis labels and a title.
- **Determinism**: every script must seed random generators
  (`numpy.random.seed(0)`).
- **Oracle**: ULP plots use `mpmath.mp.dps = 64` for the reference values; the
  simdmath output is read from a CSV produced by `cargo run --example ulp_dump
--release` (planned for v0.2).

## Status

This directory is a **skeleton**. Figures will be added alongside the v0.2
release; v0.1 ships without rendered figures. Where book chapters reference
figures, they describe the intended plot in text.
