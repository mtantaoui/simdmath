# Changelog

All notable changes to `simdmath` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- KaTeX-rendered math in rustdoc and the new mdBook (`book/`).
- Mathematical-reference book scaffolding under `book/` covering foundations,
  per-function derivations, and appendices.
- ULP measurement methodology document (`docs/ulp-methodology.md`).
- Full `Cargo.toml` publish metadata (description, license, repository,
  keywords, categories, MSRV, exclude list, `[package.metadata.docs.rs]`).
- README rewritten with badges, target table, install instructions, usage
  examples, function inventory, and prior-art comparison.

### Changed

- `.cargo/config.toml`: replaced `-C target-cpu=native` (which leaked to
  downstream consumers) with rustdoc-only flags for KaTeX injection.
- AVX2 CI: added `+fma` to `RUSTFLAGS`.
- NEON CI: added a `fmt` job.
- AVX-512 CI: switched the `test` job to `cargo test --no-run` because
  GitHub-hosted runners do not provide AVX-512 instructions.

### Removed

- Stale `commit_message.txt` working file.

## [0.1.0] — unreleased

Initial release. See README for the supported feature set.
