# Copilot Instructions for simdmath

## Project Overview

SIMD math library in Rust with implementations for AVX2, AVX-512, and ARM NEON.
All math functions must achieve **≤1 ULP** accuracy.

## Architecture Pattern

When implementing a new math function (e.g., `atan`):

1. **Constants**: Add to `src/arch/consts/<func>.rs`
2. **AVX2**: `src/arch/avx2/<func>.rs` with `_mm256_<func>_ps` and `_mm256_<func>_pd`
3. **AVX-512**: `src/arch/avx512/<func>.rs` with `_mm512_<func>_ps` and `_mm512_<func>_pd`
4. **NEON**: `src/arch/neon/<func>.rs` with `v<func>_f32` and `v<func>_f64`

## Documentation Requirements

Every function needs a `# Precision` doc section:
```rust
/// # Precision
///
/// **≤ 1 ULP** error across the entire domain.
```

## Algorithm Source

Port from musl libc. Use exact musl constants, NOT `std::f32::consts`.

## Testing Checklist

- [ ] Special values (0, -0, ±1, NaN, ±∞)
- [ ] Domain errors return NaN
- [ ] All SIMD lanes independent
- [ ] ULP sweep test

## NEON Notes

- `vbslq(mask, true_val, false_val)` - order differs from x86
- No `vmvnq_u64` - use `veorq_u64(x, all_ones)`
- FMA: `vfmaq(c, a, b)` = a*b + c (accumulator first)
