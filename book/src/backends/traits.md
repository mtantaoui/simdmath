# The `Load`, `Store`, `Align`, and `Math` traits

Every backend in this crate implements the same four traits, defined in
[`src/ops/simd.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/ops/simd.rs).
This chapter walks through each trait method, its safety contract, and the
per-backend implementation strategy. It also documents the v0.1 fix that
changed `store_at` from a `*const T` signature to `*mut T` for
pointer-provenance soundness.

## Trait roster

| Trait      | Purpose                                                  |
|------------|----------------------------------------------------------|
| `Align<T>` | Pointer alignment checks                                 |
| `Load<T>`  | Loading data from memory into a SIMD register            |
| `Store<T>` | Writing a SIMD register back to memory                   |
| `Math`     | Element-wise / fully-parallel mathematical operations    |

All four are `pub(crate)` — register types are not part of the v0.1 public
API. The public surface is the [`VecMath`](../functions/sin.md) trait on
`Vec<T>`.

## `Align<T>`

```rust,ignore
pub(crate) trait Align<T> {
    fn is_aligned(ptr: *const T) -> bool;
}
```

The single method reports whether a pointer meets the **natural alignment**
of the implementing register type:

| Backend  | Register type   | Required alignment |
|----------|------------------|--------------------|
| AVX2     | `__m256` / `__m256d` | 32 bytes        |
| AVX-512  | `__m512` / `__m512d` | 64 bytes        |
| NEON     | `float32x4_t` / `float64x2_t` | 16 bytes |

The check is the standard pointer-as-`usize` modulus:

```rust,ignore
fn is_aligned(ptr: *const f32) -> bool {
    (ptr as usize).is_multiple_of(core::mem::align_of::<__m256>())
}
```

This is consumed by `Load::load` and `Store::store_at` to dispatch to the
faster aligned variant on x86; on NEON the result is informational because
the unaligned variant is the same instruction.

## `Load<T>`

```rust,ignore
pub(crate) trait Load<T> {
    type Output;

    unsafe fn load(ptr: *const T, size: usize) -> Self::Output;
    unsafe fn load_aligned(ptr: *const T) -> Self::Output;
    unsafe fn load_unaligned(ptr: *const T) -> Self::Output;
    unsafe fn load_partial(ptr: *const T, size: usize) -> Self::Output;
    unsafe fn broadcast(val: T) -> Self::Output;
    unsafe fn zero() -> Self::Output;
}
```

### `load(ptr, size)`

The primary entry point for full-register loads. Inspects the pointer with
`Align::is_aligned` and forwards to either `load_aligned` or
`load_unaligned`. Asserts in debug builds that `size == LANE_COUNT`.

**Safety**: `ptr` must be non-null and valid for `size` reads.

### `load_aligned` / `load_unaligned`

The two backend-specific full-register paths.

| Backend  | Aligned                       | Unaligned                        |
|----------|-------------------------------|----------------------------------|
| AVX2     | `_mm256_load_ps`              | `_mm256_loadu_ps`                |
| AVX-512  | `_mm512_load_ps`              | `_mm512_loadu_ps`                |
| NEON     | `vld1q_f32`                   | `vld1q_f32` (same instruction)   |

**Safety**: `ptr` must be non-null. `load_aligned` additionally requires
the pointer to be aligned to the register boundary; passing a misaligned
pointer to `load_aligned` will fault on x86 (general protection).

### `load_partial(ptr, size)`

The "tail" variant used when fewer than `LANE_COUNT` elements remain.
**Inactive lanes are zeroed**, so subsequent arithmetic is well-defined
(no spurious infinities or NaNs from uninitialised memory).

| Backend  | Strategy                                                  |
|----------|-----------------------------------------------------------|
| AVX2     | `_mm256_maskload_ps` driven by a precomputed sign-bit table |
| AVX-512  | `_mm512_maskz_loadu_ps` with `mask = (1 << size) - 1`    |
| NEON     | element-wise scalar fill into a stack buffer + `vld1q_*` |

**Safety**: `ptr` non-null and valid for `size` reads; `size < LANE_COUNT`.

### `broadcast(val)` and `zero()`

Set every lane to `val` (resp. `0`). Compile to `_mm256_set1_*`,
`_mm512_set1_*`, `vdupq_n_*`. The `size` field is initialised to
`LANE_COUNT` so that an immediate `store_at` writes a full register.

## `Store<T>`

```rust,ignore
pub(crate) trait Store<T> {
    type Output;

    unsafe fn store_at(&self, ptr: *mut T);
    unsafe fn stream_at(&self, ptr: *mut T);
    unsafe fn store_aligned_at(&self, ptr: *mut T);
    unsafe fn store_unaligned_at(&self, ptr: *mut T);
    unsafe fn store_at_partial(&self, ptr: *mut T);
}
```

Each method mirrors a `Load` counterpart: full / aligned / unaligned /
partial, plus a streaming variant for write-once buffers.

### `store_at` / `store_aligned_at` / `store_unaligned_at`

Same dispatch shape as the corresponding `load*` methods. Maps to
`_mm256_store_ps` / `_mm256_storeu_ps`, `_mm512_store_ps` /
`_mm512_storeu_ps`, `vst1q_f32`. **Safety** matches the load contract:
non-null, valid for a full-register write, aligned where required.

### `stream_at`

Non-temporal (streaming) store that bypasses the CPU cache. Useful for
large output buffers that will not be re-read soon. Maps to
`_mm256_stream_ps`, `_mm512_stream_ps`, etc. A subsequent `_mm_sfence` is
required before any reader can observe the stored bytes; the trait does
not insert this fence.

### `store_at_partial`

Writes only the first `self.size` lanes. Inactive lanes are **not**
written — the destination memory is left unchanged. This is what makes it
safe to apply tail handling at the end of a slice without overrunning the
caller's allocation.

| Backend  | Strategy                                                  |
|----------|-----------------------------------------------------------|
| AVX2     | `_mm256_maskstore_ps` driven by the sign-bit table        |
| AVX-512  | `_mm512_mask_storeu_ps` with `mask = (1 << size) - 1`     |
| NEON     | element-wise scalar copy from the register                |

## The `*mut T` signature: pointer-provenance soundness fix

Earlier development snapshots of the crate had `store_at` typed as

```rust,ignore
unsafe fn store_at(&self, ptr: *const T);     // historical
```

with the implementation casting the pointer to `*mut T` internally:

```rust,ignore
_mm256_storeu_ps(ptr as *mut f32, self.elements);
```

This was unsound under [Stacked Borrows](https://plv.mpi-sws.org/rustbelt/stacked-borrows/)
and the in-progress [Tree Borrows](https://perso.crans.org/vanille/treebor/)
aliasing models: writing through a pointer derived from a `*const T`
violates pointer provenance because the original pointer never had write
provenance, and Miri reports this with `error: trying to retag for
SharedReadWrite, but parent tag <…> does not have an appropriate item in
the borrow stack`.

The v0.1 fix changed the signature to `*mut T`:

```rust,ignore
unsafe fn store_at(&self, ptr: *mut T);       // current
```

This pushes the responsibility of producing a `*mut`-provenance pointer
onto the caller — typically `Vec::as_mut_ptr()`, which produces a
write-provenance pointer by construction. The internal store no longer
laundering provenance, and Miri runs cleanly on the test suite.

The change is **API-breaking only at the trait level**; the trait is
`pub(crate)` and the `Vec<T>` `VecMath` callers all hold mutable
references already, so the practical impact on callers was zero.

## `Math`

The `Math` trait is the per-register element-wise / fully-parallel
arithmetic surface. Each method comes in two tiers:

- **Sequential** (`fn name`) — applies the scalar libm operation lane-by-
  lane. Used for parity testing.
- **Parallel** (`fn par_name`) — applies the SIMD-native polynomial
  kernel to all lanes simultaneously.

Where a hardware instruction exists (`_mm256_sqrt_ps`,
`_mm256_xor_ps` for sign-bit clear in `abs`), it is used directly. The
transcendentals (sin, cos, tan, asin, acos, atan, exp, ln, pow, cbrt) all
fall in the second category and dispatch to the per-function modules
documented in their dedicated chapters.

## Per-architecture implementation strategy

| Trait method   | AVX2                      | AVX-512                       | NEON                           |
|----------------|---------------------------|-------------------------------|--------------------------------|
| `is_aligned`   | mod 32                    | mod 64                        | mod 16                         |
| `load`         | aligned-or-unaligned `ps`/`pd` | aligned-or-unaligned `ps`/`pd` | unified `vld1q_*` |
| `load_partial` | `maskload` + sign-bit table | `maskz_loadu` + bit mask    | scalar fill + `vld1q_*`        |
| `store_at`     | aligned-or-unaligned `ps`/`pd` | aligned-or-unaligned `ps`/`pd` | unified `vst1q_*` |
| `store_at_partial` | `maskstore`            | `mask_storeu`                 | scalar copy                    |
| `stream_at`    | `_mm256_stream_*`          | `_mm512_stream_*`             | (not implemented; falls back)  |

## Where to look in the source

| Topic | File |
|-------|------|
| Trait definitions       | [`ops/simd.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/ops/simd.rs) |
| AVX2 impls              | [`avx2/f32x8.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/f32x8.rs), [`avx2/f64x4.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx2/f64x4.rs) |
| AVX-512 impls           | [`avx512/f32x16.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx512/f32x16.rs), [`avx512/f64x8.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/avx512/f64x8.rs) |
| NEON impls              | [`neon/f32x4.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/f32x4.rs), [`neon/f64x2.rs`](https://github.com/mtantaoui/simdmath/blob/main/src/arch/neon/f64x2.rs) |

## See also

- [AVX2](./avx2.md), [AVX-512](./avx512.md), [NEON](./neon.md) backends.
- [Compile-time dispatch](./dispatch.md) — how the right impl is picked.
