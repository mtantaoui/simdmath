#[cfg(not(doc))]
compile_error!(
    "SSE backend is not yet implemented. simdmath requires AVX2 (x86_64) or NEON (aarch64). \
     Enable AVX2 with RUSTFLAGS='-C target-feature=+avx2,+fma' or target an aarch64 platform."
);
