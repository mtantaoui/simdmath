"""Shared ULP-error helpers used by figure-generation scripts.

ULP error of a finite, non-zero approximation `f_hat` to the true value `f`
is defined here as

    ulp_error(f_hat, f) = |to_bits(f_hat) - to_bits(f_rounded)|

where `f_rounded` is `f` rounded to nearest in the target precision. This
matches the methodology described in `docs/ulp-methodology.md`.
"""

from __future__ import annotations

import math
import struct


def f64_bits(x: float) -> int:
    """Reinterpret an f64 as a 64-bit unsigned integer."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def f64_from_bits(bits: int) -> float:
    """Reinterpret a 64-bit unsigned integer as an f64."""
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def ulp_error_f64(f_hat: float, f_ref: float) -> float:
    """Return |to_bits(f_hat) - to_bits(f_ref)| as a float.

    Both inputs must be finite. NaN/infinity handling is delegated to the
    caller.
    """
    if not (math.isfinite(f_hat) and math.isfinite(f_ref)):
        raise ValueError("ulp_error_f64 expects finite arguments")
    a = f64_bits(f_hat)
    b = f64_bits(f_ref)
    if a >> 63:
        a = (1 << 63) - (a & ((1 << 63) - 1))
    if b >> 63:
        b = (1 << 63) - (b & ((1 << 63) - 1))
    return float(abs(a - b))
