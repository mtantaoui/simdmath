"""Skeleton ULP-sweep figure generator (placeholder for v0.2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("function", help="Function name (e.g. sin, cos, exp)")
    parser.add_argument("csv", type=Path, help="CSV input from cargo run --example ulp_dump")
    parser.add_argument("out", type=Path, help="Output figure path (.svg or .png)")
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(
            f"figures: {args.csv} does not exist. The `ulp_dump` example is "
            "planned for v0.2; until then this script is a placeholder.",
            file=sys.stderr,
        )
        return 1

    print(f"figures: would generate {args.out} for {args.function} from {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
