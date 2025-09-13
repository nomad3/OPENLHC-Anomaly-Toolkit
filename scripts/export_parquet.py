from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--tree", type=str, default="Events")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    try:
        import uproot  # type: ignore
        import awkward as ak  # type: ignore
    except Exception as e:
        raise SystemExit("This script requires uproot and awkward to read ROOT files.")

    try:
        import pyarrow  # noqa: F401
    except Exception:
        raise SystemExit("Writing Parquet requires 'pyarrow'. Please install it: pip install pyarrow")

    with uproot.open(args.input) as f:  # type: ignore
        tree = f[args.tree]
        arr = tree.arrays(library="ak")  # type: ignore
        import pandas as pd
        df = ak.to_pandas(arr)  # type: ignore
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
