from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_parquet(output: Path, n: int = 10000, d: int = 5, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    X_b = rng.normal(0, 1, size=(n, d))
    cols = [f"f{i+1}" for i in range(d)]
    df = pd.DataFrame(X_b, columns=cols)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(output)
    except Exception:
        output.with_suffix(".csv").write_text(df.to_csv(index=False))


def maybe_download(url: str, output: Path) -> None:
    import urllib.request
    output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-parquet", type=Path, default=Path("data/processed/synth.parquet"))
    p.add_argument("--download-url", type=str, default="")
    p.add_argument("--download-output", type=Path, default=Path("data/cms/sample.root"))
    args = p.parse_args()

    if args.download_url:
        maybe_download(args.download_url, args.download_output)
        print(f"Downloaded to {args.download_output}")
    else:
        generate_synthetic_parquet(args.synthetic_parquet)
        print(f"Generated synthetic dataset at {args.synthetic_parquet}")


if __name__ == "__main__":
    main()
