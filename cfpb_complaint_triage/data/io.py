"""CSV/Parquet I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet file by suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def write_table(dataframe: pd.DataFrame, path: Path) -> None:
    """Write dataframe to CSV or Parquet by suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        dataframe.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        dataframe.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported file format: {path}")
