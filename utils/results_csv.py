"""
Shared utility for upserting experiment results into a single CSV.

The composite key is (experiment, model).  A new run replaces any existing
row with the same key so the CSV always reflects the latest results.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


RESULTS_CSV = "results/experiment_results.csv"


def upsert_results_csv(csv_path: str, rows: list[dict]) -> None:
    """
    Upsert experiment result rows into a shared CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV (created if absent; parent dirs created automatically).
    rows : list[dict]
        Each dict must have at least 'experiment' and 'model'.
        Any additional keys become columns; missing values are NaN.
    """
    if not rows:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)
    key_set = set(zip(new_df["experiment"].tolist(), new_df["model"].tolist()))

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        keep = ~old_df.apply(
            lambda r: (r.get("experiment"), r.get("model")) in key_set, axis=1
        )
        old_df = old_df[keep]
        all_cols = list(dict.fromkeys(old_df.columns.tolist() + new_df.columns.tolist()))
        result = pd.concat(
            [old_df.reindex(columns=all_cols), new_df.reindex(columns=all_cols)],
            ignore_index=True,
        )
    else:
        result = new_df

    result.to_csv(csv_path, index=False)
    print(f"Results upserted → {csv_path}  ({len(rows)} row(s))")
