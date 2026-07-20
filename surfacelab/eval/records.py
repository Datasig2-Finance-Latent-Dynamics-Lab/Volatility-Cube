"""
Per-(model, day, split) result accumulator → records.csv + summary.csv.

The summary schema matches the project convention:
    model, split, rmse, rmse_liquid, rmse_illiquid, mae, n_rows, butterfly_pct, calendar_pct
RMSE columns are pooled across days via the stored squared-error sums (not a mean of
per-day RMSEs), so they are exact.
"""
from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from surfacelab.eval.metrics import rmse


class Records:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, model: str, day, split: str, stats: dict,
            butterfly_pct: float = 0.0, calendar_pct: float = 0.0) -> None:
        self.rows.append({
            "model": model, "day": day, "split": split,
            "sq": stats.get("sq", 0.0), "n": stats.get("n", 0),
            "abs": stats.get("abs", 0.0),
            "sq_liquid": stats.get("sq_liquid", 0.0), "n_liquid": stats.get("n_liquid", 0),
            "sq_illiquid": stats.get("sq_illiquid", 0.0),
            "n_illiquid": stats.get("n_illiquid", 0),
            "call_sq": stats.get("call_sq", 0.0), "call_n": stats.get("call_n", 0),
            "call_oob": stats.get("call_oob", 0),
            "call_oob_spread_sum": stats.get("call_oob_spread_sum", 0.0),
            "butterfly_pct": butterfly_pct, "calendar_pct": calendar_pct,
        })

    # ── aggregation ────────────────────────────────────────────────────────────
    def summary(self) -> list[dict]:
        groups: dict = {}
        for r in self.rows:
            groups.setdefault((r["model"], r["split"]), []).append(r)
        out = []
        for (model, split), rs in sorted(groups.items()):
            S = lambda key: sum(r[key] for r in rs)  # noqa: E731
            n, call_n, call_oob = S("n"), S("call_n"), S("call_oob")
            out.append({
                "model": model, "split": split,
                "rmse": rmse(S("sq"), n),
                "rmse_liquid": rmse(S("sq_liquid"), S("n_liquid")),
                "rmse_illiquid": rmse(S("sq_illiquid"), S("n_illiquid")),
                "mae": (S("abs") / n) if n else float("nan"),
                "call_rmse": rmse(S("call_sq"), call_n),
                "call_oob_pct": (100.0 * call_oob / call_n)
                if call_n else float("nan"),
                # how far outside the quotes we are WHEN wrong, in spread-widths (MEAN over
                # the out-of-spread points only).  Dimensionless: ~1 == one spread out.
                "call_oob_spread_mean": (S("call_oob_spread_sum") / call_oob)
                if call_oob else float("nan"),
                "n_rows": len(rs),
                "butterfly_pct": float(np.mean([r["butterfly_pct"] for r in rs])),
                "calendar_pct": float(np.mean([r["calendar_pct"] for r in rs])),
            })
        return out

    # ── persistence ────────────────────────────────────────────────────────────
    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._write_csv(out / "records.csv", self.rows)
        self._write_csv(out / "summary.csv", self.summary())

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def print_table(self, title: str = "") -> None:
        s = self.summary()
        if title:
            print(f"\n=== {title} ===")
        hdr = (f"{'model':24s} {'split':14s} {'rmse':>9s} {'rmse_liq':>9s} "
               f"{'rmse_illiq':>10s} {'mae':>9s} {'call_rmse':>9s} {'oob%':>6s} "
               f"{'oob_sprds':>9s} {'bfly%':>6s}")
        print(hdr); print("-" * len(hdr))
        for r in s:
            print(f"{r['model']:24s} {r['split']:14s} {r['rmse']:9.5f} "
                  f"{r['rmse_liquid']:9.5f} {r['rmse_illiquid']:10.5f} "
                  f"{r['mae']:9.5f} {r['call_rmse']:9.5f} {r['call_oob_pct']:6.2f} "
                  f"{r['call_oob_spread_mean']:9.5f} {r['butterfly_pct']:6.2f}")
