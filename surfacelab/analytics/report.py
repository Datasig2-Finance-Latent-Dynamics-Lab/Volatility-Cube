"""
Self-contained HTML comparison report for an evaluation run.

Builds one `report.html` under the output dir: a sortable summary table (every model ×
split, with RMSE / liquid / illiquid / MAE / no-arb %), plus embedded plots — an
RMSE-vs-context curve (independent) or an RMSE-over-time decay curve (sequential), and a
per-model reconstruction panel on a representative day.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from surfacelab.analytics.plots import (
    plot_reconstruction, plot_rmse_vs_ctx, plot_rmse_vs_ctx_lastday, plot_rmse_decay,
    plot_spread_vs_ctx, plot_spread_vs_ctx_lastday, plot_spread_decay,
)

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222}
h1{font-size:20px} h2{font-size:15px;margin-top:28px}
table{border-collapse:collapse;font-size:13px;margin:8px 0}
th,td{border:1px solid #ddd;padding:4px 9px;text-align:right}
th{background:#f4f4f4} td:first-child,th:first-child{text-align:left}
tr:nth-child(even){background:#fafafa}
img{max-width:100%;border:1px solid #eee;margin:6px 0}
.best{background:#e7f5e7;font-weight:600}
"""


def _fmt(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.5f}"


def _table(summary):
    cols = ["model", "split", "rmse", "rmse_liquid", "rmse_illiquid", "mae",
            "call_rmse", "call_oob_pct", "call_oob_spread_mean", "butterfly_pct", "calendar_pct"]
    # best rmse per split (highlight)
    best = {}
    for r in summary:
        s = r["split"]
        if not np.isnan(r["rmse"]) and (s not in best or r["rmse"] < best[s][1]):
            best[s] = (r["model"], r["rmse"])
    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = []
    for r in summary:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"<td>{v if c in ('model', 'split') else _fmt(v)}</td>")
        cls = " class='best'" if best.get(r["split"], (None,))[0] == r["model"] else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


_REGIME_LABEL = {"unif_": "uniform sampling", "extrap_": "extrapolation sampling",
                 "ctx_": "context"}


def _split_sort_key(s: str):
    """Order splits by (regime prefix, context size), e.g. extrap_8 < extrap_80; a split with
    no trailing integer sorts at size 0.  Splits the name once (rsplit semantics preserved)."""
    head, tail = s.rsplit("_", 1) if "_" in s else (s, s)
    return (head, int(tail) if tail.isdigit() else 0)


def _ctx_prefixes(records) -> list[str]:
    """Sampling-regime prefixes present as `<prefix><int>` splits (e.g. 'unif_', 'extrap_')."""
    prefixes = set()
    for r in records.rows:
        s = r["split"]
        if s.endswith("_excl"):                 # leave-one-out: unif_10_excl → unif_10
            s = s[:-5]
        if "_" in s:
            head, tail = s.rsplit("_", 1)
            if tail.isdigit():
                prefixes.add(head + "_")
    return sorted(prefixes)


def build_report(records, models, dataset, out_dir: str, *, mode: str = "independent",
                 title: str | None = None, target_asset=None) -> str:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    title = title or out.name
    summary = records.summary()
    prefixes = _ctx_prefixes(records)

    # leave-one-out / asymmetric runs score ONE asset; resolve it so the reconstruction
    # panel shows only that asset (predicted, in the asymmetric case, from full peers).
    names = list(dataset.meta.get("asset_names", []))
    tgt = None
    if target_asset is not None:
        tgt = names.index(target_asset) if isinstance(target_asset, str) else int(target_asset)

    # ── plots ──────────────────────────────────────────────────────────────────
    imgs = []
    if mode == "sequential":
        # one RMSE-over-time decay plot per regime/context split, each with a spread companion
        splits = sorted({r["split"] for r in records.rows}, key=_split_sort_key)
        for s in splits:
            fn = f"rmse_decay_{s}.png"
            plot_rmse_decay(records, split=s, out_path=str(out / fn),
                            title=f"Sequential RMSE over time — {s}")
            imgs.append((f"RMSE over time — {s}", fn))
            fns = f"miss_decay_{s}.png"
            plot_spread_decay(records, split=s, out_path=str(out / fns),
                              title=f"Sequential spread miss over time — {s}")
            imgs.append((f"Spread miss over time — {s}", fns))
        # RMSE vs context size, one plot per sampling regime: the window-MEAN (from the
        # summary, the headline curve) and the final-day curve (after state has propagated).
        for pre in prefixes:
            lbl = _REGIME_LABEL.get(pre, pre.rstrip("_"))
            fn = f"rmse_vs_ctx_{pre.rstrip('_')}.png"
            plot_rmse_vs_ctx(summary, out_path=str(out / fn),
                             title=f"RMSE vs context size (mean over window) — {lbl}", prefix=pre)
            imgs.append((f"RMSE vs context size (mean) — {lbl}", fn))
            fns = f"miss_vs_ctx_{pre.rstrip('_')}.png"
            plot_spread_vs_ctx(summary, out_path=str(out / fns),
                               title=f"Spread miss vs context size (mean over window) — {lbl}",
                               prefix=pre)
            imgs.append((f"Spread miss vs context size (mean) — {lbl}", fns))
        for pre in prefixes:
            lbl = _REGIME_LABEL.get(pre, pre.rstrip("_"))
            fn = f"rmse_vs_ctx_lastday_{pre.rstrip('_')}.png"
            if plot_rmse_vs_ctx_lastday(records, out_path=str(out / fn),
                                        title=f"RMSE vs context (last day) — {lbl}",
                                        prefix=pre):
                imgs.append((f"RMSE vs context size, last day — {lbl}", fn))
            fns = f"miss_vs_ctx_lastday_{pre.rstrip('_')}.png"
            if plot_spread_vs_ctx_lastday(records, out_path=str(out / fns),
                                          title=f"Spread miss vs context (last day) — {lbl}",
                                          prefix=pre):
                imgs.append((f"Spread miss vs context size, last day — {lbl}", fns))
    else:
        for pre in prefixes:
            lbl = _REGIME_LABEL.get(pre, pre.rstrip("_"))
            fn = f"rmse_vs_ctx_{pre.rstrip('_')}.png"
            plot_rmse_vs_ctx(summary, out_path=str(out / fn),
                             title=f"RMSE vs context size — {lbl}", prefix=pre)
            imgs.append((f"RMSE vs context size — {lbl}", fn))
            fns = f"miss_vs_ctx_{pre.rstrip('_')}.png"
            plot_spread_vs_ctx(summary, out_path=str(out / fns),
                               title=f"Spread miss vs context size — {lbl}", prefix=pre)
            imgs.append((f"Spread miss vs context size — {lbl}", fns))

    # representative day; for a single-asset run, ensure the target is present that day
    val_days = dataset.val_idx()
    day = int(val_days[len(val_days) // 2])
    if tgt is not None:
        present = [int(d) for d in val_days
                   if d > 0 and (dataset.asset_ids[int(d), dataset.valid_points(int(d))] == tgt).any()]
        if present:
            day = present[len(present) // 2]
    recon = []
    recon_kw = dict(only_asset=tgt, sparse_asset=tgt, n_ctx_sparse=3) if tgt is not None else {}
    for m in models:
        fn = f"recon_{m.name}.png"
        try:
            plot_reconstruction(m, dataset, day, n_ctx=50, out_path=str(out / fn), **recon_kw)
            recon.append((m.name, fn))
        except Exception as e:  # keep the report even if one model can't plot
            print(f"  (recon plot failed for {m.name}: {e})")

    # ── html ───────────────────────────────────────────────────────────────────
    html = [f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>",
            f"<style>{_CSS}</style></head><body>",
            f"<h1>surfacelab — {title}</h1>",
            f"<p>{mode} evaluation · {len(models)} models · "
            f"{len(dataset.val_idx())} eval days · {dataset.n_assets} assets</p>",
            "<h2>Summary</h2>", _table(summary)]
    for label, fn in imgs:
        html.append(f"<h2>{label}</h2><img src='{fn}'>")
    html.append(f"<h2>Reconstruction (day {day})</h2>")
    for name, fn in recon:
        html.append(f"<h3>{name}</h3><img src='{fn}'>")
    html.append("</body></html>")

    path = out / "report.html"
    path.write_text("\n".join(html))
    return str(path)
