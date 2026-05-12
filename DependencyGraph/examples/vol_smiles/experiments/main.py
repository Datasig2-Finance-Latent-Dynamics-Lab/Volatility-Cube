"""
Backtest: (SVI , SVI-JW , BSpline) x (data , temporal , tiered)

Runs on the last 5 dates only.

Running
-------
    python -m DependencyGraph.examples.vol_smiles.experiments.main

Input CSV columns (CSV_PATH):
    date, underlying, expiry, dte, T, logmoneyness, iv, weight
"""

from __future__ import annotations

import warnings
import pandas as pd
from tqdm import tqdm

from ....source.core.dependency import StaticDependencies
from ....source.core.graph import Graph
from ....source.distances.state import L2ParameterDistance
from ....source.distances.graph import NodewiseGraphDistance
from ....source.losses.combined import CombinedLoss
from ....source.losses.temporal import TemporalLoss
from ....source.losses.graph import L2DependencyGraphLoss
from ....source.updater import SeparableGraphUpdater, GraphUpdater
from ....source.experiments.splitter import NodeMaskingSplitter
from ..factory import ObservationFactory, GraphFactory, fit_svi_jw
from ..losses.data import VolDataLoss
from ..rollers import StickyStrikeRoller
from ..curves.bspline import fit_bspline
from ..updater import BSplineUpdater
from ..nodes import CurveNode


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH    = "Data/options_surface_sample.csv" # Input your own. I am not too sure on the legality of uploading massive data to a public github repo.
UNDERLYINGS = ["AAPL", "MSFT", "NVDA", "AMZN", "SPY"]

LAMBDA_TEMPORAL = 0.8
LAMBDA_GRAPH    = 0.5
TRAIN_FRAC      = 0.5
SEED            = 67

N_INTERIOR = 9
DEGREE     = 3
N_EXPIRIES = 5
N_DATES    = 5

SPY_STOCK_WEIGHT   = 0.60
STOCK_SPY_WEIGHT   = 0.20
STOCK_STOCK_WEIGHT = 0.40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_expiries(df: pd.DataFrame, n: int = N_EXPIRIES) -> list[pd.Timestamp]:
    first_date = df["date"].min()
    viable = df[(df["date"] == first_date) & (df["dte"] >= 14)]["expiry"].unique()
    counts = df[df["expiry"].isin(viable)].groupby("expiry").size()
    return list(pd.to_datetime(counts.nlargest(n).index.sort_values()))


def build_tiered_dependencies(
    underlyings: list[str],
    expiries: list[pd.Timestamp],
    df: pd.DataFrame,
    date: pd.Timestamp,
) -> StaticDependencies:
    """
    Directed tiered weights:
      SPY → stock : SPY_STOCK_WEIGHT (0.60)
      stock → SPY : STOCK_SPY_WEIGHT  (0.20)
      stock ↔ stock : STOCK_STOCK_WEIGHT (0.40) symmetric
    """
    day_df = df[df["date"] == date]
    wts: dict = {}
    for expiry in expiries:
        nodes = []
        for u in underlyings:
            s = day_df[(day_df["underlying"] == u) & (day_df["expiry"] == expiry)]
            if s.empty:
                continue
            nodes.append(CurveNode(u, expiry, float(s["T"].iloc[0])))
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1:]:
                if n1.underlying == "SPY":
                    wts[(n1, n2)] = SPY_STOCK_WEIGHT
                    wts[(n2, n1)] = STOCK_SPY_WEIGHT
                elif n2.underlying == "SPY":
                    wts[(n1, n2)] = STOCK_SPY_WEIGHT
                    wts[(n2, n1)] = SPY_STOCK_WEIGHT
                else:
                    wts[(n1, n2)] = STOCK_STOCK_WEIGHT
                    wts[(n2, n1)] = STOCK_STOCK_WEIGHT
    return StaticDependencies(wts)


def build_initial_graph(model_type: str, df: pd.DataFrame, date: pd.Timestamp,
                        expiries: list[pd.Timestamp], deps: StaticDependencies) -> Graph:
    if model_type == "bspline":
        day_df = df[df["date"] == date]
        nodes: dict = {}
        for underlying in UNDERLYINGS:
            for expiry in expiries:
                s = day_df[(day_df["underlying"] == underlying) & (day_df["expiry"] == expiry)]
                if s.empty:
                    continue
                T = float(s["T"].iloc[0])
                nodes[CurveNode(underlying, expiry, T)] = fit_bspline(
                    s["logmoneyness"].values, s["iv"].values, T,
                    weights=s["weight"].values, n_interior=N_INTERIOR, degree=DEGREE,
                )
        return Graph(date, nodes, deps)
    fit_fn = fit_svi_jw if model_type == "jw" else None
    return GraphFactory(UNDERLYINGS, expiries, deps, fit_fn=fit_fn).build(df, date)


def weighted_iv_mse(graph: Graph, obs_set) -> float:
    total, total_w = 0.0, 0.0
    for nid in graph.node_ids():
        if not isinstance(nid, CurveNode):
            continue
        for obs in obs_set.for_node(nid):
            k, iv_obs = obs.data
            iv_fit   = graph.get(nid).implied_vol(k, nid.T)
            total   += obs.weight * (iv_fit - iv_obs) ** 2
            total_w += obs.weight
    return total / total_w if total_w > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df   = pd.read_csv(CSV_PATH, parse_dates=["date", "expiry"])
    dates = sorted(df["date"].unique())[-N_DATES:]

    expiries    = discover_expiries(df)
    tiered_deps = build_tiered_dependencies(UNDERLYINGS, expiries, df, dates[0])
    no_deps     = StaticDependencies({})

    print(f"Expiries ({len(expiries)}): {[e.date() for e in expiries]}")
    print(f"Running on {len(dates)} dates: {dates[0].date()} – {dates[-1].date()}")

    obs_factory = ObservationFactory(UNDERLYINGS, expiries)
    roller      = StickyStrikeRoller()
    graph_dist  = NodewiseGraphDistance(L2ParameterDistance())
    splitter    = NodeMaskingSplitter(node_mask_prob=0.3, train_frac=TRAIN_FRAC, seed=SEED)

    data_loss = VolDataLoss()
    temp_loss = TemporalLoss(graph_dist)

    # (updater, model_type, deps)
    experiments: dict[str, tuple] = {
        "svi_data": (
            SeparableGraphUpdater(CombinedLoss(data_loss=data_loss, lambda_data=1.0)),
            "svi", no_deps,
        ),
        "svi_temporal": (
            SeparableGraphUpdater(CombinedLoss(
                data_loss=data_loss, temporal_loss=temp_loss, roller=roller,
                lambda_data=1.0, lambda_temporal=LAMBDA_TEMPORAL,
            )),
            "svi", no_deps,
        ),
        "svi_tiered": (
            GraphUpdater(CombinedLoss(
                data_loss=data_loss, temporal_loss=temp_loss,
                graph_loss=L2DependencyGraphLoss(), roller=roller,
                lambda_data=1.0, lambda_temporal=LAMBDA_TEMPORAL, lambda_graph=LAMBDA_GRAPH,
            )),
            "svi", tiered_deps,
        ),
        "jw_data": (
            SeparableGraphUpdater(CombinedLoss(data_loss=data_loss, lambda_data=1.0)),
            "jw", no_deps,
        ),
        "jw_temporal": (
            SeparableGraphUpdater(CombinedLoss(
                data_loss=data_loss, temporal_loss=temp_loss, roller=roller,
                lambda_data=1.0, lambda_temporal=LAMBDA_TEMPORAL,
            )),
            "jw", no_deps,
        ),
        "jw_tiered": (
            GraphUpdater(CombinedLoss(
                data_loss=data_loss, temporal_loss=temp_loss,
                graph_loss=L2DependencyGraphLoss(), roller=roller,
                lambda_data=1.0, lambda_temporal=LAMBDA_TEMPORAL, lambda_graph=LAMBDA_GRAPH,
            )),
            "jw", tiered_deps,
        ),
        "bspline_data": (
            BSplineUpdater(lambda_data=1.0),
            "bspline", no_deps,
        ),
        "bspline_temporal": (
            BSplineUpdater(lambda_data=1.0, lambda_temporal=LAMBDA_TEMPORAL),
            "bspline", no_deps,
        ),
        "bspline_tiered": (
            BSplineUpdater(lambda_data=1.0, lambda_temporal=LAMBDA_TEMPORAL, lambda_graph=LAMBDA_GRAPH),
            "bspline", tiered_deps,
        ),
    }

    prior_graphs: dict[str, Graph | None] = {label: None for label in experiments}
    records: list[dict] = []

    for date in tqdm(dates, desc="dates", unit="day"):
        obs_set             = obs_factory.build(df, date)
        train_obs, test_obs = splitter.split(obs_set)
        row: dict           = {"date": date}

        for label, (updater, model_type, deps) in experiments.items():
            prior = prior_graphs[label]

            if prior is None:
                x0 = build_initial_graph(model_type, df, date, expiries, deps)
            else:
                dt = int((date - prior.date).days)
                x0 = roller.roll(prior, dt)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fitted = updater.update(x0, train_obs, prior_graph=prior)

            prior_graphs[label]   = fitted
            row[f"{label}_train"] = weighted_iv_mse(fitted, train_obs)
            row[f"{label}_test"]  = weighted_iv_mse(fitted, test_obs)

        records.append(row)

    results = pd.DataFrame(records)

    col_w    = max(len(k) for k in experiments) + 2
    means    = {lbl: results[f"{lbl}_test"].mean() for lbl in experiments}
    baseline = means["svi_data"]

    print("\n" + "=" * 65)
    print("TEST IV-MSE  (weighted, held-out strikes, masked-node imputation)")
    print("=" * 65)
    print(f"  {'model':{col_w}}  {'mean MSE':>10}  {'Δ vs svi_data':>14}")
    print(f"  {'-'*col_w}  {'-'*10}  {'-'*14}")

    for group_name, prefix in [("SVI", "svi"), ("SVI-JW", "jw"), ("B-spline", "bspline")]:
        labels = [l for l in experiments if l.startswith(prefix + "_")]
        print(f"\n  [{group_name}]")
        for lbl in labels:
            m     = means[lbl]
            delta = m - baseline
            flag  = " ◀" if delta < -1e-7 else ""
            print(f"  {lbl:{col_w}}  {m:10.6f}  {delta:+14.6f}{flag}")

    print("\n  ◀ = improvement over svi_data baseline")

    tail_cols = [f"{l}_test" for l in experiments]
    tail = results[["date"] + tail_cols].copy()
    tail.columns = ["date"] + list(experiments.keys())
    print(f"\nTest MSE — all {len(tail)} dates:")
    print(tail.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    return results


if __name__ == "__main__":
    main()
