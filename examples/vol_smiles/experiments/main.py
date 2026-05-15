"""
Demonstrates DependencyGraph.experiments with implied-vol smile models.

Model specs are compared on a (prior_date, test_date) pair.

Run:
    python -m examples.vol_smiles.experiments.main

Results on html.
"""

# Imports

import numpy as np
import pandas as pd

from DependencyGraph.source.edge import DeltaEdgeState
from DependencyGraph.losses.combined import CombinedLoss
from DependencyGraph.losses.temporal import (
    TemporalLoss,
    NodewiseGraphDistance,
    L2ParameterDistance,
)
from DependencyGraph.losses.graph import GraphLoss
from DependencyGraph.time_stepping.updater import SeparableGraphUpdater, GraphUpdater
from DependencyGraph.experiments.experiment import Experiment, ModelSpec
from DependencyGraph.experiments.splitter import NodeMaskingSplitter
from examples.vol_smiles.experiments.comparison import SmileModelComparison

from examples.vol_smiles.source.factory import ObservationFactory, GraphFactory
from examples.vol_smiles.source.curves.bspline import fit_bspline
from examples.vol_smiles.source.curves.svi import fit_svi_jw
from examples.vol_smiles.source.nodes import CurveNode
from examples.vol_smiles.losses.data import VolDataLoss
from examples.vol_smiles.time_stepping.rollers import VolRoller
from examples.vol_smiles.time_stepping.updater import BSplineUpdater


# Static tiered cross-asset edges

def build_tiered_smile_edges(
    underlyings: list[str],
    expiries: list[pd.Timestamp],
    spy_to_stock: float = 0.8,
    stock_to_spy: float = 0.2,
    stock_to_stock: float = 0.4,
) -> dict:
    """
    Connect same-expiry nodes across underlyings with tiered DeltaEdgeState precision.

    Precision weights encode the idea that SPY leads individual stocks:
      SPY → stock : high coupling  (SPY moves drive stock moves)
      stock → SPY : low coupling   (individual stocks have little effect on SPY)
      stock → stock : medium coupling

    """
    edges: dict = {}
    for expiry in expiries:
        nodes = [CurveNode(u, expiry) for u in underlyings]
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1:]:
                u1, u2 = n1.underlying, n2.underlying
                if u1 == "SPY":
                    w12, w21 = spy_to_stock, stock_to_spy
                elif u2 == "SPY":
                    w12, w21 = stock_to_spy, spy_to_stock
                else:
                    w12 = w21 = stock_to_stock
                edges[(n1, n2)] = DeltaEdgeState(precision=w12)
                edges[(n2, n1)] = DeltaEdgeState(precision=w21)
    return edges


# Factored f * g edges

def build_factored_smile_edges(
    underlyings: list[str],
    expiries: list[pd.Timestamp],
    lambda_f: float = 1.0,
    min_precision: float = 0.05,
    spy_to_stock: float = 0.5,
    stock_to_spy: float = 0.1,
    stock_to_stock: float = 0.2,
) -> dict:
    """
    
    """
    edges: dict = {}

    for j, expiry1 in enumerate(expiries):
        for expiry2 in expiries[j + 1:]:
            nodes1 = [CurveNode(u, expiry1) for u in underlyings]
            nodes2 = [CurveNode(u, expiry2) for u in underlyings]

            for n1 in nodes1:
                for n2 in nodes2:
                    dt_days = abs((n1.expiry - n2.expiry).days)
                    f = float(np.exp(-lambda_f * (dt_days / 365.0)))

                    if f < min_precision:
                        f = 0.0

                    u1, u2 = n1.underlying, n2.underlying
                    if u1 == "SPY":
                        w12, w21 = spy_to_stock, stock_to_spy
                    elif u2 == "SPY":
                        w12, w21 = stock_to_spy, spy_to_stock
                    else:
                        w12 = w21 = stock_to_stock

                    edges[(n1, n2)] = DeltaEdgeState(precision=w12 * f)
                    edges[(n2, n1)] = DeltaEdgeState(precision=w21 * f)

    return edges

# Main

def main() -> None:
    DATA_PATH = "Data/options_surface_sample.csv"
    df = pd.read_csv(DATA_PATH)
    df = df[df["type"] == "call"]
    df["date"]   = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])

    underlyings = sorted(df["underlying"].unique())
    expiries    = sorted(df["expiry"].unique())
    dates       = sorted(df["date"].unique())

    prior_date = dates[-2]
    test_date  = dates[-1]

    tiered_edges   = build_tiered_smile_edges(underlyings, expiries)
    factored_edges = build_factored_smile_edges(underlyings, expiries)

    print(f"Underlyings    : {underlyings}")
    print(f"Expiries       : {len(expiries)}")
    print(f"Prior date     : {prior_date.date()}")
    print(f"Test date      : {test_date.date()}")
    print(f"Tiered edges   : {len(tiered_edges)} directed edges")
    print(f"Factored edges : {len(factored_edges)} directed edges\n")

    obs_factory   = ObservationFactory(underlyings, expiries)
    temporal_loss = TemporalLoss(NodewiseGraphDistance(L2ParameterDistance()))

    # ---- svijw_data -----------------------------------------------------------
    svijw_data_spec = ModelSpec(
        name="svijw_data",
        build_graph=GraphFactory(underlyings, expiries, edges={}, fit_fn=fit_svi_jw).build,
        updater=SeparableGraphUpdater(
            CombinedLoss(
                data_loss=VolDataLoss(),
                lambda_data=1.0,
            ),
            roller=None,
        ),
        roller=VolRoller(),
        static_edges={},
    )

    # ---- svijw_temporal -------------------------------------------------------
    svijw_temporal_spec = ModelSpec(
        name="svijw_temporal",
        build_graph=GraphFactory(underlyings, expiries, edges={}, fit_fn=fit_svi_jw).build,
        updater=SeparableGraphUpdater(
            CombinedLoss(
                data_loss=VolDataLoss(),
                temporal_loss=temporal_loss,
                lambda_data=1.0,
                lambda_temporal=0.1,
            ),
            roller=VolRoller(),
        ),
        roller=VolRoller(),
        static_edges={},
    )

    # ---- bspline_temporal ---------------------------------------------------
    bspline_temporal_spec = ModelSpec(
        name="bspline_temporal",
        build_graph=GraphFactory(
            underlyings, expiries, edges={}, fit_fn=fit_bspline
        ).build,
        updater=BSplineUpdater(
            lambda_data=1.0,
            lambda_temporal=0.1,
        ),
        roller=VolRoller(),
        static_edges={},
    )

    # ---- bspline_tiered_graph --------------------------------------------
    bspline_spec = ModelSpec(
        name="bspline_tiered_graph",
        build_graph=GraphFactory(
            underlyings, expiries, edges={}, fit_fn=fit_bspline
        ).build,
        updater=BSplineUpdater(
            lambda_data=1.0,
            lambda_temporal=0.1,
            lambda_graph=1.0,
        ),
        roller=VolRoller(),
        static_edges=tiered_edges,
    )

    # ---- bspline_factored_graph ---------------------------------------------
    bspline_factored_spec = ModelSpec(
        name="bspline_factored_graph",
        build_graph=GraphFactory(
            underlyings, expiries, edges={}, fit_fn=fit_bspline
        ).build,
        updater=BSplineUpdater(
            lambda_data=1.0,
            lambda_temporal=0.1,
            lambda_graph=1.0,
        ),
        roller=VolRoller(),
        static_edges=factored_edges,
    )


    # ---- Experiment ---------------------------------------------------------
    splitter   = NodeMaskingSplitter(node_mask_prob=0.2, train_frac=0.1)
    comparison = SmileModelComparison(VolDataLoss(), splitter=splitter)


    experiment = Experiment(
        df=df,
        models=[svijw_data_spec, svijw_temporal_spec, bspline_temporal_spec, bspline_spec, bspline_factored_spec],
        build_obs=obs_factory.build,
        splitter=splitter,
        output_fn=comparison,
    )

    print("Running experiment (no training phase needed for static edges)...")
    result = experiment.fit(prior_date, test_date)
    result.print_table("Vol Smiles — Model Comparison")
    print("(Masked column: nodes with zero train observations — tests cross-asset imputation)")
    result.to_html("vol_smiles_comparison.html", title="Vol Smiles — Model Comparison")


if __name__ == "__main__":
    main()
