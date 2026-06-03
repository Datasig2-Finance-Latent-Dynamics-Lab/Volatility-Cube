"""
Vol smile dependency graph experiment on Group Tech US market data.

Trains on all dates except the last N_EVAL_DAYS, then evaluates on the last
N_EVAL_DAYS with a rolling forward update. Metrics are aggregated across the
eval window; plots are shown for the last eval day.

Run:
    .venv/bin/python -m dgraph.examples.vol_smiles.experiments.main
"""

# Imports

import numpy as np
import pandas as pd

from dgraph.source.edge import DeltaEdgeState
from dgraph.losses.combined import CombinedLoss
from dgraph.losses.temporal import (
    TemporalLoss,
    NodewiseGraphDistance,
    L2ParameterDistance,
)
from dgraph.losses.graph import GraphLoss
from dgraph.time_stepping.updater import SeparableGraphUpdater, GraphUpdater
from dgraph.examples.vol_smiles.time_stepping.updater import BSplineUpdater
from dgraph.experiments.experiment import Experiment, ModelSpec
from dgraph.experiments.splitter import NodeMaskingSplitter

from dgraph.examples.vol_smiles.experiments.comparison import SmileModelComparison
from dgraph.examples.vol_smiles.source.factory import ObservationFactory, GraphFactory
from dgraph.examples.vol_smiles.source.curves.bspline import fit_bspline
from dgraph.examples.vol_smiles.source.curves.svi import fit_svi_jw
from dgraph.examples.vol_smiles.source.nodes import CurveNode
from dgraph.examples.vol_smiles.losses.data import VolDataLoss
from dgraph.examples.vol_smiles.losses.node import SviJWNodeLoss, BSplineNALoss
from dgraph.examples.vol_smiles.time_stepping.rollers import VolRoller
from dgraph.examples.vol_smiles.source.edges import build_tiered_smile_edges, build_factored_smile_edges

# Configs

LAMBDA_DATA = 4.0
LAMBDA_TEMPORAL = 0.05
LAMBDA_GRAPH = 0.05
LAMBDA_NODE = 10000.0
PRECISION_GAIN = 2.0 * LAMBDA_DATA
NODE_MASK_FRAC = 0.3
TRAIN_FRAC = 0.05  

N_EVAL_DAYS = 5  # last N days used for evaluation; everything else is training

DATA_PATH    = "/home/alvaro/projects/dissertation/data/scripts/bulk_download/output/group_tech_us.csv"
OUT_PATH     = "results/vol_smiles_dgraph/vol_smiles_comparison.html"
METRICS_PATH = "results/vol_smiles_dgraph/vol_smiles_metrics.json"
RESULTS_CSV  = "results/experiment_results.csv"



# Main

def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df[df["type"] == "call"]
    df["date"]   = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])

    all_dates  = sorted(df["date"].unique())
    # 1 init day + N_EVAL_DAYS rolling eval days
    test_dates = all_dates[-(N_EVAL_DAYS + 1):]

    # Underlyings and expiries derived from the test window — avoids iterating
    # over hundreds of already-expired options from the full history.
    test_df     = df[df["date"].isin(set(test_dates))]
    underlyings = sorted(test_df["underlying"].unique())
    expiries    = sorted(test_df["expiry"].unique())

    tiered_edges   = build_tiered_smile_edges(underlyings, expiries)
    factored_edges = build_factored_smile_edges(underlyings, expiries)

    print(f"Underlyings    : {underlyings}")
    print(f"Total dates    : {len(all_dates)}")
    print(f"Eval window    : {test_dates[1].date()} to {test_dates[-1].date()} ({N_EVAL_DAYS} days)")
    print(f"Expiries       : {len(expiries)} (in eval window)")
    print(f"Tiered edges   : {len(tiered_edges)} directed edges")
    print(f"Factored edges : {len(factored_edges)} directed edges\n")

    obs_factory   = ObservationFactory(underlyings, expiries)
    temporal_loss = TemporalLoss(NodewiseGraphDistance(L2ParameterDistance()))
    node_loss     = SviJWNodeLoss(weight=1e4)

    svijw_data_spec = ModelSpec(
        name="svijw_data",
        build_graph=GraphFactory(underlyings, expiries, edges={}, fit_fn=fit_svi_jw).build,
        updater=SeparableGraphUpdater(
            CombinedLoss(
                data_loss=VolDataLoss(),
                node_loss=node_loss,
                lambda_data=LAMBDA_DATA,
                lambda_node=LAMBDA_NODE,
            ),
            roller=None,
            precision_gain=PRECISION_GAIN,
        ),
        roller=VolRoller(),
        static_edges={},
    )

    svijw_temporal_spec = ModelSpec(
        name="svijw_temporal",
        build_graph=GraphFactory(underlyings, expiries, edges={}, fit_fn=fit_svi_jw).build,
        updater=SeparableGraphUpdater(
            CombinedLoss(
                data_loss=VolDataLoss(),
                temporal_loss=temporal_loss,
                node_loss=node_loss,
                lambda_data=LAMBDA_DATA,
                lambda_temporal=LAMBDA_TEMPORAL,
                lambda_node=LAMBDA_NODE,
            ),
            roller=VolRoller(),
            precision_gain=PRECISION_GAIN,
        ),
        roller=VolRoller(),
        static_edges={},
    )

    svijw_temporal_graph_spec = ModelSpec(
        name="svijw_temporal_graph",
        build_graph=GraphFactory(underlyings, expiries, edges={}, fit_fn=fit_svi_jw).build,
        updater=SeparableGraphUpdater(
            CombinedLoss(
                data_loss=VolDataLoss(),
                temporal_loss=temporal_loss,
                graph_loss=GraphLoss(),
                node_loss=node_loss,
                lambda_data=LAMBDA_DATA,
                lambda_temporal=LAMBDA_TEMPORAL,
                lambda_graph=LAMBDA_GRAPH,
                lambda_node=LAMBDA_NODE,
            ),
            roller=VolRoller(),
            precision_gain=PRECISION_GAIN,
        ),
        roller=VolRoller(),
        static_edges=tiered_edges,
    )

    # BSplines.

    bspline_node_loss = BSplineNALoss(weight=1.0)

    bspline_temporal_spec = ModelSpec(
        name="bspline_temporal",
        build_graph=GraphFactory(
            underlyings, expiries, edges={}, fit_fn=fit_bspline
        ).build,
        updater=BSplineUpdater(
            lambda_data=LAMBDA_DATA,
            lambda_temporal=LAMBDA_TEMPORAL,
        ),
        roller=VolRoller(),
        static_edges={},
    )

    bspline_tiered_spec = ModelSpec(
        name="bspline_tiered_graph",
        build_graph=GraphFactory(
            underlyings, expiries, edges={}, fit_fn=fit_bspline
        ).build,
        updater=BSplineUpdater(
            lambda_data=LAMBDA_DATA,
            lambda_temporal=LAMBDA_TEMPORAL,
            lambda_graph=LAMBDA_GRAPH,
        ),
        roller=VolRoller(),
        static_edges=tiered_edges,
    )

    bspline_factored_spec = ModelSpec(
        name="bspline_factored_graph",
        build_graph=GraphFactory(
            underlyings, expiries, edges={}, fit_fn=fit_bspline
        ).build,
        updater=GraphUpdater(
            CombinedLoss(
                data_loss=VolDataLoss(),
                temporal_loss=temporal_loss,
                graph_loss=GraphLoss(),
                node_loss=bspline_node_loss,
                lambda_data=LAMBDA_DATA,
                lambda_temporal=LAMBDA_TEMPORAL,
                lambda_graph=LAMBDA_GRAPH,
                lambda_node=LAMBDA_NODE,
            ),
            roller=VolRoller(),
            precision_gain=PRECISION_GAIN,
        ),
        roller=VolRoller(),
        static_edges=factored_edges,
    )


    # Experiment
    splitter   = NodeMaskingSplitter(node_mask_prob = NODE_MASK_FRAC, train_frac = TRAIN_FRAC)
    comparison = SmileModelComparison(VolDataLoss(), splitter=splitter)


    experiment = Experiment(
        df=df,
        models=[svijw_data_spec, svijw_temporal_spec, svijw_temporal_graph_spec, bspline_temporal_spec, bspline_tiered_spec], #bspline_factored_spec
        build_obs=obs_factory.build,
        splitter=splitter,
        output_fn=comparison,
    )

    print(f"Running rolling experiment over {N_EVAL_DAYS} eval dates…")
    experiment.run(test_dates)

    comparison.print_table("Vol Smiles — Model Comparison")
    print("(Masked column: nodes with zero train observations — tests cross-asset imputation)")
    comparison.save_metrics(METRICS_PATH, results_csv=RESULTS_CSV, experiment="vol_smiles_dgraph")
    comparison.to_html(OUT_PATH, title="Vol Smiles — Model Comparison")


if __name__ == "__main__":
    main()
