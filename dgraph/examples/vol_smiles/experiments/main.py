"""
Demonstrates dgraph.experiments with implied-vol smile models.

Model specs are compared on a (prior_date, test_date) pair.

Run:
    .venv/bin/python -m dgraph.examples.vol_smiles.experiments.main

Results on html.
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

# Precision accumulated per unit observation weight after each SVI update.
# Set to 2 * lambda_data so it stays on the same scale as the data Hessian.
PRECISION_GAIN = 2.0 * LAMBDA_DATA

NODE_MASK_FRAC = 0.5

TRAIN_FRAC = 0.05



# Main

def main() -> None:
    DATA_PATH = "data/market/options_surface_sample.csv"
    df = pd.read_csv(DATA_PATH)
    df = df[df["type"] == "call"]
    df["date"]   = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])

    underlyings = sorted(df["underlying"].unique())
    expiries    = sorted(df["expiry"].unique())
    dates       = sorted(df["date"].unique())[0:2]

    tiered_edges   = build_tiered_smile_edges(underlyings, expiries)
    factored_edges = build_factored_smile_edges(underlyings, expiries)

    print(f"Underlyings    : {underlyings}")
    print(f"Expiries       : {len(expiries)}")
    print(f"Dates          : {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)")
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
        updater=GraphUpdater(
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
        updater=SeparableGraphUpdater(
            CombinedLoss(
                data_loss=VolDataLoss(),
                temporal_loss=temporal_loss,
                node_loss=bspline_node_loss,
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

    bspline_tiered_spec = ModelSpec(
        name="bspline_tiered_graph",
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

    print(f"Running sequential experiment over {len(dates)} days...")
    result = experiment.run(dates)[-1]
    result.print_table("Vol Smiles — Model Comparison")
    print("(Masked column: nodes with zero train observations — tests cross-asset imputation)")
    result.to_html("results/vol_smiles_dgraph/vol_smiles_comparison.html", title="Vol Smiles — Model Comparison")


if __name__ == "__main__":
    main()
