"""
Demonstrates DependencyGraph.experiments with SSVI surface models.

Three model specs:

  ssvi_data
      State: SSVI  (6 params per underlying: v_0, v_inf, κ, ρ, η, γ)
      Loss: Data + NodeLoss  (λ = 1, 1)
      Updater: SeparableGraphUpdater
      Roller: SurfaceRoller (warm-start only; no temporal prior)
      Edges: none

  ssvi_temporal
      State: SSVI
      Loss: Data + Temporal + NodeLoss  (λ = 1, 0.2, 1)
      Updater: SeparableGraphUpdater
      Roller: SurfaceRoller
      Edges: none

  ssvi_temporal_graph
      State: SSVI
      Loss: Data + Temporal + Graph + NodeLoss  (λ = 1, 0.2, 0.3, 1)
      Updater: GraphUpdater  (joint scipy L-BFGS-B over all underlyings)
      Roller: SurfaceRoller
      build_edges: learns M_{i→j} by OLS on Δθ sequences from training history
                   Residual: Δθ_i − M_{i→j} · Δθ_j

Experiment flow:
    1. train(train_dates)
           For ssvi_temporal_graph: independently fits SSVI on each date,
           computes Δθ increments, and runs OLS to learn M per directed pair.
           Other models have no build_edges so train() is a no-op for them.
    2. fit(prior_date, test_date)
           30% of underlying nodes are fully masked (no train observations).
           The graph-regularised model is expected to impute them better via
           cross-asset coupling encoded in the learned edge matrices.

Run:
    python -m examples.vol_surface.experiments.main
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from DependencyGraph.source.graph import Graph
from DependencyGraph.source.edge import DeltaEdgeState
from DependencyGraph.losses.combined import CombinedLoss
from DependencyGraph.losses.graph import GraphLoss
from DependencyGraph.losses.temporal import (
    TemporalLoss,
    NodewiseGraphDistance,
    L2ParameterDistance,
)
from DependencyGraph.time_stepping.updater import SeparableGraphUpdater, GraphUpdater
from DependencyGraph.experiments.experiment import Experiment, ModelSpec
from DependencyGraph.experiments.splitter import NodeMaskingSplitter
from DependencyGraph.experiments.comparison import ModelComparison

from examples.vol_smiles.source.nodes import SurfaceNode
from examples.vol_surface.source.factory import (
    SurfaceObservationFactory,
    SurfaceGraphFactory,
)
from examples.vol_surface.losses.data import SurfaceDataLoss
from examples.vol_surface.losses.node import SSVINodeLoss
from examples.vol_surface.time_stepping.rollers import SurfaceRoller



def learn_surface_edges(
    history: list[Graph],
    underlyings: list[str],
    precision: float = 1.0,
) -> dict:
    """
    Learn linear parameter dependency matrices from a sequence of independently-fitted
    surface graphs.

    For each directed pair (i → j), fits M_{i→j} by OLS over consecutive increments:
        Δθ_j^t  ≈  M_{i→j} · Δθ_i^t

    The resulting DeltaEdgeState encodes both the precision weight and the learned
    linear map, so the graph loss penalises ||Δθ_j - M · Δθ_i||² rather than plain
    parameter-space distance.
    """
    nodes = [SurfaceNode(u) for u in underlyings]

    param_seqs: dict = {nid: [] for nid in nodes}
    for g in history:
        for nid in nodes:
            if nid in g.nodes:
                param_seqs[nid].append(g.get(nid).parameters())

    delta_seqs: dict = {}
    for nid in nodes:
        ps = param_seqs[nid]
        if len(ps) >= 2:
            delta_seqs[nid] = np.array(
                [ps[t + 1] - ps[t] for t in range(len(ps) - 1)]
            )  # shape: (T-1, p)

    edges: dict = {}
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i+1:]:
            if n1 not in delta_seqs or n2 not in delta_seqs:
                continue

            D1 = delta_seqs[n2]   # (T-1, p) — target (Δθ_j)

            D2 = delta_seqs[n1]   # (T-1, p) — predictor (Δθ_i)

            n_obs = min(len(D1), len(D2))
            D1, D2 = D1[:n_obs], D2[:n_obs]

            # OLS: minimise ||D1 − D2 @ M.T||_F²
            # lstsq(D2, D1) → X  s.t.  D2 @ X ≈ D1;  then M = X.T
            X12, _, _, _ = np.linalg.lstsq(D2, D1, rcond=None)
            M12 = X12.T   # M12 @ Δθ_i ≈ Δθ_j

            X21, _, _, _ = np.linalg.lstsq(D1, D2, rcond=None)
            M21 = X21.T   # M21 @ Δθ_j ≈ Δθ_i

            edges[(n1, n2)] = DeltaEdgeState(precision=precision, matrix=M12)
            edges[(n2, n1)] = DeltaEdgeState(precision=precision, matrix=M21)

    return edges


def learn_surface_edges_diagonal(
    history: list[Graph],
    underlyings: list[str],
    precision: float = 1.0,
) -> dict:
    """
    Like learn_surface_edges but fits a diagonal M: one scalar per parameter.

    For each parameter k and directed pair (i → j):
        Δθ_j[k]  ≈  m_k · Δθ_i[k]

    m_k = (Δθ_i[k] · Δθ_j[k]) / (Δθ_i[k] · Δθ_i[k])   (no-intercept OLS)

    This avoids fitting cross-parameter mixing terms, which are mostly noise
    given the limited training data.
    """
    nodes = [SurfaceNode(u) for u in underlyings]

    param_seqs: dict = {nid: [] for nid in nodes}
    for g in history:
        for nid in nodes:
            if nid in g.nodes:
                param_seqs[nid].append(g.get(nid).parameters())

    delta_seqs: dict = {}
    for nid in nodes:
        ps = param_seqs[nid]
        if len(ps) >= 2:
            delta_seqs[nid] = np.array(
                [ps[t + 1] - ps[t] for t in range(len(ps) - 1)]
            )  # shape: (T-1, p)

    edges: dict = {}
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i+1:]:
            if n1 not in delta_seqs or n2 not in delta_seqs:
                continue

            D1 = delta_seqs[n2]   # (T-1, p)
            D2 = delta_seqs[n1]   # (T-1, p)
            n_obs = min(len(D1), len(D2))
            D1, D2 = D1[:n_obs], D2[:n_obs]

            # Per-parameter no-intercept OLS: m_k = (d_i[k]·d_j[k]) / (d_i[k]·d_i[k])
            denom12 = (D2 * D2).sum(axis=0)
            denom21 = (D1 * D1).sum(axis=0)
            m12 = np.where(denom12 > 0, (D2 * D1).sum(axis=0) / denom12, 0.0)
            m21 = np.where(denom21 > 0, (D1 * D2).sum(axis=0) / denom21, 0.0)

            edges[(n1, n2)] = DeltaEdgeState(precision=precision, matrix=np.diag(m12))
            edges[(n2, n1)] = DeltaEdgeState(precision=precision, matrix=np.diag(m21))

    return edges


def main() -> None:

    # Dataframe and train test date splitting.

    DATA_PATH = "Data/options_surface_sample.csv"
    df = pd.read_csv(DATA_PATH)
    df["date"]   = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])

    underlyings = sorted(df["underlying"].unique())
    dates       = sorted(df["date"].unique())

    n_train     = 110
    train_dates = dates[:n_train]
    prior_date  = dates[n_train] # Change if needed.
    test_date   = dates[n_train + 1]

    print(f"Underlyings  : {underlyings}")
    print(
        f"Train dates  : {train_dates[0].date()} to {train_dates[-1].date()} "
        f"({len(train_dates)} dates)"
    )
    print(f"Prior date   : {prior_date.date()}")
    print(f"Test date    : {test_date.date()}")

    obs_factory     = SurfaceObservationFactory(underlyings)
    surface_factory = SurfaceGraphFactory(underlyings, edges={})
    roller          = SurfaceRoller()
    temporal_loss   = TemporalLoss(NodewiseGraphDistance(L2ParameterDistance()))
    node_loss       = SSVINodeLoss()

    # ---- ssvi_data ----------------------------------------------------------
    ssvi_data_spec = ModelSpec(
        name="ssvi_data",
        build_graph=surface_factory.build,
        updater=SeparableGraphUpdater(
            CombinedLoss(
                data_loss=SurfaceDataLoss(),
                node_loss=node_loss,
                lambda_data=1.0,
                lambda_node=1.0,
            ),
            roller=None,
        ),
        roller=roller,    # only decays precision as no prior in loss
        static_edges={},
    )

    # ---- ssvi_temporal ------------------------------------------------------
    ssvi_temporal_spec = ModelSpec(
        name="ssvi_temporal",
        build_graph=surface_factory.build,
        updater=SeparableGraphUpdater(
            CombinedLoss(
                data_loss=SurfaceDataLoss(),
                temporal_loss=temporal_loss,
                node_loss=node_loss,
                lambda_data=1.0,
                lambda_temporal=0.2,
                lambda_node=1.0,
            ),
            roller=roller,
        ),
        roller=roller,
        static_edges={},
    )

    # ---- ssvi_temporal_graph ------------------------------------------------
    ssvi_graph_spec = ModelSpec(
        name="ssvi_temporal_graph",
        build_graph=surface_factory.build,
        updater=GraphUpdater(
            CombinedLoss(
                data_loss=SurfaceDataLoss(),
                temporal_loss=temporal_loss,
                graph_loss=GraphLoss(),
                node_loss=node_loss,
                lambda_data=1.0,
                lambda_temporal=0.05,
                lambda_graph=0.05,
                lambda_node=1.0,
            ),
            roller=roller,
        ),
        roller=roller,
        build_edges=lambda history: learn_surface_edges(
            history, underlyings, precision=0.5
        ),
    )

    # ---- ssvi_temporal_graph_diag -------------------------------------------
    ssvi_graph_diag_spec = ModelSpec(
        name="ssvi_temporal_graph_diag",
        build_graph=surface_factory.build,
        updater=GraphUpdater(
            CombinedLoss(
                data_loss=SurfaceDataLoss(),
                temporal_loss=temporal_loss,
                graph_loss=GraphLoss(),
                node_loss=node_loss,
                lambda_data=1.0,
                lambda_temporal=0.05,
                lambda_graph=0.05,
                lambda_node=1.0,
            ),
            roller=roller,
        ),
        roller=roller,
        build_edges=lambda history: learn_surface_edges_diagonal(
            history, underlyings, precision=0.5
        ),
    )

    # ---- Experiment ---------------------------------------------------------
    splitter   = NodeMaskingSplitter(node_mask_prob=0.2, train_frac=0.05)
    comparison = ModelComparison(SurfaceDataLoss(), splitter=splitter)

    experiment = Experiment(
        df=df,
        models=[ssvi_data_spec, ssvi_temporal_spec, ssvi_graph_spec, ssvi_graph_diag_spec],
        build_obs=obs_factory.build,
        splitter=splitter,
        output_fn=comparison,
    )

    print(
        f"\nTraining dependency model "
        f"({len(train_dates)} × {len(underlyings)} independent SSVI fits)..."
    )
    experiment.train(train_dates)
    print("Training complete.\n")

    print("Running experiment...")
    result = experiment.fit(prior_date, test_date)
    result.print_table("Vol Surface — Model Comparison (weighted IV MSE)")
    print(
        "\n(Masked: nodes with zero train observations — tests cross-asset imputation)"
    )


if __name__ == "__main__":
    main()
