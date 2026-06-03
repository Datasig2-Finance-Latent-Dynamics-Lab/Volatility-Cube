"""
Vol surface dependency graph experiment.

Seven models compared on a rolling out-of-sample backtest:

  ssvi_data                — SSVI, data + node loss
  ssvi_temporal            — SSVI, + temporal regularisation
  ssvi_temporal_graph      — SSVI, + learned full OLS edge matrices M_{i→j}
  ssvi_temporal_graph_diag — SSVI, + diagonal OLS edges (one scalar per param)
  pca_data                 — functional-PCA state, data only
  pca_temporal             — functional-PCA, + temporal
  pca_temporal_graph       — functional-PCA, + scalar edges from PC1 correlation

Run:
    python -m dgraph.examples.vol_surface.experiments.main
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dgraph.source.edge import DeltaEdgeState
from dgraph.source.graph import Graph
from dgraph.losses.combined import CombinedLoss
from dgraph.losses.graph import GraphLoss
from dgraph.losses.temporal import L2ParameterDistance, NodewiseGraphDistance, TemporalLoss
from dgraph.time_stepping.updater import GraphUpdater, SeparableGraphUpdater
from dgraph.experiments.experiment import Experiment, ModelSpec
from dgraph.experiments.splitter import NodeMaskingSplitter

from dgraph.examples.vol_smiles.source.nodes import SurfaceNode
from dgraph.examples.vol_surface.experiments.comparison import SurfaceModelComparison
from dgraph.examples.vol_surface.losses.data import SurfaceDataLoss
from dgraph.examples.vol_surface.losses.node import SSVINodeLoss
from dgraph.examples.vol_surface.source.factory import SurfaceObservationFactory, SurfaceGraphFactory
from dgraph.examples.vol_surface.source.states.pca import PCASurfaceState, VolSurfacePCA
from dgraph.examples.vol_surface.source.states.ssvi import fit_ssvi
from dgraph.examples.vol_surface.time_stepping.rollers import SurfaceRoller




# =========================================================================
# Config
# =========================================================================

# Data split — train on all dates except the last N_EVAL_DAYS
N_EVAL_DAYS = 5

# PCA basis
N_PCA_COMPONENTS = 5
N_K_GRID         = 20
N_T_GRID         = 15

# SSVI loss weights
SSVI_LAMBDA_DATA     = 1.0
SSVI_LAMBDA_NODE     = 1.0
SSVI_LAMBDA_TEMPORAL = 0.2   # ssvi_temporal
SSVI_GRAPH_LAMBDA_TEMPORAL = 0.05  # ssvi_temporal_graph*
SSVI_GRAPH_LAMBDA_GRAPH    = 0.05
SSVI_EDGE_PRECISION  = 0.5

# PCA loss weights
PCA_LAMBDA_DATA              = 1.0
PCA_LAMBDA_TEMPORAL          = 0.05
PCA_GRAPH_LAMBDA_DATA        = 5.0
PCA_GRAPH_LAMBDA_TEMPORAL    = 0.05
PCA_GRAPH_LAMBDA_GRAPH       = 0.05

# Splitter
NODE_MASK_PROB = 0.1
TRAIN_FRAC     = 0.10   # was 0.05 — more obs per update, same runtime

# Input 
DATA_PATH = "/home/alvaro/projects/dissertation/data/scripts/bulk_download/output/group_tech_us.csv"

# Output
OUT_PATH     = "results/vol_surface_dgraph/vol_surface_comparison.html"
METRICS_PATH = "results/vol_surface_dgraph/vol_surface_metrics.json"
RESULTS_CSV  = "results/experiment_results.csv"

    # =========================================================================


# ============================================================================
# SSVI edge learning helpers
# ============================================================================

def learn_surface_edges(
    history: list[Graph],
    underlyings: list[str],
    precision: float = 1.0,
) -> dict:
    """
    For each directed pair (i → j) fit M_{i→j} by OLS over consecutive
    parameter increments:  Δθ_j^t ≈ M_{i→j} · Δθ_i^t
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
            delta_seqs[nid] = np.array([ps[t + 1] - ps[t] for t in range(len(ps) - 1)])

    edges: dict = {}
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i + 1:]:
            if n1 not in delta_seqs or n2 not in delta_seqs:
                continue
            D1 = delta_seqs[n2]
            D2 = delta_seqs[n1]
            n_obs = min(len(D1), len(D2))
            D1, D2 = D1[:n_obs], D2[:n_obs]
            X12, _, _, _ = np.linalg.lstsq(D2, D1, rcond=None)
            X21, _, _, _ = np.linalg.lstsq(D1, D2, rcond=None)
            edges[(n1, n2)] = DeltaEdgeState(precision=precision, matrix=X12.T)
            edges[(n2, n1)] = DeltaEdgeState(precision=precision, matrix=X21.T)

    return edges


def learn_surface_edges_diagonal(
    history: list[Graph],
    underlyings: list[str],
    precision: float = 1.0,
) -> dict:
    """
    Like learn_surface_edges but diagonal M: one scalar per parameter.
    Per-parameter no-intercept OLS:  m_k = (Δθ_i[k] · Δθ_j[k]) / ‖Δθ_i[k]‖²
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
            delta_seqs[nid] = np.array([ps[t + 1] - ps[t] for t in range(len(ps) - 1)])

    edges: dict = {}
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i + 1:]:
            if n1 not in delta_seqs or n2 not in delta_seqs:
                continue
            D1 = delta_seqs[n2]
            D2 = delta_seqs[n1]
            n_obs = min(len(D1), len(D2))
            D1, D2 = D1[:n_obs], D2[:n_obs]
            denom12 = (D2 * D2).sum(axis=0)
            denom21 = (D1 * D1).sum(axis=0)
            m12 = np.where(denom12 > 0, (D2 * D1).sum(axis=0) / denom12, 0.0)
            m21 = np.where(denom21 > 0, (D1 * D2).sum(axis=0) / denom21, 0.0)
            edges[(n1, n2)] = DeltaEdgeState(precision=precision, matrix=np.diag(m12))
            edges[(n2, n1)] = DeltaEdgeState(precision=precision, matrix=np.diag(m21))

    return edges


# ============================================================================
# PCA helpers
# ============================================================================

def fit_surface_pca(
    df: pd.DataFrame,
    dates: list,
    underlyings: list[str],
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    n_components: int = 8,
) -> VolSurfacePCA:
    """
    Fit functional PCA on total-variance surfaces from training dates.
    For each (date, underlying): fit SSVI, evaluate on the (k, T) grid,
    collect as a row. PCA is run on the resulting matrix.
    """
    KK, TT = np.meshgrid(k_grid, T_grid, indexing="ij")
    K_flat, T_flat = KK.ravel(), TT.ravel()

    rows: list[np.ndarray] = []
    for date in dates:
        day_df = df[df["date"] == date]
        for underlying in underlyings:
            u_df = day_df[day_df["underlying"] == underlying]
            if u_df.empty:
                continue
            try:
                state = fit_ssvi(
                    k=u_df["logmoneyness"].values,
                    T=u_df["T"].values,
                    iv=u_df["iv"].values,
                    weights=u_df["weight"].values,
                )
                rows.append(state.total_variance(K_flat, T_flat))
            except Exception:
                continue

    pca = VolSurfacePCA(k_grid, T_grid, n_components=n_components)
    pca.fit(np.array(rows))
    return pca


class PCAGraphFactory:
    """
    Builds a Graph whose nodes hold PCASurfaceState objects.
    Fits SSVI per underlying then projects onto the shared PCA basis.
    Underlyings with no data get zero coefficients (mean surface).
    """

    def __init__(
        self,
        underlyings: list[str],
        pca: VolSurfacePCA,
        edges: dict | None = None,
    ) -> None:
        self.underlyings = underlyings
        self.pca = pca
        self.edges = edges or {}
        KK, TT = np.meshgrid(pca.k_grid, pca.T_grid, indexing="ij")
        self._K_flat = KK.ravel()
        self._T_flat = TT.ravel()

    def build(self, df: pd.DataFrame, date: pd.Timestamp) -> Graph:
        day_df = df[df["date"] == date]
        nodes: dict[SurfaceNode, PCASurfaceState] = {}
        for underlying in self.underlyings:
            u_df = day_df[day_df["underlying"] == underlying]
            if u_df.empty:
                coeffs = np.zeros(self.pca.n_components)
            else:
                try:
                    ssvi = fit_ssvi(
                        k=u_df["logmoneyness"].values,
                        T=u_df["T"].values,
                        iv=u_df["iv"].values,
                        weights=u_df["weight"].values,
                    )
                    coeffs = self.pca.transform(ssvi.total_variance(self._K_flat, self._T_flat))
                except Exception:
                    coeffs = np.zeros(self.pca.n_components)
            nodes[SurfaceNode(underlying)] = PCASurfaceState(coefficients=coeffs, pca=self.pca)
        return Graph(date, nodes, self.edges)


def learn_scalar_pca_edges(
    history: list[Graph],
    underlyings: list[str],
) -> dict:
    """
    Scalar edges: precision = |Pearson ρ| of the PC1-coefficient series
    between each pair of underlyings.
    """
    nodes = [SurfaceNode(u) for u in underlyings]
    pc1: dict = {nid: [] for nid in nodes}
    for g in history:
        for nid in nodes:
            if nid in g.nodes:
                pc1[nid].append(g.get(nid).parameters()[0])

    edges: dict = {}
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i + 1:]:
            s1 = np.array(pc1[n1])
            s2 = np.array(pc1[n2])
            n_obs = min(len(s1), len(s2))
            if n_obs < 2:
                continue
            w = abs(float(np.corrcoef(s1[:n_obs], s2[:n_obs])[0, 1]))
            edges[(n1, n2)] = DeltaEdgeState(precision=w)
            edges[(n2, n1)] = DeltaEdgeState(precision=w)
    return edges


# ============================================================================
# Experiment
# ============================================================================

def main() -> None:

    df = pd.read_csv(DATA_PATH)
    df = df[df["type"] == "call"].copy()
    df["date"]   = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])

    underlyings = sorted(df["underlying"].unique())
    dates       = sorted(df["date"].unique())
    train_dates = dates[:-N_EVAL_DAYS]           # all but last 5 days
    test_dates  = dates[-(N_EVAL_DAYS + 1):]     # last 5 eval days + 1 init day

    print(f"Underlyings  : {underlyings}")
    print(f"Total dates  : {len(dates)}")
    print(f"Train / Eval : {len(train_dates)} / {N_EVAL_DAYS} dates")

    # PCA basis must be fitted before model specs are created
    k_grid = np.linspace(df["logmoneyness"].quantile(0.02), df["logmoneyness"].quantile(0.98), N_K_GRID)
    T_grid = np.exp(np.linspace(np.log(df["T"].min()), np.log(df["T"].max()), N_T_GRID))
    print(f"\nFitting PCA basis on {len(train_dates)} training dates…")
    pca = fit_surface_pca(df, train_dates, underlyings, k_grid, T_grid, n_components=N_PCA_COMPONENTS)
    cumvar = pca.explained_variance_ratio_.cumsum()
    for i, (ev, cum) in enumerate(zip(pca.explained_variance_ratio_, cumvar)):
        print(f"  PC{i + 1}: {ev:.1%}  (cumulative {cum:.1%})")

    # Shared components
    obs_factory   = SurfaceObservationFactory(underlyings)
    ssvi_factory  = SurfaceGraphFactory(underlyings, edges={})
    pca_factory   = PCAGraphFactory(underlyings, pca)
    roller        = SurfaceRoller()
    temporal_loss = TemporalLoss(NodewiseGraphDistance(L2ParameterDistance()))
    node_loss     = SSVINodeLoss()
    data_loss     = SurfaceDataLoss()

    models = [
        # ---- SSVI models ------------------------------------------------
        ModelSpec(
            name="ssvi_data",
            build_graph=ssvi_factory.build,
            updater=SeparableGraphUpdater(
                CombinedLoss(
                    data_loss=data_loss, node_loss=node_loss,
                    lambda_data=SSVI_LAMBDA_DATA, lambda_node=SSVI_LAMBDA_NODE,
                ),
                roller=None,
            ),
            roller=roller,
            static_edges={},
        ),
        ModelSpec(
            name="ssvi_temporal",
            build_graph=ssvi_factory.build,
            updater=SeparableGraphUpdater(
                CombinedLoss(
                    data_loss=data_loss, temporal_loss=temporal_loss, node_loss=node_loss,
                    lambda_data=SSVI_LAMBDA_DATA, lambda_temporal=SSVI_LAMBDA_TEMPORAL,
                    lambda_node=SSVI_LAMBDA_NODE,
                ),
                roller=roller,
            ),
            roller=roller,
            static_edges={},
        ),
        ModelSpec(
            name="ssvi_temporal_graph",
            build_graph=ssvi_factory.build,
            updater=GraphUpdater(
                CombinedLoss(
                    data_loss=data_loss, temporal_loss=temporal_loss,
                    graph_loss=GraphLoss(), node_loss=node_loss,
                    lambda_data=SSVI_LAMBDA_DATA, lambda_temporal=SSVI_GRAPH_LAMBDA_TEMPORAL,
                    lambda_graph=SSVI_GRAPH_LAMBDA_GRAPH, lambda_node=SSVI_LAMBDA_NODE,
                ),
                roller=roller,
            ),
            roller=roller,
            build_edges=lambda history: learn_surface_edges(history, underlyings, precision=SSVI_EDGE_PRECISION),
        ),
        ModelSpec(
            name="ssvi_temporal_graph_diag",
            build_graph=ssvi_factory.build,
            updater=GraphUpdater(
                CombinedLoss(
                    data_loss=data_loss, temporal_loss=temporal_loss,
                    graph_loss=GraphLoss(), node_loss=node_loss,
                    lambda_data=SSVI_LAMBDA_DATA, lambda_temporal=SSVI_GRAPH_LAMBDA_TEMPORAL,
                    lambda_graph=SSVI_GRAPH_LAMBDA_GRAPH, lambda_node=SSVI_LAMBDA_NODE,
                ),
                roller=roller,
            ),
            roller=roller,
            build_edges=lambda history: learn_surface_edges_diagonal(history, underlyings, precision=SSVI_EDGE_PRECISION),
        ),
        # ---- PCA models -------------------------------------------------
        ModelSpec(
            name="pca_data",
            build_graph=pca_factory.build,
            updater=SeparableGraphUpdater(
                CombinedLoss(data_loss=data_loss, lambda_data=PCA_LAMBDA_DATA),
                roller=None,
            ),
            roller=roller,
            static_edges={},
        ),
        ModelSpec(
            name="pca_temporal",
            build_graph=pca_factory.build,
            updater=SeparableGraphUpdater(
                CombinedLoss(
                    data_loss=data_loss, temporal_loss=temporal_loss,
                    lambda_data=PCA_LAMBDA_DATA, lambda_temporal=PCA_LAMBDA_TEMPORAL,
                ),
                roller=roller,
            ),
            roller=roller,
            static_edges={},
        ),
        ModelSpec(
            name="pca_temporal_graph",
            build_graph=pca_factory.build,
            updater=GraphUpdater(
                CombinedLoss(
                    data_loss=data_loss, temporal_loss=temporal_loss, graph_loss=GraphLoss(),
                    lambda_data=PCA_GRAPH_LAMBDA_DATA, lambda_temporal=PCA_GRAPH_LAMBDA_TEMPORAL,
                    lambda_graph=PCA_GRAPH_LAMBDA_GRAPH,
                ),
                roller=roller,
            ),
            roller=roller,
            build_edges=lambda history: learn_scalar_pca_edges(history, underlyings),
        ),
    ]

    splitter   = NodeMaskingSplitter(node_mask_prob=NODE_MASK_PROB, train_frac=TRAIN_FRAC)
    comparison = SurfaceModelComparison(data_loss, splitter=splitter)

    experiment = Experiment(
        df=df,
        models=models,
        build_obs=obs_factory.build,
        splitter=splitter,
        output_fn=comparison,
    )

    print(f"\nTraining edge models on {len(train_dates)} dates…")
    experiment.train(train_dates)
    print("Training complete.\n")

    print(f"Running rolling experiment over {N_EVAL_DAYS} eval dates…")
    experiment.run(test_dates)

    comparison.print_table("Vol Surface — Model Comparison")
    comparison.save_metrics(METRICS_PATH, results_csv=RESULTS_CSV, experiment="vol_surface_dgraph")
    comparison.to_html(OUT_PATH, title="Vol Surface — Model Comparison")


if __name__ == "__main__":
    main()
