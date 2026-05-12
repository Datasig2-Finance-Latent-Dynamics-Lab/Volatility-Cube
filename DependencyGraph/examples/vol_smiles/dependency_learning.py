# TODO: read and understand this in detail, move to appropiate places and maybe make better structure. This is messy.
# Not used for the moment. Idea is to learn dependencis from past data.

"""
Learning edge values from Δθ sequences extracted by a data-only backtest.

The learner fits the graph at each historical date using only the data loss
(no temporal or graph regularisation), then estimates edge values from the
resulting parameter change sequences Δθ(t) = θ(t) - θ(t-1).

Two implementations:

  ScalarDependencyLearner
      w_ij = |Corr(Δθ_i_flat, Δθ_j_flat)|  — a single scalar per pair.
      Use with L2DependencyGraphLoss.

  MatrixDependencyLearner
      A_ij from OLS regression of Δθ_i on Δθ_j — a (p × p) matrix per directed pair.
      Use with MatrixDependencyGraphLoss (also defined here).

Usage example:

    roller  = StickyStrikeRoller()
    learner = ScalarDependencyLearner(roller, min_corr=0.1)
    deps    = learner.fit(df, initial_graph, obs_factory)

    # Then use deps in build_tiered_dependencies or pass directly to Graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ...source.core.dependency import Dependencies, StaticDependencies
from ...source.core.graph import Graph
from ...source.core.node import NodeId
from ...source.core.roller import Roller
from ...source.losses.combined import CombinedLoss
from ...source.losses.graph import GraphLoss
from ...source.updater import SeparableGraphUpdater
from .factory import ObservationFactory
from .losses.data import VolDataLoss


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DependencyLearner(ABC):
    """
    Estimates edge values from Δθ sequences obtained by a data-only backtest.

    The internal fitting loop uses the provided updater (defaulting to
    SeparableGraphUpdater with VolDataLoss) with prior_graph=None at every
    step, so each date is fitted independently without temporal anchoring.

    Parameters
    ----------
    roller      : used to warm-start each fitting step (rolls the previous
                  fitted graph forward to the current date).
    updater     : updater to use for fitting.  If None, a SeparableGraphUpdater
                  with VolDataLoss is constructed automatically.
    min_periods : minimum number of consecutive Δθ observations required for
                  a node pair to produce an edge.
    """

    def __init__(
        self,
        roller: Roller,
        updater=None,
        min_periods: int = 2,
    ):
        self.roller = roller
        self.min_periods = min_periods
        self._updater = updater or SeparableGraphUpdater(
            CombinedLoss(data_loss=VolDataLoss())
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame | str,
        initial_graph: Graph,
        obs_factory: ObservationFactory,
        dates: list[pd.Timestamp] | None = None,
    ) -> Dependencies:
        """
        Run the data-only backtest, collect Δθ sequences, and return Dependencies.

        Parameters
        ----------
        df            : raw options DataFrame or path to CSV.
        initial_graph : graph for the first date (warm-start and node template).
        obs_factory   : builds ObservationSets from the DataFrame.
        dates         : dates to include; defaults to all unique dates in df.
        """
        if isinstance(df, str):
            df = pd.read_csv(df, parse_dates=["date", "expiry"])

        if dates is None:
            dates = sorted(df["date"].unique())

        deltas = self._collect_deltas(df, initial_graph, obs_factory, dates)
        node_ids = list(deltas.keys())
        return self._estimate_dependencies(deltas, node_ids)

    # ------------------------------------------------------------------
    # Internal: backtest loop
    # ------------------------------------------------------------------

    def _collect_deltas(
        self,
        df: pd.DataFrame,
        initial_graph: Graph,
        obs_factory: ObservationFactory,
        dates: list[pd.Timestamp],
    ) -> dict[NodeId, np.ndarray]:
        """
        Return a dict mapping each NodeId to a (T-1, p) array of Δθ values.

        Dates with missing observations for a node are skipped for that node,
        so pairwise alignment is handled per-pair in _estimate_dependencies.
        """
        # dated_thetas[nid][date] = parameter vector at that date
        dated_thetas: dict[NodeId, dict[pd.Timestamp, np.ndarray]] = {}

        graph = initial_graph
        prior: Graph | None = None

        for date in dates:
            obs = obs_factory.build(df, pd.Timestamp(date))
            if len(obs) == 0:
                continue

            if prior is not None:
                dt = int((date - prior.date).days)
                graph = self.roller.roll(prior, dt)

            graph = self._updater.update(graph, obs, prior_graph=None)
            prior = graph

            for nid in graph.node_ids():
                dated_thetas.setdefault(nid, {})[date] = graph.get(nid).parameters()

        # Build aligned Δθ sequences per node
        deltas: dict[NodeId, np.ndarray] = {}
        for nid, by_date in dated_thetas.items():
            sorted_dates = sorted(by_date.keys())
            if len(sorted_dates) < self.min_periods + 1:
                continue
            arr = np.stack([by_date[d] for d in sorted_dates])  # (T, p)
            deltas[nid] = np.diff(arr, axis=0)                  # (T-1, p)

        return deltas

    # ------------------------------------------------------------------
    # To implement in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _estimate_dependencies(
        self,
        deltas: dict[NodeId, np.ndarray],
        node_ids: list[NodeId],
    ) -> Dependencies:
        """Estimate edge values from Δθ sequences and return a Dependencies object."""
        ...

    # ------------------------------------------------------------------
    # Shared helper: aligned delta pair
    # ------------------------------------------------------------------

    @staticmethod
    def _aligned_pair(
        deltas: dict[NodeId, np.ndarray],
        ni: NodeId,
        nj: NodeId,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Return (Δθ_i, Δθ_j) aligned to the same time axis by taking the
        shorter of the two sequences from their shared start.  Returns None
        if either node is missing or too short.
        """
        if ni not in deltas or nj not in deltas:
            return None
        di, dj = deltas[ni], deltas[nj]
        n = min(len(di), len(dj))
        if n < 2:
            return None
        return di[:n], dj[:n]


# ---------------------------------------------------------------------------
# Scalar learner
# ---------------------------------------------------------------------------

class ScalarDependencyLearner(DependencyLearner):
    """
    Scalar edges: w_ij = |Corr(Δθ_i_flat, Δθ_j_flat)|.

    The two delta arrays are flattened over the (time, parameter) axes before
    computing Pearson correlation, so each (t, k) entry is treated as one
    observation.  This is the natural scalar summary for the L2 graph loss
    Σ w_ij ||Δθ_i - Δθ_j||² — a high absolute correlation means the two nodes
    tend to move together across all parameter dimensions simultaneously.

    Edges are symmetric and undirected.

    Parameters
    ----------
    roller      : warm-start roller.
    min_corr    : absolute correlation threshold; pairs below this get no edge.
    updater     : optional custom updater (defaults to data-only VolDataLoss).
    min_periods : minimum Δθ observations needed to form an edge.
    """

    def __init__(
        self,
        roller: Roller,
        min_corr: float = 0.0,
        updater=None,
        min_periods: int = 2,
    ):
        super().__init__(roller, updater, min_periods)
        self.min_corr = min_corr

    def _estimate_dependencies(
        self,
        deltas: dict[NodeId, np.ndarray],
        node_ids: list[NodeId],
    ) -> StaticDependencies:
        edges: dict[tuple[NodeId, NodeId], float] = {}

        for i, ni in enumerate(node_ids):
            for nj in node_ids[i + 1:]:
                pair = self._aligned_pair(deltas, ni, nj)
                if pair is None:
                    continue
                di, dj = pair[0].flatten(), pair[1].flatten()
                corr_mat = np.corrcoef(di, dj)
                w = abs(float(corr_mat[0, 1]))

                if np.isnan(w) or w < self.min_corr:
                    continue

                edges[(ni, nj)] = w
                edges[(nj, ni)] = w

        return StaticDependencies(edges)


# ---------------------------------------------------------------------------
# Matrix learner
# ---------------------------------------------------------------------------

class MatrixDependencyLearner(DependencyLearner):
    """
    Matrix edges: A_ij estimated by OLS regression of Δθ_i on Δθ_j.

    For the directed edge i←j, A_ij is the (p × p) matrix solving:

        min_A  Σ_t ||Δθ_i(t) - A Δθ_j(t)||²_2

    which has the closed-form solution:

        A_ij = (Σ_t Δθ_i(t) Δθ_j(t)^T)(Σ_t Δθ_j(t) Δθ_j(t)^T)^{-1}

    Edges are directed: A_ij (i←j) and A_ji (j←i) are estimated independently,
    so A_ij ≈ A_ji^{-1} in expectation but is not enforced.

    Use MatrixDependencyGraphLoss (defined below) with these edges — not
    L2DependencyGraphLoss, which expects scalar floats.

    Parameters
    ----------
    roller      : warm-start roller.
    min_r2      : R² threshold on the training data; pairs with a weaker
                  linear fit get no edge.
    updater     : optional custom updater.
    min_periods : minimum Δθ observations needed to fit A reliably.
    rcond       : regularisation cutoff passed to np.linalg.lstsq.
    """

    def __init__(
        self,
        roller: Roller,
        min_r2: float = 0.0,
        updater=None,
        min_periods: int = 5,
        rcond: float = 1e-6,
    ):
        super().__init__(roller, updater, min_periods)
        self.min_r2 = min_r2
        self.rcond = rcond

    def _estimate_dependencies(
        self,
        deltas: dict[NodeId, np.ndarray],
        node_ids: list[NodeId],
    ) -> StaticDependencies:
        edges: dict[tuple[NodeId, NodeId], np.ndarray] = {}

        for ni in node_ids:
            for nj in node_ids:
                if ni is nj:
                    continue
                pair = self._aligned_pair(deltas, ni, nj)
                if pair is None:
                    continue
                Di, Dj = pair  # (T-1, p) each

                # OLS: A such that (A @ Δθ_j) ≈ Δθ_i
                # Each column of A.T is a separate regression of one component
                # of Δθ_i against all components of Δθ_j.
                # lstsq solves Dj @ A.T ≈ Di  →  A.T has shape (p, p)
                A_T, _, _, _ = np.linalg.lstsq(Dj, Di, rcond=self.rcond)
                A = A_T.T  # (p, p)

                if self.min_r2 > 0.0 and not self._passes_r2(Di, Dj, A):
                    continue

                edges[(ni, nj)] = A

        return StaticDependencies(edges)

    def _passes_r2(
        self, Di: np.ndarray, Dj: np.ndarray, A: np.ndarray
    ) -> bool:
        residuals = Di - (A @ Dj.T).T
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((Di - Di.mean(axis=0)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return r2 >= self.min_r2


# ---------------------------------------------------------------------------
# Graph loss for matrix edges
# ---------------------------------------------------------------------------

class MatrixDependencyGraphLoss(GraphLoss):
    """
    Graph loss for matrix-valued edges produced by MatrixDependencyLearner.

    For each directed edge (i, j) with matrix A_ij:

        loss = Σ_{(i,j)} ||Δθ_i - A_ij Δθ_j||²_2
        where Δθ_k = θ_k^new - θ_k^rolled

    This is the delta-based analogue of L2DependencyGraphLoss, but instead
    of penalising identical movements it penalises deviations from the learned
    linear relationship between movements.
    """

    def __call__(self, graph: Graph, rolled_prior: Graph) -> float:
        try:
            all_edges = graph.dependencies.edges()
        except NotImplementedError:
            return 0.0

        # Pre-compute Δθ for all nodes that appear in edges
        delta_cache: dict[NodeId, np.ndarray] = {}

        def get_delta(nid: NodeId) -> np.ndarray | None:
            if nid in delta_cache:
                return delta_cache[nid]
            state = graph.nodes.get(nid)
            if state is None:
                return None
            theta_new = state.parameters()
            prior_state = rolled_prior.nodes.get(nid)
            theta_rolled = prior_state.parameters() if prior_state is not None else np.zeros_like(theta_new)
            delta_cache[nid] = theta_new - theta_rolled
            return delta_cache[nid]

        total = 0.0
        for src, tgt, A in all_edges:
            if A is None or not isinstance(A, np.ndarray):
                continue
            d_src = get_delta(src)
            d_tgt = get_delta(tgt)
            if d_src is None or d_tgt is None:
                continue
            residual = d_src - A @ d_tgt
            total += float(residual @ residual)

        return total
