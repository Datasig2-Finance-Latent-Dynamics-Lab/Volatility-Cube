"""
Builds surface Graph objects and ObservationSets from a raw options DataFrame.

Each row is expected to have columns:
  date, underlying, expiry, dte, T, logmoneyness, iv, weight
"""
import numpy as np
import pandas as pd

from DependencyGraph.source.graph import Graph
from DependencyGraph.source.edge import DeltaEdgeState
from DependencyGraph.source.observation import Observation, ObservationSet
from ...vol_smiles.source.nodes import SurfaceNode
from .states.ssvi import SSVISurfaceState, fit_ssvi


class SurfaceObservationFactory:
    """Builds an ObservationSet for a single date. Each observation carries (k, T, iv)."""

    def __init__(self, underlyings: list[str]):
        self.underlyings = underlyings

    def build(self, df: pd.DataFrame, date: pd.Timestamp) -> ObservationSet:

        day_df = df[df["date"] == date]
        observations: list[Observation] = []

        for underlying in self.underlyings:
            u_df = day_df[day_df["underlying"] == underlying]
            if u_df.empty:
                continue
            nid = SurfaceNode(underlying)
            for _, row in u_df.iterrows():
                observations.append(
                    Observation(nid, (row["logmoneyness"], row["T"], row["iv"]), float(row["weight"]))
                )

        return ObservationSet(observations, date)


class SurfaceGraphFactory:
    """Builds a Graph with one SSVISurfaceState per underlying."""

    def __init__(self, underlyings: list[str], edges: dict | None = None):
        self.underlyings = underlyings
        self.edges = edges or {}

    def build(self, df: pd.DataFrame, date: pd.Timestamp) -> Graph:
        day_df = df[df["date"] == date]
        nodes: dict[SurfaceNode, SSVISurfaceState] = {}

        for underlying in self.underlyings:
            u_df = day_df[day_df["underlying"] == underlying]
            if u_df.empty:
                continue
            state = fit_ssvi(
                k=u_df["logmoneyness"].values,
                T=u_df["T"].values,
                iv=u_df["iv"].values,
                weights=u_df["weight"].values,
            )
            nodes[SurfaceNode(underlying)] = state

        return Graph(date, nodes, self.edges)


def build_tiered_surface_edges(
    underlyings: list[str],
    spy_to_stock: float = 0.60,
    stock_to_spy: float = 0.20,
    stock_to_stock: float = 0.40
) -> dict:
    """
    Assumes no matrix dependence between states, simply builds edges with given precisions.
    Good for quick testing.
    """
    edges: dict = {}
    nodes = [SurfaceNode(u) for u in underlyings]
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i + 1:]:
            u1, u2 = n1.underlying, n2.underlying
            if u1 == "SPY":
                w12, w21 = spy_to_stock, stock_to_spy
            elif u2 == "SPY":
                w12, w21 = stock_to_spy, spy_to_stock
            else:
                w12 = w21 = stock_to_stock

            edges[(n1, n2)] = DeltaEdgeState(precision=w12, matrix=None)
            edges[(n2, n1)] = DeltaEdgeState(precision=w21, matrix=None)

    return edges
