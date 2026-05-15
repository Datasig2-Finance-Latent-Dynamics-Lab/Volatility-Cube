from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from DependencyGraph.source.graph import Graph
from DependencyGraph.source.observation import ObservationSet
from DependencyGraph.time_stepping.roller import Roller
from .splitter import Splitter


@dataclass
class ModelSpec:
    """
    One model variant, which consists of: graph builder + loss + updater.
    Where graph builder constructs an specific graph from a dataframe. 

    build_graph : (df, date) -> Graph
        Independently fits a graph for a date using all available data (no prior,
        no regularisation).  Used for the training history and prior fitting.
        It is used as a "perfect" prior, since it is trained in a lot of data.
        Should return a Graph with empty edges; the Experiment injects the learned
        or static edge dict before every updater call.

    updater
        Any object with .update(graph, obs, prior_graph=None) -> Graph.
        Called on the test date with partial observations (train split only).

    roller : Roller | None
        Used to warm-start the optimiser: x0 = roller.roll(prior, dt).
        If None, the prior nodes are carried forward as-is with the test date.

    build_edges : (graph_history: list[Graph]) -> dict[tuple[NodeId, NodeId], EdgeState] | None
        Called after train() with the sequence of independently-fitted graphs.
        Use it to learn an edge dict from the parameter-change (Δθ) history.
        If None, static_edges is used without any training step.

    static_edges : dict[tuple[NodeId, NodeId], EdgeState]
        Edge dict used directly when build_edges is None, or as a fallback
        when fit() is called without a preceding train().
    """

    name: str
    build_graph: Callable[[pd.DataFrame, pd.Timestamp], Graph]
    updater: Any
    roller: Roller | None = None
    build_edges: Callable[[list[Graph]], dict] | None = None
    static_edges: dict = field(default_factory=dict)


class Experiment:
    """
    Two-phase experiment framework: train to learn edges, then evaluate on a
    specific (prior_date, test_date) pair.

    train(dates)
        For each model that has build_edges: independently fits build_graph on
        each date and passes the resulting history to build_edges to produce an
        edge dict.  No updater is called — the assumption is that with enough
        data the independent fits are accurate enough for dependency estimation.
        Models with no build_edges skip this phase and use static_edges directly.

    fit(prior_date, test_date) -> Any
        1. Fits a prior graph for each model on prior_date with all observations.
        2. Splits test_date observations via splitter into (train_obs, test_obs).
        3. Rolls the prior forward and runs updater.update(x0, train_obs, prior).
        4. Returns output_fn(fitted_graphs, train_obs, test_obs).

    train() may be called once and fit() called multiple times on different
    date pairs without repeating the training step.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        models: list[ModelSpec],
        build_obs: Callable[[pd.DataFrame, pd.Timestamp], ObservationSet],
        splitter: Splitter,
        output_fn: Callable[[dict[str, Graph], ObservationSet, ObservationSet], Any],
    ):
        self.df = df
        self.models = models
        self.build_obs = build_obs
        self.splitter = splitter
        self.output_fn = output_fn

        # Edges are set to static edges by default.
        self._edges: dict[str, dict] = {m.name: m.static_edges for m in models}


    def train(self, dates: list[pd.Timestamp]) -> None:
        """
        Build parameter histories and learn edges for models that need it.

        Models without build_edges are left untouched (static_edges already set).
        """
        for model in self.models:
            if model.build_edges is None:
                continue
            history = [model.build_graph(self.df, date) for date in dates]
            self._edges[model.name] = model.build_edges(history)

    def fit(self, prior_date: pd.Timestamp, test_date: pd.Timestamp) -> Any:

        """
        Evaluate all models on test_date given a prior fitted on prior_date.

        prior_date: fitted with all data (no split) to produce a high-quality prior.
        test_date:  observations are split; train split feeds the updater, test
                    split is held out for evaluation by output_fn.
        """

        test_obs_full = self.build_obs(self.df, test_date)
        train_obs, test_obs = self.splitter.split(test_obs_full)

        dt = (test_date - prior_date).days / 365
        fitted_graphs: dict[str, Graph] = {}

        for model in self.models:
            edges = self._edges[model.name]
            # build prior from ALL available observations at prior date
            prior = Graph(prior_date, model.build_graph(self.df, prior_date).nodes, edges)

            # Roll if possible
            if model.roller is not None:
                rolled = model.roller.roll(prior, dt)
                x0 = Graph(rolled.date, rolled.nodes, edges)
            else:
                x0 = Graph(test_date, prior.nodes, edges)

            fitted = model.updater.update(x0, train_obs, prior_graph=prior)

            fitted_graphs[model.name] = fitted

        return self.output_fn(fitted_graphs, train_obs, test_obs)
