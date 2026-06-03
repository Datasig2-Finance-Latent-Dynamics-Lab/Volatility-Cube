from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

from dgraph.source.graph import Graph
from dgraph.source.observation import ObservationSet
from dgraph.time_stepping.roller import Roller
from .splitter import Splitter

@dataclass
class ModelSpec:
    """
    One model variant, which consists of: graph builder + loss + updater.
    Where graph builder constructs a specific graph from a dataframe. 

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
        when run() is called without a preceding train().
    """

    name: str
    build_graph: Callable[[pd.DataFrame, pd.Timestamp], Graph]
    updater: Any
    roller: Roller | None = None
    build_edges: Callable[[list[Graph]], dict] | None = None
    static_edges: dict = field(default_factory=dict)


class Experiment:
    """
    Two-phase experiment framework: fit and train to learn edges, then run sequentially
    over a sequence of dates.

    train(dates)
        For each model that has build_edges: independently fits build_graph on
        each date and passes the resulting history to build_edges to produce an
        edge dict.  No updater is called — the assumption is that with enough
        data the independent fits are accurate enough for dependency estimation.
        Models with no build_edges skip this phase and use static_edges directly.

    run(dates) -> list[Any]
        1. On dates[0]: fits build_graph with all available data (perfect fit).
        2. On each subsequent date: splits observations via splitter, rolls the
           previous day's graph forward, and runs updater.update(x0, train_obs, prior).
        3. Calls output_fn(fitted_graphs, train_obs, test_obs) for each update step
           and returns the list of results (one per date after the first).

    train() may be called once and run() called multiple times on different
    date sequences without repeating the training step.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        models: list[ModelSpec],
        build_obs: Callable[[pd.DataFrame, pd.Timestamp], ObservationSet],
        splitter: Splitter,
        output_fn: Callable[[dict[str, Graph], ObservationSet, ObservationSet], Any],
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe for experiment.
            models (list[ModelSpec]): List of modelspecs to test the data on.
            build_obs (Callable[[pd.DataFrame, pd.Timestamp], ObservationSet]): Observation builder from dataframe
            splitter (Splitter): How to split data into fit and test at each updater call.
            output_fn (Callable[[dict[str, Graph], ObservationSet, ObservationSet], Any]): Output function of results.
        """
        self.df = df
        self.models = models
        self.build_obs = build_obs
        self.splitter = splitter
        self.output_fn = output_fn

        # Edges are set to static edges by default.
        self._edges: dict[str, dict] = {m.name: m.static_edges for m in models}


    def train(self, dates: list[pd.Timestamp]) -> None:
        """For each model that has build_edges: independently fits build_graph on
        each date and passes the resulting history to build_edges to produce an
        edge dict.  No updater is called, the assumption is that with enough
        data the independent fits are accurate enough for dependency estimation.
        Models with no build_edges skip this phase and use static_edges directly.

        Args:
            dates: List of dates to train edges on.
        """
        for model in self.models:
            if model.build_edges is None:
                continue
            history = [model.build_graph(self.df, date) for date in dates] # Create history of "perfectly" fitted graphs.
            self._edges[model.name] = model.build_edges(history) # Builds edges from "perfect" graphs.

    def run(self, dates: list[pd.Timestamp]) -> list[Any]:
        """1. On dates[0]: fits build_graph with all available data (perfect fit).
        2. On each subsequent date: splits observations via splitter, rolls the
           previous day's graph forward, and runs updater.update(x0, train_obs, prior).
        3. Calls output_fn(fitted_graphs, train_obs, test_obs) for each update step
           and returns the list of results (one per date after the first).

        Args:
            dates: Dates to run the experiment.
        """
        if len(dates) < 2:
            raise ValueError("run() requires at least 2 dates.")

        # Day 0: perfect initialisation for each model.
        current_graphs: dict[str, Graph] = {}
        for model in self.models:
            edges = self._edges[model.name]
            g = model.build_graph(self.df, dates[0])
            current_graphs[model.name] = Graph(dates[0], g.nodes, edges)

        # Day 1+
        results: list[Any] = []
        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]
            dt = (curr_date - prev_date).days / 365

            obs_full = self.build_obs(self.df, curr_date) # Build observations
            train_obs, test_obs = self.splitter.split(obs_full) # Split observations

            print(f"{curr_date.date()}  observations used: {len(train_obs)}")

            fitted_graphs: dict[str, Graph] = {}
            for model in tqdm(self.models):
                edges = self._edges[model.name] # Get edges
                prior = current_graphs[model.name] # Get prior

                if model.roller is not None: # Attempt to roll the graph
                    rolled = model.roller.roll(prior, dt)
                    x0 = Graph(rolled.date, rolled.nodes, edges)
                else:
                    x0 = Graph(curr_date, prior.nodes, edges)

                fitted = model.updater.update(x0, train_obs, prior_graph=prior) # Update the graph
                
                fitted_graphs[model.name] = fitted

            print(f"Graph has {len(fitted.node_ids())} nodes")

            current_graphs = fitted_graphs
            results.append(self.output_fn(fitted_graphs, train_obs, test_obs)) # Save output for the day.

        return results
