# DependencyGraph

A framework for fitting and tracking a graph of parametric states through time, where all states are updated jointly at each time step by minimising a structured loss function.

## Concept

The core object is a **Graph**: a set of nodes, each carrying a **State** (a vector of parameters), connected by directed weighted **edges**. At each point in time new observations arrive — possibly only for a subset of nodes — and the framework updates *all* node states simultaneously by solving:

```
min   λ_data · L_data(graph, observations)
    + λ_temporal · L_temporal(graph, rolled_prior)
    + λ_graph · L_graph(graph - rolled_prior)
    + λ_node · L_node(graph)
```

Each loss component is optional and independently weighted.  `rolled_prior` is the
previous graph advanced forward by `dt` days; it is computed once per step by
`CombinedLoss` (which owns the `Roller`) and shared by both temporal and graph losses.

### Loss components

| Component | Purpose |
|---|---|
| **Data loss** | Penalises misfit between the fitted states and the incoming observations. |
| **Temporal loss** | Penalises node-wise distance between the new graph and the *rolled prior* (the previous graph advanced to today). Encourages temporal smoothness. |
| **Graph loss** | Encodes cross-node dependency: penalises `Σ w_ij ‖Δθ_i − Δθ_j‖²` where `Δθ_k = θ_k^new − θ_k^rolled`. |
| **Node loss** | An independent penalty (e.g. no-arbitrage constraints). |

### Architecture

```
source/
  core/
    graph.py          # Graph: nodes + edges, parameter flattening
    node.py           # NodeId
    observation.py    # Observation / ObservationSet
    dependency.py     # Dependencies ABC + StaticDependencies
    roller.py         # Roller ABC: advances a graph forward
    state.py          # State ABC
  losses/
    combined.py       # CombinedLoss: owns Roller, rolls prior once, passes to temporal+graph
    data.py           # DataLoss ABC
    temporal.py       # TemporalLoss
    graph.py          # GraphLoss ABC + L2DependencyGraphLoss: penalises Δθ_i − Δθ_j
    node.py           # NodeLoss ABC
  distances/
    state.py          # StateDistance ABC (L2 parameter distance)
    graph.py          # GraphDistance ABC (node-wise)
  updater.py          # GraphUpdater / SeparableGraphUpdater: wraps scipy.minimize
```

Everything in `source/` is application-agnostic. Concrete models live in `examples/`.

```
examples/vol_smiles/
  nodes.py            # CurveNode (underlying, expiry, T) and SurfaceNode
  curves/
    base.py           # CurveState ABC for parametric curves
    svi.py            # SVI (raw + Jump-Wing) state
    bspline.py        # Cubic B-spline state
  losses/
    data.py           # VolDataLoss, CalendarSpreadPenalty
  distances.py        # L2TotalVarianceDistance
  rollers.py          # VolRoller, StickyStrikeRoller, StickyDeltaRoller
  dependencies.py     # UniformCrossAssetDependencies, CorrelationDependencies
  factory.py          # ObservationFactory, GraphFactory, fit helpers
  updater.py          # BSplineUpdater: analytic linear-system solver
  experiments/
    main.py           # Backtest: (SVI | B-spline) × (data | +temporal | +temporal+graph)
```

---

## Vol smiles example

`examples/vol_smiles/` applies the framework to fitting implied volatility smiles.

### Nodes

Each node is a **`CurveNode(underlying, T)`** identifying a single vol smile for a given underlying asset (e.g. `"AAPL"`) at a given time-to-expiry `T` (in years).OoooIncl

### Edges

Edge values are typed as `Any`, with `None` meaning no edge. The framework doesn't care about what an edge value represents; it is the `GraphLoss` implementation that decides how to use it. Two concrete scalar dependency classes are provided in `dependencies.py`:

| Class | Edge value (weight) |
|---|---|
| `UniformCrossAssetDependencies` | Fixed scalar — same weight for every same-expiry cross-asset pair. |
| `CorrelationDependencies` | Pairwise absolute return correlation $|\rho_{ij}|$, with an optional minimum threshold. |

`L2DependencyGraphLoss` (in `source/`) expects scalar float edge values and penalises $\sum_{(i,j)} w_{ij} \|\theta_i - \theta_j\|^2$. Custom `GraphLoss` subclasses can interpret edge values differently (e.g. a matrix encoding an OLS relationsh

### Temporal rolling

`VolRoller` advances the graph forward by converting calendar days to years (`dt_years = dt / 365`) and subtracting from each node's `T`. SVI parameters are carried forward unchanged as the simplest possible prior; a richer roller could model term-structure dynamics.

### Data

Each observation is a `(log-moneyness, implied_vol)` pair associated with a `CurveNode`. The data loss is weighted MSE in implied volatility space.

### Experiments

`main.py` runs a backtest over a sequence of dates comparing model configurations, 
three curve parameterisations crossed with three regularisation regimes:

| Label | Curve | Loss components |
|---|---|---|
| `svi_data` | SVI | Data |
| `svi_data_temporal` | SVI | Data + Temporal |
| `svi_data_temporal_graph` | SVI | Data + Temporal + Graph (joint optimiser) |
| `jw_data` | SVI | Data |
| `jw_data_temporal` | SVI | Data + Temporal |
| `jw_data_temporal_graph` | SVI | Data + Temporal + Graph (joint optimiser) |
| `bspline_data` | B-spline | Data |
| `bspline_data_temporal` | B-spline | Data + Temporal |
| `bspline_data_temporal_graph` | B-spline | Data + Temporal + Graph |

At each date the observation set is split into train and test, a fraction of nodes are fully
masked (zero train observations) to test cross-asset imputation via the graph loss.
The fitted graph from the previous day is rolled forward and used 
as the temporal/graph prior, so each model's trajectory diverges over time.

Obviously results are not too meaningfull since the priors are not good.

### Running

```bash
source .venv/bin/activate
python -m DependencyGraph.examples.vol_smiles.experiments.main
```

Input data is expected at `Data/options_surface_sample.csv` with columns:
`date, underlying, expiry, dte, T, logmoneyness, iv, weight`.
