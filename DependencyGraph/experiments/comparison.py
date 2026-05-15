from __future__ import annotations

import dataclasses
import json

import numpy as np

from DependencyGraph.source.graph import Graph
from DependencyGraph.source.observation import ObservationSet
from DependencyGraph.losses.data import DataLoss
from DependencyGraph.experiments.splitter import NodeMaskingSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_label(nid) -> str:
    if hasattr(nid, "underlying") and hasattr(nid, "expiry"):
        return f"{nid.underlying} {nid.expiry.strftime('%Y-%m-%d')}"
    if hasattr(nid, "underlying"):
        return str(nid.underlying)
    return str(nid)


def _get_param_names(state) -> list[str]:
    _skip = {"T", "precision", "_dm_cache", "knots", "degree"}
    try:
        fields = [
            f.name for f in dataclasses.fields(state)
            if f.name not in _skip and not f.name.startswith("_")
        ]
        if len(fields) == state.n_params:
            return fields
    except (TypeError, AttributeError):
        pass
    return [f"c{i}" for i in range(state.n_params)]


def _serialize_precision(prec) -> float | list | None:
    if prec is None:
        return None
    if isinstance(prec, np.ndarray):
        return prec.tolist()
    return float(prec)


# ---------------------------------------------------------------------------
# HTML template (placeholders: ___TITLE___, ___APP_DATA_JSON___,
#                               ___EXTRA_HEAD___, ___EXTRA_NODE_HTML___,
#                               ___EXTRA_INIT_JS___)
# ---------------------------------------------------------------------------

_HTML_BASE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>___TITLE___</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
___EXTRA_HEAD___
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1500px; margin: 0 auto; padding: 24px; background: #f4f6f8; color: #333; }
  h1   { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }
  h2   { color: #34495e; margin-top: 0; font-size: 1.05em; }
  .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
          box-shadow: 0 2px 4px rgba(0,0,0,.08); }
  select { font-size: 14px; padding: 6px 12px; border-radius: 6px;
           border: 1px solid #bdc3c7; background: white; width: 100%;
           max-width: 520px; margin-bottom: 10px; cursor: pointer; }
  select:focus { outline: none; border-color: #3498db; }
</style>
</head>
<body>
<h1>___TITLE___</h1>

<div class="card">
  <h2>Summary</h2>
  <div id="summary-div"></div>
</div>

<div class="card">
  <h2>Node Inspector</h2>
  <select id="node-select" onchange="onNodeChange(this.value)"></select>
  <div id="node-params-div"></div>
  ___EXTRA_NODE_HTML___
</div>

<script>
const D = ___APP_DATA_JSON___;
const COLORS = ['#2196F3','#FF5722','#4CAF50','#9C27B0','#FF9800','#00BCD4','#E91E63','#009688'];

// ---- populate dropdowns ------------------------------------------------
(function() {
  const ns = document.getElementById('node-select');
  D.node_ids.forEach(id => {
    const o = document.createElement('option');
    o.value = id; o.textContent = id; ns.appendChild(o);
  });
})();

// ---- summary table -----------------------------------------------------
(function() {
  const splits = D.has_split
    ? ['train','test','test_unmasked','test_masked']
    : ['train','test'];
  const sLabels = {train:'Train', test:'Test',
                   test_unmasked:'Unmasked', test_masked:'Masked'};
  const hdr = ['Model'];
  splits.forEach(s => D.metrics_keys.forEach(m =>
    hdr.push(sLabels[s] + ' ' + m.toUpperCase())));

  const rows = D.model_names.map(name => {
    const r = D.summary[name];
    const row = [name];
    splits.forEach(s => D.metrics_keys.forEach(m => {
      const v = r[s] ? r[s][m] : null;
      row.push(v != null ? v.toFixed(5) : 'N/A');
    }));
    return row;
  });

  Plotly.newPlot('summary-div', [{
    type: 'table',
    header: { values: hdr, fill: {color:'#2c3e50'}, font: {color:'white', size:12},
              align: 'center', line: {color:'white', width:1} },
    cells: {
      values: hdr.map((_, ci) => rows.map(r => r[ci])),
      fill:   { color: [rows.map((_, i) => i%2===0 ? '#EAF2FF' : 'white')] },
      font:   { size: 12 },
      align:  ['left', ...Array(hdr.length - 1).fill('center')],
      line:   { color: '#dee2e6', width: 0.5 }
    }
  }], { margin:{t:5,b:5,l:5,r:5}, height: 50 + D.model_names.length * 26 },
  { responsive: true, displayModeBar: false });
})();

// ---- helpers -----------------------------------------------------------
function fmtPrec(p) {
  if (p == null) return '—';
  if (typeof p === 'number') return p.toFixed(4);
  if (Array.isArray(p)) {
    const diag = Array.isArray(p[0]) ? p.map((r, i) => r[i]) : p;
    return '[' + diag.slice(0,4).map(x => (+x).toFixed(3)).join(', ')
               + (diag.length > 4 ? '…' : '') + ']';
  }
  return String(p);
}

// ---- node inspector ----------------------------------------------------
function renderNodeParams(nodeId) {
  const nd = D.nodes[nodeId];
  if (!nd) return;
  const rows = D.model_names.map(m => {
    const md = nd.models[m];
    if (!md) return [m, '—', '—'];
    const names = md.param_names && md.param_names.length === md.params.length
      ? md.param_names
      : md.params.map((_, i) => 'c' + i);
    const paramsStr = names.map((n, i) => n + '=' + (+md.params[i]).toFixed(5)).join(',  ');
    return [m, paramsStr, fmtPrec(md.precision)];
  });
  Plotly.newPlot('node-params-div', [{
    type: 'table',
    header: { values: ['Model', 'Parameters', 'Precision'], fill: {color:'#1a6b8a'},
              font: {color:'white', size:12}, align: ['center','left','center'],
              line: {color:'white', width:1} },
    cells: {
      values: [rows.map(r=>r[0]), rows.map(r=>r[1]), rows.map(r=>r[2])],
      fill:   { color: [rows.map((_, i) => i%2===0 ? '#E8F4F8' : 'white')] },
      font:   { size: 12 },
      align:  ['left', 'left', 'center'],
      line:   { color: '#dee2e6', width: 0.5 }
    }
  }], { margin:{t:5,b:5,l:5,r:5}, height: 50 + D.model_names.length * 26 },
  { responsive: true, displayModeBar: false });
}

function onNodeChange(nodeId) {
  renderNodeParams(nodeId);
  ___EXTRA_INIT_JS___
}

// ---- initialise --------------------------------------------------------
if (D.node_ids.length > 0) onNodeChange(D.node_ids[0]);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# ModelComparison
# ---------------------------------------------------------------------------

class ModelComparison:
    """
    Computes and displays train / test data-loss for a set of fitted graphs.

    Stores fitted graphs and observations for use in to_html().  Calling
    compare() (or using the instance as output_fn) replaces previous results.

    Metrics reported depend on the data_loss implementation:
      - Always available: MSE
      - If data_loss has a metrics() method: also MAE and MAPE

    Pass a NodeMaskingSplitter to also report test MSE broken down by masked
    vs unmasked nodes.
    """

    def __init__(
        self,
        data_loss: DataLoss,
        splitter: NodeMaskingSplitter | None = None,
    ):
        self.data_loss = data_loss
        self.splitter  = splitter
        self.results:     dict[str, dict] = {}
        self._graphs:     dict[str, Graph] = {}
        self._train_obs:  ObservationSet | None = None
        self._test_obs:   ObservationSet | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compare(
        self,
        graphs: dict[str, Graph],
        train_obs: ObservationSet,
        test_obs:  ObservationSet,
    ) -> "ModelComparison":
        self._graphs    = dict(graphs)
        self._train_obs = train_obs
        self._test_obs  = test_obs

        masked_obs, unmasked_obs = self._split_test(test_obs)
        self.results = {}
        for name, graph in graphs.items():
            self.results[name] = {
                "train":         self._eval(graph, train_obs),
                "test":          self._eval(graph, test_obs),
                "test_unmasked": self._eval(graph, unmasked_obs) if unmasked_obs else None,
                "test_masked":   self._eval(graph, masked_obs)   if masked_obs   else None,
            }
        return self

    def __call__(
        self,
        graphs: dict[str, Graph],
        train_obs: ObservationSet,
        test_obs:  ObservationSet,
    ) -> "ModelComparison":
        return self.compare(graphs, train_obs, test_obs)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_table(self, title: str = "Model Comparison") -> None:
        if not self.results:
            print("No results — call compare() first.")
            return

        first    = next(iter(self.results.values()))
        mk_keys  = list(first["test"].keys())
        has_split = any(r.get("test_masked") is not None for r in self.results.values())

        splits = ["train", "test"]
        if has_split:
            splits += ["test_unmasked", "test_masked"]
        slabels = {"train": "Train", "test": "Test",
                   "test_unmasked": "Unmasked", "test_masked": "Masked"}

        col = max(len(n) for n in self.results) + 2
        w   = 11

        def _fmt(d, key):
            if d is None:
                return " " * (w - 3) + "N/A"
            return f"{d[key]:{w}.6f}"

        hdr = f"{'Model':<{col}}"
        for s in splits:
            for m in mk_keys:
                hdr += f" {slabels[s]+'.'+m.upper():>{w}}"
        print(f"\n{title}")
        print(hdr)
        print("-" * (col + len(splits) * len(mk_keys) * (w + 1)))

        for name, r in self.results.items():
            row = f"{name:<{col}}"
            for s in splits:
                for m in mk_keys:
                    row += " " + _fmt(r.get(s), m)
            print(row)

    def to_html(self, path: str, title: str = "Model Comparison") -> None:
        """Write an interactive HTML report to *path*."""
        html = self._build_html(title)
        with open(path, "w") as f:
            f.write(html)
        print(f"Saved report to {path}")

    # ------------------------------------------------------------------
    # Internal builders (override in subclasses)
    # ------------------------------------------------------------------

    def _build_html(self, title: str) -> str:
        app_data = self._build_app_data()
        app_data.update(self._extra_app_data())

        html = _HTML_BASE
        html = html.replace("___TITLE___",         title)
        html = html.replace("___APP_DATA_JSON___",  json.dumps(app_data))
        html = html.replace("___EXTRA_HEAD___",     self._extra_head())
        html = html.replace("___EXTRA_NODE_HTML___", self._extra_node_html())
        html = html.replace("___EXTRA_INIT_JS___",   self._extra_init_js())
        return html

    def _build_app_data(self) -> dict:
        model_names = list(self._graphs.keys())

        # Collect all node IDs in stable order
        all_nids: list = []
        seen_nids: set = set()
        for graph in self._graphs.values():
            for nid in graph.node_ids():
                if nid not in seen_nids:
                    all_nids.append(nid)
                    seen_nids.add(nid)

        # Node data — param_names stored per model entry so mixed-type graphs work
        node_ids = [_node_label(n) for n in all_nids]
        node_data: dict = {}
        for i, nid in enumerate(all_nids):
            label = node_ids[i]
            node_data[label] = {"models": {}}
            for mname, graph in self._graphs.items():
                if nid in graph.nodes:
                    state = graph.get(nid)
                    node_data[label]["models"][mname] = {
                        "param_names": _get_param_names(state),
                        "params":      [round(float(p), 7) for p in state.parameters()],
                        "precision":   _serialize_precision(state.precision),
                    }

        # Summary metrics
        first_r   = next(iter(self.results.values()))
        mk_keys   = list(first_r["test"].keys())
        has_split = any(r.get("test_masked") is not None for r in self.results.values())

        summary: dict = {}
        for name, r in self.results.items():
            summary[name] = {}
            for split in ["train", "test", "test_unmasked", "test_masked"]:
                d = r.get(split)
                summary[name][split] = (
                    {k: round(v, 8) for k, v in d.items()} if d else None
                )

        return {
            "model_names":  model_names,
            "node_ids":     node_ids,
            "nodes":        node_data,
            "summary":      summary,
            "metrics_keys": mk_keys,
            "has_split":    has_split,
        }

    # Subclass hooks
    def _extra_app_data(self) -> dict:
        return {}

    def _extra_head(self) -> str:
        return ""

    def _extra_node_html(self) -> str:
        return ""

    def _extra_init_js(self) -> str:
        return ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval(self, graph: Graph, obs: ObservationSet) -> dict[str, float]:
        if hasattr(self.data_loss, "metrics"):
            return self.data_loss.metrics(graph, obs)
        return {"mse": float(self.data_loss(graph, obs))}

    def _split_test(
        self, test_obs: ObservationSet
    ) -> tuple[ObservationSet | None, ObservationSet | None]:
        if self.splitter is None:
            return None, None
        masked_ids = getattr(self.splitter, "masked_nodes", set())
        if not masked_ids:
            return None, None
        masked   = [o for o in test_obs.observations if o.node_id in masked_ids]
        unmasked = [o for o in test_obs.observations if o.node_id not in masked_ids]
        masked_obs   = ObservationSet(masked,   test_obs.date) if masked   else None
        unmasked_obs = ObservationSet(unmasked, test_obs.date) if unmasked else None
        return masked_obs, unmasked_obs
