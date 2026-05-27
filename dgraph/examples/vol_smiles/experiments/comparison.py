from __future__ import annotations

import numpy as np

from dgraph.source.graph import Graph
from dgraph.experiments.comparison import ModelComparison, _node_label
from dgraph.experiments.splitter import NodeMaskingSplitter
from dgraph.losses.data import DataLoss


class SmileModelComparison(ModelComparison):
    """
    ModelComparison extended with per-node IV smile plots.

    Node inspector shows:
      - Parameter table (from base class)
      - IV smile chart: train/test scatter + fitted curves per model
    """

    def __init__(
        self,
        data_loss: DataLoss,
        splitter: NodeMaskingSplitter | None = None,
        k_min: float = -0.5,
        k_max: float = 0.5,
        n_curve_points: int = 120,
    ):
        super().__init__(data_loss, splitter)
        self.k_min = k_min
        self.k_max = k_max
        self.n_curve_points = n_curve_points

    # ------------------------------------------------------------------
    # Extra app data: smile observations + fitted curves
    # ------------------------------------------------------------------

    def _extra_app_data(self) -> dict:
        if not self._graphs or self._train_obs is None:
            return {}

        all_nids: list = []
        seen: set = set()
        for graph in self._graphs.values():
            for nid in graph.node_ids():
                if nid not in seen:
                    all_nids.append(nid)
                    seen.add(nid)

        k_grid = np.linspace(self.k_min, self.k_max, self.n_curve_points)
        smile_obs: dict = {}
        smile_curves: dict = {}
        node_metrics: dict = {}
        has_metrics = hasattr(self.data_loss, "metrics")

        # test_obs from NodeMaskingSplitter contains ALL observations; compute
        # the held-out set as test minus train (by object identity).
        train_ids = {id(o) for o in self._train_obs.observations}

        for nid in all_nids:
            label = _node_label(nid)

            train_list = [
                {"k": float(o.data[0]), "iv": float(o.data[1])}
                for o in self._train_obs.for_node(nid)
            ]
            test_list = [
                {"k": float(o.data[0]), "iv": float(o.data[1])}
                for o in self._test_obs.for_node(nid)
                if id(o) not in train_ids
            ]
            smile_obs[label] = {"train": train_list, "test": test_list}

            smile_curves[label] = {}
            node_metrics[label] = {}
            for mname, graph in self._graphs.items():
                if nid not in graph.nodes:
                    continue
                state = graph.get(nid)
                try:
                    iv_grid = np.clip(state.implied_vol(k_grid), 1e-8, None)
                    smile_curves[label][mname] = {
                        "k_grid":  k_grid.tolist(),
                        "iv_grid": [round(float(v), 6) for v in iv_grid],
                    }
                except Exception:
                    pass

                mini = Graph(graph.date, {nid: state}, {})
                entry: dict = {}
                for split, obs in [("train", self._train_obs), ("test", self._test_obs)]:
                    if not obs.for_node(nid):
                        entry[split] = None
                        continue
                    if has_metrics:
                        m = self.data_loss.metrics(mini, obs)
                    else:
                        m = {"mse": round(float(self.data_loss(mini, obs)), 8)}
                    entry[split] = {k: round(v, 8) for k, v in m.items()}
                node_metrics[label][mname] = entry

        return {"smile_obs": smile_obs, "smile_curves": smile_curves, "node_metrics": node_metrics}

    # ------------------------------------------------------------------
    # HTML extension hooks
    # ------------------------------------------------------------------

    def _extra_head(self) -> str:
        return """<script>
function renderSmilePlot(nodeId) {
  const obs    = D.smile_obs    && D.smile_obs[nodeId]    || {train:[], test:[]};
  const curves = D.smile_curves && D.smile_curves[nodeId] || {};
  const traces = [];

  if (obs.train.length > 0) {
    traces.push({
      x: obs.train.map(o => o.k),
      y: obs.train.map(o => +(o.iv * 100).toFixed(4)),
      mode: 'markers', name: 'Train obs',
      marker: {symbol:'circle-open', size:7, color:'#555',
               line:{color:'#555', width:1.5}},
    });
  }
  if (obs.test.length > 0) {
    traces.push({
      x: obs.test.map(o => o.k),
      y: obs.test.map(o => +(o.iv * 100).toFixed(4)),
      mode: 'markers', name: 'Test obs',
      marker: {symbol:'diamond', size:8, color:'#111'},
    });
  }

  D.model_names.forEach((m, i) => {
    const c = curves[m];
    if (!c) return;
    traces.push({
      x: c.k_grid,
      y: c.iv_grid.map(v => +(v * 100).toFixed(4)),
      mode: 'lines', name: m,
      line: {color: COLORS[i % COLORS.length], width: 2},
    });
  });

  Plotly.react('smile-plot-div', traces, {
    xaxis: {title: 'Log-moneyness', zeroline: true,
            zerolinecolor: '#ccc', zerolinewidth: 1},
    yaxis: {title: 'Implied Vol (%)'},
    legend: {orientation: 'h', y: -0.28, font: {size: 11}},
    margin: {t: 20, b: 90, l: 55, r: 20},
    height: 380,
    plot_bgcolor: '#fafafa',
  }, {responsive: true, displayModeBar: false});
}

function renderNodeMetrics(nodeId) {
  const nm = D.node_metrics && D.node_metrics[nodeId];
  if (!nm) return;
  const mk = D.metrics_keys || ['mse'];
  const fmt = v => v != null ? v.toExponential(3) : '—';
  const hdr = ['Model'];
  mk.forEach(m => hdr.push('Train ' + m.toUpperCase()));
  mk.forEach(m => hdr.push('Test ' + m.toUpperCase()));
  const rows = D.model_names.map(mname => {
    const md = nm[mname];
    const row = [mname];
    mk.forEach(m => row.push(md && md.train ? fmt(md.train[m]) : '—'));
    mk.forEach(m => row.push(md && md.test  ? fmt(md.test[m])  : '—'));
    return row;
  });
  Plotly.newPlot('node-metrics-div', [{
    type: 'table',
    header: { values: hdr, fill: {color:'#6a1b9a'},
              font: {color:'white', size:11}, align: 'center',
              line: {color:'white', width:1} },
    cells: {
      values: hdr.map((_, ci) => rows.map(r => r[ci])),
      fill:   { color: [rows.map((_, i) => i%2===0 ? '#F3E5F5' : 'white')] },
      font:   { size: 11 },
      align:  ['left', ...Array(hdr.length - 1).fill('center')],
      line:   { color: '#dee2e6', width: 0.5 }
    }
  }], { margin:{t:5,b:5,l:5,r:5}, height: 50 + D.model_names.length * 26 },
  { responsive: true, displayModeBar: false });
}
</script>"""

    def _extra_node_html(self) -> str:
        return (
            '<div id="smile-plot-div" style="margin-top:14px;"></div>'
            '<div id="node-metrics-div" style="margin-top:14px;"></div>'
        )

    def _extra_init_js(self) -> str:
        return "renderSmilePlot(nodeId); renderNodeMetrics(nodeId);"
