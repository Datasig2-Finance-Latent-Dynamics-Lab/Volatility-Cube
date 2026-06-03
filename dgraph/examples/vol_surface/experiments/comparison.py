from __future__ import annotations

import numpy as np

from dgraph.experiments.comparison import ModelComparison, _node_label
from dgraph.experiments.splitter import NodeMaskingSplitter
from dgraph.losses.data import DataLoss
from utils.pricing import bs_call_from_iv


class SurfaceModelComparison(ModelComparison):
    """
    ModelComparison extended with interactive 3D vol surface plots.

    Node inspector panel shows:
      - Parameter table (from base class)
      - 3D surface: train/test observations + one surface per model
      - 2D smile slice: pick any maturity from the T grid via a dropdown
    """

    def __init__(
        self,
        data_loss: DataLoss,
        splitter: NodeMaskingSplitter | None = None,
        n_k: int = 40,
        n_T: int = 18,
    ):
        super().__init__(data_loss, splitter)
        self.n_k = n_k
        self.n_T = n_T

    # ------------------------------------------------------------------
    # Extra app data: surface grids + scattered observations
    # ------------------------------------------------------------------

    def _extra_app_data(self) -> dict:
        if not self._graphs or self._train_obs is None:
            return {}

        all_obs = list(self._test_obs.observations)
        k_vals = [float(o.data[0]) for o in all_obs]
        T_vals = [float(o.data[1]) for o in all_obs]

        k_grid = np.linspace(np.percentile(k_vals, 2), np.percentile(k_vals, 98), self.n_k)
        T_grid = np.exp(
            np.linspace(np.log(max(min(T_vals), 0.01)), np.log(max(T_vals)), self.n_T)
        )

        # z[t_idx][k_idx] is what Plotly Surface expects (row = y = T)
        TT, KK = np.meshgrid(T_grid, k_grid, indexing="ij")  # (n_T, n_k)
        K_flat = KK.ravel()
        T_flat = TT.ravel()

        all_nids: list = []
        seen: set = set()
        for graph in self._graphs.values():
            for nid in graph.node_ids():
                if nid not in seen:
                    all_nids.append(nid)
                    seen.add(nid)

        train_ids = {id(o) for o in self._train_obs.observations}

        surface_data: dict = {}
        for nid in all_nids:
            label = _node_label(nid)

            def _spt(o):
                k_v   = float(o.data[0])
                T_v   = float(o.data[1])
                iv_v  = float(o.data[2])
                bid_n = o.data[3] if len(o.data) > 3 else None
                ask_n = o.data[4] if len(o.data) > 4 else None
                d = {
                    "k":    round(k_v,       5),
                    "T":    round(T_v,       5),
                    "iv":   round(iv_v * 100, 4),
                    "call": round(float(bs_call_from_iv(iv_v, k_v, T_v)) * 100, 5),
                }
                if bid_n is not None:
                    d["bid"] = round(float(bid_n) * 100, 5)
                if ask_n is not None:
                    d["ask"] = round(float(ask_n) * 100, 5)
                return d

            train_pts = [_spt(o) for o in self._train_obs.for_node(nid)]
            test_pts  = [
                _spt(o)
                for o in self._test_obs.for_node(nid)
                if id(o) not in train_ids
            ]

            model_surfaces: dict = {}
            call_surfaces:  dict = {}
            for mname, graph in self._graphs.items():
                if nid not in graph.nodes:
                    continue
                state = graph.get(nid)
                try:
                    iv_flat   = state.implied_vol(K_flat, T_flat)
                    call_flat = state.call_price(K_flat, T_flat) * 100.0
                    iv_2d   = (iv_flat * 100.0).reshape(self.n_T, self.n_k)
                    call_2d = call_flat.reshape(self.n_T, self.n_k)
                    model_surfaces[mname] = [
                        [round(float(v), 4) for v in row] for row in iv_2d
                    ]
                    call_surfaces[mname] = [
                        [round(float(v), 5) for v in row] for row in call_2d
                    ]
                except Exception:
                    pass

            surface_data[label] = {
                "k_grid": [round(float(v), 5) for v in k_grid],
                "T_grid": [round(float(v), 5) for v in T_grid],
                "obs": {"train": train_pts, "test": test_pts},
                "models":       model_surfaces,
                "call_models":  call_surfaces,
            }

        return {"surface_data": surface_data}

    # ------------------------------------------------------------------
    # HTML / JS hooks
    # ------------------------------------------------------------------

    def _extra_head(self) -> str:
        return """<style>
  .slice-row { display:flex; align-items:center; gap:12px; margin-top:14px; }
  .slice-row label { font-weight:500; white-space:nowrap; }
  .slice-row select { width:220px; }
</style>
<script>
let _surfaceNodeId = null;

// ---------- 3-D surface plot ----------
function renderSurfacePlot(nodeId) {
  const sd = D.surface_data && D.surface_data[nodeId];
  if (!sd) return;
  _surfaceNodeId = nodeId;

  const traces = [];

  if (sd.obs.train.length > 0) {
    traces.push({
      type: 'scatter3d', mode: 'markers', name: 'Train obs',
      x: sd.obs.train.map(o => o.k),
      y: sd.obs.train.map(o => o.T),
      z: sd.obs.train.map(o => o.iv),
      marker: {size: 2.5, color: '#888', opacity: 0.75},
    });
  }
  if (sd.obs.test.length > 0) {
    traces.push({
      type: 'scatter3d', mode: 'markers', name: 'Test obs',
      x: sd.obs.test.map(o => o.k),
      y: sd.obs.test.map(o => o.T),
      z: sd.obs.test.map(o => o.iv),
      marker: {size: 4, color: '#111', symbol: 'diamond', opacity: 1},
    });
  }

  const colorscales = [
    [[0,'#BBDEFB'],[1,'#0D47A1']],
    [[0,'#C8E6C9'],[1,'#1B5E20']],
    [[0,'#FFCCBC'],[1,'#BF360C']],
    [[0,'#E1BEE7'],[1,'#4A148C']],
    [[0,'#FFF9C4'],[1,'#F57F17']],
  ];

  D.model_names.forEach((m, i) => {
    const z = sd.models[m];
    if (!z) return;
    traces.push({
      type: 'surface',
      x: sd.k_grid, y: sd.T_grid, z: z,
      colorscale: colorscales[i % colorscales.length],
      opacity: 0.70,
      name: m,
      showscale: false,
      hovertemplate: 'k=%{x:.3f}  T=%{y:.3f}<br>IV=%{z:.2f}%<br>' + m + '<extra></extra>',
    });
  });

  Plotly.react('surface-3d-div', traces, {
    scene: {
      xaxis: {title: {text:'Log-moneyness', font:{size:11}}},
      yaxis: {title: {text:'Maturity (yr)',  font:{size:11}}},
      zaxis: {title: {text:'IV (%)',          font:{size:11}}},
      camera: {eye: {x:-1.6, y:-1.6, z:1.1}},
      aspectmode: 'manual',
      aspectratio: {x:1.2, y:1.0, z:0.7},
    },
    legend: {orientation:'h', y:-0.04, font:{size:11}},
    margin: {t:10, b:10, l:0, r:0},
    height: 520,
  }, {responsive:true});
}

// ---------- T-slice dropdown ----------
function populateTSelect(nodeId) {
  const sd = D.surface_data && D.surface_data[nodeId];
  if (!sd) return;
  const sel = document.getElementById('T-select');
  sel.innerHTML = '';
  sd.T_grid.forEach((T, i) => {
    const o = document.createElement('option');
    o.value = i;
    o.textContent = 'T = ' + T.toFixed(3) + ' yr';
    sel.appendChild(o);
  });
  // default to roughly the 6-month maturity
  const midIdx = sd.T_grid.reduce((best, T, i) =>
    Math.abs(T - 0.5) < Math.abs(sd.T_grid[best] - 0.5) ? i : best, 0);
  sel.value = midIdx;
  renderSmileSlice(nodeId, midIdx);
  renderCallSmileSlice(nodeId, midIdx);
}

// ---------- 2-D smile slice ----------
function renderSmileSlice(nodeId, tIdx) {
  const sd = D.surface_data && D.surface_data[nodeId];
  if (!sd) return;
  const T = sd.T_grid[tIdx];

  const traces = [];

  // Observations within ±30 % of T (by relative distance)
  const bw = T * 0.30;
  const trainNear = sd.obs.train.filter(o => Math.abs(o.T - T) <= bw);
  const testNear  = sd.obs.test.filter(o  => Math.abs(o.T - T) <= bw);

  if (trainNear.length > 0) {
    traces.push({
      x: trainNear.map(o => o.k), y: trainNear.map(o => o.iv),
      mode: 'markers', name: 'Train obs',
      marker: {symbol:'circle-open', size:7, color:'#555',
               line:{color:'#555', width:1.5}},
    });
  }
  if (testNear.length > 0) {
    traces.push({
      x: testNear.map(o => o.k), y: testNear.map(o => o.iv),
      mode: 'markers', name: 'Test obs',
      marker: {symbol:'diamond', size:8, color:'#111'},
    });
  }

  const COLORS = ['#2196F3','#E64A19','#388E3C','#7B1FA2','#F57C00'];
  D.model_names.forEach((m, i) => {
    const z = sd.models[m];
    if (!z) return;
    traces.push({
      x: sd.k_grid, y: z[tIdx],
      mode: 'lines', name: m,
      line: {color: COLORS[i % COLORS.length], width: 2},
    });
  });

  Plotly.react('smile-slice-div', traces, {
    xaxis: {title:'Log-moneyness', zeroline:true,
            zerolinecolor:'#ddd', zerolinewidth:1},
    yaxis: {title:'IV (%)'},
    title: {text:'Smile slice  T ≈ ' + T.toFixed(3) + ' yr', font:{size:13}},
    legend: {orientation:'h', y:-0.28, font:{size:11}},
    margin: {t:40, b:90, l:55, r:20},
    height: 320,
    plot_bgcolor: '#fafafa',
  }, {responsive:true, displayModeBar:false});
}

// ---------- Call price slice ----------
function renderCallSmileSlice(nodeId, tIdx) {
  const sd = D.surface_data && D.surface_data[nodeId];
  if (!sd) return;
  const T = sd.T_grid[tIdx];

  const traces = [];
  const bw = T * 0.30;

  const trainNear = sd.obs.train.filter(o => Math.abs(o.T - T) <= bw);
  const testNear  = sd.obs.test.filter(o  => Math.abs(o.T - T) <= bw);

  function withField(arr, f) { return arr.filter(o => o[f] !== undefined); }

  // Train bid / ask
  const trBid = withField(trainNear, 'bid');
  const trAsk = withField(trainNear, 'ask');
  if (trBid.length > 0) {
    traces.push({
      x: trBid.map(o => o.k), y: trBid.map(o => o.bid),
      mode: 'markers', name: 'Bid (train)',
      marker: {symbol:'triangle-down', size:7, color:'#1565C0', opacity:0.75},
    });
  }
  if (trAsk.length > 0) {
    traces.push({
      x: trAsk.map(o => o.k), y: trAsk.map(o => o.ask),
      mode: 'markers', name: 'Ask (train)',
      marker: {symbol:'triangle-up', size:7, color:'#B71C1C', opacity:0.75},
    });
  }

  // Test bid / ask
  const teBid = withField(testNear, 'bid');
  const teAsk = withField(testNear, 'ask');
  if (teBid.length > 0) {
    traces.push({
      x: teBid.map(o => o.k), y: teBid.map(o => o.bid),
      mode: 'markers', name: 'Bid (test)',
      marker: {symbol:'triangle-down-open', size:9, color:'#0D47A1',
               line:{color:'#0D47A1', width:1.5}},
    });
  }
  if (teAsk.length > 0) {
    traces.push({
      x: teAsk.map(o => o.k), y: teAsk.map(o => o.ask),
      mode: 'markers', name: 'Ask (test)',
      marker: {symbol:'triangle-up-open', size:9, color:'#B71C1C',
               line:{color:'#B71C1C', width:1.5}},
    });
  }

  const COLORS = ['#2196F3','#E64A19','#388E3C','#7B1FA2','#F57C00'];
  D.model_names.forEach((m, i) => {
    const z = sd.call_models && sd.call_models[m];
    if (!z) return;
    traces.push({
      x: sd.k_grid, y: z[tIdx],
      mode: 'lines', name: m,
      line: {color: COLORS[i % COLORS.length], width: 2},
    });
  });

  Plotly.react('call-slice-div', traces, {
    xaxis: {title:'Log-moneyness', zeroline:true,
            zerolinecolor:'#ddd', zerolinewidth:1},
    yaxis: {title:'Call Price (% of forward, discount=1)'},
    title: {text:'Call price slice  T ≈ ' + T.toFixed(3) + ' yr', font:{size:13}},
    legend: {orientation:'h', y:-0.28, font:{size:11}},
    margin: {t:40, b:90, l:65, r:20},
    height: 320,
    plot_bgcolor: '#fafafa',
  }, {responsive:true, displayModeBar:false});
}
</script>"""

    def _extra_node_html(self) -> str:
        return """
<div id="surface-3d-div" style="margin-top:16px;"></div>
<div class="slice-row">
  <label>Smile slice:</label>
  <select id="T-select"
          onchange="renderSmileSlice(_surfaceNodeId, +this.value); renderCallSmileSlice(_surfaceNodeId, +this.value)"></select>
</div>
<div id="smile-slice-div"></div>
<div id="call-slice-div" style="margin-top:14px;"></div>"""

    def _extra_init_js(self) -> str:
        return "renderSurfacePlot(nodeId); populateTSelect(nodeId);"
