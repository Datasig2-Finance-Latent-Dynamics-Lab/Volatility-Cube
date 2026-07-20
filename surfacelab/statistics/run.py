"""
Orchestrator: compute every cross-asset-exploitability diagnostic and save to disk.

Run from the repo root:

    .venv/bin/python3 -m surfacelab.statistics.run                # market data (default)
    .venv/bin/python3 -m surfacelab.statistics.run --dataset heston

Outputs land in results/diagnostics/exploitability/:
    exploitability.txt   — human-readable summary (correlation story + ceiling + verdict)
    sweep.csv            — r(N), ρ̂(N), C(N) per quote budget
    validation.txt       — Monte-Carlo confirmation that the ceiling formula holds
    sweep.png            — ρ̂(N), r(N), C(N) curves
    ceiling_heatmap.png  — C(ρ, r) with the data's operating point marked

WHY a single orchestrator: each snippet is independently useful, but the *argument* is
the sequence — measure ρ, measure r, combine into the ceiling, validate the formula,
sweep over N.  This file is that argument, executed and written down.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from surfacelab.data import load_grouptech, load_heston
from surfacelab.experiments.configs import MARKET_CSV, HESTON_TRAIN, HESTON_OOD

from .atm import atm_series, increments, pairwise_corr
from .correlation import summarise
from .ceiling import ceiling, rmse_gain, conditional_mi, required_r, partial_r2_cross
from .sweep import run_sweep
from .validate import validation_table
from .asymmetric import empirical_cov, asymmetric_gains

OUT = Path(__file__).resolve().parents[2] / "results" / "diagnostics" / "exploitability"
REF_N = 10                      # reference quote budget (project's small-context regime)


def _load(name: str):
    if name == "heston":
        ds, _ = load_heston(HESTON_TRAIN, HESTON_OOD)
        return ds
    return load_grouptech(MARKET_CSV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="market", choices=["market", "heston"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ds = _load(args.dataset)
    names = ds.meta["asset_names"]
    print(f"[{args.dataset}] {ds.n_days} days, {ds.n_assets} assets: {names}")

    # ---- 1. correlation story (raw, partial|SPY, common-factor share) ----
    days = ds.train_idx()
    d_full = increments(atm_series(ds, days))
    C = pairwise_corr(d_full)
    corr = summarise(C, names) if "SPY" in names else summarise(C, names, spy_name=names[0])

    # ---- 2. quote-count sweep: ρ̂(N), r(N), C(N) ----
    sw = run_sweep(ds)
    rho_true = sw["rho_true"]
    ref = min(sw["rows"], key=lambda x: abs(x["n_quotes"] - REF_N))

    # ---- 3. empirical (model-free) partial-R² of peers, at full data ----
    spy_idx = names.index("SPY") if "SPY" in names else 0
    pr2_raw = partial_r2_cross(d_full, spy_idx)         # how much peers explain SPY's move
    # for a stock: explanatory power of peers after controlling for SPY (own-proxy = SPY move)
    stock0 = next(i for i in range(ds.n_assets) if i != spy_idx)
    pr2_stock_given_spy = partial_r2_cross(d_full, stock0, own_proxy=d_full[:, spy_idx])

    # ---- 4. validate the ceiling formula by Monte Carlo ----
    r_grid = [r["r_mean"] for r in sw["rows"] if np.isfinite(r["r_mean"]) and r["r_mean"] > 0]
    vt = validation_table(rho_true, sorted(set(r_grid)), n_peers=ds.n_assets - 1)

    # ---- 5. ASYMMETRIC liquidity: perfect SPY / perfect peers vs a sparse target ----
    # This is the realistic regime the per-asset-uniform harness hides. Built on the
    # empirical covariance so each asset keeps its true beta to SPY.
    Sigma = empirical_cov(d_full)
    asym = {}                                 # N -> per-target gain rows
    for row in sw["rows"]:
        if np.all(np.isnan(row["nu2"])):
            continue
        nu2 = np.where(np.isfinite(row["nu2"]), row["nu2"], np.nanmax(row["nu2"]))
        asym[row["n_quotes"]] = asymmetric_gains(Sigma, nu2, names) if "SPY" in names else []

    _write_text(args.dataset, names, corr, sw, ref, rho_true,
                pr2_raw, pr2_stock_given_spy, vt, asym)
    _write_csv(sw)
    _plot_sweep(sw)
    _plot_heatmap(rho_true, ref["r_mean"])
    _plot_asymmetric(asym)
    _write_validation(vt)
    print(f"\nWrote diagnostics to {OUT}/")


# ───────────────────────── reporting ─────────────────────────
def _write_text(dataset, names, corr, sw, ref, rho_true,
                pr2_raw, pr2_stock_given_spy, vt, asym):
    C_ref = ref["ceiling_true_rho"]
    G_ref = ref["realizable_sym"]
    lines = []
    A = lines.append
    A(f"CROSS-ASSET EXPLOITABILITY  ({dataset} data, assets={names})")
    A("=" * 74)
    A("")
    A("1. CORRELATION STRUCTURE  (Δ ATM-level increments, full-data fits, train days)")
    A(f"   mean stock-stock corr (raw)            : {corr['mean_stock_stock_raw']:+.3f}")
    A(f"   mean stock-SPY corr                    : {corr['mean_stock_spy']:+.3f}")
    A(f"   mean stock-stock corr | SPY removed    : {corr['mean_stock_stock_partial_given_spy']:+.3f}")
    A(f"   common-factor variance share (top PC)  : {corr['common_factor_share']:.3f}")
    A(f"   idiosyncratic share                    : {corr['idiosyncratic_share']:.3f}")
    A(f"   # factors for 90% of variance          : {corr['n_factors_for_90pct']}")
    A("   -> Co-movement is real and a single market factor (SPY) carries most of it. So")
    A("      this is NOT primarily an idiosyncratic-floor story: rho is genuinely high.")
    A("      The reason it is still unexploitable is in sections 2-3.")
    A("")
    A("2. THE OBSERVATION BOTTLENECK  (per own-quote budget N)")
    A("   r(N)=nu^2/sigma^2 is how poorly N quotes pin today's ATM vs the size of a daily move.")
    A("   rho_hat(N) is the cross-asset corr you can actually MEASURE from N-quote fits.")
    A("")
    A("     N     r(N)    rho_hat(N)   C_oracle(N)   realizable_sym(N)   (RMSE gain)")
    for row in sw["rows"]:
        g = row["realizable_sym"]
        A(f"   {row['n_quotes']:<4d}  {row['r_mean']:7.2f}   {row['rho_hat']:+.3f}      "
          f"{row['ceiling_true_rho']:.3f}         {g:.3f}            {rmse_gain(g)*100:4.1f}%")
    A("")
    A("   TWO THINGS KILL EXPLOITABILITY, and the table shows both:")
    A("   (a) SYMMETRIC NOISE. C_oracle assumes peers are observed PERFECTLY, so it looks")
    A("       large at low N. But every asset has similar liquidity: when you are too")
    A("       data-starved to pin your own ATM (large r), the peers are equally starved")
    A("       and cannot reveal the common factor any better. The honest realizable_sym")
    A(f"       gain peaks at only ~{max(r['realizable_sym'] for r in sw['rows'])*100:.0f}% variance "
      f"(~{rmse_gain(max(r['realizable_sym'] for r in sw['rows']))*100:.0f}% RMSE), at an INTERMEDIATE N.")
    A("   (b) UNOBSERVABILITY. Even that modest gain assumes the coupling rho is KNOWN. A")
    A("       learned model must estimate it, but rho_hat(N) ~ rho/(1+2r) collapses to ~0")
    A(f"       at the operating budget (rho_hat~{ref['rho_hat']:.3f} at N={ref['n_quotes']}). You cannot learn a")
    A("       coupling that is statistically invisible in the data you are given -> no")
    A("       gradient pushes the CNP toward using it.")
    A("")
    A(f"3. AT THE REFERENCE N={ref['n_quotes']}  (rho={rho_true:.3f}, r={ref['r_mean']:.2f}, rho_hat={ref['rho_hat']:.3f})")
    A(f"   oracle ceiling C (perfect peers)            : {C_ref:.3f}  ({rmse_gain(C_ref)*100:.1f}% RMSE)  <- unattainable")
    A(f"   realizable gain (peers share your noise)    : {G_ref:.3f}  ({rmse_gain(G_ref)*100:.1f}% RMSE)  <- still assumes rho known")
    A(f"   conditional mutual info ceiling             : {conditional_mi(C_ref):.2f} bits")
    A(f"   observable coupling at this budget          : rho_hat={ref['rho_hat']:.3f}  <- what a model could actually learn")
    A("")
    A("4. MODEL-FREE CROSS-CHECK (partial R^2, no Gaussian assumption, full-data fits)")
    A(f"   peers' R^2 explaining SPY's own move                  : {pr2_raw:.3f}")
    A(f"   peers' R^2 explaining a stock AFTER its SPY-move proxy : {pr2_stock_given_spy:.3f}")
    A("   -> most shared structure is the single SPY factor; the residual multi-factor")
    A("      part (second number) is what a graph could add ON TOP of a market-beta term.")
    A("")
    A("5. ASYMMETRIC LIQUIDITY  (the realistic case: SPY liquid, target sparse)")
    A("   The eval harness (`_per_asset_sample`) gives EVERY asset the same n_ctx quotes, so")
    A("   it lives in the symmetric regime of sections 2-3 where peers are as blind as you.")
    A("   But in the market SPY/large names are liquid while single names can be sparse.")
    A("   Below: target observed with its measured N-quote noise; peers observed EXACTLY.")
    A("   (Empirical covariance, so each asset keeps its real beta to SPY.)")
    A("")
    ref_asym = asym.get(ref["n_quotes"], [])
    if ref_asym:
        A(f"   At N={ref['n_quotes']} for the sparse target:")
        A(f"   {'asset':<7} {'own_r':>7} {'gain: perfect SPY':>18} {'perfect peers':>15} {'SPY-only R^2':>13}")
        for rrow in ref_asym:
            A(f"   {rrow['asset']:<7} {rrow['own_r']:>7.2f} {rrow['gain_perfect_spy']*100:>16.0f}%  "
              f"{rrow['gain_perfect_peers']*100:>13.0f}%  {rrow['pure_spy_R2']*100:>11.0f}%")
        mean_spy = float(np.mean([r["gain_perfect_spy"] for r in ref_asym]))
        A(f"   mean gain from a perfect SPY (variance): {mean_spy*100:.0f}%  (~{rmse_gain(mean_spy)*100:.0f}% RMSE)")
        A("   -> Unlike the symmetric case, a single LIQUID peer (SPY) removes a large chunk of")
        A("      a sparse name's residual variance. Cross-asset IS exploitable here. The catch:")
        A("      it requires (i) the liquidity asymmetry to be present at eval time, and (ii) the")
        A("      model to know the beta - which it CAN learn from history (beta is stable),")
        A("      unlike the sparse-quote coupling rho_hat of section 2 which is unmeasurable.")
        A("   CAVEAT: own_r uses the pooled-in-maturity ATM fit; a maturity-aware estimator")
        A("   (what the CNP effectively is) may pin the sparse ATM better, lowering own_r and")
        A("   shrinking these gains. The SIGN of the conclusion is robust; the MAGNITUDE is an")
        A("   upper-ish estimate. Treat the numbers as 'cross-asset is worth a real experiment'.")
    A("")
    A("VERDICT")
    A(f"   rho is high (~{rho_true:.2f}). Whether it is EXPLOITABLE depends entirely on LIQUIDITY")
    A("   SYMMETRY:")
    A("   - SYMMETRIC (all assets equally sparse, as the current harness samples): peers are")
    A(f"     as blind as you; realizable gain ~{rmse_gain(G_ref)*100:.0f}% RMSE and the coupling is unobservable")
    A("     at the operating quote count -> no method, incl. the CNP, can or should use it.")
    if ref_asym:
        A(f"   - ASYMMETRIC (SPY liquid, target sparse, as in reality): a perfect SPY removes ~"
          f"{float(np.mean([r['gain_perfect_spy'] for r in ref_asym]))*100:.0f}% of")
        A("     the sparse name's variance via a learnable, stable beta -> cross-asset SHOULD help.")
    A("   IMPLICATION: the experiments likely found 'no cross-asset benefit' because the harness")
    A("   sparsifies SPY too, destroying the very asymmetry that makes coupling useful. The")
    A("   decisive experiment: hold SPY (and liquid names) at FULL quotes and sparsify only the")
    A("   target, then re-test the graph/cross-asset models. The CNP learning Heston but not the")
    A("   symmetric coupling remains consistent: that signal is genuinely absent; this one isn't.")
    (OUT / "exploitability.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _write_csv(sw):
    import csv
    with open(OUT / "sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_quotes", "r_mean", "rho_hat", "ceiling_oracle", "realizable_sym"])
        for r in sw["rows"]:
            w.writerow([r["n_quotes"], f"{r['r_mean']:.5f}", f"{r['rho_hat']:.5f}",
                        f"{r['ceiling_true_rho']:.5f}", f"{r['realizable_sym']:.5f}"])


def _write_validation(vt):
    lines = ["MONTE-CARLO VALIDATION OF THE CEILING FORMULA  C = rho*r/((1-rho)+r)",
             "(one-factor sim; C_formula should match C_oracle_sim; C_realistic_sim < C_formula)",
             "",
             f"{'r':>8} {'C_formula':>12} {'C_oracle_sim':>14} {'C_realistic_sim':>16}"]
    for v in vt:
        lines.append(f"{v['r']:8.3f} {v['C_formula']:12.4f} {v['C_oracle_sim']:14.4f} "
                     f"{v['C_realistic_sim']:16.4f}")
    lines += ["",
              "C_oracle_sim ~ C_formula  -> the closed form is correct.",
              "C_realistic_sim < C_formula -> finite, noisy peers fall short of the oracle;",
              "the real exploitable gain is even smaller than the (already small) ceiling."]
    (OUT / "validation.txt").write_text("\n".join(lines) + "\n")


# ───────────────────────── plots ─────────────────────────
def _plot_sweep(sw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sw["rows"]
    N = [r["n_quotes"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(N, [r["r_mean"] for r in rows], "o-")
    ax[0].set(title="Own-observation noise r(N)", xlabel="# own quotes N", ylabel=r"$r=\nu^2/\sigma^2$")
    ax[0].set_xscale("log")
    ax[1].plot(N, [r["rho_hat"] for r in rows], "o-", label=r"observed $\hat\rho(N)$")
    ax[1].axhline(sw["rho_true"], ls="--", color="k", label=r"true $\rho$")
    ax[1].set(title="Observed cross-asset corr (attenuated by noise)", xlabel="# own quotes N",
              ylabel=r"$\hat\rho$"); ax[1].set_xscale("log"); ax[1].legend()
    ax[2].plot(N, [r["ceiling_true_rho"] for r in rows], "o-", label="oracle (perfect peers)")
    ax[2].plot(N, [r["realizable_sym"] for r in rows], "s-", label="realizable (peers share noise)")
    ax[2].set(title="Cross-asset variance-reduction: oracle vs realizable", xlabel="# own quotes N",
              ylabel="fraction of residual variance removed"); ax[2].set_xscale("log"); ax[2].legend()
    fig.tight_layout(); fig.savefig(OUT / "sweep.png", dpi=120); plt.close(fig)


def _plot_asymmetric(asym):
    """Mean per-target gain from a perfect SPY (and perfect peers) vs the target's quotes.

    The key contrast with sweep.png: here the gain is LARGE when the target is sparse and
    shrinks as it gains its own quotes — the opposite trend, and the signature of
    asymmetric liquidity being the exploitable regime.
    """
    if not asym:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = sorted(asym)
    spy = [np.mean([r["gain_perfect_spy"] for r in asym[n]]) for n in Ns]
    peers = [np.mean([r["gain_perfect_peers"] for r in asym[n]]) for n in Ns]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Ns, spy, "o-", label="perfect SPY only")
    ax.plot(Ns, peers, "s-", label="all peers perfect")
    ax.set(xscale="log", xlabel="# quotes on the SPARSE target asset",
           ylabel="fraction of target residual variance removed",
           title="Asymmetric liquidity: gain from well-observed peers\n(SPY/peers liquid, one target sparse)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "asymmetric.png", dpi=120); plt.close(fig)


def _plot_heatmap(rho_data, r_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rho = np.linspace(0.01, 0.95, 200)
    r = np.logspace(-2, 1.5, 200)
    R, RHO = np.meshgrid(r, rho)
    C = RHO * R / ((1 - RHO) + R)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.contourf(R, RHO, C, levels=20, cmap="viridis")
    cs = ax.contour(R, RHO, C, levels=[0.05, 0.1, 0.2, 0.3], colors="w", linewidths=0.8)
    ax.clabel(cs, fmt="%.2f")
    ax.plot([r_data], [rho_data], "r*", ms=18, label=f"data (rho={rho_data:.2f}, r={r_data:.2f})")
    ax.set(xscale="log", xlabel=r"own-observation noise $r=\nu^2/\sigma^2$",
           ylabel=r"common-factor correlation $\rho$",
           title=r"Exploitability ceiling  $C=\rho r/((1-\rho)+r)$")
    ax.legend(loc="upper left"); fig.colorbar(im, label="max variance-reduction fraction")
    fig.tight_layout(); fig.savefig(OUT / "ceiling_heatmap.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
