"""
Heston multi-asset data generator.

All four Heston parameters evolve as correlated OU processes;
surfaces are priced via the Lewis Fourier formula.

It also generates a smaller dataset Out Of Distribution (OOD)
to check wether the transformer has learned the underlying structure
or the specific Heston path.

Run from the repo root:
    python -m surfacelab.data.generate_heston                  # default config
    python -m surfacelab.data.generate_heston --n_days 500     # quick test
    python -m surfacelab.data.generate_heston --out_dir /tmp   # custom output dir

Writes heston_multiasset_training.npz and heston_multiasset_ood_test.npz, the two files
`surfacelab.data.heston.load_heston` reads (see configs.HESTON_TRAIN / HESTON_OOD).
"""
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class HestonDGPConfig: # Heston Data Generating Process Config
    # Scale
    n_assets:   int   = 5
    n_days:     int   = 5000
    n_days_ood: int   = 200
    seed:       int   = 42
    seed_ood:   int   = 999

    # Market conventions
    s0: float = 100.0
    r:  float = 0.05

    # Observation grid
    maturity_bins: list = field(default_factory=lambda:
        [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    lm_std:  float = 0.25
    lm_clip: float = 0.50
    t_mean:  float = 0.50   # maturity sampling weight centre
    t_std:   float = 0.80

    # Data layout
    n_ctx_pool: int = 400
    n_tgt_pool: int = 400

    # Train / val split (chronological)
    n_train: int = 4500

    # OU parameters — theta (long-run variance, natural-space OU)
    theta_bar:  list = field(default_factory=lambda: [0.040, 0.060, 0.030, 0.080, 0.050])
    kappa_th:   list = field(default_factory=lambda: [0.050, 0.040, 0.060, 0.030, 0.050])
    sigma_th:   list = field(default_factory=lambda: [0.006, 0.008, 0.005, 0.010, 0.007])
    theta_min:  float = 0.005
    theta_max:  float = 0.500

    # OU parameters — kappa (MR speed, log-space OU)
    kappa_bar:  list = field(default_factory=lambda: [2.0, 1.5, 3.0, 1.0, 4.0])
    kappa_k:    list = field(default_factory=lambda: [0.04]*5)
    sigma_k:    list = field(default_factory=lambda: [0.08]*5)
    kappa_min:  float = 0.3
    kappa_max:  float = 10.0

    # OU parameters — xi (vol-of-vol, log-space OU)
    xi_bar:     list = field(default_factory=lambda: [0.30, 0.50, 0.20, 0.55, 0.40])
    kappa_xi:   list = field(default_factory=lambda: [0.04]*5)
    sigma_xi:   list = field(default_factory=lambda: [0.10]*5)
    xi_min:     float = 0.05
    xi_max:     float = 1.50

    # OU parameters — rho (spot-var corr, natural-space OU)
    rho_bar:    list = field(default_factory=lambda: [-0.70, -0.50, -0.80, -0.35, -0.60])
    kappa_rho:  list = field(default_factory=lambda: [0.04]*5)
    sigma_rho:  list = field(default_factory=lambda: [0.025]*5)
    rho_min:    float = -0.95
    rho_max:    float = -0.05

    # Cross-asset correlation matrix (shared by all 4 OU processes)
    # This is for simplicity, there would be no problem using different 
    # correlation matrices.
    theta_corr: list = field(default_factory=lambda: [
        [1.00, 0.72, 0.63, 0.67, 0.76],
        [0.72, 1.00, 0.58, 0.62, 0.70],
        [0.63, 0.58, 1.00, 0.55, 0.62],
        [0.67, 0.62, 0.55, 1.00, 0.65],
        [0.76, 0.70, 0.62, 0.65, 1.00],
    ])

    # Fourier pricer
    u_max:  float = 200.0
    n_gl:   int   = 96

    out_dir: str = str(Path(__file__).resolve().parents[2] / "data" / "synthetic")


# ── OU simulation ─────────────────────────────────────────────────────────────

def simulate_ou_natural(
    n_days: int,
    x_bar: np.ndarray,
    kappa: np.ndarray,
    sigma: np.ndarray,
    chol: np.ndarray,
    x_min: float,
    x_max: float,
    seed: int,
) -> np.ndarray:
    """Standard OU in natural space with cross-asset correlation via Cholesky."""
    N   = len(x_bar)
    rng = np.random.default_rng(seed)
    X   = np.zeros((n_days, N))
    X[0] = x_bar.copy()
    for d in range(1, n_days):
        eps   = chol @ rng.standard_normal(N)
        drift = kappa * (x_bar - X[d - 1])
        X[d]  = np.clip(X[d - 1] + drift + sigma * eps, x_min, x_max)
    return X


def simulate_ou_logspace(
    n_days: int,
    x_bar: np.ndarray,
    kappa: np.ndarray,
    sigma: np.ndarray,
    chol: np.ndarray,
    x_min: float,
    x_max: float,
    seed: int,
) -> np.ndarray:
    """Log-space OU: OU on log(x), then exponentiate and clip (guarantees positivity)."""
    N       = len(x_bar)
    rng     = np.random.default_rng(seed)
    log_bar = np.log(x_bar)
    logX    = np.zeros((n_days, N))
    logX[0] = log_bar.copy()
    for d in range(1, n_days):
        eps    = chol @ rng.standard_normal(N)
        drift  = kappa * (log_bar - logX[d - 1])
        logX[d] = logX[d - 1] + drift + sigma * eps
    return np.clip(np.exp(logX), x_min, x_max)


# ── Heston Fourier pricer (Lewis formula) ────────────────────────────────────

def _build_gl_nodes(u_max: float, n_gl: int):
    xi_gl, w_gl = leggauss(n_gl)
    u_nodes = (xi_gl + 1.0) / 2.0 * u_max
    w_nodes = w_gl  / 2.0 * u_max
    return u_nodes, w_nodes


def _heston_cf_lewis(u_real, T, v0, kappa, theta, xi, rho):
    u = u_real - 0.5j
    a = kappa - rho * xi * 1j * u
    d = np.sqrt(a ** 2 + xi ** 2 * (u ** 2 + 1j * u))
    g = (a - d) / (a + d)
    h = np.exp(-d * T)
    C = (kappa * theta / xi ** 2
         * ((a - d) * T - 2.0 * np.log((1.0 - g * h) / (1.0 - g))))
    D = (a - d) / xi ** 2 * (1.0 - h) / (1.0 - g * h)
    return np.exp(C + D * v0)


def heston_call_lewis(S0, r, K_arr, T, v0, kappa, theta, xi, rho, u_nodes, w_nodes):
    F    = S0 * np.exp(r * T)
    k    = np.log(K_arr / F)
    disc = np.exp(-r * T)
    cf   = _heston_cf_lewis(u_nodes, T, v0, kappa, theta, xi, rho)
    base = cf / (u_nodes ** 2 + 0.25)
    M    = np.real(np.exp(-1j * np.outer(k, u_nodes)) * base[None, :])
    I    = M @ w_nodes
    calls = S0 - np.sqrt(K_arr * F) * disc / np.pi * I
    return np.clip(calls, np.maximum(S0 - K_arr * disc, 0.0), S0)


# ── Implied vol solver ────────────────────────────────────────────────────────

def implied_vol_vec(prices, K, T, S0, r, n_iter=80):
    intrinsic = np.maximum(S0 - K * np.exp(-r * T), 0.0)
    valid     = prices > intrinsic + 1e-10
    iv        = np.full_like(prices, 0.30)
    sqT       = np.sqrt(np.where(T > 0, T, 1e-10))

    for _ in range(n_iter):
        d1   = (np.log(S0 / K) + (r + 0.5 * iv ** 2) * T) / (iv * sqT + 1e-14)
        bs   = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - iv * sqT)
        vega = S0 * norm.pdf(d1) * sqT
        mask = valid & (vega > 1e-14)
        iv   = np.where(mask, iv - (bs - prices) / vega, iv)
        iv   = np.where(valid, iv.clip(1e-8, 10.0), iv)

    p_final = S0 * norm.cdf(
        (np.log(S0 / K) + (r + 0.5 * iv ** 2) * T) / (iv * sqT + 1e-14)
    ) - K * np.exp(-r * T) * norm.cdf(
        (np.log(S0 / K) + (r + 0.5 * iv ** 2) * T) / (iv * sqT + 1e-14) - iv * sqT
    )
    bad = ~valid | (np.abs(p_final - prices) > 1e-5)
    return np.where(bad, np.nan, iv).astype(np.float32)


# ── Point sampling ────────────────────────────────────────────────────────────
# Samples points in the surface, maturities at fixed dsicrete values,
# logmoneyness in continuum.

def _build_t_weights(maturity_bins, t_mean, t_std):
    bins  = np.asarray(maturity_bins)
    raw_w = np.exp(-0.5 * ((bins - t_mean) / t_std) ** 2)
    return (raw_w / raw_w.sum()).astype(np.float64)


def sample_surface_points(rng, n_points, maturity_bins, t_weights, n_assets, lm_std, lm_clip):
    lm  = rng.normal(0.0, lm_std, n_points).clip(-lm_clip, lm_clip).astype(np.float32)
    t_i = rng.choice(len(maturity_bins), size=n_points, p=t_weights)
    T   = np.asarray(maturity_bins)[t_i].astype(np.float32)
    aid = rng.integers(0, n_assets, n_points).astype(np.int8)
    return lm, T, aid


def compute_ivs(lm_arr, T_arr, aid_arr, v0_d, kappa_d, theta_d, xi_d, rho_d,
                S0, r, n_assets, u_nodes, w_nodes):
    n      = len(lm_arr)
    K_arr  = S0 * np.exp(lm_arr.astype(np.float64))
    prices = np.zeros(n, dtype=np.float64)
    for a in range(n_assets):
        mask_a = aid_arr == a
        if not mask_a.any():
            continue
        for T in np.unique(T_arr[mask_a]):
            mask = mask_a & (T_arr == T)
            prices[mask] = heston_call_lewis(
                S0=S0, r=r, K_arr=K_arr[mask], T=float(T),
                v0=float(v0_d[a]), kappa=float(kappa_d[a]),
                theta=float(theta_d[a]), xi=float(xi_d[a]), rho=float(rho_d[a]),
                u_nodes=u_nodes, w_nodes=w_nodes,
            )
    return implied_vol_vec(prices, K_arr, T_arr.astype(np.float64), S0, r)


# ── Main generation loop ──────────────────────────────────────────────────────

def _run_loop(n_days, theta_path, kappa_path, xi_path, rho_path,
              n_total, n_assets, maturity_bins, t_weights, S0, r,
              lm_std, lm_clip, u_nodes, w_nodes, rng, desc):
    all_lm     = np.zeros((n_days, n_total), dtype=np.float32)
    all_T      = np.zeros((n_days, n_total), dtype=np.float32)
    all_aid    = np.zeros((n_days, n_total), dtype=np.int8)
    all_iv     = np.full((n_days, n_total), np.nan, dtype=np.float32)
    all_params = np.zeros((n_days, n_assets, 5), dtype=np.float32)
    n_dropped  = 0

    for d in tqdm(range(n_days), desc=desc):
        theta_d = theta_path[d]
        kappa_d = kappa_path[d]
        xi_d    = xi_path[d]
        rho_d   = rho_path[d]
        v0_d    = theta_d

        lm_d, T_d, aid_d = sample_surface_points(
            rng, n_total, maturity_bins, t_weights, n_assets, lm_std, lm_clip)
        iv_d = compute_ivs(lm_d, T_d, aid_d, v0_d, kappa_d, theta_d, xi_d, rho_d,
                           S0, r, n_assets, u_nodes, w_nodes)

        nan_frac = np.isnan(iv_d).mean()
        if nan_frac > 0.05:
            n_dropped += 1
            if d > 0:
                all_lm[d]  = all_lm[d - 1]
                all_T[d]   = all_T[d - 1]
                all_aid[d] = all_aid[d - 1]
                all_iv[d]  = all_iv[d - 1]
            continue

        bad = np.isnan(iv_d)
        if bad.any():
            iv_d[bad] = float(np.nanmean(iv_d))

        all_lm[d]     = lm_d
        all_T[d]      = T_d
        all_aid[d]    = aid_d
        all_iv[d]     = iv_d
        all_params[d] = np.column_stack([v0_d, kappa_d, theta_d, xi_d, rho_d])

    if n_dropped:
        print(f"  Dropped {n_dropped}/{n_days} days (>{5}% NaN IVs)")
    return all_lm, all_T, all_aid, all_iv, all_params


# ── Public entry point ────────────────────────────────────────────────────────

def generate(cfg: HestonDGPConfig | None = None) -> dict[str, str]:
    """
    Generate training and OOD datasets according to cfg.
    Returns {'train': path, 'ood': path}.
    """
    if cfg is None:
        cfg = HestonDGPConfig()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_assets   = cfg.n_assets
    mat_bins   = np.asarray(cfg.maturity_bins)
    t_weights  = _build_t_weights(mat_bins, cfg.t_mean, cfg.t_std)
    n_total    = cfg.n_ctx_pool + cfg.n_tgt_pool
    u_nodes, w_nodes = _build_gl_nodes(cfg.u_max, cfg.n_gl)

    theta_corr = np.asarray(cfg.theta_corr)
    eigvals    = np.linalg.eigvalsh(theta_corr)
    assert eigvals.min() > 0, "theta_corr is not positive definite"
    chol = np.linalg.cholesky(theta_corr)

    theta_bar  = np.asarray(cfg.theta_bar)
    kappa_bar  = np.asarray(cfg.kappa_bar)
    xi_bar     = np.asarray(cfg.xi_bar)
    rho_bar    = np.asarray(cfg.rho_bar)

    kappa_th   = np.asarray(cfg.kappa_th);   sigma_th  = np.asarray(cfg.sigma_th)
    kappa_k    = np.asarray(cfg.kappa_k);    sigma_k   = np.asarray(cfg.sigma_k)
    kappa_xi   = np.asarray(cfg.kappa_xi);   sigma_xi  = np.asarray(cfg.sigma_xi)
    kappa_rho  = np.asarray(cfg.kappa_rho);  sigma_rho = np.asarray(cfg.sigma_rho)

    # ── Training data ─────────────────────────────────────────────────────────
    print(f"Simulating OU paths ({cfg.n_days} days, {n_assets} assets)...")
    theta_path = simulate_ou_natural(cfg.n_days, theta_bar, kappa_th, sigma_th, chol,
                                      cfg.theta_min, cfg.theta_max, cfg.seed)
    kappa_path = simulate_ou_logspace(cfg.n_days, kappa_bar, kappa_k,  sigma_k,  chol,
                                       cfg.kappa_min, cfg.kappa_max, cfg.seed + 1)
    xi_path    = simulate_ou_logspace(cfg.n_days, xi_bar,    kappa_xi, sigma_xi, chol,
                                       cfg.xi_min, cfg.xi_max, cfg.seed + 2)
    rho_path   = simulate_ou_natural(cfg.n_days, rho_bar,   kappa_rho, sigma_rho, chol,
                                      cfg.rho_min, cfg.rho_max, cfg.seed + 3)
    

    t0  = time.time()
    rng = np.random.default_rng(cfg.seed + 10)
    lm, T, aid, iv, params = _run_loop(
        cfg.n_days, theta_path, kappa_path, xi_path, rho_path,
        n_total, n_assets, mat_bins, t_weights,
        cfg.s0, cfg.r, cfg.lm_std, cfg.lm_clip,
        u_nodes, w_nodes, rng, desc="Training days",
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Fall back to a 90/10 chronological split on short runs, otherwise a small --n_days
    # lands entirely in train and the loader gets no validation days.
    n_train = cfg.n_train if cfg.n_train < cfg.n_days else int(0.9 * cfg.n_days)
    split   = np.zeros(cfg.n_days, dtype=np.int8)
    split[n_train:] = 1

    train_path = out_dir / "heston_multiasset_training.npz"
    np.savez_compressed(
        train_path,
        lm=lm, maturity=T, asset_id=aid, iv=iv, params=params,
        split=split,
        ctx_max=np.int32(cfg.n_ctx_pool),
        theta_path=theta_path, kappa_path=kappa_path,
        xi_path=xi_path, rho_path=rho_path,
        theta_corr=theta_corr,
    )
    size_mb = train_path.stat().st_size / 1e6
    print(f"  Saved {train_path}  ({size_mb:.1f} MB)  "
          f"train={n_train}  val={cfg.n_days-n_train}")

    # ── OOD data ──────────────────────────────────────────────────────────────
    print(f"\nSimulating OOD OU paths ({cfg.n_days_ood} days, seed={cfg.seed_ood})...")
    theta_ood = simulate_ou_natural(cfg.n_days_ood, theta_bar, kappa_th, sigma_th, chol,
                                     cfg.theta_min, cfg.theta_max, cfg.seed_ood)
    kappa_ood = simulate_ou_logspace(cfg.n_days_ood, kappa_bar, kappa_k,  sigma_k,  chol,
                                      cfg.kappa_min, cfg.kappa_max, cfg.seed_ood + 1)
    xi_ood    = simulate_ou_logspace(cfg.n_days_ood, xi_bar,    kappa_xi, sigma_xi, chol,
                                      cfg.xi_min, cfg.xi_max, cfg.seed_ood + 2)
    rho_ood   = simulate_ou_natural(cfg.n_days_ood, rho_bar,   kappa_rho, sigma_rho, chol,
                                     cfg.rho_min, cfg.rho_max, cfg.seed_ood + 3)

    t0      = time.time()
    rng_ood = np.random.default_rng(cfg.seed_ood + 10)
    lm_ood, T_ood, aid_ood, iv_ood, params_ood = _run_loop(
        cfg.n_days_ood, theta_ood, kappa_ood, xi_ood, rho_ood,
        n_total, n_assets, mat_bins, t_weights,
        cfg.s0, cfg.r, cfg.lm_std, cfg.lm_clip,
        u_nodes, w_nodes, rng_ood, desc="OOD days",
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    ood_path = out_dir / "heston_multiasset_ood_test.npz"
    np.savez_compressed(
        ood_path,
        lm=lm_ood, maturity=T_ood, asset_id=aid_ood, iv=iv_ood, params=params_ood,
        ctx_max=np.int32(cfg.n_ctx_pool),
        theta_path=theta_ood, kappa_path=kappa_ood,
        xi_path=xi_ood, rho_path=rho_ood,
        theta_corr=theta_corr,
    )
    size_ood = ood_path.stat().st_size / 1e6
    print(f"  Saved {ood_path}  ({size_ood:.1f} MB)")

    return {"train": str(train_path), "ood": str(ood_path)}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Heston multi-asset surface data")
    parser.add_argument("--n_days",     type=int,   default=None)
    parser.add_argument("--n_days_ood", type=int,   default=None)
    parser.add_argument("--n_assets",  type=int,   default=None)
    parser.add_argument("--seed",      type=int,   default=None)
    parser.add_argument("--out_dir",   type=str,   default=None)
    args = parser.parse_args()

    cfg = HestonDGPConfig()
    if args.n_days:     cfg.n_days     = args.n_days
    if args.n_days_ood: cfg.n_days_ood = args.n_days_ood
    if args.n_assets:   cfg.n_assets   = args.n_assets
    if args.seed:       cfg.seed       = args.seed
    if args.out_dir:    cfg.out_dir    = args.out_dir

    generate(cfg)
