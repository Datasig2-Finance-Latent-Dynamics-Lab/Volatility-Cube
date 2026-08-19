"""
Unified experiment runner.

    .venv/bin/python -m surfacelab.experiments.run --config market_all_methods
    .venv/bin/python -m surfacelab.experiments.run --config market_all_methods_sequential
    .venv/bin/python -m surfacelab.experiments.run --config market_all_methods --quick
    .venv/bin/python -m surfacelab.experiments.run --config market_exclude_aapl
    .venv/bin/python -m surfacelab.experiments.run --config market_exclude_aapl_sequential

Builds the configured models, trains/loads each, runs the independent or sequential
harness, writes records.csv + summary.csv under results/surfacelab/<config>/, and prints
the summary table.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from surfacelab.data import compute_bspline_prior
from surfacelab.models import registry
from surfacelab.eval import (run, run_sequential, run_exclude, run_sequential_exclude,
                             run_target_asymmetric, run_models)
from surfacelab.experiments.configs import (get_experiment, make_experiment,
                                            EXPERIMENTS, LOADERS, CNPTrainConfig)

warnings.filterwarnings("ignore")


def _finish(rec, exp, args, models, dataset, mode) -> None:
    """Save records, print the summary table, and (unless --no-report) build the HTML report.
    Shared by the composable and legacy harness paths."""
    out_dir = args.out or exp.out_dir
    rec.save(out_dir)
    rec.print_table(exp.name)
    if not args.no_report:
        from surfacelab.analytics.report import build_report
        path = build_report(rec, models, dataset, out_dir, mode=mode, title=exp.name,
                            target_asset=getattr(exp, "asymmetric_target", None)
                            or getattr(exp, "exclude_asset", None))
        print(f"Report: {path}")
    print(f"\nWrote {out_dir}/records.csv and summary.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="surfacelab experiment runner")
    ap.add_argument("--config", required=True,
                    help="experiment name: a key in configs.py, or ANY name (e.g. a generated "
                         "'iv_surface_xxx') when --dataset and --models are also given")
    ap.add_argument("--dataset", choices=sorted(LOADERS), default=None,
                    help="ad-hoc run: dataset loader (heston | market | market_thesis)")
    ap.add_argument("--models", default=None,
                    help="ad-hoc run: comma-separated registry names, e.g. bspline,prior,ssvi")
    ap.add_argument("--mode", choices=["independent", "sequential"], default="independent",
                    help="ad-hoc run: harness mode (default independent)")
    ap.add_argument("--no-prior", action="store_true",
                    help="ad-hoc run: skip the B-spline prior computation")
    ap.add_argument("--sequential", action="store_true",
                    help="force sequential mode regardless of the config default")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain CNP models from scratch instead of loading checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override CNP training epochs")
    ap.add_argument("--quick", action="store_true",
                    help="subsample data for a fast smoke run")
    ap.add_argument("--days", type=int, default=None,
                    help="cap evaluation to the last N eval days (fast checks)")
    ap.add_argument("--out", default=None, help="override output dir")
    ap.add_argument("--no-report", action="store_true", help="skip HTML report generation")
    ap.add_argument("--prior_ctx", choices=["full", "match"], default=None,
                    help="independent harness: seed temporal models with yesterday's full "
                         "surface ('full', perfect-prior reference) or a context-matched "
                         "sparse sample ('match', the fair regime). Default: the config's.")
    args = ap.parse_args()

    if args.config not in EXPERIMENTS and (args.dataset or args.models):
        if not (args.dataset and args.models):
            ap.error("an ad-hoc --config needs BOTH --dataset and --models")
        names = [registry.resolve(m.strip()) for m in args.models.split(",") if m.strip()]
        exp = make_experiment(args.config, args.dataset, [(n, {}) for n in names],
                              mode=args.mode, needs_prior=not args.no_prior)
        print(f"Ad-hoc experiment '{exp.name}': dataset={args.dataset} models={names} "
              f"mode={exp.mode}")
    else:
        exp = get_experiment(args.config)
    mode = "sequential" if args.sequential else exp.mode
    prior_ctx = args.prior_ctx or exp.prior_ctx

    print(f"Loading data for '{exp.name}' …")
    dataset, _ood = exp.loader()
    if args.quick:
        keep = min(dataset.n_days, 120)
        dataset = dataset.subset(list(range(dataset.n_days - keep, dataset.n_days)))
    if exp.needs_prior and dataset.prior_targets is None:
        print("Computing B-spline prior …")
        dataset.prior_targets = compute_bspline_prior(dataset)
    print(f"  {dataset.n_days} days, {dataset.n_assets} assets, ctx_max={dataset.ctx_max}")

    def _cnp_cfg():
        c = CNPTrainConfig(device=args.device)
        if args.epochs is not None:
            c.train.n_epochs = args.epochs
        return c

    # build + train/load models
    cnp_names = ("cnp", "cnp_delta")
    models = []
    for name, kwargs in exp.models:
        kw = dict(kwargs)
        if args.retrain and name in cnp_names:
            kw.pop("checkpoint", None)
            kw["config"] = _cnp_cfg()
        elif name in cnp_names:
            kw["device"] = args.device
            # provide a training config so a missing checkpoint triggers load-or-train
            # (e.g. market has no cached CNP yet); ignored when the checkpoint exists.
            kw.setdefault("config", _cnp_cfg())
        model = registry.build(name, **kw)
        print(f"  preparing {model.name} …", flush=True)
        model.train(dataset, saved=True, force=args.retrain and name in cnp_names)
        models.append(model)

    # optionally cap evaluation to the last N days (fast checks)
    eval_days = None
    if args.days is not None:
        val = dataset.val_idx(); val = np.sort(val[val > 0])
        eval_days = val[-args.days:]
        print(f"  capping to last {len(eval_days)} eval days: {list(map(int, eval_days))}")

    # NEW composable path: a Model-bundle list run through the unified loop.
    if getattr(exp, "specs", None) is not None:
        fitters = {m.name: m for m in models}
        model_specs = exp.specs(fitters)
        print(f"Running composable harness … {len(model_specs)} model bundles "
              f"({len(fitters)} fitters)", flush=True)
        rec = run_models(model_specs, dataset, eval_days=eval_days)
        _finish(rec, exp, args, models, dataset, mode)
        return

    exclude = getattr(exp, "exclude_asset", None)
    asym = getattr(exp, "asymmetric_target", None)
    print(f"Running {mode} harness …"
          + (f" (asymmetric target: {asym}, peers full)" if asym else "")
          + (f" (leave-one-out: {exclude})" if exclude else "")
          + f"  [prior_ctx={prior_ctx}]", flush=True)
    if asym:
        rec = run_target_asymmetric(models, dataset, asym, ctx_sizes=exp.ctx_sizes,
                                    prior_ctx=prior_ctx, eval_days=eval_days)
    elif exclude:
        if mode == "sequential":
            rec = run_sequential_exclude(models, dataset, exclude,
                                         ctx_sizes=exp.seq_ctx_sizes, prior_ctx=prior_ctx,
                                         eval_days=eval_days)
        else:
            rec = run_exclude(models, dataset, exclude, ctx_sizes=exp.ctx_sizes,
                              prior_ctx=prior_ctx, eval_days=eval_days)
    elif mode == "sequential":
        rec = run_sequential(models, dataset, ctx_sizes=exp.seq_ctx_sizes, prior_ctx=prior_ctx,
                             eval_days=eval_days)
    else:
        rec = run(models, dataset, ctx_sizes=exp.ctx_sizes, prior_ctx=prior_ctx,
                  eval_days=eval_days)

    _finish(rec, exp, args, models, dataset, mode)


if __name__ == "__main__":
    main()
