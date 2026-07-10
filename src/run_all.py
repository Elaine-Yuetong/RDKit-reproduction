"""Run the full Phase-1 grid and consolidate results.

Grid (40 configs):
  - XGBoost: 5 targets x 3 feature sets x 2 splits            = 30
  - RandomForest (sanity): 5 targets x concat x 2 splits      = 10

Reuses the cached features and the frozen splits (seed=42) from src.data /
src.featurize; every result is merged into results/metrics.json and a parity
plot saved. Resumable: configs already present in metrics.json are skipped.

Usage:
  python -m src.run_all --csv data/qm9.csv          # run (or resume) the grid
  python -m src.run_all --csv data/qm9.csv --table  # just print the table
"""

from __future__ import annotations

import argparse
import json

from src.data import load_qm9, TARGETS
from src.evaluate import metric_key, METRICS_JSON
from src.train import fit_config, get_features, EV_TARGETS

FEATURES = ["desc", "fp", "concat"]
SPLITS = ["random", "scaffold"]


def build_plan():
    """(target, features, split, model) tuples in feature-major order so each
    feature matrix is loaded from cache exactly once."""
    plan = []
    for feat in FEATURES:
        for target in TARGETS:
            for split in SPLITS:
                plan.append((target, feat, split, "xgb"))
        if feat == "concat":                       # RF sanity: concat only
            for target in TARGETS:
                for split in SPLITS:
                    plan.append((target, feat, split, "rf"))
    return plan


def _load_metrics():
    try:
        with open(METRICS_JSON) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def run_grid(csv):
    df = load_qm9(csv_path=csv)
    plan = build_plan()
    total = len(plan)
    metrics = _load_metrics()
    done = 0

    # Feature-major: load each matrix once, run all its configs, then free it.
    for feat in FEATURES:
        feat_cfgs = [c for c in plan if c[1] == feat]
        idx_of = {c: plan.index(c) + 1 for c in feat_cfgs}
        pending = [c for c in feat_cfgs
                   if metric_key(c[0], c[1], c[2], c[3]) not in metrics]
        if not pending:
            for c in feat_cfgs:
                done += 1
                k = metric_key(*c)
                print(f"[{done:2d}/{total}] SKIP  {k}  "
                      f"test_MAE={metrics[k]['test_mae']:.4f} {metrics[k]['unit']}")
            continue

        X = get_features(feat, df)                 # loaded once for this block
        for c in feat_cfgs:
            target, features, split, model = c
            done += 1
            k = metric_key(*c)
            if k in metrics:
                print(f"[{done:2d}/{total}] SKIP  {k}  "
                      f"test_MAE={metrics[k]['test_mae']:.4f} {metrics[k]['unit']}")
                continue
            record, info, _ = fit_config(
                df, X, target, features, split, model, verbose=False)
            metrics[k] = record
            es = ""
            if model == "xgb":
                es = ("early-stop" if info.get("early_stopped")
                      else "CEILING")
                es = f"  n_est={record['n_estimators']:<4} ({es})"
            print(f"[{done:2d}/{total}] DONE  {k}  "
                  f"test_MAE={record['test_mae']:.4f} {record['unit']}"
                  f"  R2={record['test_r2']:.4f}{es}")
        del X

    print(f"\nGrid complete: {total} configs in metrics.json.")
    return metrics


def print_table(metrics=None):
    """Consolidated table: target x features x split -> test MAE."""
    if metrics is None:
        metrics = _load_metrics()

    def cell(target, feat, split, model):
        k = metric_key(target, feat, split, model)
        return f"{metrics[k]['test_mae']:.4f}" if k in metrics else "  --  "

    print("\n=== TEST MAE (eV for homo/lumo/gap; native for mu, alpha) ===")
    print("XGBoost grid:")
    hdr = (f"{'target':7s} {'feat':7s} | {'random':>8s} {'scaffold':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for target in TARGETS:
        unit = "eV" if target in EV_TARGETS else "nat"
        for feat in FEATURES:
            print(f"{target:7s} {feat:7s} | "
                  f"{cell(target, feat, 'random', 'xgb'):>8s} "
                  f"{cell(target, feat, 'scaffold', 'xgb'):>9s}   [{unit}]")

    print("\nRandomForest (concat only, sanity):")
    print(hdr)
    print("-" * len(hdr))
    for target in TARGETS:
        unit = "eV" if target in EV_TARGETS else "nat"
        print(f"{target:7s} {'concat':7s} | "
              f"{cell(target, 'concat', 'random', 'rf'):>8s} "
              f"{cell(target, 'concat', 'scaffold', 'rf'):>9s}   [{unit}]")


def main():
    ap = argparse.ArgumentParser(description="Run the full Phase-1 grid.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--table", action="store_true",
                    help="only print the consolidated table, do not train")
    args = ap.parse_args()

    if args.table:
        print_table()
        return
    metrics = run_grid(args.csv)
    print_table(metrics)


if __name__ == "__main__":
    main()
