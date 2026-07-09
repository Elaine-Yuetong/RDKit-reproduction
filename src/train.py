"""Train one model on one (target, feature set, split) config (Phase 1).

Scientific source of truth: METHODS.md.
  - Targets already in eV for homo/lumo/gap (converted once in src.data);
    NO target scaling here (trees don't need it).
  - Splits (seed=42) come from src.data and are REUSED, never re-created.
  - gap random-split test MAE should land ~0.25-0.5 eV. If far outside,
    diagnose in this order: Hartree->eV conversion, target-column mapping,
    train/test leakage.

Usage:
  python -m src.train --csv data/qm9.csv --target gap --features concat \
      --split random --model xgb
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.data import load_qm9, random_split, scaffold_split, SEED
from src.evaluate import mae_r2, update_metrics, parity_plot

# eV for these; native units otherwise (mu: Debye, alpha: Bohr^3).
EV_TARGETS = {"homo", "lumo", "gap"}

# XGB tuning grid (small, per CLAUDE.md: "light hyperparam search only").
XGB_MAX_DEPTH = [6, 8, 10]
XGB_LEARNING_RATE = [0.05, 0.1]
XGB_N_ESTIMATORS = 2000          # ceiling; early stopping picks the real count
XGB_EARLY_STOPPING = 50


def get_features(kind, df):
    from src.featurize import load_features
    if kind == "desc":
        X, _ = load_features("desc", df)
        return X
    if kind in ("fp", "concat"):
        return load_features(kind, df)
    raise ValueError(f"unknown feature kind: {kind!r}")


def get_split(split, df):
    if split == "random":
        return random_split(df, seed=SEED)
    if split == "scaffold":
        return scaffold_split(df)
    raise ValueError(f"unknown split: {split!r}")


def assert_clean_partition(train_idx, val_idx, test_idx, n):
    parts = [train_idx, val_idx, test_idx]
    total = sum(len(p) for p in parts)
    union = len(set(np.concatenate(parts)))
    assert total == n == union, (
        f"split is not a clean partition: total={total} union={union} n={n}")


def train_xgb(Xtr, ytr, Xva, yva):
    """Grid over max_depth x learning_rate, early-stopped on the validation
    set, pick the best by validation MAE. Returns (best_model, info)."""
    best = None
    for md in XGB_MAX_DEPTH:
        for lr in XGB_LEARNING_RATE:
            model = XGBRegressor(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=md,
                learning_rate=lr,
                tree_method="hist",
                n_jobs=-1,
                early_stopping_rounds=XGB_EARLY_STOPPING,
                eval_metric="mae",
                random_state=SEED,
            )
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            val_mae = float(model.best_score)            # MAE at best_iteration
            n_trees = int(model.best_iteration) + 1
            print(f"    max_depth={md} lr={lr:<4} -> "
                  f"val_MAE={val_mae:.4f}  best_trees={n_trees}")
            if best is None or val_mae < best["val_mae"]:
                best = {"model": model, "max_depth": md, "learning_rate": lr,
                        "val_mae": val_mae, "n_estimators": n_trees}
    info = {
        "best_params": {"max_depth": best["max_depth"],
                        "learning_rate": best["learning_rate"]},
        "n_estimators": best["n_estimators"],
    }
    return best["model"], info


def train_rf(Xtr, ytr):
    """RandomForest sanity model: fixed n_estimators=300, no tuning."""
    model = RandomForestRegressor(
        n_estimators=300, n_jobs=-1, random_state=SEED)
    model.fit(Xtr, ytr)
    return model, {"best_params": None, "n_estimators": 300}


def main():
    ap = argparse.ArgumentParser(description="Train one Phase-1 baseline config.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--features", required=True, choices=["desc", "fp", "concat"])
    ap.add_argument("--split", required=True, choices=["random", "scaffold"])
    ap.add_argument("--model", required=True, choices=["xgb", "rf"])
    args = ap.parse_args()

    df = load_qm9(csv_path=args.csv)
    if args.target not in df.columns:
        raise SystemExit(
            f"target {args.target!r} not in columns: {list(df.columns)}")

    X = get_features(args.features, df)
    y = df[args.target].to_numpy(dtype=np.float64)
    n = len(df)
    assert X.shape[0] == n, f"feature rows {X.shape[0]} != df len {n}"

    # Reuse the canonical splits; never re-split here.
    train_idx, val_idx, test_idx = get_split(args.split, df)
    assert_clean_partition(train_idx, val_idx, test_idx, n)

    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    Xte, yte = X[test_idx], y[test_idx]

    unit = "eV" if args.target in EV_TARGETS else "native"
    print(f"\nConfig: target={args.target} features={args.features} "
          f"split={args.split} model={args.model}")
    print(f"Rows: train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)}  |  X dim={X.shape[1]}  |  target unit={unit}")

    if args.model == "xgb":
        print("  XGBoost grid (max_depth x learning_rate), early-stopped on val:")
        model, info = train_xgb(Xtr, ytr, Xva, yva)
    else:
        print("  RandomForest sanity model (n_estimators=300, no tuning)...")
        model, info = train_rf(Xtr, ytr)

    # Predict val + test. For xgb, the early-stopped best_iteration is used.
    val_metrics = mae_r2(yva, model.predict(Xva))
    test_metrics = mae_r2(yte, model.predict(Xte))   # test scored exactly once

    record = {
        "target": args.target, "features": args.features,
        "split": args.split, "model": args.model,
        "unit": unit,
        "val_mae": val_metrics["mae"], "val_r2": val_metrics["r2"],
        "test_mae": test_metrics["mae"], "test_r2": test_metrics["r2"],
        "best_params": info["best_params"],
        "n_estimators": info["n_estimators"],
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    key = update_metrics(record)
    plot_path = parity_plot(yte, model.predict(Xte), args.target,
                            args.features, args.split, args.model, unit=unit)

    print(f"\n  best_params      : {info['best_params']}")
    print(f"  n_estimators     : {info['n_estimators']}"
          f"{'  (early-stopped)' if args.model == 'xgb' else ''}")
    print(f"  val  MAE / R2    : {val_metrics['mae']:.4f} {unit} / "
          f"{val_metrics['r2']:.4f}")
    print(f"  test MAE / R2    : {test_metrics['mae']:.4f} {unit} / "
          f"{test_metrics['r2']:.4f}")
    print(f"  metrics key      : {key}")
    print(f"  parity plot      : {plot_path}")


if __name__ == "__main__":
    main()
