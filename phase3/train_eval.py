"""Train/evaluate Phase 3 ESP models on the real in-house DFT labels.

This is ESP-only for the current checkpoint. It reuses the cached Phase 3
descriptor + Morgan features and the Phase 1 XGBoost configuration family.
No QM9, Colab, or Phase 2 assets are touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from xgboost import XGBRegressor

from phase3.data import DATA_DIR, load_unique_labels
from phase3.featurize_real import DESC_NPY, FP_NPY, META_JSON
from src.data import SEED

TARGET = "esp_vmin_mean_kcal_per_mol"
UNIT = "kcal/mol"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "phase3_esp_metrics.json"

XGB_MAX_DEPTH = [6, 8]
XGB_LEARNING_RATE = [0.05, 0.1]
XGB_N_ESTIMATORS = 4000
XGB_EARLY_STOPPING = 50
INNER_VAL_FRAC = 0.2
LABEL_NOISE_FLOOR = {
    "duplicate_std_median": 0.34,
    "duplicate_std_mean": 2.26,
    "unit": UNIT,
}


def load_concat_features(df) -> np.ndarray:
    """Load cached Phase 3 descriptor + fingerprint features as concat float32."""
    for path in (DESC_NPY, FP_NPY, META_JSON):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"missing feature cache file: {path}. Run "
                "./venv/bin/python -m phase3.featurize_real first."
            )

    with open(META_JSON) as f:
        meta = json.load(f)
    if int(meta.get("n_rows", -1)) != len(df):
        raise ValueError(
            f"feature cache row count {meta.get('n_rows')} != labels {len(df)}"
        )

    desc = np.load(DESC_NPY)
    fp = np.load(FP_NPY)
    if desc.shape[0] != len(df) or fp.shape[0] != len(df):
        raise ValueError(
            f"feature rows do not match labels: desc={desc.shape}, "
            f"fp={fp.shape}, labels={len(df)}"
        )
    return np.hstack([desc.astype(np.float32), fp.astype(np.float32)])


def mae_r2(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_mean_baseline(y_train, y_test) -> dict[str, float]:
    pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=np.float64)
    out = mae_r2(y_test, pred)
    out["train_mean"] = float(np.mean(y_train))
    return out


def make_inner_train_val(train_idx: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic held-out slice from the outer training fold."""
    # Binning stabilizes target coverage in the inner validation slice without
    # scaling the target or changing the outer evaluation protocol.
    y_train = y[train_idx]
    bins = np.digitize(y_train, np.quantile(y_train, [0.2, 0.4, 0.6, 0.8]))
    tr, va = train_test_split(
        train_idx,
        test_size=INNER_VAL_FRAC,
        random_state=SEED,
        shuffle=True,
        stratify=bins,
    )
    return np.asarray(tr, dtype=int), np.asarray(va, dtype=int)


def train_xgb_tuned(X, y, train_idx: np.ndarray, verbose: bool = True):
    """Tune max_depth x learning_rate by inner validation MAE."""
    inner_train_idx, inner_val_idx = make_inner_train_val(train_idx, y)
    Xtr, ytr = X[inner_train_idx], y[inner_train_idx]
    Xva, yva = X[inner_val_idx], y[inner_val_idx]

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
            val_mae = float(model.best_score)
            n_trees = int(model.best_iteration) + 1
            if verbose:
                print(
                    f"      max_depth={md} lr={lr:<4} -> "
                    f"inner_val_MAE={val_mae:.4f} best_trees={n_trees}"
                )
            if best is None or val_mae < best["val_mae"]:
                best = {
                    "model": model,
                    "val_mae": val_mae,
                    "max_depth": md,
                    "learning_rate": lr,
                    "n_estimators": n_trees,
                }

    return best["model"], {
        "best_params": {
            "max_depth": best["max_depth"],
            "learning_rate": best["learning_rate"],
        },
        "inner_val_mae": best["val_mae"],
        "n_estimators": best["n_estimators"],
        "early_stopped": best["n_estimators"] < XGB_N_ESTIMATORS,
        "n_inner_train": int(len(inner_train_idx)),
        "n_inner_val": int(len(inner_val_idx)),
    }


def eval_one_split(name: str, X, y, train_idx: np.ndarray, test_idx: np.ndarray,
                   fold: int | str) -> dict:
    print(f"\n{name} fold={fold} n_train={len(train_idx)} n_test={len(test_idx)}")
    baseline = train_mean_baseline(y[train_idx], y[test_idx])
    print(
        f"    train-mean baseline: MAE={baseline['mae']:.4f} {UNIT} "
        f"R2={baseline['r2']:.4f}"
    )
    model, info = train_xgb_tuned(X, y, train_idx, verbose=True)
    pred = model.predict(X[test_idx])
    metrics = mae_r2(y[test_idx], pred)
    print(
        f"    XGB: MAE={metrics['mae']:.4f} {UNIT} R2={metrics['r2']:.4f} "
        f"params={info['best_params']} n_estimators={info['n_estimators']}"
    )
    return {
        "fold": fold,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "train_mean_baseline_mae": baseline["mae"],
        "train_mean_baseline_r2": baseline["r2"],
        "train_mean": baseline["train_mean"],
        **info,
    }


def summarize(records: list[dict]) -> dict:
    maes = np.array([r["mae"] for r in records], dtype=float)
    r2s = np.array([r["r2"] for r in records], dtype=float)
    bmaes = np.array([r["train_mean_baseline_mae"] for r in records], dtype=float)
    br2s = np.array([r["train_mean_baseline_r2"] for r in records], dtype=float)
    return {
        "mean_mae": float(maes.mean()),
        "std_mae": float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
        "mean_r2": float(r2s.mean()),
        "std_r2": float(r2s.std(ddof=1)) if len(r2s) > 1 else 0.0,
        "mean_train_mean_baseline_mae": float(bmaes.mean()),
        "std_train_mean_baseline_mae": float(bmaes.std(ddof=1)) if len(bmaes) > 1 else 0.0,
        "mean_train_mean_baseline_r2": float(br2s.mean()),
        "std_train_mean_baseline_r2": float(br2s.std(ddof=1)) if len(br2s) > 1 else 0.0,
    }


def scaffold_groups(scaffolds) -> np.ndarray:
    groups = []
    for i, scaffold in enumerate(scaffolds):
        if scaffold == "":
            groups.append(f"acyclic_{i}")
        else:
            groups.append(f"scaffold_{scaffold}")
    return np.asarray(groups, dtype=object)


def run_random_cv(X, y) -> dict:
    records = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        records.append(eval_one_split("random_cv", X, y, train_idx, test_idx, fold))
    return {"folds": records, "summary": summarize(records)}


def run_scaffold_cv(X, y, df) -> dict:
    groups = scaffold_groups(df["murcko_scaffold"].fillna("").astype(str))
    empty_count = int((df["murcko_scaffold"].fillna("") == "").sum())
    records = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise AssertionError(f"scaffold group leakage in fold {fold}: {list(overlap)[:5]}")
        records.append(eval_one_split("scaffold_cv", X, y, train_idx, test_idx, fold))
    return {
        "empty_scaffold_handling": "each empty-scaffold molecule assigned its own group",
        "empty_scaffold_count": empty_count,
        "n_groups": int(len(set(groups))),
        "folds": records,
        "summary": summarize(records),
    }


def run_source_splits(X, y, df) -> dict:
    source = df["source_kind"].astype(str).to_numpy()
    baseline_idx = np.flatnonzero(source == "baseline")
    agent_idx = np.flatnonzero(source == "agent")
    if len(baseline_idx) == 0 or len(agent_idx) == 0:
        raise ValueError("source split requires non-empty baseline and agent sets")

    records = [
        eval_one_split("source_split", X, y, baseline_idx, agent_idx, "baseline_to_agent"),
        eval_one_split("source_split", X, y, agent_idx, baseline_idx, "agent_to_baseline"),
    ]
    return {"directions": records}


def print_summary_table(results: dict) -> None:
    print("\n=== Phase 3 ESP XGBoost Summary (concat features) ===")
    print(f"Target: {TARGET} ({UNIT})")
    print(
        "DFT duplicate-label noise floor for ESP: "
        f"median std={LABEL_NOISE_FLOOR['duplicate_std_median']:.2f} {UNIT}, "
        f"mean std={LABEL_NOISE_FLOOR['duplicate_std_mean']:.2f} {UNIT}"
    )
    print("\nScheme                  MAE mean +/- std      R2 mean +/- std       Train-mean MAE")
    print("-" * 86)
    for key, label in [
        ("random_5fold_cv", "Random 5-fold CV"),
        ("scaffold_5fold_cv", "Scaffold 5-fold CV"),
    ]:
        s = results[key]["summary"]
        print(
            f"{label:23s} "
            f"{s['mean_mae']:7.3f} +/- {s['std_mae']:<7.3f} "
            f"{s['mean_r2']:7.3f} +/- {s['std_r2']:<7.3f} "
            f"{s['mean_train_mean_baseline_mae']:7.3f} +/- "
            f"{s['std_train_mean_baseline_mae']:<7.3f}"
        )
    for rec in results["source_split"]["directions"]:
        print(
            f"Source {rec['fold']:16s} "
            f"{rec['mae']:7.3f} +/- {'NA':<7s} "
            f"{rec['r2']:7.3f} +/- {'NA':<7s} "
            f"{rec['train_mean_baseline_mae']:7.3f} +/- {'NA':<7s}"
        )

    print("\nPer-fold MAE:")
    for key, label in [
        ("random_5fold_cv", "random"),
        ("scaffold_5fold_cv", "scaffold"),
    ]:
        pieces = [
            f"{r['fold']}={r['mae']:.3f} (baseline {r['train_mean_baseline_mae']:.3f})"
            for r in results[key]["folds"]
        ]
        print(f"  {label}: " + ", ".join(pieces))
    for r in results["source_split"]["directions"]:
        print(
            f"  source {r['fold']}: {r['mae']:.3f} "
            f"(baseline {r['train_mean_baseline_mae']:.3f})"
        )


def main() -> None:
    df = load_unique_labels()
    y = df[TARGET].to_numpy(dtype=np.float64)
    X = load_concat_features(df)

    print("Phase 3 ESP train/eval")
    print(f"data_dir={DATA_DIR}")
    print(f"n_molecules={len(df)} X_shape={X.shape}")
    print(f"target={TARGET} unit={UNIT}")
    print("model=XGBoost concat features; grid=max_depth {6,8} x lr {0.05,0.1}")
    print("early_stopping=50 on a held-out slice of each training fold")
    print(
        "reference_noise_floor="
        f"median_duplicate_std {LABEL_NOISE_FLOOR['duplicate_std_median']} {UNIT}, "
        f"mean_duplicate_std {LABEL_NOISE_FLOOR['duplicate_std_mean']} {UNIT}"
    )

    results = {
        "target": TARGET,
        "unit": UNIT,
        "features": "concat",
        "model": "xgb",
        "seed": SEED,
        "xgb_grid": {
            "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE,
            "n_estimators_ceiling": XGB_N_ESTIMATORS,
            "early_stopping_rounds": XGB_EARLY_STOPPING,
            "tree_method": "hist",
            "n_jobs": -1,
        },
        "label_noise_floor": LABEL_NOISE_FLOOR,
        "n_molecules": int(len(df)),
        "n_features": int(X.shape[1]),
        "random_5fold_cv": run_random_cv(X, y),
        "scaffold_5fold_cv": run_scaffold_cv(X, y, df),
        "source_split": run_source_splits(X, y, df),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved metrics: {RESULTS_PATH}")
    print_summary_table(results)


if __name__ == "__main__":
    main()
