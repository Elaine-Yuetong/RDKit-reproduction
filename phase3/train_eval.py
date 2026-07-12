"""Train/evaluate Phase 3 models on the real in-house DFT labels.

It reuses the cached Phase 3 descriptor + Morgan features and the Phase 1
XGBoost configuration family. No QM9, Colab, or Phase 2 assets are touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from xgboost import XGBRegressor

from phase3.data import DATA_DIR, load_unique_labels
from phase3.featurize_real import DESC_NPY, FP_NPY, META_JSON
from src.data import SEED

UNIT = "kcal/mol"
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

TARGET_CONFIGS = {
    "esp_vmin_mean_kcal_per_mol": {
        "short_name": "ESP",
        "result_path": RESULTS_DIR / "phase3_esp_metrics.json",
        "label_noise_floor": {
            "duplicate_std_median": 0.34,
            "duplicate_std_mean": 2.26,
            "unit": UNIT,
        },
        "outlier_diagnostic": False,
    },
    "zn_e_bind_mean_kcal_per_mol": {
        "short_name": "Zn binding",
        "result_path": RESULTS_DIR / "phase3_zn_metrics.json",
        "label_noise_floor": {
            "duplicate_std_median": 1.73,
            "duplicate_std_mean": 6.56,
            "unit": UNIT,
        },
        "outlier_diagnostic": True,
    },
}

XGB_MAX_DEPTH = [6, 8]
XGB_LEARNING_RATE = [0.05, 0.1]
XGB_N_ESTIMATORS = 4000
XGB_EARLY_STOPPING = 50
INNER_VAL_FRAC = 0.2


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


def outlier_indices_for_target(y: np.ndarray) -> np.ndarray:
    """Global |z| > 4 diagnostic indices. These are never removed from training."""
    z = (y - y.mean()) / y.std(ddof=1)
    return np.flatnonzero(np.abs(z) > 4)


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


def eval_one_split(
    name: str,
    X,
    y,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int | str,
    outlier_idx: np.ndarray | None = None,
) -> dict:
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
    record = {
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
    if outlier_idx is not None:
        outlier_set = set(map(int, outlier_idx))
        keep = np.array([int(idx) not in outlier_set for idx in test_idx], dtype=bool)
        n_excluded = int((~keep).sum())
        if keep.any():
            diag_mae = float(mean_absolute_error(y[test_idx][keep], pred[keep]))
        else:
            diag_mae = None
        record["diagnostic_mae_excluding_global_outliers"] = diag_mae
        record["diagnostic_n_outliers_excluded"] = n_excluded
        record["diagnostic_n_eval_no_outliers"] = int(keep.sum())
        if diag_mae is None:
            print("    diagnostic no-outlier MAE: NA (all test rows excluded)")
        else:
            print(
                f"    diagnostic no-outlier MAE: {diag_mae:.4f} {UNIT} "
                f"(excluded {n_excluded} global |z|>4 rows from test metric only)"
            )
    return record


def summarize(records: list[dict]) -> dict:
    maes = np.array([r["mae"] for r in records], dtype=float)
    r2s = np.array([r["r2"] for r in records], dtype=float)
    bmaes = np.array([r["train_mean_baseline_mae"] for r in records], dtype=float)
    br2s = np.array([r["train_mean_baseline_r2"] for r in records], dtype=float)
    summary = {
        "mean_mae": float(maes.mean()),
        "std_mae": float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
        "mean_r2": float(r2s.mean()),
        "std_r2": float(r2s.std(ddof=1)) if len(r2s) > 1 else 0.0,
        "mean_train_mean_baseline_mae": float(bmaes.mean()),
        "std_train_mean_baseline_mae": float(bmaes.std(ddof=1)) if len(bmaes) > 1 else 0.0,
        "mean_train_mean_baseline_r2": float(br2s.mean()),
        "std_train_mean_baseline_r2": float(br2s.std(ddof=1)) if len(br2s) > 1 else 0.0,
    }
    diag = [
        r.get("diagnostic_mae_excluding_global_outliers")
        for r in records
        if r.get("diagnostic_mae_excluding_global_outliers") is not None
    ]
    if diag:
        d = np.asarray(diag, dtype=float)
        summary["diagnostic_mean_mae_excluding_global_outliers"] = float(d.mean())
        summary["diagnostic_std_mae_excluding_global_outliers"] = (
            float(d.std(ddof=1)) if len(d) > 1 else 0.0
        )
        summary["diagnostic_total_outliers_excluded"] = int(
            sum(r.get("diagnostic_n_outliers_excluded", 0) for r in records)
        )
    return summary


def scaffold_groups(scaffolds) -> np.ndarray:
    groups = []
    for i, scaffold in enumerate(scaffolds):
        if scaffold == "":
            groups.append(f"acyclic_{i}")
        else:
            groups.append(f"scaffold_{scaffold}")
    return np.asarray(groups, dtype=object)


def run_random_cv(X, y, outlier_idx: np.ndarray | None = None) -> dict:
    records = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        records.append(
            eval_one_split("random_cv", X, y, train_idx, test_idx, fold, outlier_idx)
        )
    return {"folds": records, "summary": summarize(records)}


def run_scaffold_cv(X, y, df, outlier_idx: np.ndarray | None = None) -> dict:
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
        records.append(
            eval_one_split("scaffold_cv", X, y, train_idx, test_idx, fold, outlier_idx)
        )
    return {
        "empty_scaffold_handling": "each empty-scaffold molecule assigned its own group",
        "empty_scaffold_count": empty_count,
        "n_groups": int(len(set(groups))),
        "folds": records,
        "summary": summarize(records),
    }


def run_source_splits(X, y, df, outlier_idx: np.ndarray | None = None) -> dict:
    source = df["source_kind"].astype(str).to_numpy()
    baseline_idx = np.flatnonzero(source == "baseline")
    agent_idx = np.flatnonzero(source == "agent")
    if len(baseline_idx) == 0 or len(agent_idx) == 0:
        raise ValueError("source split requires non-empty baseline and agent sets")

    records = [
        eval_one_split(
            "source_split", X, y, baseline_idx, agent_idx,
            "baseline_to_agent", outlier_idx
        ),
        eval_one_split(
            "source_split", X, y, agent_idx, baseline_idx,
            "agent_to_baseline", outlier_idx
        ),
    ]
    return {"directions": records}


def _scheme_line(label: str, summary: dict) -> str:
    return (
        f"{label:23s} "
        f"{summary['mean_mae']:7.3f} +/- {summary['std_mae']:<7.3f} "
        f"{summary['mean_r2']:7.3f} +/- {summary['std_r2']:<7.3f} "
        f"{summary['mean_train_mean_baseline_mae']:7.3f} +/- "
        f"{summary['std_train_mean_baseline_mae']:<7.3f}"
    )


def print_summary_table(results: dict) -> None:
    short_name = results["target_config"]["short_name"]
    noise = results["label_noise_floor"]
    print(f"\n=== Phase 3 {short_name} XGBoost Summary (concat features) ===")
    print(f"Target: {results['target']} ({UNIT})")
    print(
        f"DFT duplicate-label noise floor for {short_name}: "
        f"median std={noise['duplicate_std_median']:.2f} {UNIT}, "
        f"mean std={noise['duplicate_std_mean']:.2f} {UNIT}"
    )
    print("\nScheme                  MAE mean +/- std      R2 mean +/- std       Train-mean MAE")
    print("-" * 86)
    for key, label in [
        ("random_5fold_cv", "Random 5-fold CV"),
        ("scaffold_5fold_cv", "Scaffold 5-fold CV"),
    ]:
        s = results[key]["summary"]
        print(_scheme_line(label, s))
        if "diagnostic_mean_mae_excluding_global_outliers" in s:
            print(
                f"{'  diagnostic no-out':23s} "
                f"{s['diagnostic_mean_mae_excluding_global_outliers']:7.3f} +/- "
                f"{s['diagnostic_std_mae_excluding_global_outliers']:<7.3f} "
                f"{'(MAE only; |z|>4 rows excluded from test metric)':>36s}"
            )
    for rec in results["source_split"]["directions"]:
        print(
            f"Source {rec['fold']:16s} "
            f"{rec['mae']:7.3f} +/- {'NA':<7s} "
            f"{rec['r2']:7.3f} +/- {'NA':<7s} "
            f"{rec['train_mean_baseline_mae']:7.3f} +/- {'NA':<7s}"
        )
        if "diagnostic_mae_excluding_global_outliers" in rec:
            diag = rec["diagnostic_mae_excluding_global_outliers"]
            diag_txt = "NA" if diag is None else f"{diag:.3f}"
            print(
                f"{'  diagnostic no-out':23s} {diag_txt:>7s} +/- {'NA':<7s} "
                f"(MAE only; excluded {rec['diagnostic_n_outliers_excluded']} rows)"
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


def comparison_row(results: dict) -> list[str]:
    rand = results["random_5fold_cv"]["summary"]
    scaf = results["scaffold_5fold_cv"]["summary"]
    source = {r["fold"]: r for r in results["source_split"]["directions"]}
    row = [
        results["target_config"]["short_name"],
        f"{rand['mean_mae']:.3f} +/- {rand['std_mae']:.3f}",
        f"{scaf['mean_mae']:.3f} +/- {scaf['std_mae']:.3f}",
        f"{source['baseline_to_agent']['mae']:.3f}",
        f"{source['agent_to_baseline']['mae']:.3f}",
    ]
    if "diagnostic_mean_mae_excluding_global_outliers" in rand:
        row.extend([
            f"{rand['diagnostic_mean_mae_excluding_global_outliers']:.3f}",
            f"{scaf['diagnostic_mean_mae_excluding_global_outliers']:.3f}",
            f"{source['baseline_to_agent']['diagnostic_mae_excluding_global_outliers']:.3f}",
            f"{source['agent_to_baseline']['diagnostic_mae_excluding_global_outliers']:.3f}",
        ])
    else:
        row.extend(["NA", "NA", "NA", "NA"])
    return row


def print_comparison_table(*results_objects: dict) -> None:
    print("\n=== Phase 3 ESP vs Zn MAE Comparison (kcal/mol) ===")
    header = (
        f"{'Target':11s} {'Random CV':>18s} {'Scaffold CV':>18s} "
        f"{'Base->Agent':>12s} {'Agent->Base':>12s} "
        f"{'Rand no-out':>11s} {'Scaf no-out':>11s} "
        f"{'B->A no-out':>11s} {'A->B no-out':>11s}"
    )
    print(header)
    print("-" * len(header))
    for results in results_objects:
        row = comparison_row(results)
        print(
            f"{row[0]:11s} {row[1]:>18s} {row[2]:>18s} "
            f"{row[3]:>12s} {row[4]:>12s} "
            f"{row[5]:>11s} {row[6]:>11s} {row[7]:>11s} {row[8]:>11s}"
        )


def config_for_target(target: str) -> dict:
    try:
        return TARGET_CONFIGS[target]
    except KeyError as exc:
        choices = ", ".join(TARGET_CONFIGS)
        raise SystemExit(f"unknown target {target!r}; choose one of: {choices}") from exc


def normalize_loaded_results(results: dict) -> dict:
    """Add target_config for metrics written before --target was introduced."""
    if "target_config" not in results:
        cfg = config_for_target(results["target"])
        results["target_config"] = {
            "short_name": cfg["short_name"],
            "outlier_diagnostic": cfg["outlier_diagnostic"],
        }
    return results


def run_target(target: str) -> dict:
    cfg = config_for_target(target)
    df = load_unique_labels()
    y = df[target].to_numpy(dtype=np.float64)
    X = load_concat_features(df)
    outlier_idx = None
    if cfg["outlier_diagnostic"]:
        outlier_idx = outlier_indices_for_target(y)

    print(f"Phase 3 {cfg['short_name']} train/eval")
    print(f"data_dir={DATA_DIR}")
    print(f"n_molecules={len(df)} X_shape={X.shape}")
    print(f"target={target} unit={UNIT}")
    print("model=XGBoost concat features; grid=max_depth {6,8} x lr {0.05,0.1}")
    print("early_stopping=50 on a held-out slice of each training fold")
    print(
        "reference_noise_floor="
        f"median_duplicate_std {cfg['label_noise_floor']['duplicate_std_median']} {UNIT}, "
        f"mean_duplicate_std {cfg['label_noise_floor']['duplicate_std_mean']} {UNIT}"
    )
    if outlier_idx is not None:
        print(
            "diagnostic_outliers="
            f"{len(outlier_idx)} global |z|>4 rows; kept in headline train/eval, "
            "excluded only for secondary MAE lines"
        )

    results = {
        "target": target,
        "target_config": {
            "short_name": cfg["short_name"],
            "outlier_diagnostic": cfg["outlier_diagnostic"],
        },
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
        "label_noise_floor": cfg["label_noise_floor"],
        "global_outlier_diagnostic": None if outlier_idx is None else {
            "rule": "|z| > 4 on the full deduped target distribution",
            "n_outliers": int(len(outlier_idx)),
            "indices": [int(i) for i in outlier_idx],
            "canonical_smiles": df.iloc[outlier_idx]["canonical_smiles"].tolist(),
            "target_values": [float(v) for v in y[outlier_idx]],
            "note": "Outliers are genuine signal and remain in headline training/evaluation.",
        },
        "n_molecules": int(len(df)),
        "n_features": int(X.shape[1]),
        "random_5fold_cv": run_random_cv(X, y, outlier_idx),
        "scaffold_5fold_cv": run_scaffold_cv(X, y, df, outlier_idx),
        "source_split": run_source_splits(X, y, df, outlier_idx),
    }

    result_path = cfg["result_path"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved metrics: {result_path}")
    print_summary_table(results)
    return results


def maybe_print_esp_zn_comparison(current_results: dict) -> None:
    candidates = []
    for target, cfg in TARGET_CONFIGS.items():
        path = cfg["result_path"]
        if path.exists():
            candidates.append(normalize_loaded_results(json.loads(path.read_text())))
        elif current_results["target"] == target:
            candidates.append(current_results)
    # Keep order stable: ESP first, Zn second, no duplicates.
    by_target = {r["target"]: r for r in candidates}
    ordered = [by_target[t] for t in TARGET_CONFIGS if t in by_target]
    if len(ordered) >= 2:
        print_comparison_table(*ordered)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate Phase 3 DFT targets.")
    ap.add_argument(
        "--target",
        default="esp_vmin_mean_kcal_per_mol",
        choices=sorted(TARGET_CONFIGS),
        help="Target column to evaluate. Default preserves the original ESP run.",
    )
    args = ap.parse_args()
    results = run_target(args.target)
    maybe_print_esp_zn_comparison(results)


if __name__ == "__main__":
    main()
