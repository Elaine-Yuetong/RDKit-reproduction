"""A/B/C ChemBERTa vs RDKit comparison on the Phase 3 mentor split.

Feature sets:
  A      = RDKit descriptors + Morgan fingerprints
  B_mean = ChemBERTa mean-pooled token embeddings
  B_cls  = ChemBERTa CLS-token embeddings
  C_mean = A concatenated with B_mean
  C_cls  = A concatenated with B_cls

The split is fixed by the mentor's protocol: train on baseline molecules and
test on agent-campaign molecules. No feature extraction happens here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from phase3.data import DATA_DIR, load_unique_labels
from phase3.task1_element_breakdown import (
    QM9_COMPATIBLE_ELEMENTS,
    TARGETS,
    atom_symbols_from_smiles,
)
from phase3.train_eval import RESULTS_DIR, UNIT, load_concat_features, train_xgb_tuned
from src.data import SEED

CHEMBERTA_DIR = DATA_DIR / "features_chemberta"
CHEMBERTA_MEAN_NPY = CHEMBERTA_DIR / "chemberta_mean.npy"
CHEMBERTA_CLS_NPY = CHEMBERTA_DIR / "chemberta_cls.npy"
CHEMBERTA_LABELS_CSV = CHEMBERTA_DIR / "labels_chemberta.csv"
TASK1_RESULTS_JSON = RESULTS_DIR / "phase3_task1_element_breakdown.json"
RESULT_PATH = RESULTS_DIR / "phase3_task2_abc_metrics.json"

FEATURE_LABELS = {
    "A_rdkit_concat": "A: RDKit concat",
    "B1_chemberta_mean": "B1: ChemBERTa mean",
    "B2_chemberta_cls": "B2: ChemBERTa CLS",
    "C_mean": "C_mean: RDKit+mean",
    "C_cls": "C_cls: RDKit+CLS",
}
GROUP_ORDER = [
    "all_agent_test",
    "qm9_compat_test",
    "exotic_test",
    "exotic_S_only_test",
]


def load_chemberta_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    for path in (CHEMBERTA_MEAN_NPY, CHEMBERTA_CLS_NPY, CHEMBERTA_LABELS_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"missing ChemBERTa cache file: {path}. Run "
                "./venv/bin/python -m phase3.extract_chemberta_feats first."
            )

    labels = pd.read_csv(CHEMBERTA_LABELS_CSV)
    required = {"row_id", "canonical_smiles"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"missing columns in {CHEMBERTA_LABELS_CSV}: {missing}")
    expected_row_id = np.arange(len(df), dtype=int)
    if not np.array_equal(labels["row_id"].to_numpy(dtype=int), expected_row_id):
        raise ValueError("ChemBERTa row_id column is not 0..n-1 in label order")

    rdkit_smiles = df["canonical_smiles"].astype(str).to_numpy()
    chemberta_smiles = labels["canonical_smiles"].astype(str).to_numpy()
    mismatches = np.flatnonzero(rdkit_smiles != chemberta_smiles)
    if len(mismatches):
        preview = [
            {
                "row": int(i),
                "load_unique_labels": str(rdkit_smiles[i]),
                "labels_chemberta": str(chemberta_smiles[i]),
            }
            for i in mismatches[:10]
        ]
        raise ValueError(
            f"ChemBERTa label order mismatch at {len(mismatches)} rows; "
            f"first mismatches: {preview}"
        )

    mean = np.load(CHEMBERTA_MEAN_NPY).astype(np.float32, copy=False)
    cls = np.load(CHEMBERTA_CLS_NPY).astype(np.float32, copy=False)
    expected_shape = (len(df), 384)
    if mean.shape != expected_shape or cls.shape != expected_shape:
        raise ValueError(
            f"unexpected ChemBERTa shapes: mean={mean.shape}, cls={cls.shape}, "
            f"expected={expected_shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(cls).all():
        raise ValueError("ChemBERTa features contain NaN or Inf")
    return mean, cls


def load_task1_reference() -> dict[str, dict[str, dict]]:
    data = json.loads(TASK1_RESULTS_JSON.read_text())
    out: dict[str, dict[str, dict]] = {}
    for target_record in data["targets"]:
        out[target_record["target"]] = {
            group["group"]: group for group in target_record["groups"]
        }
    return out


def group_indices(df: pd.DataFrame, element_sets: list[set[str]]) -> dict[str, np.ndarray]:
    source = df["source_kind"].astype(str).to_numpy()
    test_idx = np.flatnonzero(source == "agent")
    qm9 = np.asarray(
        [
            int(idx)
            for idx in test_idx
            if element_sets[int(idx)].issubset(QM9_COMPATIBLE_ELEMENTS)
        ],
        dtype=int,
    )
    exotic = np.asarray(
        [
            int(idx)
            for idx in test_idx
            if not element_sets[int(idx)].issubset(QM9_COMPATIBLE_ELEMENTS)
        ],
        dtype=int,
    )
    exotic_s = np.asarray(
        [int(idx) for idx in exotic if "S" in element_sets[int(idx)]],
        dtype=int,
    )
    return {
        "all_agent_test": test_idx,
        "qm9_compat_test": qm9,
        "exotic_test": exotic,
        "exotic_S_only_test": exotic_s,
    }


def mae_r2(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"mae": None, "r2": None}
    result: dict[str, float | None] = {
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": None,
    }
    if len(y_true) >= 2:
        result["r2"] = float(r2_score(y_true, pred))
    return result


def evaluate_groups(
    y: np.ndarray,
    pred_by_abs_idx: dict[int, float],
    groups: dict[str, np.ndarray],
) -> dict[str, dict]:
    out = {}
    for name in GROUP_ORDER:
        idx = groups[name]
        pred = np.asarray([pred_by_abs_idx[int(i)] for i in idx], dtype=np.float64)
        metrics = mae_r2(y[idx], pred)
        out[name] = {"n": int(len(idx)), **metrics}
    return out


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def print_reference_bar(target: str, task1_ref: dict[str, dict]) -> None:
    print(f"\nTask-1 A-only reference for {target} ({UNIT})")
    print(f"{'group':22s} {'n':>5s} {'MAE':>8s} {'R2':>8s}")
    print("-" * 48)
    for group in GROUP_ORDER:
        row = task1_ref[target][group]
        print(f"{group:22s} {row['n']:5d} {row['mae']:8.3f} {row['r2']:8.3f}")


def print_target_table(target: str, records: list[dict]) -> None:
    print(f"\n=== {target} A/B/C comparison ({UNIT}) ===")
    header = (
        f"{'feature set':24s} "
        f"{'overall MAE/R2':>18s} {'qm9 MAE/R2':>18s} "
        f"{'exotic MAE/R2':>18s} {'exotic_S MAE/R2':>18s}"
    )
    print(header)
    print("-" * len(header))
    for rec in records:
        groups = rec["groups"]
        print(
            f"{FEATURE_LABELS[rec['feature_set']]:24s} "
            f"{fmt(groups['all_agent_test']['mae'])}/{fmt(groups['all_agent_test']['r2']):>6s} "
            f"{fmt(groups['qm9_compat_test']['mae'])}/{fmt(groups['qm9_compat_test']['r2']):>6s} "
            f"{fmt(groups['exotic_test']['mae'])}/{fmt(groups['exotic_test']['r2']):>6s} "
            f"{fmt(groups['exotic_S_only_test']['mae'])}/{fmt(groups['exotic_S_only_test']['r2']):>6s}"
        )


def run_feature_target(
    feature_name: str,
    X: np.ndarray,
    target: str,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: dict[str, np.ndarray],
) -> dict:
    print(
        f"\nTraining {feature_name} target={target} "
        f"X_shape={X.shape} n_train={len(train_idx)} n_test={len(test_idx)}"
    )
    model, info = train_xgb_tuned(X, y, train_idx, verbose=False)
    pred = model.predict(X[test_idx])
    pred_by_abs_idx = {
        int(idx): float(value) for idx, value in zip(test_idx, pred, strict=True)
    }
    group_metrics = evaluate_groups(y, pred_by_abs_idx, groups)
    overall = group_metrics["all_agent_test"]
    print(
        f"  MAE={overall['mae']:.4f} {UNIT} R2={overall['r2']:.4f} "
        f"params={info['best_params']} n_estimators={info['n_estimators']}"
    )
    return {
        "feature_set": feature_name,
        "n_features": int(X.shape[1]),
        "best_params": info["best_params"],
        "n_estimators": info["n_estimators"],
        "inner_val_mae": info["inner_val_mae"],
        "groups": group_metrics,
    }


def main() -> None:
    df = load_unique_labels().reset_index(drop=True)
    source = df["source_kind"].astype(str).to_numpy()
    train_idx = np.flatnonzero(source == "baseline")
    test_idx = np.flatnonzero(source == "agent")
    if len(train_idx) != 2140 or len(test_idx) != 1523:
        raise ValueError(
            f"mentor split count mismatch: train={len(train_idx)} test={len(test_idx)}"
        )

    X_a = load_concat_features(df).astype(np.float32, copy=False)
    X_mean, X_cls = load_chemberta_features(df)
    print("Alignment check passed: ChemBERTa labels match load_unique_labels row order.")
    print(f"n_molecules={len(df)} train_baseline={len(train_idx)} test_agent={len(test_idx)}")
    print(f"A RDKit concat shape={X_a.shape}")
    print(f"B_mean ChemBERTa shape={X_mean.shape}")
    print(f"B_cls ChemBERTa shape={X_cls.shape}")

    feature_sets = {
        "A_rdkit_concat": X_a,
        "B1_chemberta_mean": X_mean,
        "B2_chemberta_cls": X_cls,
        "C_mean": np.hstack([X_a, X_mean]).astype(np.float32, copy=False),
        "C_cls": np.hstack([X_a, X_cls]).astype(np.float32, copy=False),
    }

    element_sets = [
        atom_symbols_from_smiles(smiles) for smiles in df["canonical_smiles"].astype(str)
    ]
    groups = group_indices(df, element_sets)
    for name in GROUP_ORDER:
        print(f"{name}_n={len(groups[name])}")

    task1_ref = load_task1_reference()
    results = {
        "seed": SEED,
        "unit": UNIT,
        "split": "mentor baseline_to_agent",
        "n_molecules": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "alignment_check": "passed: canonical_smiles match row-by-row",
        "feature_sets": {name: int(X.shape[1]) for name, X in feature_sets.items()},
        "targets": [],
    }

    for target in TARGETS:
        y = df[target].to_numpy(dtype=np.float64)
        print_reference_bar(target, task1_ref)
        records = [
            run_feature_target(name, X, target, y, train_idx, test_idx, groups)
            for name, X in feature_sets.items()
        ]
        print_target_table(target, records)
        results["targets"].append(
            {
                "target": target,
                "task1_reference_A_only": task1_ref[target],
                "records": records,
            }
        )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved metrics: {RESULT_PATH}")


if __name__ == "__main__":
    main()
