"""Element-class error breakdown for the Phase 3 mentor source split.

This reproduces the existing baseline->agent XGBoost source split, then
reports errors on the agent test set split by QM9-compatible versus
exotic-element chemistry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, r2_score

from phase3.data import load_unique_labels
from phase3.train_eval import (
    RESULTS_DIR,
    TARGET_CONFIGS,
    UNIT,
    load_concat_features,
    train_xgb_tuned,
)
from src.data import SEED

QM9_COMPATIBLE_ELEMENTS = {"C", "N", "O", "F"}
TARGETS = [
    "esp_vmin_mean_kcal_per_mol",
    "zn_e_bind_mean_kcal_per_mol",
]
RESULT_PATH = RESULTS_DIR / "phase3_task1_element_breakdown.json"


def atom_symbols_from_smiles(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse canonical_smiles={smiles!r}")
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def mae_r2_or_none(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"mae": None, "r2": None}
    out: dict[str, float | None] = {
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": None,
    }
    if len(y_true) >= 2:
        out["r2"] = float(r2_score(y_true, pred))
    return out


def group_record(
    label: str,
    idx: np.ndarray,
    y: np.ndarray,
    pred_by_abs_idx: dict[int, float],
    train_mean: float,
) -> dict:
    y_true = y[idx]
    pred = np.asarray([pred_by_abs_idx[int(i)] for i in idx], dtype=np.float64)
    metrics = mae_r2_or_none(y_true, pred)
    baseline_pred = np.full_like(y_true, fill_value=train_mean, dtype=np.float64)
    baseline = mae_r2_or_none(y_true, baseline_pred)
    return {
        "group": label,
        "n": int(len(idx)),
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "train_mean_baseline_mae": baseline["mae"],
        "train_mean_baseline_r2": baseline["r2"],
    }


def load_saved_baseline_to_agent(target: str) -> dict:
    path = Path(TARGET_CONFIGS[target]["result_path"])
    saved = json.loads(path.read_text())
    for rec in saved["source_split"]["directions"]:
        if rec["fold"] == "baseline_to_agent":
            return rec
    raise KeyError(f"baseline_to_agent not found in {path}")


def run_target(target: str, X: np.ndarray, df, element_sets: list[set[str]]) -> dict:
    source = df["source_kind"].astype(str).to_numpy()
    train_idx = np.flatnonzero(source == "baseline")
    test_idx = np.flatnonzero(source == "agent")
    if len(train_idx) != 2140 or len(test_idx) != 1523:
        raise ValueError(
            f"mentor split count mismatch: train={len(train_idx)} test={len(test_idx)}"
        )

    y = df[target].to_numpy(dtype=np.float64)
    print(f"\n=== {target} ===")
    print(f"mentor split: train baseline={len(train_idx)} test agent={len(test_idx)}")
    model, info = train_xgb_tuned(X, y, train_idx, verbose=True)
    pred_test = model.predict(X[test_idx])
    pred_by_abs_idx = {
        int(idx): float(pred) for idx, pred in zip(test_idx, pred_test, strict=True)
    }
    train_mean = float(np.mean(y[train_idx]))

    all_agent = test_idx
    qm9_compat = np.asarray(
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

    groups = [
        group_record("all_agent_test", all_agent, y, pred_by_abs_idx, train_mean),
        group_record("qm9_compat_test", qm9_compat, y, pred_by_abs_idx, train_mean),
        group_record("exotic_test", exotic, y, pred_by_abs_idx, train_mean),
        group_record("exotic_S_only_test", exotic_s, y, pred_by_abs_idx, train_mean),
    ]

    overall = groups[0]
    saved = load_saved_baseline_to_agent(target)
    mae_diff = abs(float(overall["mae"]) - float(saved["mae"]))
    r2_diff = abs(float(overall["r2"]) - float(saved["r2"]))
    reproduction = {
        "saved_mae": float(saved["mae"]),
        "saved_r2": float(saved["r2"]),
        "current_mae": float(overall["mae"]),
        "current_r2": float(overall["r2"]),
        "mae_abs_diff": float(mae_diff),
        "r2_abs_diff": float(r2_diff),
        "matches_saved_within_1e-6": bool(mae_diff <= 1e-6 and r2_diff <= 1e-6),
    }

    print(
        "saved baseline_to_agent: "
        f"MAE={saved['mae']:.6f} {UNIT} R2={saved['r2']:.6f}"
    )
    print(
        "current baseline_to_agent: "
        f"MAE={overall['mae']:.6f} {UNIT} R2={overall['r2']:.6f} "
        f"(diff MAE={mae_diff:.3g}, R2={r2_diff:.3g})"
    )
    print_table(target, groups)

    return {
        "target": target,
        "unit": UNIT,
        "split": "baseline_to_agent",
        "features": "concat",
        "model": "xgb",
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "best_params": info["best_params"],
        "n_estimators": info["n_estimators"],
        "inner_val_mae": info["inner_val_mae"],
        "reproduction_check": reproduction,
        "groups": groups,
    }


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def print_table(target: str, groups: list[dict]) -> None:
    print(f"\nElement breakdown for {target} ({UNIT})")
    print(f"{'test group':24s} {'n':>6s} {'MAE':>10s} {'R2':>10s} {'train-mean-MAE':>16s}")
    print("-" * 70)
    for row in groups:
        print(
            f"{row['group']:24s} {row['n']:6d} "
            f"{fmt(row['mae']):>10s} {fmt(row['r2']):>10s} "
            f"{fmt(row['train_mean_baseline_mae']):>16s}"
        )


def main() -> None:
    df = load_unique_labels()
    X = load_concat_features(df)
    element_sets = [
        atom_symbols_from_smiles(smiles) for smiles in df["canonical_smiles"].astype(str)
    ]
    source_counts = df["source_kind"].value_counts().to_dict()
    print("Phase 3 task 1 element-class breakdown")
    print(f"n_molecules={len(df)} X_shape={X.shape} seed={SEED}")
    print(f"source_kind_counts={source_counts}")
    print(f"qm9_compatible_elements={sorted(QM9_COMPATIBLE_ELEMENTS)}")

    results = {
        "seed": SEED,
        "n_molecules": int(len(df)),
        "n_features": int(X.shape[1]),
        "qm9_compatible_elements": sorted(QM9_COMPATIBLE_ELEMENTS),
        "source_kind_counts": {str(k): int(v) for k, v in source_counts.items()},
        "targets": [run_target(target, X, df, element_sets) for target in TARGETS],
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved metrics: {RESULT_PATH}")


if __name__ == "__main__":
    main()
