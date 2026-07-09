"""Metrics + parity plots for the QM9 RDKit baseline (Phase 1).

Metrics (MAE, R2 on val and test) are merged into results/metrics.json keyed
by (target, features, split, model). Parity plots go to results/plots/.
Energies for homo/lumo/gap are already in eV (converted once in src.data).
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")
METRICS_JSON = os.path.join(RESULTS_DIR, "metrics.json")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def mae_r2(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def metric_key(target, features, split, model) -> str:
    return f"{target}|{features}|{split}|{model}"


def update_metrics(record: dict, path: str = METRICS_JSON) -> str:
    """Merge one record into results/metrics.json keyed by
    (target, features, split, model). Returns the key used."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metrics = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            metrics = {}
    key = metric_key(record["target"], record["features"],
                     record["split"], record["model"])
    metrics[key] = record
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return key


def parity_plot(y_true, y_pred, target, features, split, model,
                unit="eV", out_dir: str = PLOTS_DIR) -> str:
    """Scatter of predicted vs true (test set) with a y=x line and the test
    MAE annotated. Returns the saved file path."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.15, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "r-", lw=1.2, label="y = x")
    ax.set_xlabel(f"True {target} ({unit})")
    ax.set_ylabel(f"Predicted {target} ({unit})")
    ax.set_title(f"{target} | {features} | {split} | {model} (test)")
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.05, 0.95,
            f"MAE = {mae:.4f} {unit}\n$R^2$ = {r2:.4f}\nn = {len(y_true)}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right")
    fig.tight_layout()

    fname = f"{target}_{features}_{split}_{model}.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
