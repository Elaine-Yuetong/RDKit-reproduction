"""Retrospective coverage-aware sampling evaluation for Phase 4.

This script tests whether an "eyes-predict-then-fill-empty-property-cells"
policy covers true ESP/Zn property space more uniformly than structural
diversity selection, under the same fixed budget.

Fairness rules:
  1. Pickers never use pool ground-truth labels.
  2. Ground-truth labels are revealed only in evaluate().
  3. DimeNet++ eyes are trained only on the seed set, never on the pool.

Intended runtime: GPU box with torch_geometric installed. Do not run locally
unless the Phase 4 PyG environment is available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from phase4_generation.coverage_map import ESP_TARGET, RESULT_PATH, ZN_TARGET, normalized_entropy
from phase4_generation.train_painn import (
    DEFAULT_CACHE,
    SEED,
    TARGETS,
    forward_model,
    import_torch_stack,
    load_records,
    mae_r2,
    make_model,
    predict,
    records_to_data,
    run_epoch,
    set_seed,
    train_val_split,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "phase4_generation" / "figures"
DEFAULT_SAMPLING_OUT_DIR = ROOT / "phase4_generation" / "coverage_sampling_runs"
FP_SIZE = 2048
FP_RADIUS = 2


# ---------------------------------------------------------------------------
# SECTION 1 - load_pool_and_seed(scenario)


def _clone_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return shallow record copies so split bookkeeping cannot mutate cache data."""
    return [dict(record) for record in records]


def _geometry_filtered_records(args, torch, Data) -> list[dict[str, Any]]:
    records = load_records(args.cache)
    kept_data = records_to_data(records, "esp", torch, Data, args.min_atom_dist)
    kept_row_ids = {int(data.row_id) for data in kept_data}
    return [record for record in records if int(record["row_id"]) in kept_row_ids]


def load_pool_and_seed(scenario: str, args, torch, Data) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load private cached records and split them into seed and pool records.

    The returned records still contain labels for later evaluation, but picker
    functions must not read those target keys.
    """
    records = _geometry_filtered_records(args, torch, Data)
    if scenario == "random":
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(records))
        n_seed = max(1, int(round(len(records) * args.seed_frac)))
        seed_records = [records[int(i)] for i in order[:n_seed]]
        pool_records = [records[int(i)] for i in order[n_seed:]]
    elif scenario == "source":
        seed_records = [record for record in records if str(record["source_kind"]) == "baseline"]
        pool_records = [record for record in records if str(record["source_kind"]) == "agent"]
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    if not seed_records or not pool_records:
        raise ValueError(
            f"scenario={scenario} produced empty split: "
            f"seed={len(seed_records)} pool={len(pool_records)}"
        )
    print(
        f"load_pool_and_seed: scenario={scenario} seed={len(seed_records)} "
        f"pool={len(pool_records)} seed_frac={args.seed_frac}"
    )
    return _clone_records(seed_records), _clone_records(pool_records)


def load_pool_and_exact_random_seed(
    seed_size: int,
    split_seed: int,
    args,
    torch,
    Data,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a random seed/pool split with exactly seed_size seed molecules."""
    records = _geometry_filtered_records(args, torch, Data)
    if seed_size < 1:
        raise ValueError("seed_size must be >= 1")
    if seed_size >= len(records):
        raise ValueError(f"seed_size={seed_size} leaves no pool molecules from n={len(records)}")
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(records))
    seed_records = [records[int(i)] for i in order[:seed_size]]
    pool_records = [records[int(i)] for i in order[seed_size:]]
    print(
        f"load_pool_and_exact_random_seed: split_seed={split_seed} "
        f"seed_size={len(seed_records)} pool={len(pool_records)}"
    )
    return _clone_records(seed_records), _clone_records(pool_records)


# ---------------------------------------------------------------------------
# SECTION 2 - train_eyes_on_seed(seed_records, target, args)


def _target_values(records: list[dict[str, Any]], target: str) -> np.ndarray:
    key = TARGETS[target]["record_key"]
    return np.asarray([float(record[key]) for record in records], dtype=np.float64)


def train_eyes_on_seed(
    seed_records: list[dict[str, Any]],
    target: str,
    args,
    torch,
    Data,
    DataLoader,
    device,
) -> dict[str, Any]:
    """Train one DimeNet++ predictor on seed records only."""
    if len(seed_records) < 5:
        raise ValueError(f"not enough seed records to train {target}: n={len(seed_records)}")

    eye_args = SimpleNamespace(**vars(args))
    eye_args.model = "dimenetpp"
    eye_args.lr = args.eyes_lr
    eye_args.grad_clip = args.grad_clip

    dataset = records_to_data(seed_records, target, torch, Data, min_atom_dist=0.0)
    all_idx = np.arange(len(dataset), dtype=int)
    train_idx, val_idx = train_val_split(all_idx, args.val_frac, args.seed)
    train_values = _target_values([seed_records[int(i)] for i in train_idx], target)
    target_mean = float(train_values.mean())
    target_std = float(train_values.std(ddof=0))
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError(f"{target} seed target std is invalid: {target_std}")

    model, model_class, model_hparams = make_model(eye_args, torch)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=eye_args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader([dataset[int(i)] for i in train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([dataset[int(i)] for i in val_idx], batch_size=args.batch_size, shuffle=False)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            target_mean,
            target_std,
            device,
            torch,
            eye_args.grad_clip,
        )
        y_val, pred_val = predict(model, val_loader, target_mean, target_std, device, torch)
        val_metrics = mae_r2(y_val, pred_val)
        val_mae = val_metrics["mae"]
        if math.isfinite(val_mae) and val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            if not math.isfinite(val_mae):
                print(
                    f"WARNING: {target} eyes epoch={epoch:03d} non-finite val_mae; "
                    "skipping best update"
                )
            patience_left -= 1
        if epoch == 1 or epoch % args.log_every == 0 or patience_left == 0:
            print(
                f"eyes target={target} epoch={epoch:03d} "
                f"train_norm_mae={train_loss:.4f} val_mae={val_mae:.4f}"
            )
        if patience_left == 0:
            break

    if best_state is None:
        raise RuntimeError(f"{target} eyes never produced a finite validation checkpoint")
    model.load_state_dict(best_state)
    print(f"eyes target={target} BEST epoch={best_epoch} val_mae={best_val:.4f}")
    return {
        "target": target,
        "model": model,
        "model_class": model_class,
        "model_hparams": model_hparams,
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val),
        "n_seed_train": int(len(train_idx)),
        "n_seed_val": int(len(val_idx)),
    }


# ---------------------------------------------------------------------------
# SECTION 3 - pickers


def _fingerprints(pool_records: list[dict[str, Any]]):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
    fps = []
    bad = []
    for i, record in enumerate(pool_records):
        mol = Chem.MolFromSmiles(str(record["canonical_smiles"]))
        if mol is None:
            bad.append((i, int(record["row_id"]), str(record["canonical_smiles"])))
            fps.append(None)
        else:
            fps.append(gen.GetFingerprint(mol))
    if bad:
        raise ValueError(f"RDKit failed to parse pool SMILES for diversity picker: {bad[:10]}")
    return fps


def _simple_maxmin_pick(fps, n_pick: int) -> list[int]:
    selected = [0]
    min_dists = np.ones(len(fps), dtype=np.float64)
    min_dists[0] = -1.0
    while len(selected) < n_pick:
        last_fp = fps[selected[-1]]
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(last_fp, fps), dtype=np.float64)
        dists = 1.0 - sims
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0
        selected.append(int(np.argmax(min_dists)))
    return selected


def pick_by_diversity(pool_records: list[dict[str, Any]], n_pick: int, seed: int = SEED) -> list[int]:
    """Pick structurally diverse molecules using only Morgan fingerprints."""
    n_pick = min(n_pick, len(pool_records))
    if n_pick <= 0:
        return []
    fps = _fingerprints(pool_records)
    try:
        from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

        picker = MaxMinPicker()
        picks = list(picker.LazyBitVectorPick(fps, len(fps), n_pick, seed=int(seed)))
        if len(picks) == n_pick:
            return [int(i) for i in picks]
        print("WARNING: MaxMinPicker returned fewer picks than requested; falling back to simple max-min")
    except Exception as exc:
        print(f"WARNING: RDKit MaxMinPicker unavailable/failed ({type(exc).__name__}); using simple max-min")
    return _simple_maxmin_pick(fps, n_pick)


def pick_by_random(pool_records: list[dict[str, Any]], n_pick: int, rng: np.random.Generator) -> list[int]:
    """Pick a uniform random subset of pool indices without replacement."""
    n_pick = min(n_pick, len(pool_records))
    if n_pick <= 0:
        return []
    return [int(i) for i in rng.choice(len(pool_records), size=n_pick, replace=False)]


def _records_to_prediction_data(records: list[dict[str, Any]], torch, Data) -> list[Any]:
    dataset = []
    for record in records:
        z = torch.tensor(record["z"], dtype=torch.long)
        pos = torch.tensor(record["pos"], dtype=torch.float)
        data = Data(z=z, pos=pos, y=torch.tensor([0.0], dtype=torch.float))
        data.row_id = int(record["row_id"])
        data.source_kind = str(record["source_kind"])
        dataset.append(data)
    return dataset


def _predict_eye_values(
    pool_records: list[dict[str, Any]],
    eye: dict[str, Any],
    args,
    torch,
    Data,
    DataLoader,
    device,
) -> np.ndarray:
    model = eye["model"]
    model.eval()
    dataset = _records_to_prediction_data(pool_records, torch, Data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    preds = []
    dropped_row_ids = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = forward_model(model, batch) * eye["target_std"] + eye["target_mean"]
            row_ids = batch.row_id.detach().cpu().numpy().astype(int).tolist()
            pred_values = pred.detach().cpu().numpy().astype(float).tolist()
            finite = np.isfinite(pred_values)
            for ok, value, row_id in zip(finite, pred_values, row_ids):
                if ok:
                    preds.append(float(value))
                else:
                    preds.append(float("nan"))
                    if len(dropped_row_ids) < 20:
                        dropped_row_ids.append(int(row_id))
    if dropped_row_ids:
        print(
            f"WARNING: {eye['target']} eyes produced non-finite pool predictions; "
            f"bad row_ids up to 20={dropped_row_ids}"
        )
    return np.asarray(preds, dtype=np.float64)


def _cell_ids(x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    x_bin = np.searchsorted(x_edges, x, side="right") - 1
    y_bin = np.searchsorted(y_edges, y, side="right") - 1
    x_bin = np.clip(x_bin, 0, len(x_edges) - 2)
    y_bin = np.clip(y_bin, 0, len(y_edges) - 2)
    return np.stack([x_bin, y_bin], axis=1).astype(int)


def pick_by_coverage(
    pool_records: list[dict[str, Any]],
    eyes_esp: dict[str, Any],
    eyes_zn: dict[str, Any],
    n_pick: int,
    grid: dict[str, Any],
    args,
    torch,
    Data,
    DataLoader,
    device,
    pool_predictions: dict[str, Any] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Pick molecules from emptiest predicted ESP/Zn cells using only eyes outputs."""
    n_pick = min(n_pick, len(pool_records))
    if n_pick <= 0:
        return [], {"esp": [], "zn": []}
    if pool_predictions is None:
        pred_esp = _predict_eye_values(pool_records, eyes_esp, args, torch, Data, DataLoader, device)
        pred_zn = _predict_eye_values(pool_records, eyes_zn, args, torch, Data, DataLoader, device)
    else:
        pred_esp = np.asarray(pool_predictions["esp"], dtype=np.float64)
        pred_zn = np.asarray(pool_predictions["zn"], dtype=np.float64)
    finite = np.isfinite(pred_esp) & np.isfinite(pred_zn)
    if not np.any(finite):
        raise RuntimeError("coverage picker has no finite eyes predictions for the pool")

    x_edges = np.asarray(grid["bin_edges"][ESP_TARGET], dtype=np.float64)
    y_edges = np.asarray(grid["bin_edges"][ZN_TARGET], dtype=np.float64)
    cells = _cell_ids(pred_esp, pred_zn, x_edges, y_edges)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    by_cell: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for idx, ok in enumerate(finite):
        if not ok:
            continue
        cell = (int(cells[idx, 0]), int(cells[idx, 1]))
        dist_to_center = (
            (float(pred_esp[idx]) - float(x_centers[cell[0]])) ** 2
            + (float(pred_zn[idx]) - float(y_centers[cell[1]])) ** 2
        )
        by_cell.setdefault(cell, []).append((dist_to_center, int(idx)))
    for candidates in by_cell.values():
        candidates.sort(key=lambda item: (item[0], item[1]))

    selected: list[int] = []
    selected_counts = {cell: 0 for cell in by_cell}
    while len(selected) < n_pick and by_cell:
        best_cell = min(
            by_cell,
            key=lambda cell: (
                selected_counts[cell],
                -len(by_cell[cell]),
                cell[0],
                cell[1],
            ),
        )
        _, idx = by_cell[best_cell].pop(0)
        selected.append(idx)
        selected_counts[best_cell] += 1
        if not by_cell[best_cell]:
            del by_cell[best_cell]
    return selected, {
        "esp": pred_esp.tolist(),
        "zn": pred_zn.tolist(),
        "finite_prediction_count": int(np.count_nonzero(finite)),
        "nonfinite_prediction_count": int(len(pool_records) - np.count_nonzero(finite)),
    }


# ---------------------------------------------------------------------------
# SECTION 4 - evaluate(picked_idx, pool_records, grid)


def _true_property_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    esp = np.asarray([float(record["esp_vmin_mean"]) for record in records], dtype=np.float64)
    zn = np.asarray([float(record["zn_e_bind_mean"]) for record in records], dtype=np.float64)
    return esp, zn


def _cell_metrics(esp: np.ndarray, zn: np.ndarray, grid: dict[str, Any]) -> dict[str, Any]:
    x_edges = np.asarray(grid["bin_edges"][ESP_TARGET], dtype=np.float64)
    y_edges = np.asarray(grid["bin_edges"][ZN_TARGET], dtype=np.float64)
    counts, _, _ = np.histogram2d(esp, zn, bins=[x_edges, y_edges])
    counts = counts.astype(int)
    entropy, entropy_norm = normalized_entropy(counts, int((len(x_edges) - 1) * (len(y_edges) - 1)))
    return {
        "n": int(len(esp)),
        "occupied_cells": int(np.count_nonzero(counts)),
        "shannon_entropy_natural_log": float(entropy),
        "normalized_entropy": float(entropy_norm),
        "cell_counts": counts.tolist(),
        "property_range_spread": {
            "esp_min": float(np.min(esp)) if len(esp) else float("nan"),
            "esp_max": float(np.max(esp)) if len(esp) else float("nan"),
            "esp_range": float(np.max(esp) - np.min(esp)) if len(esp) else float("nan"),
            "zn_min": float(np.min(zn)) if len(zn) else float("nan"),
            "zn_max": float(np.max(zn)) if len(zn) else float("nan"),
            "zn_range": float(np.max(zn) - np.min(zn)) if len(zn) else float("nan"),
        },
    }


def _eyes_pool_metrics(pool_records: list[dict[str, Any]], pool_predictions: dict[str, Any]) -> dict[str, Any]:
    true_esp, true_zn = _true_property_arrays(pool_records)
    pred_esp = np.asarray(pool_predictions["esp"], dtype=np.float64)
    pred_zn = np.asarray(pool_predictions["zn"], dtype=np.float64)
    return {
        "esp": mae_r2(true_esp, pred_esp),
        "zn": mae_r2(true_zn, pred_zn),
    }


def predict_pool_with_eyes(
    pool_records: list[dict[str, Any]],
    eyes_esp: dict[str, Any],
    eyes_zn: dict[str, Any],
    args,
    torch,
    Data,
    DataLoader,
    device,
) -> dict[str, Any]:
    """Predict ESP/Zn for a pool using current eyes, without using labels."""
    return {
        "esp": _predict_eye_values(pool_records, eyes_esp, args, torch, Data, DataLoader, device).tolist(),
        "zn": _predict_eye_values(pool_records, eyes_zn, args, torch, Data, DataLoader, device).tolist(),
    }


def evaluate(
    picked_idx: list[int],
    pool_records: list[dict[str, Any]],
    grid: dict[str, Any],
    pool_predictions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reveal ground truth only here and score true property-space coverage."""
    picked_records = [pool_records[int(i)] for i in picked_idx]
    esp, zn = _true_property_arrays(picked_records)
    metrics = _cell_metrics(esp, zn, grid)
    metrics["picked_row_ids"] = [int(record["row_id"]) for record in picked_records]
    if pool_predictions is not None:
        metrics["eyes_pool_metrics"] = _eyes_pool_metrics(pool_records, pool_predictions)
    return metrics


def plot_sampling(
    pool_records: list[dict[str, Any]],
    random_idx: list[int],
    diversity_idx: list[int],
    coverage_idx: list[int],
    grid: dict[str, Any],
    args,
) -> Path:
    import matplotlib.pyplot as plt

    pool_esp, pool_zn = _true_property_arrays(pool_records)
    rand_esp, rand_zn = _true_property_arrays([pool_records[int(i)] for i in random_idx])
    div_esp, div_zn = _true_property_arrays([pool_records[int(i)] for i in diversity_idx])
    cov_esp, cov_zn = _true_property_arrays([pool_records[int(i)] for i in coverage_idx])
    x_edges = np.asarray(grid["bin_edges"][ESP_TARGET], dtype=np.float64)
    y_edges = np.asarray(grid["bin_edges"][ZN_TARGET], dtype=np.float64)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"coverage_sampling_{args.scenario}_budget{args.budget}.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, title, esp, zn, color in [
        (axes[0], "random picked", rand_esp, rand_zn, "tab:gray"),
        (axes[1], "structural diversity picked", div_esp, div_zn, "tab:orange"),
        (axes[2], "eyes coverage picked", cov_esp, cov_zn, "tab:blue"),
    ]:
        ax.scatter(pool_esp, pool_zn, s=8, c="0.85", alpha=0.35, linewidths=0, label="pool")
        ax.scatter(esp, zn, s=18, c=color, alpha=0.85, linewidths=0, label="picked")
        for edge in x_edges:
            ax.axvline(edge, color="0.9", lw=0.7, zorder=0)
        for edge in y_edges:
            ax.axhline(edge, color="0.9", lw=0.7, zorder=0)
        ax.set_title(title)
        ax.set_xlabel("ESP minimum (kcal/mol)")
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("Zn binding (kcal/mol)")
    fig.suptitle(f"Retrospective coverage sampling: {args.scenario}, budget={args.budget}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_iterative_rounds(
    original_pool_records: list[dict[str, Any]],
    picked_round_records: list[list[dict[str, Any]]],
    grid: dict[str, Any],
    args,
) -> Path:
    import matplotlib.pyplot as plt

    pool_esp, pool_zn = _true_property_arrays(original_pool_records)
    x_edges = np.asarray(grid["bin_edges"][ESP_TARGET], dtype=np.float64)
    y_edges = np.asarray(grid["bin_edges"][ZN_TARGET], dtype=np.float64)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"coverage_sampling_{args.scenario}_iterative.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pool_esp, pool_zn, s=8, c="0.88", alpha=0.35, linewidths=0, label="original pool")
    cmap = plt.get_cmap("viridis", max(1, len(picked_round_records)))
    for round_idx, records in enumerate(picked_round_records, start=1):
        if not records:
            continue
        esp, zn = _true_property_arrays(records)
        ax.scatter(
            esp,
            zn,
            s=26,
            color=cmap(round_idx - 1),
            alpha=0.9,
            linewidths=0,
            label=f"round {round_idx}",
        )
    for edge in x_edges:
        ax.axvline(edge, color="0.9", lw=0.7, zorder=0)
    for edge in y_edges:
        ax.axhline(edge, color="0.9", lw=0.7, zorder=0)
    ax.set_xlabel("ESP minimum (kcal/mol)")
    ax.set_ylabel("Zn binding (kcal/mol)")
    ax.set_title(
        f"Iterative coverage sampling: {args.scenario}, "
        f"{args.n_rounds}x{args.round_budget}"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sweep(metrics: dict[str, Any]) -> Path:
    import matplotlib.pyplot as plt

    seed_sizes = sorted([int(size) for size in metrics["aggregate"]], reverse=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "coverage_sampling_sweep.png"

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for picker, label, color in [
        ("random", "random", "tab:gray"),
        ("structural_diversity", "structural diversity", "tab:orange"),
        ("eyes_coverage", "eyes coverage", "tab:blue"),
    ]:
        means = [
            metrics["aggregate"][str(size)]["pickers"][picker]["normalized_entropy"]["mean"]
            for size in seed_sizes
        ]
        stds = [
            metrics["aggregate"][str(size)]["pickers"][picker]["normalized_entropy"]["std"]
            for size in seed_sizes
        ]
        ax.errorbar(seed_sizes, means, yerr=stds, marker="o", capsize=3, label=label, color=color)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("seed size (DFT-labeled molecules)")
    ax.set_ylabel("picked-set normalized entropy")
    ax.set_title(f"Coverage-sampling seed-size sweep, budget={metrics['budget']}")
    ax.legend(loc="best")
    ax.grid(True, which="both", color="0.9", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _load_grid(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing coverage map JSON: {path}")
    grid = json.loads(path.read_text())
    for target in [ESP_TARGET, ZN_TARGET]:
        if target not in grid.get("bin_edges", {}):
            raise ValueError(f"coverage grid missing bin edges for {target}")
    return grid


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return {"mean": float(finite.mean()), "std": std}


def _aggregate_seed_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    pickers = ["random", "structural_diversity", "eyes_coverage"]
    aggregate: dict[str, Any] = {"pickers": {}, "eyes_pool_r2": {}}
    for picker in pickers:
        rows = [result["pickers"][picker] for result in seed_results]
        aggregate["pickers"][picker] = {
            "normalized_entropy": _mean_std([row["normalized_entropy"] for row in rows]),
            "occupied_cells": _mean_std([row["occupied_cells"] for row in rows]),
            "esp_range": _mean_std([row["property_range_spread"]["esp_range"] for row in rows]),
            "zn_range": _mean_std([row["property_range_spread"]["zn_range"] for row in rows]),
        }
    aggregate["eyes_pool_r2"]["esp"] = _mean_std(
        [result["eyes_pool_metrics"]["esp"]["r2"] for result in seed_results]
    )
    aggregate["eyes_pool_r2"]["zn"] = _mean_std(
        [result["eyes_pool_metrics"]["zn"]["r2"] for result in seed_results]
    )
    return aggregate


def _aggregate_sweep_results(per_rep: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    seed_sizes = sorted({int(row["seed_size"]) for row in per_rep}, reverse=True)
    for seed_size in seed_sizes:
        rows = [row for row in per_rep if int(row["seed_size"]) == seed_size]
        aggregate[str(seed_size)] = {
            "n_reps": int(len(rows)),
            "eyes_pool_r2": {
                "esp": _mean_std([row["eyes_pool_metrics"]["esp"]["r2"] for row in rows]),
                "zn": _mean_std([row["eyes_pool_metrics"]["zn"]["r2"] for row in rows]),
            },
            "pickers": {},
        }
        for picker in ["random", "structural_diversity", "eyes_coverage"]:
            picker_rows = [row["pickers"][picker] for row in rows]
            aggregate[str(seed_size)]["pickers"][picker] = {
                "normalized_entropy": _mean_std(
                    [picker_row["normalized_entropy"] for picker_row in picker_rows]
                ),
                "occupied_cells": _mean_std(
                    [picker_row["occupied_cells"] for picker_row in picker_rows]
                ),
            }
        random_h = aggregate[str(seed_size)]["pickers"]["random"]["normalized_entropy"]
        coverage_h = aggregate[str(seed_size)]["pickers"]["eyes_coverage"]["normalized_entropy"]
        aggregate[str(seed_size)]["coverage_minus_random_entropy"] = {
            "mean": float(coverage_h["mean"] - random_h["mean"]),
            "std": float(math.sqrt(coverage_h["std"] ** 2 + random_h["std"] ** 2)),
        }
    return aggregate


def _fmt_mean_std(stats: dict[str, float], digits: int = 4) -> str:
    return f"{stats['mean']:.{digits}f}+/-{stats['std']:.{digits}f}"


def _print_single_summary(metrics: dict[str, Any]) -> None:
    print("\nCoverage-sampling evaluation")
    print(f"scenario={metrics['scenario']} seed={metrics['seed']} budget={metrics['budget']}")
    print(
        f"seed_set={metrics['n_seed']} pool={metrics['n_pool']} "
        f"seed_frac={metrics['seed_frac']} grid_bins={metrics['n_bins']}x{metrics['n_bins']}"
    )
    print()
    print(f"{'picker':22s} {'occupied':>10s} {'H_norm':>10s} {'ESP_range':>11s} {'Zn_range':>11s}")
    print("-" * 70)
    for name in ["random", "structural_diversity", "eyes_coverage"]:
        row = metrics["pickers"][name]
        spread = row["property_range_spread"]
        print(
            f"{name:22s} {row['occupied_cells']:10d} {row['normalized_entropy']:10.4f} "
            f"{spread['esp_range']:11.4f} {spread['zn_range']:11.4f}"
        )
    eyes = metrics["eyes_pool_metrics"]
    print()
    print(
        "eyes pool metrics (labels revealed only for evaluation): "
        f"ESP MAE={eyes['esp']['mae']:.4f}, R2={eyes['esp']['r2']:.4f}; "
        f"Zn MAE={eyes['zn']['mae']:.4f}, R2={eyes['zn']['r2']:.4f}"
    )


def _print_aggregate_summary(metrics: dict[str, Any]) -> None:
    aggregate = metrics["aggregate"]
    print("\nAggregate coverage-sampling summary")
    print(
        f"scenario={metrics['scenario']} budget={metrics['budget']} "
        f"n_seeds={metrics['n_seeds']} seeds={metrics['seeds']}"
    )
    print()
    print(f"{'picker':22s} {'H_norm(mean+/-std)':>22s} {'occupied(mean+/-std)':>24s}")
    print("-" * 72)
    for name in ["random", "structural_diversity", "eyes_coverage"]:
        row = aggregate["pickers"][name]
        print(
            f"{name:22s} {_fmt_mean_std(row['normalized_entropy']):>22s} "
            f"{_fmt_mean_std(row['occupied_cells'], digits=2):>24s}"
        )
    print()
    print(
        "eyes pool R2 mean+/-std: "
        f"ESP {_fmt_mean_std(aggregate['eyes_pool_r2']['esp'])}; "
        f"Zn {_fmt_mean_std(aggregate['eyes_pool_r2']['zn'])}"
    )


def _print_iterative_summary(metrics: dict[str, Any]) -> None:
    print("\nIterative coverage-sampling evaluation")
    print(
        f"scenario={metrics['scenario']} seed={metrics['seed']} "
        f"round_budget={metrics['round_budget']} n_rounds={metrics['n_rounds']}"
    )
    print()
    print(
        f"{'round':>5s} {'n_seed':>8s} {'ESP_R2':>10s} {'Zn_R2':>10s} "
        f"{'cum_n':>8s} {'cum_H_norm':>12s} {'cum_occupied':>13s}"
    )
    print("-" * 78)
    for row in metrics["rounds"]:
        print(
            f"{row['round']:5d} {row['n_seed_so_far']:8d} "
            f"{row['eyes_pool_r2']['esp']:10.4f} {row['eyes_pool_r2']['zn']:10.4f} "
            f"{row['cumulative_picked_count']:8d} "
            f"{row['cumulative_coverage']['normalized_entropy']:12.4f} "
            f"{row['cumulative_coverage']['occupied_cells']:13d}"
        )
    iterative = metrics["iterative_final"]["normalized_entropy"]
    single = metrics["single_shot_baseline"]["normalized_entropy"]
    print()
    print(
        "final comparison: "
        f"iterative cumulative H_norm={iterative:.4f} vs "
        f"single-shot H_norm={single:.4f}"
    )


def _print_sweep_summary(metrics: dict[str, Any]) -> None:
    print("\nSeed-size sweep summary")
    print(
        f"scenario=random budget={metrics['budget']} sweep_reps={metrics['sweep_reps']} "
        f"base_seed={metrics['base_seed']}"
    )
    print()
    print(
        f"{'seed_size':>9s} {'eyesESP_R2':>17s} {'random_H':>17s} "
        f"{'diversity_H':>17s} {'coverage_H':>17s} {'coverage-random':>17s}"
    )
    print("-" * 98)
    for seed_size in metrics["seed_sizes"]:
        row = metrics["aggregate"][str(seed_size)]
        print(
            f"{seed_size:9d} "
            f"{_fmt_mean_std(row['eyes_pool_r2']['esp']):>17s} "
            f"{_fmt_mean_std(row['pickers']['random']['normalized_entropy']):>17s} "
            f"{_fmt_mean_std(row['pickers']['structural_diversity']['normalized_entropy']):>17s} "
            f"{_fmt_mean_std(row['pickers']['eyes_coverage']['normalized_entropy']):>17s} "
            f"{_fmt_mean_std(row['coverage_minus_random_entropy']):>17s}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrospective Phase 4 coverage-sampling eval.")
    parser.add_argument("--scenario", choices=["random", "source"], default="random")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seed_frac", type=float, default=0.3)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--round_budget", type=int, default=50)
    parser.add_argument("--n_rounds", type=int, default=4)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep_sizes", default="1200,800,500,300,150,80,40")
    parser.add_argument("--sweep_reps", type=int, default=3)
    parser.add_argument("--coverage_map", type=Path, default=RESULT_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_SAMPLING_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eyes_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_rbf", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--min_atom_dist", type=float, default=0.1)
    parser.add_argument("--max_z", type=int, default=100)
    parser.add_argument("--max_num_neighbors", type=int, default=32)
    parser.add_argument("--dimenet_num_blocks", type=int, default=4)
    parser.add_argument("--dimenet_int_emb_size", type=int, default=64)
    parser.add_argument("--dimenet_basis_emb_size", type=int, default=8)
    parser.add_argument("--dimenet_out_emb_channels", type=int, default=256)
    parser.add_argument("--dimenet_num_spherical", type=int, default=7)
    parser.add_argument("--dimenet_num_radial", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="Tiny GPU smoke run.")
    parser.add_argument("--smoke_n", type=int, default=256)
    return parser


def run_one_seed(
    args,
    run_seed: int,
    grid: dict[str, Any],
    torch,
    Data,
    DataLoader,
    device,
    make_figure: bool,
) -> dict[str, Any]:
    """Run one full fair retrospective evaluation for a single RNG seed."""
    run_args = SimpleNamespace(**vars(args))
    run_args.seed = int(run_seed)
    set_seed(run_args.seed, torch)
    rng = np.random.default_rng(run_args.seed)
    n_bins = int(grid["n_bins"])

    print("\n" + "=" * 78)
    print(f"starting coverage-sampling seed={run_args.seed}")
    print("=" * 78)

    seed_records, pool_records = load_pool_and_seed(run_args.scenario, run_args, torch, Data)
    if run_args.smoke:
        seed_keep = rng.choice(len(seed_records), size=min(run_args.smoke_n, len(seed_records)), replace=False)
        pool_keep = rng.choice(len(pool_records), size=min(run_args.smoke_n, len(pool_records)), replace=False)
        seed_records = [seed_records[int(i)] for i in seed_keep]
        pool_records = [pool_records[int(i)] for i in pool_keep]
        print(f"SMOKE mode: seed={len(seed_records)} pool={len(pool_records)} budget={run_args.budget}")

    eyes_esp = train_eyes_on_seed(seed_records, "esp", run_args, torch, Data, DataLoader, device)
    eyes_zn = train_eyes_on_seed(seed_records, "zn", run_args, torch, Data, DataLoader, device)

    budget = min(run_args.budget, len(pool_records))
    random_idx = pick_by_random(pool_records, budget, rng)
    diversity_idx = pick_by_diversity(pool_records, budget, seed=run_args.seed)
    coverage_idx, pool_predictions = pick_by_coverage(
        pool_records, eyes_esp, eyes_zn, budget, grid, run_args, torch, Data, DataLoader, device
    )

    random_metrics = evaluate(random_idx, pool_records, grid)
    diversity_metrics = evaluate(diversity_idx, pool_records, grid)
    coverage_metrics = evaluate(coverage_idx, pool_records, grid, pool_predictions)
    eyes_pool_metrics = coverage_metrics.pop("eyes_pool_metrics")
    fig_path = (
        plot_sampling(pool_records, random_idx, diversity_idx, coverage_idx, grid, run_args)
        if make_figure
        else None
    )

    metrics = {
        "scenario": run_args.scenario,
        "seed": int(run_args.seed),
        "budget": int(budget),
        "seed_frac": float(run_args.seed_frac),
        "n_seed": int(len(seed_records)),
        "n_pool": int(len(pool_records)),
        "n_bins": int(n_bins),
        "coverage_map": str(run_args.coverage_map),
        "fairness": {
            "pickers_use_ground_truth": False,
            "ground_truth_revealed_only_in_evaluate": True,
            "eyes_trained_only_on_seed": True,
        },
        "eyes": {
            "esp": {k: v for k, v in eyes_esp.items() if k != "model"},
            "zn": {k: v for k, v in eyes_zn.items() if k != "model"},
        },
        "eyes_pool_metrics": eyes_pool_metrics,
        "pickers": {
            "random": random_metrics,
            "structural_diversity": diversity_metrics,
            "eyes_coverage": coverage_metrics,
        },
        "figure": str(fig_path) if fig_path is not None else None,
    }
    _print_single_summary(metrics)
    return metrics


def _remove_indices(records: list[dict[str, Any]], selected_idx: list[int]) -> list[dict[str, Any]]:
    selected = {int(i) for i in selected_idx}
    return [record for i, record in enumerate(records) if i not in selected]


def run_iterative(args, grid: dict[str, Any], torch, Data, DataLoader, device) -> dict[str, Any]:
    """Run active-learning-style rounds with reveal/add/retrain after each batch."""
    run_args = SimpleNamespace(**vars(args))
    run_args.seed = int(args.seed)
    set_seed(run_args.seed, torch)
    rng = np.random.default_rng(run_args.seed)
    n_bins = int(grid["n_bins"])

    if run_args.round_budget < 1:
        raise ValueError("--round_budget must be >= 1")
    if run_args.n_rounds < 1:
        raise ValueError("--n_rounds must be >= 1")

    seed_records, remaining_pool = load_pool_and_seed(run_args.scenario, run_args, torch, Data)
    if run_args.smoke:
        seed_keep = rng.choice(len(seed_records), size=min(run_args.smoke_n, len(seed_records)), replace=False)
        pool_keep = rng.choice(len(remaining_pool), size=min(run_args.smoke_n, len(remaining_pool)), replace=False)
        seed_records = [seed_records[int(i)] for i in seed_keep]
        remaining_pool = [remaining_pool[int(i)] for i in pool_keep]
        print(
            f"SMOKE iterative mode: seed={len(seed_records)} pool={len(remaining_pool)} "
            f"round_budget={run_args.round_budget} n_rounds={run_args.n_rounds}"
        )

    original_seed_count = len(seed_records)
    original_pool = _clone_records(remaining_pool)
    total_single_shot_budget = min(run_args.round_budget * run_args.n_rounds, len(original_pool))
    cumulative_picked: list[dict[str, Any]] = []
    picked_round_records: list[list[dict[str, Any]]] = []
    rounds: list[dict[str, Any]] = []
    single_shot_baseline = None

    for round_idx in range(1, run_args.n_rounds + 1):
        if not remaining_pool:
            print(f"stopping iterative loop early at round {round_idx}: remaining_pool is empty")
            break

        print("\n" + "=" * 78)
        print(
            f"iterative round={round_idx} seed_records={len(seed_records)} "
            f"remaining_pool={len(remaining_pool)}"
        )
        print("=" * 78)

        eyes_esp = train_eyes_on_seed(seed_records, "esp", run_args, torch, Data, DataLoader, device)
        eyes_zn = train_eyes_on_seed(seed_records, "zn", run_args, torch, Data, DataLoader, device)

        # Accuracy measurement uses labels only for reporting. These labels are
        # not fed back into training until after the current batch is picked.
        pool_predictions = predict_pool_with_eyes(
            remaining_pool, eyes_esp, eyes_zn, run_args, torch, Data, DataLoader, device
        )
        eyes_pool = _eyes_pool_metrics(remaining_pool, pool_predictions)

        if round_idx == 1:
            single_idx, single_predictions = pick_by_coverage(
                original_pool,
                eyes_esp,
                eyes_zn,
                total_single_shot_budget,
                grid,
                run_args,
                torch,
                Data,
                DataLoader,
                device,
                pool_predictions=pool_predictions if original_pool is remaining_pool else None,
            )
            single_shot_baseline = evaluate(single_idx, original_pool, grid, single_predictions)
            single_shot_baseline["budget"] = int(total_single_shot_budget)

        batch_budget = min(run_args.round_budget, len(remaining_pool))
        picked_idx, _ = pick_by_coverage(
            remaining_pool,
            eyes_esp,
            eyes_zn,
            batch_budget,
            grid,
            run_args,
            torch,
            Data,
            DataLoader,
            device,
            pool_predictions=pool_predictions,
        )

        # Reveal only the selected batch now: move it into the training seed for
        # the next round. Unpicked pool labels remain unused by training.
        batch_records = [remaining_pool[int(i)] for i in picked_idx]
        cumulative_picked.extend(batch_records)
        picked_round_records.append(batch_records)
        remaining_pool = _remove_indices(remaining_pool, picked_idx)
        seed_records.extend(batch_records)

        cum_esp, cum_zn = _true_property_arrays(cumulative_picked)
        cumulative_coverage = _cell_metrics(cum_esp, cum_zn, grid)
        rounds.append(
            {
                "round": int(round_idx),
                "n_seed_so_far": int(len(seed_records)),
                "n_remaining_pool": int(len(remaining_pool)),
                "eyes_pool_mae": {
                    "esp": float(eyes_pool["esp"]["mae"]),
                    "zn": float(eyes_pool["zn"]["mae"]),
                },
                "eyes_pool_r2": {
                    "esp": float(eyes_pool["esp"]["r2"]),
                    "zn": float(eyes_pool["zn"]["r2"]),
                },
                "picked_this_round": int(len(batch_records)),
                "picked_row_ids": [int(record["row_id"]) for record in batch_records],
                "cumulative_picked_count": int(len(cumulative_picked)),
                "cumulative_coverage": cumulative_coverage,
            }
        )

    if single_shot_baseline is None:
        raise RuntimeError("iterative run did not produce a single-shot baseline")
    fig_path = plot_iterative_rounds(original_pool, picked_round_records, grid, run_args)
    iterative_final = rounds[-1]["cumulative_coverage"] if rounds else _cell_metrics(
        np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), grid
    )

    metrics = {
        "mode": "iterative",
        "scenario": run_args.scenario,
        "seed": int(run_args.seed),
        "round_budget": int(run_args.round_budget),
        "n_rounds": int(run_args.n_rounds),
        "total_single_shot_budget": int(total_single_shot_budget),
        "seed_frac": float(run_args.seed_frac),
        "n_original_seed": int(original_seed_count),
        "n_original_pool": int(len(original_pool)),
        "n_final_seed": int(len(seed_records)),
        "n_final_remaining_pool": int(len(remaining_pool)),
        "n_bins": int(n_bins),
        "coverage_map": str(run_args.coverage_map),
        "fairness": {
            "pickers_use_ground_truth": False,
            "ground_truth_revealed_only_after_batch_picked": True,
            "eyes_trained_only_on_current_seed": True,
            "single_shot_trained_only_on_original_seed": True,
        },
        "rounds": rounds,
        "iterative_final": iterative_final,
        "single_shot_baseline": single_shot_baseline,
        "figure": str(fig_path),
    }
    _print_iterative_summary(metrics)
    return metrics


def _parse_sweep_sizes(value: str) -> list[int]:
    sizes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        size = int(part)
        if size < 1:
            raise ValueError(f"sweep seed size must be >= 1, got {size}")
        sizes.append(size)
    if not sizes:
        raise ValueError("--sweep_sizes produced an empty list")
    return sorted(dict.fromkeys(sizes), reverse=True)


def run_sweep(args, grid: dict[str, Any], torch, Data, DataLoader, device) -> dict[str, Any]:
    """Run exact-seed-size random-scenario sweep for smart sampling robustness."""
    if args.sweep_reps < 1:
        raise ValueError("--sweep_reps must be >= 1")
    seed_sizes = _parse_sweep_sizes(args.sweep_sizes)
    per_rep: list[dict[str, Any]] = []
    n_bins = int(grid["n_bins"])

    print("\nSeed-size sweep mode: forcing scenario=random")
    print(f"seed_sizes={seed_sizes} sweep_reps={args.sweep_reps} budget={args.budget}")

    for seed_size in seed_sizes:
        for rep in range(args.sweep_reps):
            run_seed = int(args.seed + rep)
            run_args = SimpleNamespace(**vars(args))
            run_args.scenario = "random"
            run_args.seed = run_seed
            set_seed(run_seed, torch)
            rng = np.random.default_rng(run_seed)

            print("\n" + "=" * 78)
            print(f"sweep seed_size={seed_size} rep={rep + 1}/{args.sweep_reps} split_seed={run_seed}")
            print("=" * 78)

            seed_records, pool_records = load_pool_and_exact_random_seed(
                seed_size, run_seed, run_args, torch, Data
            )
            eyes_esp = train_eyes_on_seed(seed_records, "esp", run_args, torch, Data, DataLoader, device)
            eyes_zn = train_eyes_on_seed(seed_records, "zn", run_args, torch, Data, DataLoader, device)

            pool_predictions = predict_pool_with_eyes(
                pool_records, eyes_esp, eyes_zn, run_args, torch, Data, DataLoader, device
            )
            eyes_pool = _eyes_pool_metrics(pool_records, pool_predictions)

            budget = min(run_args.budget, len(pool_records))
            random_idx = pick_by_random(pool_records, budget, rng)
            diversity_idx = pick_by_diversity(pool_records, budget, seed=run_seed)
            coverage_idx, _ = pick_by_coverage(
                pool_records,
                eyes_esp,
                eyes_zn,
                budget,
                grid,
                run_args,
                torch,
                Data,
                DataLoader,
                device,
                pool_predictions=pool_predictions,
            )

            random_metrics = evaluate(random_idx, pool_records, grid)
            diversity_metrics = evaluate(diversity_idx, pool_records, grid)
            coverage_metrics = evaluate(coverage_idx, pool_records, grid)
            per_rep.append(
                {
                    "seed_size": int(seed_size),
                    "rep": int(rep),
                    "split_seed": int(run_seed),
                    "budget": int(budget),
                    "n_seed": int(len(seed_records)),
                    "n_pool": int(len(pool_records)),
                    "eyes_pool_metrics": eyes_pool,
                    "pickers": {
                        "random": random_metrics,
                        "structural_diversity": diversity_metrics,
                        "eyes_coverage": coverage_metrics,
                    },
                }
            )

    aggregate = _aggregate_sweep_results(per_rep)
    metrics = {
        "mode": "seed_size_sweep",
        "scenario": "random",
        "budget": int(args.budget),
        "base_seed": int(args.seed),
        "seed_sizes": seed_sizes,
        "sweep_reps": int(args.sweep_reps),
        "n_bins": int(n_bins),
        "coverage_map": str(args.coverage_map),
        "fairness": {
            "pickers_use_ground_truth": False,
            "ground_truth_revealed_only_in_evaluate": True,
            "eyes_trained_only_on_seed": True,
        },
        "per_rep_results": per_rep,
        "aggregate": aggregate,
    }
    fig_path = plot_sweep(metrics)
    metrics["figure"] = str(fig_path)
    _print_sweep_summary(metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    if args.n_seeds < 1:
        raise ValueError("--n_seeds must be >= 1")
    if args.sweep and args.iterative:
        raise ValueError("--sweep and --iterative are mutually exclusive")
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 3)
        args.budget = min(args.budget, 30)
        args.round_budget = min(args.round_budget, 10)
        args.n_rounds = min(args.n_rounds, 2)
        args.sweep_sizes = ",".join(str(v) for v in _parse_sweep_sizes(args.sweep_sizes)[:2])
        args.sweep_reps = min(args.sweep_reps, 1)

    torch, Data, DataLoader = import_torch_stack()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("model=DimeNetPlusPlus eyes; pool pickers never use ground-truth labels")

    grid = _load_grid(args.coverage_map)
    if args.sweep:
        args.scenario = "random"
        metrics = run_sweep(args, grid, torch, Data, DataLoader, device)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        run_path = args.out_dir / "coverage_sampling_sweep.json"
        result_path = RESULTS_DIR / "phase4_coverage_sampling_sweep.json"
        payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        run_path.write_text(payload)
        result_path.write_text(payload)
        print(f"saved_metrics={result_path}")
        print(f"saved_run_copy={run_path}")
        print(f"saved_figure={metrics['figure']}")
        return

    if args.iterative:
        metrics = run_iterative(args, grid, torch, Data, DataLoader, device)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        run_path = args.out_dir / f"coverage_sampling_{args.scenario}_iterative.json"
        result_path = RESULTS_DIR / f"phase4_coverage_sampling_{args.scenario}_iterative.json"
        payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        run_path.write_text(payload)
        result_path.write_text(payload)
        print(f"saved_metrics={result_path}")
        print(f"saved_run_copy={run_path}")
        print(f"saved_figure={metrics['figure']}")
        return

    seeds = [int(args.seed + k) for k in range(args.n_seeds)]
    seed_results = [
        run_one_seed(args, run_seed, grid, torch, Data, DataLoader, device, make_figure=(i == 0))
        for i, run_seed in enumerate(seeds)
    ]
    aggregate = _aggregate_seed_results(seed_results)
    budget = int(seed_results[0]["budget"])

    if args.n_seeds == 1:
        metrics = {
            **seed_results[0],
            "n_seeds": 1,
            "seeds": seeds,
            "per_seed_results": seed_results,
            "aggregate": aggregate,
        }
        filename = f"coverage_sampling_{args.scenario}_budget{budget}.json"
        result_filename = f"phase4_coverage_sampling_{args.scenario}_budget{budget}.json"
    else:
        metrics = {
            "scenario": args.scenario,
            "budget": budget,
            "seed_frac": float(args.seed_frac),
            "n_seeds": int(args.n_seeds),
            "seeds": seeds,
            "n_bins": int(grid["n_bins"]),
            "coverage_map": str(args.coverage_map),
            "fairness": {
                "pickers_use_ground_truth": False,
                "ground_truth_revealed_only_in_evaluate": True,
                "eyes_trained_only_on_seed": True,
            },
            "per_seed_results": seed_results,
            "aggregate": aggregate,
            "figure": seed_results[0]["figure"],
        }
        filename = f"coverage_sampling_{args.scenario}_budget{budget}_seeds{args.n_seeds}.json"
        result_filename = f"phase4_coverage_sampling_{args.scenario}_budget{budget}_seeds{args.n_seeds}.json"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = args.out_dir / filename
    result_path = RESULTS_DIR / result_filename
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    run_path.write_text(payload)
    result_path.write_text(payload)

    _print_aggregate_summary(metrics)
    print(f"saved_metrics={result_path}")
    print(f"saved_run_copy={run_path}")
    if metrics.get("figure"):
        print(f"saved_figure={metrics['figure']}")


if __name__ == "__main__":
    main()
