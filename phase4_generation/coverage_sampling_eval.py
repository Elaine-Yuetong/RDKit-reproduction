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
    DEFAULT_OUT_DIR,
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
) -> tuple[list[int], dict[str, Any]]:
    """Pick molecules from emptiest predicted ESP/Zn cells using only eyes outputs."""
    n_pick = min(n_pick, len(pool_records))
    if n_pick <= 0:
        return [], {"esp": [], "zn": []}
    pred_esp = _predict_eye_values(pool_records, eyes_esp, args, torch, Data, DataLoader, device)
    pred_zn = _predict_eye_values(pool_records, eyes_zn, args, torch, Data, DataLoader, device)
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
    diversity_idx: list[int],
    coverage_idx: list[int],
    grid: dict[str, Any],
    args,
) -> Path:
    import matplotlib.pyplot as plt

    pool_esp, pool_zn = _true_property_arrays(pool_records)
    div_esp, div_zn = _true_property_arrays([pool_records[int(i)] for i in diversity_idx])
    cov_esp, cov_zn = _true_property_arrays([pool_records[int(i)] for i in coverage_idx])
    x_edges = np.asarray(grid["bin_edges"][ESP_TARGET], dtype=np.float64)
    y_edges = np.asarray(grid["bin_edges"][ZN_TARGET], dtype=np.float64)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"coverage_sampling_{args.scenario}_budget{args.budget}.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, title, esp, zn, color in [
        (axes[0], "structural diversity picked", div_esp, div_zn, "tab:orange"),
        (axes[1], "eyes coverage picked", cov_esp, cov_zn, "tab:blue"),
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


def _load_grid(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing coverage map JSON: {path}")
    grid = json.loads(path.read_text())
    for target in [ESP_TARGET, ZN_TARGET]:
        if target not in grid.get("bin_edges", {}):
            raise ValueError(f"coverage grid missing bin edges for {target}")
    return grid


def _print_summary(metrics: dict[str, Any]) -> None:
    print("\nCoverage-sampling evaluation")
    print(f"scenario={metrics['scenario']} budget={metrics['budget']}")
    print(
        f"seed={metrics['n_seed']} pool={metrics['n_pool']} "
        f"seed_frac={metrics['seed_frac']} grid_bins={metrics['n_bins']}x{metrics['n_bins']}"
    )
    print()
    print(f"{'picker':22s} {'occupied':>10s} {'H_norm':>10s} {'ESP_range':>11s} {'Zn_range':>11s}")
    print("-" * 70)
    for name in ["structural_diversity", "eyes_coverage"]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrospective Phase 4 coverage-sampling eval.")
    parser.add_argument("--scenario", choices=["random", "source"], required=True)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seed_frac", type=float, default=0.3)
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


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 3)
        args.budget = min(args.budget, 30)

    torch, Data, DataLoader = import_torch_stack()
    set_seed(args.seed, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("model=DimeNetPlusPlus eyes; pool pickers never use ground-truth labels")

    grid = _load_grid(args.coverage_map)
    n_bins = int(grid["n_bins"])
    seed_records, pool_records = load_pool_and_seed(args.scenario, args, torch, Data)
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        seed_keep = rng.choice(len(seed_records), size=min(args.smoke_n, len(seed_records)), replace=False)
        pool_keep = rng.choice(len(pool_records), size=min(args.smoke_n, len(pool_records)), replace=False)
        seed_records = [seed_records[int(i)] for i in seed_keep]
        pool_records = [pool_records[int(i)] for i in pool_keep]
        print(f"SMOKE mode: seed={len(seed_records)} pool={len(pool_records)} budget={args.budget}")

    eyes_esp = train_eyes_on_seed(seed_records, "esp", args, torch, Data, DataLoader, device)
    eyes_zn = train_eyes_on_seed(seed_records, "zn", args, torch, Data, DataLoader, device)

    budget = min(args.budget, len(pool_records))
    diversity_idx = pick_by_diversity(pool_records, budget, seed=args.seed)
    coverage_idx, pool_predictions = pick_by_coverage(
        pool_records, eyes_esp, eyes_zn, budget, grid, args, torch, Data, DataLoader, device
    )

    diversity_metrics = evaluate(diversity_idx, pool_records, grid)
    coverage_metrics = evaluate(coverage_idx, pool_records, grid, pool_predictions)
    eyes_pool_metrics = coverage_metrics.pop("eyes_pool_metrics")
    fig_path = plot_sampling(pool_records, diversity_idx, coverage_idx, grid, args)

    metrics = {
        "scenario": args.scenario,
        "budget": int(budget),
        "seed_frac": float(args.seed_frac),
        "n_seed": int(len(seed_records)),
        "n_pool": int(len(pool_records)),
        "n_bins": int(n_bins),
        "coverage_map": str(args.coverage_map),
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
            "structural_diversity": diversity_metrics,
            "eyes_coverage": coverage_metrics,
        },
        "figure": str(fig_path),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = args.out_dir / f"coverage_sampling_{args.scenario}_budget{budget}.json"
    result_path = RESULTS_DIR / f"phase4_coverage_sampling_{args.scenario}_budget{budget}.json"
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    run_path.write_text(payload)
    result_path.write_text(payload)

    _print_summary(metrics)
    print(f"saved_metrics={result_path}")
    print(f"saved_run_copy={run_path}")
    print(f"saved_figure={fig_path}")


if __name__ == "__main__":
    main()
