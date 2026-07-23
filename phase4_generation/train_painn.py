"""Train/evaluate a PaiNN-style 3D predictor on Phase 4 DFT structures.

This script is intended for a GPU box with PyTorch Geometric installed. It is
kept self-contained and consumes the plain-Python cache produced by:

    ./venv/bin/python -m phase4_generation.prepare_painn_data

Do not run this locally until torch_geometric and its compiled dependencies are
installed in the target GPU environment.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "dft_real" / "painn_cache" / "painn_records.json.gz"
DEFAULT_OUT_DIR = ROOT / "phase4_generation" / "painn_runs"
SEED = 42
QM9_COMPATIBLE_Z = {1, 6, 7, 8, 9}
TARGETS = {
    "esp": {
        "record_key": "esp_vmin_mean",
        "label": "ESP minimum",
        "unit": "kcal/mol",
        "xgb_reference": {
            "random_mae": 7.35,
            "random_r2": 0.82,
            "source_mae": 18.74,
            "source_r2": 0.18,
            "exotic_s_r2": -0.24,
        },
    },
    "zn": {
        "record_key": "zn_e_bind_mean",
        "label": "Zn binding",
        "unit": "kcal/mol",
        "xgb_reference": {
            "random_mae": 18.94,
            "random_r2": 0.25,
            "source_mae": 23.37,
            "source_r2": 0.09,
            "exotic_s_r2": -0.05,
        },
    },
}


def import_torch_stack():
    """Import torch/PyG only when this training script is actually executed."""
    import torch
    from torch_geometric.data import Data

    try:
        from torch_geometric.loader import DataLoader
    except ImportError:  # pragma: no cover - compatibility for older PyG.
        from torch_geometric.data import DataLoader

    return torch, Data, DataLoader


def load_records(cache_path: Path) -> list[dict[str, Any]]:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"missing PaiNN cache: {cache_path}. Run "
            "./venv/bin/python -m phase4_generation.prepare_painn_data first."
        )
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"cache has no records: {cache_path}")
    parsed = [record for record in records if record.get("parse_status") == "ok"]
    if len(parsed) != len(records):
        print(
            "WARNING: using parsed records only; "
            f"ok={len(parsed)} total={len(records)} failed={len(records) - len(parsed)}"
        )
    return parsed


def records_to_data(records: list[dict[str, Any]], target: str, torch, Data) -> list[Any]:
    target_key = TARGETS[target]["record_key"]
    dataset = []
    for record in records:
        z = torch.tensor(record["z"], dtype=torch.long)
        pos = torch.tensor(record["pos"], dtype=torch.float)
        if pos.ndim != 2 or pos.size(-1) != 3:
            raise ValueError(f"row_id={record['row_id']} has bad pos shape {tuple(pos.shape)}")
        if z.numel() != pos.size(0):
            raise ValueError(f"row_id={record['row_id']} len(z) != len(pos)")
        y = torch.tensor([float(record[target_key])], dtype=torch.float)
        data = Data(z=z, pos=pos, y=y)
        data.row_id = int(record["row_id"])
        data.source_kind = str(record["source_kind"])
        dataset.append(data)
    return dataset


def make_model(args, torch):
    """Build PaiNN if available; otherwise fall back loudly to SchNet."""
    from torch_geometric.nn.models import SchNet

    model_hparams = {
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "cutoff": args.cutoff,
        "num_rbf": args.num_rbf,
        "max_z": args.max_z,
    }

    try:
        from torch_geometric.nn.models import PaiNN
    except ImportError:
        PaiNN = None

    if PaiNN is not None:
        attempts = [
            {
                "hidden_channels": args.hidden_channels,
                "out_channels": 1,
                "num_layers": args.num_layers,
                "num_rbf": args.num_rbf,
                "cutoff": args.cutoff,
                "max_z": args.max_z,
            },
            {
                "hidden_channels": args.hidden_channels,
                "num_layers": args.num_layers,
                "num_rbf": args.num_rbf,
                "cutoff": args.cutoff,
                "max_z": args.max_z,
            },
            {
                "hidden_channels": args.hidden_channels,
                "out_channels": 1,
                "num_layers": args.num_layers,
                "cutoff": args.cutoff,
                "max_z": args.max_z,
            },
        ]
        errors = []
        for kwargs in attempts:
            try:
                model = PaiNN(**kwargs)
                print(f"model_class=PaiNN hparams={kwargs}")
                return model, "PaiNN", kwargs
            except TypeError as exc:
                errors.append(str(exc))
        print("WARNING: torch_geometric.nn.models.PaiNN exists but constructor failed.")
        print(f"WARNING: PaiNN constructor errors: {errors}")

    print("WARNING: PaiNN is unavailable; falling back to SchNet for this run.")
    schnet_kwargs = {
        "hidden_channels": args.hidden_channels,
        "num_filters": args.hidden_channels,
        "num_interactions": args.num_layers,
        "num_gaussians": args.num_rbf,
        "cutoff": args.cutoff,
    }
    model = SchNet(**schnet_kwargs)
    print(f"model_class=SchNet hparams={schnet_kwargs}")
    return model, "SchNet", schnet_kwargs


def set_seed(seed: int, torch) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kfold_indices(n_items: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    folds = np.array_split(perm, n_splits)
    out = []
    all_idx = np.arange(n_items)
    for fold in folds:
        test = np.sort(fold)
        train = np.setdiff1d(all_idx, test, assume_unique=False)
        out.append((train, test))
    return out


def train_val_split(train_idx: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(round(len(train_idx) * val_frac)))
    val_idx = np.sort(perm[:n_val])
    inner_train_idx = np.sort(perm[n_val:])
    if len(inner_train_idx) == 0:
        raise ValueError("empty inner training split")
    return inner_train_idx, val_idx


def subset(dataset: list[Any], indices: np.ndarray) -> list[Any]:
    return [dataset[int(i)] for i in indices]


def forward_model(model, batch) -> Any:
    out = model(batch.z, batch.pos, batch.batch)
    if isinstance(out, tuple):
        out = out[0]
    return out.view(-1)


def target_array(dataset: list[Any], indices: np.ndarray) -> np.ndarray:
    return np.asarray([float(dataset[int(i)].y.item()) for i in indices], dtype=np.float64)


def normalize_targets(dataset: list[Any], indices: np.ndarray) -> tuple[float, float]:
    values = target_array(dataset, indices)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("target standard deviation is zero or invalid")
    return mean, std


def mae_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float("nan") if denom == 0 else float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
    return {"mae": mae, "r2": r2}


def run_epoch(model, loader, optimizer, mean: float, std: float, device, torch) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred_norm = forward_model(model, batch)
        y_norm = (batch.y.view(-1) - mean) / std
        loss = torch.nn.functional.l1_loss(pred_norm, y_norm)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def predict(model, loader, mean: float, std: float, device, torch) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = forward_model(model, batch) * std + mean
            y_pred.extend(pred.detach().cpu().numpy().astype(float).tolist())
            y_true.extend(batch.y.view(-1).detach().cpu().numpy().astype(float).tolist())
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)


def group_masks(dataset: list[Any], test_idx: np.ndarray) -> dict[str, np.ndarray]:
    all_test = np.asarray(test_idx, dtype=int)
    qm9 = []
    exotic = []
    sulfur = []
    for idx in all_test:
        z_set = set(int(v) for v in dataset[int(idx)].z.tolist())
        if z_set.issubset(QM9_COMPATIBLE_Z):
            qm9.append(int(idx))
        else:
            exotic.append(int(idx))
        if 16 in z_set:
            sulfur.append(int(idx))
    return {
        "all_agent_test": all_test,
        "qm9_compatible": np.asarray(qm9, dtype=int),
        "exotic": np.asarray(exotic, dtype=int),
        "exotic_S_only": np.asarray(sulfur, dtype=int),
    }


def evaluate_indices(
    model,
    dataset: list[Any],
    indices: np.ndarray,
    mean: float,
    std: float,
    batch_size: int,
    device,
    torch,
    DataLoader,
) -> dict[str, float]:
    loader = DataLoader(subset(dataset, indices), batch_size=batch_size, shuffle=False)
    y_true, y_pred = predict(model, loader, mean, std, device, torch)
    return mae_r2(y_true, y_pred)


def train_one_split(
    args,
    dataset: list[Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    fold: int | str,
    torch,
    DataLoader,
    device,
) -> dict[str, Any]:
    inner_train_idx, val_idx = train_val_split(train_idx, args.val_frac, args.seed)
    target_mean, target_std = normalize_targets(dataset, inner_train_idx)
    model, model_class, model_hparams = make_model(args, torch)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        subset(dataset, inner_train_idx),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    run_name = f"{args.target}_{split_name}_fold{fold}_{model_class.lower()}"
    ckpt_path = args.out_dir / f"{run_name}_best.pt"
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, target_mean, target_std, device, torch)
        y_val, pred_val = predict(model, val_loader, target_mean, target_std, device, torch)
        val_mae = mae_r2(y_val, pred_val)["mae"]
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if epoch == 1 or epoch % args.log_every == 0 or patience_left == 0:
            print(
                f"{split_name} fold={fold} epoch={epoch:03d} "
                f"train_norm_mae={train_loss:.4f} val_mae={val_mae:.4f}"
            )
        if patience_left == 0:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint state")
    model.load_state_dict(best_state)
    test_metrics = evaluate_indices(
        model, dataset, test_idx, target_mean, target_std, args.batch_size, device, torch, DataLoader
    )
    checkpoint = {
        "model_state_dict": best_state,
        "model_class": model_class,
        "model_hparams": model_hparams,
        "target": args.target,
        "split_name": split_name,
        "fold": fold,
        "target_mean": target_mean,
        "target_std": target_std,
        "best_val_mae": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "train_idx": [int(i) for i in train_idx],
        "inner_train_idx": [int(i) for i in inner_train_idx],
        "val_idx": [int(i) for i in val_idx],
        "test_idx": [int(i) for i in test_idx],
    }
    torch.save(checkpoint, ckpt_path)
    print(
        f"{split_name} fold={fold} BEST epoch={best_epoch} "
        f"val_mae={best_val:.4f} test_mae={test_metrics['mae']:.4f} "
        f"test_r2={test_metrics['r2']:.4f} ckpt={ckpt_path}"
    )
    return {
        "split": split_name,
        "fold": fold,
        "n_train": int(len(train_idx)),
        "n_inner_train": int(len(inner_train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch),
        "best_val_mae": float(best_val),
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "checkpoint": str(ckpt_path),
        "model_class": model_class,
        "model_hparams": model_hparams,
        "model": model,
        "normalization": {"mean": float(target_mean), "std": float(target_std)},
    }


def source_indices(dataset: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    baseline = [i for i, data in enumerate(dataset) if data.source_kind == "baseline"]
    agent = [i for i, data in enumerate(dataset) if data.source_kind == "agent"]
    return np.asarray(baseline, dtype=int), np.asarray(agent, dtype=int)


def print_xgb_reference(target: str) -> None:
    ref = TARGETS[target]["xgb_reference"]
    print(
        "reference (XGBoost task1): "
        f"random MAE={ref['random_mae']:.2f}/R2={ref['random_r2']:.2f}; "
        f"source->agent MAE={ref['source_mae']:.2f}/R2={ref['source_r2']:.2f}; "
        f"exotic_S R2={ref['exotic_s_r2']:.2f}"
    )


def print_random_summary(records: list[dict[str, Any]], unit: str) -> dict[str, float]:
    maes = np.asarray([record["test_mae"] for record in records], dtype=np.float64)
    r2s = np.asarray([record["test_r2"] for record in records], dtype=np.float64)
    summary = {
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=1)),
        "r2_mean": float(r2s.mean()),
        "r2_std": float(r2s.std(ddof=1)),
    }
    print("\nRandom 5-fold CV")
    print(f"{'fold':>4s} {'MAE':>10s} {'R2':>10s}")
    print("-" * 28)
    for record in records:
        print(f"{str(record['fold']):>4s} {record['test_mae']:10.4f} {record['test_r2']:10.4f}")
    print(
        f"mean +/- std MAE={summary['mae_mean']:.4f} +/- {summary['mae_std']:.4f} {unit}; "
        f"R2={summary['r2_mean']:.4f} +/- {summary['r2_std']:.4f}"
    )
    return summary


def evaluate_source_breakdown(
    source_record: dict[str, Any],
    dataset: list[Any],
    test_idx: np.ndarray,
    args,
    torch,
    DataLoader,
    device,
) -> dict[str, Any]:
    model = source_record.pop("model")
    mean = source_record["normalization"]["mean"]
    std = source_record["normalization"]["std"]
    groups = group_masks(dataset, test_idx)
    out = {}
    for name, idx in groups.items():
        out[name] = {
            "n": int(len(idx)),
            **evaluate_indices(model, dataset, idx, mean, std, args.batch_size, device, torch, DataLoader),
        }
    return out


def print_source_table(groups: dict[str, Any], unit: str) -> None:
    print("\nMentor source split: train baseline -> test agent")
    print(f"{'test group':22s} {'n':>6s} {'MAE':>10s} {'R2':>10s}")
    print("-" * 54)
    for name in ["all_agent_test", "qm9_compatible", "exotic", "exotic_S_only"]:
        row = groups[name]
        print(f"{name:22s} {row['n']:6d} {row['mae']:10.4f} {row['r2']:10.4f}")
    print(f"unit={unit}")


def maybe_smoke_subset(dataset: list[Any], args) -> list[Any]:
    if not args.smoke:
        return dataset
    rng = np.random.default_rng(args.seed)
    n = min(args.smoke_n, len(dataset))
    keep = set(int(i) for i in rng.choice(len(dataset), size=n, replace=False))
    smoke = [data for i, data in enumerate(dataset) if i in keep]
    print(f"SMOKE mode: using {len(smoke)} molecules, epochs={args.epochs}")
    return smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PaiNN/PyG 3D model on Phase 4 DFT data.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_rbf", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--max_z", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="Fast GPU-box gate: small subset/few epochs.")
    parser.add_argument("--smoke_n", type=int, default=256)
    parser.add_argument("--random_folds", type=int, default=5)
    args = parser.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 3)

    torch, Data, DataLoader = import_torch_stack()
    set_seed(args.seed, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.cache)
    dataset = records_to_data(records, args.target, torch, Data)
    dataset = maybe_smoke_subset(dataset, args)
    target_cfg = TARGETS[args.target]

    print("Phase 4 PaiNN predictor training")
    print(f"target={args.target} ({target_cfg['label']}, {target_cfg['unit']})")
    print(f"cache={args.cache}")
    print(f"out_dir={args.out_dir}")
    print(f"device={device}")
    print(
        "requested_hparams="
        f"hidden_channels={args.hidden_channels}, num_layers={args.num_layers}, "
        f"num_rbf={args.num_rbf}, cutoff={args.cutoff}, max_z={args.max_z}"
    )
    print_xgb_reference(args.target)

    random_records = []
    for fold_num, (train_idx, test_idx) in enumerate(
        kfold_indices(len(dataset), args.random_folds, args.seed), start=1
    ):
        rec = train_one_split(
            args, dataset, train_idx, test_idx, "random_cv", fold_num, torch, DataLoader, device
        )
        rec.pop("model")
        random_records.append(rec)
    random_summary = print_random_summary(random_records, target_cfg["unit"])

    baseline_idx, agent_idx = source_indices(dataset)
    source_record = train_one_split(
        args, dataset, baseline_idx, agent_idx, "source_baseline_to_agent", "baseline_to_agent",
        torch, DataLoader, device
    )
    source_groups = evaluate_source_breakdown(
        source_record, dataset, agent_idx, args, torch, DataLoader, device
    )
    print_source_table(source_groups, target_cfg["unit"])

    result = {
        "target": args.target,
        "target_label": target_cfg["label"],
        "unit": target_cfg["unit"],
        "seed": args.seed,
        "cache": str(args.cache),
        "device": str(device),
        "random_5fold_cv": {"folds": random_records, "summary": random_summary},
        "source_baseline_to_agent": {**source_record, "groups": source_groups},
        "xgb_task1_reference": target_cfg["xgb_reference"],
    }
    metrics_path = args.out_dir / f"{args.target}_painn_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
