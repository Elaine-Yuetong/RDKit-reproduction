"""Colab GPU SchNet training against the frozen Phase-1 split artifacts.

This script is intentionally not part of the local venv workflow. It expects
PyTorch, PyTorch Geometric, torch_cluster, and RDKit to be installed in Colab.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet

RDLogger.DisableLog("rdApp.*")

TARGET_INDEX = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smiles_sha256(smiles: list[str]) -> str:
    h = hashlib.sha256()
    h.update("\n".join(smiles).encode("utf-8"))
    return h.hexdigest()


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        raise ValueError(f"could not canonicalize SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_and_validate_splits(splits_dir: Path, split: str) -> tuple[dict, str, dict]:
    manifest_path = splits_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha256_file(manifest_path)

    rows = {}
    for part in ("train", "val", "test"):
        name = f"{split}_{part}.csv"
        path = splits_dir / name
        part_rows = read_split_csv(path)
        smiles = [r["canonical_smiles"] for r in part_rows]
        expected = manifest["files"][name]
        actual_hash = smiles_sha256(smiles)
        if len(part_rows) != expected["row_count"]:
            raise SystemExit(
                f"{name}: count mismatch {len(part_rows)} != "
                f"{expected['row_count']}")
        if actual_hash != expected["canonical_smiles_sha256"]:
            raise SystemExit(f"{name}: canonical_smiles sha256 mismatch")
        rows[part] = part_rows

    return rows, manifest_hash, manifest


def pyg_smiles(data) -> str:
    if not hasattr(data, "smiles"):
        raise AttributeError(
            "PyG QM9 item has no .smiles attribute; install a current "
            "torch_geometric build or adapt this script to the local QM9 API.")
    return str(data.smiles)


def build_pyg_index(dataset: QM9) -> dict[str, int]:
    mapping = {}
    for i, data in enumerate(dataset):
        smi = canonicalize(pyg_smiles(data))
        mapping.setdefault(smi, i)
    return mapping


def matched_indices(split_rows, pyg_by_smiles):
    smiles = [r["canonical_smiles"] for r in split_rows]
    idx = [pyg_by_smiles[s] for s in smiles if s in pyg_by_smiles]
    return idx, smiles


def subset_indices(indices: list[int], n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    idx = list(indices)
    rng.shuffle(idx)
    return idx[:min(n, len(idx))]


def target_tensor(batch, target_idx: int) -> torch.Tensor:
    if batch.y.dim() == 1:
        return batch.y.view(-1, len(TARGET_INDEX))[:, target_idx].view(-1)
    return batch.y[:, target_idx].view(-1)


def target_value(data, target_idx: int) -> torch.Tensor:
    return data.y.view(-1)[target_idx]


def evaluate(model, loader, target_idx, mean, std, device) -> dict[str, float]:
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.z, batch.pos, batch.batch).view(-1)
            pred_ev = pred * std + mean
            y_ev = target_tensor(batch, target_idx)
            ys.append(y_ev.detach().cpu())
            preds.append(pred_ev.detach().cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(preds).numpy()
    return {
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--split", choices=["random", "scaffold"], required=True)
    ap.add_argument("--target", choices=sorted(TARGET_INDEX), default="gap")
    ap.add_argument("--train_subset", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", default="phase2/pyg_qm9")
    ap.add_argument("--out_dir", default="phase2/runs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="fast end-to-end test: train_subset=2000, epochs=3")
    args = ap.parse_args()
    if args.smoke:
        args.train_subset = 2000
        args.epochs = 3

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not available; this script is intended for Colab GPU.")

    split_rows, manifest_hash, manifest = load_and_validate_splits(
        Path(args.splits_dir), args.split)

    dataset = QM9(root=args.data_root)
    pyg_by_smiles = build_pyg_index(dataset)

    train_idx, train_smiles = matched_indices(split_rows["train"], pyg_by_smiles)
    val_idx, val_smiles = matched_indices(split_rows["val"], pyg_by_smiles)
    test_idx, test_smiles = matched_indices(split_rows["test"], pyg_by_smiles)

    val_test_smiles = set(val_smiles) | set(test_smiles)
    train_idx = [
        i for i in train_idx
        if canonicalize(pyg_smiles(dataset[i])) not in val_test_smiles
    ]
    train_idx = subset_indices(train_idx, args.train_subset, args.seed)
    if not train_idx or not val_idx or not test_idx:
        raise SystemExit(
            "empty train/val/test intersection after matching PyG QM9 to "
            "the frozen split CSVs")

    coverage = {
        "pyg_qm9_size": int(len(dataset)),
        "frozen_train_count": int(len(train_smiles)),
        "frozen_val_count": int(len(val_smiles)),
        "frozen_test_count": int(len(test_smiles)),
        "matched_train_count_before_subset": int(len(matched_indices(split_rows["train"], pyg_by_smiles)[0])),
        "matched_train_count_used": int(len(train_idx)),
        "matched_val_count": int(len(val_idx)),
        "matched_test_count": int(len(test_idx)),
        "test_coverage_fraction": float(len(test_idx) / len(test_smiles)),
    }
    print(json.dumps({"coverage": coverage}, indent=2, sort_keys=True))

    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=args.batch_size)
    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size)

    target_idx = TARGET_INDEX[args.target]
    y_train = torch.stack([target_value(dataset[i], target_idx)
                           for i in train_idx])
    mean = y_train.mean().to(device)
    std = y_train.std(unbiased=False).clamp_min(1e-12).to(device)

    model = SchNet(hidden_channels=128, num_filters=128,
                   num_interactions=6, num_gaussians=50, cutoff=10.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"schnet_{args.target}_{args.split}_{len(train_idx)}"
    ckpt_path = out_dir / f"{run_name}.pt"
    state_path = out_dir / f"{run_name}_state.json"
    metrics_path = out_dir / f"{run_name}.json"

    best_val = float("inf")
    best_epoch = -1
    start_epoch = 1
    patience = 15
    stale = 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        mean = torch.tensor(float(ckpt["mean"]), device=device)
        std = torch.tensor(float(ckpt["std"]), device=device)
        start_epoch = int(ckpt.get("last_epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        if state_path.exists():
            state = json.loads(state_path.read_text())
            start_epoch = int(state.get("last_epoch", 0)) + 1
            best_val = float(state.get("best_val", best_val))
            best_epoch = int(state.get("best_epoch", best_epoch))
            stale = max(0, start_epoch - best_epoch - 1)
        print(f"resuming from {ckpt_path}: start_epoch={start_epoch} "
              f"best_val={best_val:.4f} best_epoch={best_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            y = target_tensor(batch, target_idx)
            y_norm = (y - mean) / std
            pred = model(batch.z, batch.pos, batch.batch).view(-1)
            loss = torch.nn.functional.l1_loss(pred, y_norm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * batch.num_graphs
            n_seen += batch.num_graphs

        val = evaluate(model, val_loader, target_idx, mean, std, device)
        print(f"epoch={epoch:03d} train_norm_mae={total_loss / n_seen:.4f} "
              f"val_mae_eV={val['mae']:.4f} val_r2={val['r2']:.4f}")

        if val["mae"] < best_val:
            best_val = val["mae"]
            best_epoch = epoch
            stale = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "mean": float(mean.item()),
                        "last_epoch": int(epoch),
                        "best_val": float(best_val),
                        "best_epoch": int(best_epoch),
                        "std": float(std.item()),
                        "args": vars(args)}, ckpt_path)
            state_path.write_text(json.dumps({
                "last_epoch": int(epoch),
                "best_val": float(best_val),
                "best_epoch": int(best_epoch),
            }, indent=2, sort_keys=True) + "\n")
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test = evaluate(model, test_loader, target_idx, mean, std, device)
    val = evaluate(model, val_loader, target_idx, mean, std, device)

    result = {
        "target": args.target,
        "split": args.split,
        "unit": "eV",
        "best_epoch": int(best_epoch),
        "val_mae": val["mae"],
        "val_r2": val["r2"],
        "test_mae": test["mae"],
        "test_r2": test["r2"],
        "coverage": coverage,
        "manifest_file_sha256": manifest_hash,
        "validated_manifest": manifest,
        "checkpoint": str(ckpt_path),
    }
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
