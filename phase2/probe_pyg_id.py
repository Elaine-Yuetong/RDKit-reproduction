"""Probe PyG QM9 molecule identifier fields.

Colab-only diagnostic: imports torch_geometric and downloads/loads PyG QM9.
This does not train and does not implement matching logic. It only prints the
exact identifier-like fields for the first two PyG molecules, plus the mol_id
format from our frozen split CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch_geometric.datasets import QM9


def value_or_missing(data, attr: str):
    if hasattr(data, attr):
        return getattr(data, attr)
    try:
        return data[attr]
    except Exception:
        return "<missing>"


def keys_repr(data) -> str:
    try:
        keys_obj = data.keys()
    except Exception as exc:
        return f"<keys() failed: {type(exc).__name__}: {exc}>"
    return repr(keys_obj)


def print_candidate_attrs(data, idx: int) -> None:
    print(f"\n=== PyG QM9 dataset[{idx}] ===")
    print(f"data_repr={data!r}")
    print(f"data_keys_repr={keys_repr(data)}")

    name = value_or_missing(data, "name")
    print(f"data.name type={type(name)!r} value={name!r}")

    idx_value = value_or_missing(data, "idx")
    print(f"data.idx type={type(idx_value)!r} value={idx_value!r}")

    smiles = value_or_missing(data, "smiles")
    print(f"data.smiles type={type(smiles)!r} value={smiles!r}")

    z = value_or_missing(data, "z")
    if z == "<missing>":
        print(f"data.z type={type(z)!r} value={z!r}")
    else:
        try:
            z_len = len(z)
        except TypeError:
            z_len = "<no len>"
        print(f"data.z type={type(z)!r} len={z_len} value={z!r}")

    # Print any additional key containing common identifier words, without
    # assuming PyG exposes a stable schema across versions.
    try:
        key_list = list(data.keys())
    except Exception:
        key_list = []
    for key in key_list:
        key_str = str(key).lower()
        if key_str in {"name", "idx", "smiles", "z"}:
            continue
        if any(token in key_str for token in ("id", "name", "idx", "index", "gdb")):
            value = value_or_missing(data, str(key))
            print(f"data[{key!r}] type={type(value)!r} value={value!r}")


def print_split_mol_ids(splits_dir: Path) -> None:
    path = splits_dir / "random_test.csv"
    df = pd.read_csv(path, usecols=["mol_id"], nrows=3)
    print("\n=== Frozen split mol_id sample ===")
    print(f"split_csv={path}")
    print(f"mol_id dtype={df['mol_id'].dtype!r}")
    print(f"first_3_mol_id={df['mol_id'].tolist()!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe PyG QM9 id-like fields.")
    ap.add_argument("--data_root", default="phase2/pyg_qm9")
    ap.add_argument("--splits_dir", default="data/splits")
    args = ap.parse_args()

    dataset = QM9(root=args.data_root)
    print("=== PyG QM9 dataset probe ===")
    print(f"data_root={args.data_root}")
    print(f"dataset_type={type(dataset)!r}")
    print(f"dataset_len={len(dataset)}")

    print_candidate_attrs(dataset[0], 0)
    print_candidate_attrs(dataset[1], 1)
    print_split_mol_ids(Path(args.splits_dir))


if __name__ == "__main__":
    main()

