"""Export frozen Phase-1 QM9 splits as portable CSV artifacts.

The exported files are the reproducibility contract for Phase 2: downstream
models must evaluate on the same deduped molecules and split membership.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from src.data import SEED, TARGETS, load_qm9, make_splits

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLITS_DIR = ROOT / "data" / "splits"
EXPORT_COLUMNS = ["row_index", "mol_id", "canonical_smiles", *TARGETS]
EXPECTED_DEDUP_ROWS = 133798


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        raise ValueError(f"could not canonicalize SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def smiles_sha256(smiles: pd.Series) -> str:
    h = hashlib.sha256()
    h.update("\n".join(smiles.astype(str).tolist()).encode("utf-8"))
    return h.hexdigest()


def export_splits(out_dir: Path = SPLITS_DIR) -> dict:
    df = load_qm9(verbose=False)
    if len(df) != EXPECTED_DEDUP_ROWS:
        raise SystemExit(
            f"expected {EXPECTED_DEDUP_ROWS} deduped rows, got {len(df)}")

    df = df.copy()
    df["canonical_smiles"] = [canonicalize_smiles(s) for s in df["smiles"]]
    splits = make_splits(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "dedup_row_total": int(len(df)),
        "columns": EXPORT_COLUMNS,
        "files": {},
    }

    for split_name in ("random", "scaffold"):
        for part in ("train", "val", "test"):
            key = f"{split_name}_{part}"
            idx = splits[key]
            export = df.loc[idx, ["mol_id", "canonical_smiles", *TARGETS]].copy()
            export.insert(0, "row_index", idx)
            export = export[EXPORT_COLUMNS]

            path = out_dir / f"{key}.csv"
            export.to_csv(path, index=False)
            manifest["files"][path.name] = {
                "split": split_name,
                "part": part,
                "row_count": int(len(export)),
                "canonical_smiles_sha256": smiles_sha256(
                    export["canonical_smiles"]),
                "mol_id_sha256": smiles_sha256(export["mol_id"]),
            }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)
                             + "\n")
    return manifest


def main() -> None:
    manifest = export_splits()
    print("Exported frozen splits to data/splits/")
    print("\nCounts:")
    for name, meta in manifest["files"].items():
        print(f"  {name:24s} {meta['row_count']:7d}  "
              f"smiles_sha256={meta['canonical_smiles_sha256']}  "
              f"mol_id_sha256={meta['mol_id_sha256']}")
    print("\nManifest:")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
