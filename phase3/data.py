"""Data loading helpers for the private Phase 3 DFT labels.

The raw/extracted data under data/dft_real/ are private mentor lab data and are
gitignored. This module loads the deduped, averaged label table and adds only
derived metadata needed for later leakage-aware splitting.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dft_real"
ALL_RECORDS_CSV = DATA_DIR / "labels_all_records.csv"
UNIQUE_LABELS_CSV = DATA_DIR / "labels_unique_canonical_smiles.csv"

TARGET_COLUMNS = [
    "esp_vmin_mean_kcal_per_mol",
    "zn_e_bind_mean_kcal_per_mol",
]


def _split_dataset_ids(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in re.split(r"[;,|]", value) if part.strip()]


def dataset_kind_lookup(all_records_csv: Path = ALL_RECORDS_CSV) -> dict[str, str]:
    """Return dataset_id -> dataset_kind from the record-level label file."""
    all_df = pd.read_csv(all_records_csv, usecols=["dataset_id", "dataset_kind"])
    pairs = all_df.dropna(subset=["dataset_id", "dataset_kind"]).drop_duplicates()

    lookup: dict[str, str] = {}
    for dataset_id, group in pairs.groupby("dataset_id"):
        kinds = sorted(set(group["dataset_kind"].astype(str)))
        lookup[str(dataset_id)] = kinds[0] if len(kinds) == 1 else "mixed"
    return lookup


def derive_source_kind(dataset_ids: object, lookup: dict[str, str]) -> str:
    """Map a unique-molecule dataset_ids cell to agent/baseline/mixed/unknown."""
    ids = _split_dataset_ids(dataset_ids)
    kinds = {lookup.get(dataset_id, "unknown") for dataset_id in ids}
    kinds.discard("")
    if not kinds:
        return "unknown"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def validate_canonical_smiles(df: pd.DataFrame) -> list[tuple[int, str]]:
    failures: list[tuple[int, str]] = []
    for idx, smi in df["canonical_smiles"].items():
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            failures.append((int(idx), str(smi)))
    return failures


def murcko_scaffold_smiles(smi: str) -> str:
    """Return the Murcko scaffold SMILES, using "" for acyclic molecules."""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
    except Exception as exc:  # pragma: no cover - guarded by validation
        raise ValueError(f"could not compute Murcko scaffold for {smi!r}") from exc
    return scaffold or ""


def load_unique_labels(
    unique_csv: Path = UNIQUE_LABELS_CSV,
    all_records_csv: Path = ALL_RECORDS_CSV,
) -> pd.DataFrame:
    """Load deduped Phase 3 labels with source_kind and Murcko scaffold columns.

    Targets are the averaged columns in TARGET_COLUMNS. SMILES are revalidated
    independently with RDKit; current audit shows zero failures, so any failure
    is treated as an integrity error rather than silently dropped.
    """
    df = pd.read_csv(unique_csv)
    required = {"canonical_smiles", "dataset_ids", *TARGET_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing required columns in {unique_csv}: {missing}")

    failures = validate_canonical_smiles(df)
    if failures:
        preview = failures[:10]
        raise ValueError(
            f"RDKit failed to parse {len(failures)} canonical_smiles; "
            f"first failures: {preview}"
        )

    lookup = dataset_kind_lookup(all_records_csv)
    out = df.copy()
    out["source_kind"] = [
        derive_source_kind(value, lookup) for value in out["dataset_ids"]
    ]
    out["murcko_scaffold"] = [
        murcko_scaffold_smiles(smi) for smi in out["canonical_smiles"]
    ]
    return out

