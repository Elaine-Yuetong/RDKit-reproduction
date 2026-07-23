"""Prepare Phase 3 DFT structures in a PaiNN-ready plain-Python format.

This is a format-only pass. It intentionally does not import torch or
torch_geometric, does not train a model, and does not require a GPU.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from rdkit import Chem

from phase3.data import DATA_DIR, TARGET_COLUMNS, load_unique_labels

OUT_DIR = DATA_DIR / "painn_cache"
RECORDS_JSON_GZ = OUT_DIR / "painn_records.json.gz"
LABELS_CSV = OUT_DIR / "labels_painn.csv"
EXPECTED_AUDIT_ELEMENTS = {"B", "Br", "C", "Cl", "F", "N", "O", "P", "S", "Se", "Si"}


def resolve_xyz_path(relpath: object) -> Path:
    """Resolve representative_xyz_relpath without assuming more than relativity."""
    if not isinstance(relpath, str) or not relpath.strip():
        return Path("")
    path = Path(relpath)
    if path.is_absolute():
        return path
    return DATA_DIR / path


def parse_xyz(path: Path) -> tuple[list[int], list[list[float]], list[str], str]:
    """Parse an XYZ file into atomic numbers, coordinates, symbols, and first atom line."""
    if not path.exists():
        raise FileNotFoundError(f"missing xyz file: {path}")

    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("xyz file has fewer than 3 lines")

    try:
        expected_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"first line is not an atom count: {lines[0]!r}") from exc

    atom_lines = [line.strip() for line in lines[2:] if line.strip()]
    if len(atom_lines) < expected_atoms:
        raise ValueError(
            f"xyz file has {len(atom_lines)} atom lines, expected {expected_atoms}"
        )

    periodic = Chem.GetPeriodicTable()
    z: list[int] = []
    pos: list[list[float]] = []
    symbols: list[str] = []
    for line_no, line in enumerate(atom_lines[:expected_atoms], start=3):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"line {line_no} has fewer than 4 fields: {line!r}")
        symbol = parts[0]
        atomic_number = int(periodic.GetAtomicNumber(symbol))
        if atomic_number <= 0:
            raise ValueError(f"line {line_no} has unknown element symbol: {symbol!r}")
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError as exc:
            raise ValueError(f"line {line_no} has nonnumeric coordinates: {line!r}") from exc
        symbols.append(symbol)
        z.append(atomic_number)
        pos.append(xyz)

    if len(z) != len(pos):
        raise AssertionError("internal parse error: len(z) != len(pos)")
    return z, pos, symbols, atom_lines[0]


def build_records() -> tuple[list[dict], pd.DataFrame, dict]:
    df = load_unique_labels().reset_index(drop=True)
    required = {
        "canonical_smiles",
        "representative_xyz_relpath",
        "source_kind",
        *TARGET_COLUMNS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"load_unique_labels() missing required columns: {missing}")

    print("PaiNN data preparation")
    print("Resolution scheme: if representative_xyz_relpath is absolute, use it directly;")
    print(f"otherwise resolve as DATA_DIR / representative_xyz_relpath, where DATA_DIR={DATA_DIR}")
    print()

    records: list[dict] = []
    label_rows: list[dict] = []
    examples: list[dict] = []
    failures: list[dict] = []
    n_atoms_ok: list[int] = []
    unique_elements: set[str] = set()
    len_match_failures = 0

    for row_id, row in df.iterrows():
        relpath = row["representative_xyz_relpath"]
        resolved = resolve_xyz_path(relpath)
        base_record = {
            "row_id": int(row_id),
            "canonical_smiles": str(row["canonical_smiles"]),
            "esp_vmin_mean": float(row["esp_vmin_mean_kcal_per_mol"]),
            "zn_e_bind_mean": float(row["zn_e_bind_mean_kcal_per_mol"]),
            "source_kind": str(row["source_kind"]),
            "xyz_relpath": "" if not isinstance(relpath, str) else relpath,
            "xyz_path": str(resolved),
        }

        try:
            z, pos, symbols, first_atom_line = parse_xyz(resolved)
            parse_status = "ok"
            error = ""
            n_atoms = len(z)
            if len(z) != len(pos):
                len_match_failures += 1
            n_atoms_ok.append(n_atoms)
            unique_elements.update(symbols)
            if len(examples) < 3:
                examples.append(
                    {
                        "canonical_smiles": base_record["canonical_smiles"],
                        "xyz_path": str(resolved),
                        "n_atoms": n_atoms,
                        "first_atom_line": first_atom_line,
                    }
                )
        except Exception as exc:
            parse_status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            z = []
            pos = []
            n_atoms = 0
            failures.append(
                {
                    "row_id": int(row_id),
                    "canonical_smiles": base_record["canonical_smiles"],
                    "xyz_path": str(resolved),
                    "error": error,
                }
            )

        records.append(
            {
                **base_record,
                "z": z,
                "pos": pos,
                "n_atoms": int(n_atoms),
                "parse_status": parse_status,
                "error": error,
            }
        )
        label_rows.append(
            {
                "row_id": int(row_id),
                "smiles": base_record["canonical_smiles"],
                "canonical_smiles": base_record["canonical_smiles"],
                "esp_vmin_mean_kcal_per_mol": float(row["esp_vmin_mean_kcal_per_mol"]),
                "zn_e_bind_mean_kcal_per_mol": float(row["zn_e_bind_mean_kcal_per_mol"]),
                "source_kind": str(row["source_kind"]),
                "representative_xyz_relpath": base_record["xyz_relpath"],
                "parse_status": parse_status,
                "n_atoms": int(n_atoms),
            }
        )

    labels = pd.DataFrame(label_rows)
    diagnostics = {
        "n_total": int(len(records)),
        "n_parsed_ok": int(sum(record["parse_status"] == "ok" for record in records)),
        "n_missing_or_failed": int(len(failures)),
        "failures_first_20": failures[:20],
        "first_3_examples": examples,
        "n_atoms_min": int(min(n_atoms_ok)) if n_atoms_ok else None,
        "n_atoms_max": int(max(n_atoms_ok)) if n_atoms_ok else None,
        "n_atoms_mean": float(mean(n_atoms_ok)) if n_atoms_ok else None,
        "unique_elements": sorted(unique_elements),
        "expected_audit_elements": sorted(EXPECTED_AUDIT_ELEMENTS),
        "expected_audit_elements_present": sorted(EXPECTED_AUDIT_ELEMENTS.intersection(unique_elements)),
        "expected_audit_elements_missing": sorted(EXPECTED_AUDIT_ELEMENTS.difference(unique_elements)),
        "len_z_equals_len_pos_failures": int(len_match_failures),
    }
    return records, labels, diagnostics


def save_cache(records: list[dict], labels: pd.DataFrame, diagnostics: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "plain python lists for later torch_geometric Data construction",
        "note": "No torch objects are serialized in this cache.",
        "n_records": len(records),
        "targets": TARGET_COLUMNS,
        "diagnostics": diagnostics,
        "records": records,
    }
    with gzip.open(RECORDS_JSON_GZ, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    labels.to_csv(LABELS_CSV, index=False)


def print_report(diagnostics: dict) -> None:
    print("First 3 resolved examples:")
    for i, ex in enumerate(diagnostics["first_3_examples"], start=1):
        print(
            f"{i}. {ex['canonical_smiles']} -> {ex['xyz_path']} -> "
            f"n_atoms={ex['n_atoms']} -> first_atom_line={ex['first_atom_line']}"
        )

    print()
    print("PaiNN format sanity report")
    print(f"total_molecules={diagnostics['n_total']}")
    print(f"parsed_ok={diagnostics['n_parsed_ok']}")
    print(f"missing_or_failed={diagnostics['n_missing_or_failed']}")
    if diagnostics["n_missing_or_failed"]:
        print(f"first_failures={diagnostics['failures_first_20']}")
    print(
        "n_atoms_distribution="
        f"min {diagnostics['n_atoms_min']}, "
        f"max {diagnostics['n_atoms_max']}, "
        f"mean {diagnostics['n_atoms_mean']:.2f}"
    )
    print(f"unique_elements_found={diagnostics['unique_elements']}")
    print(f"expected_audit_elements={diagnostics['expected_audit_elements']}")
    print(f"expected_audit_elements_present={diagnostics['expected_audit_elements_present']}")
    print(f"expected_audit_elements_missing={diagnostics['expected_audit_elements_missing']}")
    print(f"len_z_equals_len_pos_failures={diagnostics['len_z_equals_len_pos_failures']}")
    print(f"saved_records={RECORDS_JSON_GZ}")
    print(f"saved_labels={LABELS_CSV}")


def main() -> None:
    records, labels, diagnostics = build_records()
    save_cache(records, labels, diagnostics)
    print_report(diagnostics)


if __name__ == "__main__":
    main()
