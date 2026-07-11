"""Inspection-only audit for unpublished in-house DFT labels.

This script prints a plain-text report and writes no files. The input data are
private mentor lab data under data/dft_real/ and must remain gitignored.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dft_real"
ALL_CSV = DATA_DIR / "labels_all_records.csv"
UNIQUE_CSV = DATA_DIR / "labels_unique_canonical_smiles.csv"
STRUCTURES_DIR = DATA_DIR / "structures"

TARGETS = ["esp_vmin_kcal_per_mol", "zn_e_bind_kcal_per_mol"]
UNIQUE_STD_COLS = {
    "esp_vmin_kcal_per_mol": "esp_vmin_std_kcal_per_mol",
    "zn_e_bind_kcal_per_mol": "zn_e_bind_std_kcal_per_mol",
}


def section(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print("=" * 78)


def describe_series(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return "count=0"
    return (
        f"count={len(x)} min={x.min():.6g} q25={x.quantile(0.25):.6g} "
        f"median={x.median():.6g} mean={x.mean():.6g} "
        f"q75={x.quantile(0.75):.6g} max={x.max():.6g} "
        f"std={x.std(ddof=1):.6g}"
    )


def validate_smiles(values: pd.Series) -> list[tuple[int, str]]:
    failures = []
    for idx, smi in values.items():
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            failures.append((idx, smi))
    return failures


def print_target_stats(df: pd.DataFrame) -> None:
    section("Target completeness, distributions, and |z|>4 outliers")
    for target in TARGETS:
        x = pd.to_numeric(df[target], errors="coerce")
        present = x.notna()
        print(f"\n{target}")
        print(f"  present={int(present.sum())} missing={int((~present).sum())}")
        print(f"  {describe_series(x)}")
        xp = x[present]
        if len(xp) >= 2 and xp.std(ddof=1) > 0:
            z = (xp - xp.mean()) / xp.std(ddof=1)
            out = z[z.abs() > 4].sort_values(key=lambda s: s.abs(),
                                             ascending=False)
        else:
            out = pd.Series(dtype=float)
        print(f"  outliers_abs_z_gt_4={len(out)}")
        if len(out):
            cols = ["record_id", "dataset_kind", "canonical_smiles", target]
            for idx, zval in out.head(20).items():
                row = df.loc[idx, cols]
                print(
                    "    "
                    f"z={zval:.3f} value={row[target]:.6g} "
                    f"kind={row['dataset_kind']} record={row['record_id']} "
                    f"smiles={row['canonical_smiles']}"
                )
            if len(out) > 20:
                print(f"    ... {len(out) - 20} more not shown")


def print_source_overlap(df: pd.DataFrame) -> None:
    section("Source breakdown and unique-molecule overlap")
    print("Record-level dataset_kind counts:")
    print(df["dataset_kind"].value_counts(dropna=False).to_string())

    by_mol = df.groupby("canonical_smiles")["dataset_kind"].agg(
        lambda s: set(s.dropna().astype(str)))
    both = by_mol.apply(lambda s: {"agent", "baseline"}.issubset(s))
    agent_only = by_mol.apply(lambda s: s == {"agent"})
    baseline_only = by_mol.apply(lambda s: s == {"baseline"})
    other = ~(both | agent_only | baseline_only)
    print("\nUnique canonical_smiles source overlap:")
    print(f"  both_agent_and_baseline={int(both.sum())}")
    print(f"  agent_only={int(agent_only.sum())}")
    print(f"  baseline_only={int(baseline_only.sum())}")
    print(f"  other_or_missing_kind={int(other.sum())}")

    print("\nUnique molecules by dataset_kind membership detail:")
    membership_counts = Counter(tuple(sorted(v)) for v in by_mol)
    for kinds, count in sorted(membership_counts.items(),
                              key=lambda kv: (-kv[1], kv[0])):
        print(f"  {kinds}: {count}")


def print_duplicate_label_spread(unique: pd.DataFrame) -> None:
    section("Duplicate-label spread across repeated canonical SMILES")
    dup = unique[unique["n_records"] > 1].copy()
    print(f"unique_molecules_with_n_records_gt_1={len(dup)}")
    if dup.empty:
        return
    print("n_records distribution among duplicates:")
    print(dup["n_records"].value_counts().sort_index().to_string())
    for target, std_col in UNIQUE_STD_COLS.items():
        std = pd.to_numeric(dup[std_col], errors="coerce")
        print(f"\n{target} duplicate std column: {std_col}")
        print(f"  {describe_series(std)}")
        print(f"  nonzero_std_count={int((std.fillna(0) > 0).sum())}")


def scaffold_smiles(smi: str) -> str | None:
    try:
        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
    except Exception:
        return None


def print_scaffolds(unique: pd.DataFrame) -> None:
    section("Murcko scaffold count")
    scaffolds = [scaffold_smiles(s) for s in unique["canonical_smiles"]]
    failed = sum(s is None for s in scaffolds)
    valid = [s for s in scaffolds if s is not None]
    counts = Counter(valid)
    print(f"unique_molecules={len(unique)}")
    print(f"scaffold_failures={failed}")
    print(f"distinct_murcko_scaffolds={len(counts)}")
    if counts:
        empty = counts.get("", 0)
        print(f"empty_scaffold_count={empty}")
        print("top_20_scaffolds:")
        for scaffold, count in counts.most_common(20):
            label = scaffold if scaffold else "<empty>"
            print(f"  {count:4d}  {label}")


def main() -> None:
    all_df = pd.read_csv(ALL_CSV)
    unique = pd.read_csv(UNIQUE_CSV)

    section("Phase 3 DFT intake audit")
    print(f"data_dir={DATA_DIR}")
    print(f"labels_all_records={ALL_CSV}")
    print(f"labels_unique_canonical_smiles={UNIQUE_CSV}")
    print(f"xyz_file_count={len(list(STRUCTURES_DIR.rglob('*.xyz'))) if STRUCTURES_DIR.exists() else 0}")

    section("Row counts and deduplication")
    all_unique_count = all_df["canonical_smiles"].nunique(dropna=True)
    print(f"all_records_rows={len(all_df)}")
    print(f"unique_canonical_smiles_in_all_records={all_unique_count}")
    print(f"unique_rows_file_rows={len(unique)}")
    print(f"dedup_confirmed={len(all_df)} -> {len(unique)}")
    print(f"unique_counts_agree={all_unique_count == len(unique)}")

    print_target_stats(all_df)

    section("Independent RDKit SMILES validation")
    smiles_fail = validate_smiles(all_df["smiles"])
    canon_fail = validate_smiles(all_df["canonical_smiles"])
    unique_fail = validate_smiles(unique["canonical_smiles"])
    print(f"all_records smiles failures={len(smiles_fail)}")
    print(f"all_records canonical_smiles failures={len(canon_fail)}")
    print(f"unique canonical_smiles failures={len(unique_fail)}")
    for label, failures in (
        ("all_records smiles", smiles_fail),
        ("all_records canonical_smiles", canon_fail),
        ("unique canonical_smiles", unique_fail),
    ):
        if failures:
            print(f"  first failures for {label}:")
            for idx, smi in failures[:20]:
                print(f"    index={idx} smiles={smi!r}")

    section("Molecular size distributions")
    print(f"heavy_atom_count: {describe_series(unique['heavy_atom_count'])}")
    print(f"mol_weight: {describe_series(unique['mol_weight'])}")
    print("\nheavy_atom_count value counts:")
    print(unique["heavy_atom_count"].value_counts().sort_index().to_string())

    print_source_overlap(all_df)
    print_duplicate_label_spread(unique)
    print_scaffolds(unique)


if __name__ == "__main__":
    main()
