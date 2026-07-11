"""Diagnose PyG QM9 <-> frozen split key mismatches.

Colab-only diagnostic: imports PyTorch Geometric and loads QM9, but does not
train. The frozen split CSVs and manifest are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from torch_geometric.datasets import QM9

from phase2.train_schnet import mol_for_match, mol_key_with_status, pyg_smiles

RDLogger.DisableLog("rdApp.*")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def first_block(key: str | None) -> str | None:
    if not key:
        return None
    return key.split("-", 1)[0]


def key_for_smiles(smiles: str) -> str | None:
    key, _ = mol_key_with_status(smiles)
    return key


def canonical_after_match_path(smiles: str) -> str | None:
    mol, _ = mol_for_match(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def build_pyg_indexes(dataset: QM9):
    full = {}
    by_first = defaultdict(list)
    idx_to_key = {}
    idx_to_smiles = {}
    skipped = 0
    fallback_recovered = 0
    skipped_examples = []

    for i, data in enumerate(dataset):
        raw = pyg_smiles(data)
        key, used_fallback = mol_key_with_status(raw)
        if key is None:
            skipped += 1
            if len(skipped_examples) < 5:
                skipped_examples.append(raw)
            continue
        if used_fallback:
            fallback_recovered += 1
        full.setdefault(key, i)
        by_first[first_block(key)].append(i)
        idx_to_key[i] = key
        idx_to_smiles[i] = raw

    print("PyG index stats:")
    print(f"  pyg_size={len(dataset)}")
    print(f"  keyed={len(idx_to_key)}")
    print(f"  skipped={skipped}")
    print(f"  fallback_recovered={fallback_recovered}")
    print(f"  skipped_examples={skipped_examples}")
    return full, by_first, idx_to_key, idx_to_smiles


def print_first_20(rows, full, by_first, idx_to_key, idx_to_smiles, n):
    print(f"\nFirst {n} frozen random-test molecules:")
    print("our canonical_smiles | our InChIKey | first-block")
    for row in rows[:n]:
        smi = row["canonical_smiles"]
        key = key_for_smiles(smi)
        print(f"{smi} | {key} | {first_block(key)}")

    print(f"\nFirst {n} match diagnostics:")
    for row in rows[:n]:
        smi = row["canonical_smiles"]
        key = key_for_smiles(smi)
        block = first_block(key)
        matched = key in full
        print(f"\nOUR {smi}")
        print(f"  our_key={key}")
        print(f"  matched_full={matched}")
        if not matched and block in by_first:
            i = by_first[block][0]
            print("  first-block candidate:")
            print(f"    pyg_smiles={idx_to_smiles[i]}")
            print(f"    pyg_key={idx_to_key[i]}")
            print(f"    our_key={key}")
        elif not matched:
            print("  first-block candidate: NONE")


def summarize(rows, full, by_first):
    matched_full = 0
    matched_first_only = 0
    still_unmatched = 0
    examples = []

    for row in rows:
        smi = row["canonical_smiles"]
        key = key_for_smiles(smi)
        block = first_block(key)
        if key in full:
            matched_full += 1
        elif block in by_first:
            matched_first_only += 1
            if len(examples) < 5:
                examples.append((smi, key, by_first[block][0]))
        else:
            still_unmatched += 1
            if len(examples) < 5:
                examples.append((smi, key, None))

    print("\nFull random-test summary:")
    print(f"  test_rows={len(rows)}")
    print(f"  matched_by_full_InChIKey={matched_full}")
    print(f"  additionally_matched_by_first_block_only={matched_first_only}")
    print(f"  still_unmatched={still_unmatched}")
    print(f"  full_key_coverage={matched_full / len(rows):.6f}")
    print(f"  first_block_coverage={(matched_full + matched_first_only) / len(rows):.6f}")
    return examples


def print_unmatched_examples(examples, idx_to_key, idx_to_smiles):
    print("\nUnmatched/candidate canonical-SMILES comparison after same mol_for_match path:")
    for smi, key, pyg_idx in examples:
        our_can = canonical_after_match_path(smi)
        print(f"\nour_input={smi}")
        print(f"our_key={key}")
        print(f"our_after_match_path={our_can}")
        if pyg_idx is None:
            print("pyg_first_block_candidate=NONE")
            continue
        pyg_raw = idx_to_smiles[pyg_idx]
        pyg_can = canonical_after_match_path(pyg_raw)
        print(f"pyg_input={pyg_raw}")
        print(f"pyg_key={idx_to_key[pyg_idx]}")
        print(f"pyg_after_match_path={pyg_can}")
        print(f"plain_canonical_equal={our_can == pyg_can}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--data_root", default="phase2/pyg_qm9")
    ap.add_argument("--n_first", type=int, default=20)
    args = ap.parse_args()

    test_path = Path(args.splits_dir) / "random_test.csv"
    rows = read_csv(test_path)
    dataset = QM9(root=args.data_root)
    full, by_first, idx_to_key, idx_to_smiles = build_pyg_indexes(dataset)

    print_first_20(rows, full, by_first, idx_to_key, idx_to_smiles,
                   n=args.n_first)
    examples = summarize(rows, full, by_first)
    print_unmatched_examples(examples, idx_to_key, idx_to_smiles)


if __name__ == "__main__":
    main()
