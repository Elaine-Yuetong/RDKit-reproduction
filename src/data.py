"""QM9 data loading, cleaning, unit conversion, and splits (Phase 1).

Scientific source of truth: METHODS.md.
  - Row count after cleaning must land ~130-134k.
  - homo/lumo/gap are in Hartree in the raw CSV -> convert to eV (x27.2114).
  - gap in eV should be positive, roughly 0.5-12 eV.
  - Splits: random 80/10/10 (seed=42) AND Murcko scaffold 80/10/10.
    Build them ONCE here; never re-split per model downstream.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "data", "qm9.csv")

# NOTE: the bucket name in the original plan (deepchem-data, hyphenated) 404s.
# The live DeepChem bucket is `deepchemdata` (no hyphen).
QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

HARTREE_TO_EV = 27.2114
ENERGY_COLS = ["homo", "lumo", "gap"]        # in Hartree in the raw CSV
TARGETS = ["homo", "lumo", "gap", "mu", "alpha"]
SEED = 42

RDLogger.DisableLog("rdApp.*")  # single bad SMILES should not spam stderr


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_qm9(dest: str = CSV_PATH, url: str = QM9_URL) -> str:
    """Download the QM9 CSV to `dest` if not already present.

    Uses a streamed request with a timeout and a tqdm progress bar. Raises
    SystemExit with a clear message if the download is unreachable.
    """
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="download qm9.csv"
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
    except requests.RequestException as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        sys.exit(
            f"ERROR: could not download QM9 from {url}\n"
            f"  ({e})\n"
            f"  Check your connection or download the CSV manually to {dest}."
        )
    os.replace(tmp, dest)
    return dest


# --------------------------------------------------------------------------- #
# Load + clean + unit-convert
# --------------------------------------------------------------------------- #
def load_qm9(csv_path: str = CSV_PATH, verbose: bool = True) -> pd.DataFrame:
    """Load QM9, validate SMILES with RDKit, drop unparseable rows, and
    convert homo/lumo/gap from Hartree to eV.

    Returns a cleaned DataFrame with a fresh 0..n-1 RangeIndex. Adds a
    `canonical_smiles` column (used for the scaffold split).
    """
    download_qm9(csv_path)
    df = pd.read_csv(csv_path)

    if verbose:
        print(f"Raw CSV: {df.shape[0]} rows, {df.shape[1]} cols")
        print(f"Columns: {list(df.columns)}")

    # Fail loudly rather than guessing if the schema is not what we expect.
    required = set(["smiles"]) | set(TARGETS)
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"ERROR: QM9 CSV is missing expected columns {sorted(missing)}.\n"
            f"  Present columns: {list(df.columns)}\n"
            f"  Adapt src/data.py TARGETS/ENERGY_COLS to the real schema."
        )

    # Validate every SMILES; keep the RDKit-canonical form for scaffolds.
    canon = []
    bad = 0
    for smi in tqdm(df["smiles"].tolist(), desc="validate SMILES"):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            canon.append(None)
            bad += 1
        else:
            canon.append(Chem.MolToSmiles(mol))
    df["canonical_smiles"] = canon

    n_before = len(df)
    df = df[df["canonical_smiles"].notna()].reset_index(drop=True)
    n_parsed = len(df)

    # Drop duplicate canonical SMILES (keep first): the same molecule must not
    # straddle train/test, which would leak. QM9 has a handful (~87) of raw
    # SMILES that canonicalize to an already-seen structure.
    dup_mask = df["canonical_smiles"].duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)
    n_after = len(df)

    # Hartree -> eV, once.
    for col in ENERGY_COLS:
        df[col] = df[col] * HARTREE_TO_EV

    if verbose:
        print(f"Dropped {bad} unparseable SMILES "
              f"({n_before} -> {n_parsed} rows)")
        print(f"Dropped {n_dup} duplicate canonical SMILES "
              f"({n_parsed} -> {n_after} rows)")

    return df


# --------------------------------------------------------------------------- #
# Splits (built once, reused everywhere)
# --------------------------------------------------------------------------- #
def random_split(df: pd.DataFrame, seed: int = SEED,
                 fracs=(0.8, 0.1, 0.1)):
    """Random 80/10/10 split. Returns (train_idx, val_idx, test_idx) as
    numpy int arrays of positional indices into `df`."""
    n = len(df)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    train = perm[:n_train]
    val = perm[n_train:n_train + n_val]
    test = perm[n_train + n_val:]
    return train, val, test


def _murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES for a molecule (empty string on failure)."""
    try:
        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)
    except Exception:
        return ""


def scaffold_split(df: pd.DataFrame, fracs=(0.8, 0.1, 0.1),
                   smiles_col: str = "canonical_smiles"):
    """Deterministic Murcko-scaffold 80/10/10 split.

    Molecules sharing a scaffold stay in the same partition, so the test set
    contains unseen chemotypes (the honest generalization estimate per
    METHODS.md 2.3). Scaffold groups are filled largest-first into
    train -> val -> test. Returns (train_idx, val_idx, test_idx).
    """
    n = len(df)
    scaffolds = defaultdict(list)
    smis = df[smiles_col].tolist()
    for i, smi in enumerate(tqdm(smis, desc="scaffolds")):
        scaffolds[_murcko_scaffold(smi)].append(i)

    # Largest groups first; tie-break on scaffold string for determinism.
    groups = sorted(scaffolds.values(),
                    key=lambda idxs: (len(idxs), smis[idxs[0]]),
                    reverse=True)

    train_cutoff = int(fracs[0] * n)
    val_cutoff = int((fracs[0] + fracs[1]) * n)
    train, val, test = [], [], []
    for idxs in groups:
        # Cumulative cutoffs (DeepChem-style): a whole scaffold group stays
        # together; it spills to the next bucket only when it would overflow.
        if len(train) + len(idxs) > train_cutoff:
            if len(train) + len(val) + len(idxs) > val_cutoff:
                test.extend(idxs)
            else:
                val.extend(idxs)
        else:
            train.extend(idxs)
    return (np.array(train, dtype=int),
            np.array(val, dtype=int),
            np.array(test, dtype=int))


def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Build both splits once. Returns a dict of six positional-index arrays."""
    r_tr, r_va, r_te = random_split(df, seed=seed)
    s_tr, s_va, s_te = scaffold_split(df)
    return {
        "random_train": r_tr, "random_val": r_va, "random_test": r_te,
        "scaffold_train": s_tr, "scaffold_val": s_va, "scaffold_test": s_te,
    }


# --------------------------------------------------------------------------- #
# Main: verify against METHODS.md sanity bands
# --------------------------------------------------------------------------- #
def main():
    df = load_qm9()

    print(f"\nCleaned dataset shape: {df.shape}")

    print("\nFirst 5 rows [smiles, homo, lumo, gap] (energies in eV):")
    preview = df[["smiles", "homo", "lumo", "gap"]].head()
    print(preview.to_string(index=False))

    gap = df["gap"]
    print(f"\ngap (eV): min={gap.min():.4f} max={gap.max():.4f} "
          f"mean={gap.mean():.4f}  (expect positive, ~0.5-12 eV)")
    frac_in_band = ((gap >= 0.5) & (gap <= 12)).mean()
    print(f"gap fraction within 0.5-12 eV: {frac_in_band:.3%}")

    splits = make_splits(df)
    n = len(df)
    print("\nSix split partitions:")
    for name in ("random_train", "random_val", "random_test",
                 "scaffold_train", "scaffold_val", "scaffold_test"):
        idx = splits[name]
        print(f"  {name:16s} {len(idx):7d}  ({len(idx) / n:6.2%})")

    # Integrity: partitions within a split must be disjoint and cover all rows.
    for split in ("random", "scaffold"):
        parts = [splits[f"{split}_{p}"] for p in ("train", "val", "test")]
        total = sum(len(p) for p in parts)
        union = len(set(np.concatenate(parts)))
        assert total == n == union, (
            f"{split} split not a clean partition: "
            f"total={total} union={union} n={n}")
    print("\nSplit integrity OK: both splits are disjoint and cover all rows.")


if __name__ == "__main__":
    main()
