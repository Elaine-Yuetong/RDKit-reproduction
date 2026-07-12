"""Featurize the private Phase 3 DFT labels with the Phase 1 RDKit builders.

This command writes feature caches under data/dft_real/features/ only. That
directory is gitignored because the underlying labels are unpublished lab data.
No modeling or splitting happens here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase3.data import DATA_DIR, TARGET_COLUMNS, load_unique_labels
from src.featurize import FP_RADIUS, FP_SIZE, N_JOBS
from src.featurize import morgan_fingerprints, rdkit_descriptors

FEATURE_DIR = DATA_DIR / "features"
DESC_NPY = FEATURE_DIR / "desc.npy"
DESC_COLS_JSON = FEATURE_DIR / "desc_columns.json"
FP_NPY = FEATURE_DIR / "fp.npy"
META_JSON = FEATURE_DIR / "meta.json"


def _smiles_hash(smiles: list[str]) -> str:
    h = hashlib.sha256()
    h.update("\n".join(smiles).encode("utf-8"))
    return h.hexdigest()


def _target_stats(series: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(series, errors="coerce")
    return {
        "count": int(x.notna().sum()),
        "missing": int(x.isna().sum()),
        "min": float(x.min()),
        "median": float(x.median()),
        "mean": float(x.mean()),
        "max": float(x.max()),
        "std": float(x.std(ddof=1)),
    }


def _print_target_stats(df: pd.DataFrame) -> None:
    print("\nTarget stats:")
    for target in TARGET_COLUMNS:
        stats = _target_stats(df[target])
        print(
            f"  {target}: "
            f"count={stats['count']} missing={stats['missing']} "
            f"min={stats['min']:.6g} median={stats['median']:.6g} "
            f"mean={stats['mean']:.6g} max={stats['max']:.6g} "
            f"std={stats['std']:.6g}"
        )


def _write_cache(df: pd.DataFrame, desc: np.ndarray, names: list[str],
                 fp: np.ndarray) -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DESC_NPY, desc)
    np.save(FP_NPY, fp)
    DESC_COLS_JSON.write_text(json.dumps(names, indent=2) + "\n")

    meta = {
        "source_csv": str(DATA_DIR / "labels_unique_canonical_smiles.csv"),
        "n_rows": int(len(df)),
        "canonical_smiles_sha256": _smiles_hash(
            df["canonical_smiles"].astype(str).tolist()
        ),
        "target_columns": TARGET_COLUMNS,
        "fp_radius": FP_RADIUS,
        "fp_size": FP_SIZE,
        "descriptor_columns": names,
    }
    META_JSON.write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    df = load_unique_labels()
    smiles = df["canonical_smiles"].astype(str).tolist()

    print("Phase 3 real DFT featurization")
    print(f"data_dir={DATA_DIR}")
    print(f"n_molecules={len(df)}")
    print(f"n_jobs={N_JOBS}")

    desc, names = rdkit_descriptors(smiles, n_jobs=N_JOBS)
    fp = morgan_fingerprints(smiles, n_jobs=N_JOBS)

    assert desc.shape[0] == len(df), f"desc rows {desc.shape[0]} != {len(df)}"
    assert fp.shape[0] == len(df), f"fp rows {fp.shape[0]} != {len(df)}"

    _write_cache(df, desc, names, fp)

    mean_bits = fp.sum(axis=1, dtype=np.int64).mean()
    print("\nFeature cache written:")
    print(f"  feature_dir={FEATURE_DIR}")
    print(f"  descriptor_matrix_shape={desc.shape}")
    print(f"  descriptor_surviving_columns={len(names)}")
    print(f"  fingerprint_matrix_shape={fp.shape}")
    print(f"  fingerprint_dtype={fp.dtype}")
    print(f"  mean_bits_set_per_fingerprint={mean_bits:.2f}")
    print(f"  concat_feature_count={desc.shape[1] + fp.shape[1]}")

    print("\nReadiness summary:")
    print(f"  n_molecules={len(df)}")
    print(f"  n_descriptor_features={desc.shape[1]}")
    print(f"  n_fingerprint_features={fp.shape[1]}")
    print(f"  n_concat_features={desc.shape[1] + fp.shape[1]}")
    _print_target_stats(df)
    print("\nSource_kind counts:")
    print(df["source_kind"].value_counts(dropna=False).to_string())
    print("\nScaffolds:")
    print(f"  distinct_murcko_scaffolds={df['murcko_scaffold'].nunique(dropna=False)}")
    print(f"  empty_scaffold_count={(df['murcko_scaffold'] == '').sum()}")


if __name__ == "__main__":
    main()

