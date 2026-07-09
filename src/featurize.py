"""Feature builders for the QM9 RDKit baseline (Phase 1).

Scientific source of truth: METHODS.md.
  - RDKit 2D descriptors via Descriptors.CalcMolDescriptors (drop NaN/inf/
    constant columns; expect ~180-210 to survive).
  - Morgan fingerprints via the MODERN rdFingerprintGenerator API only
    (radius=2, fpSize=2048), stored as uint8; expect ~30-60 mean bits set.
  - concat = descriptors (float32) + fingerprints cast to float32, built
    lazily from the cached desc/fp arrays — never recomputed from scratch.

Multiprocessing note (RDKit trap): Mol objects are NOT pickled across the
Pool. Workers receive SMILES strings and parse to Mol inside the worker
process; pickling Mols is the classic "no speedup" bug.

Row alignment: load_qm9 dropped 0 rows, so feature arrays align 1:1 with the
load_qm9 DataFrame index. Row counts are asserted == len(df) before saving.
"""

from __future__ import annotations

import hashlib
import json
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from src.data import CSV_PATH, load_qm9

RDLogger.DisableLog("rdApp.*")  # silence per-molecule warnings (also in workers)

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEATURE_DIR = os.path.join(ROOT, "data", "features")
DESC_NPY = os.path.join(FEATURE_DIR, "desc.npy")
DESC_COLS_JSON = os.path.join(FEATURE_DIR, "desc_columns.json")
FP_NPY = os.path.join(FEATURE_DIR, "fp.npy")
META_JSON = os.path.join(FEATURE_DIR, "meta.json")

FP_RADIUS = 2
FP_SIZE = 2048
N_JOBS = os.cpu_count()      # all cores (16 on this Mac)
CHUNKSIZE = 256


# --------------------------------------------------------------------------- #
# Worker globals + initializers (per-process state; nothing Mol-shaped pickled)
# --------------------------------------------------------------------------- #
_DESC_NAMES = None   # canonical, ordered descriptor name list
_GEN = None          # per-process Morgan generator


def _descriptor_names():
    """Canonical ordered descriptor names, from a reference molecule."""
    ref = Chem.MolFromSmiles("CCO")
    return list(Descriptors.CalcMolDescriptors(ref).keys())


def _init_desc():
    global _DESC_NAMES
    RDLogger.DisableLog("rdApp.*")
    _DESC_NAMES = _descriptor_names()


def _init_fp():
    global _GEN
    RDLogger.DisableLog("rdApp.*")
    _GEN = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_SIZE)


def _desc_worker(smi):
    """Parse SMILES -> Mol here (not pickled), return ordered descriptor
    array (float64) or None on any failure."""
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return None
    try:
        d = Descriptors.CalcMolDescriptors(mol)
    except Exception:
        return None
    return np.array([d.get(name, np.nan) for name in _DESC_NAMES],
                    dtype=np.float64)


def _fp_worker(smi):
    """Parse SMILES -> Mol here (not pickled), return uint8 fingerprint row
    or None on failure."""
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return None
    return _GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Feature builders (take a list of SMILES strings)
# --------------------------------------------------------------------------- #
def rdkit_descriptors(smiles, n_jobs=N_JOBS):
    """Build the RDKit 2D descriptor matrix in parallel.

    Returns (X: float32 [n, k], names: list[str]) after replacing +/-inf with
    NaN and dropping any column that contains a NaN or has zero variance.
    """
    names = _descriptor_names()
    n, m = len(smiles), len(names)

    with Pool(n_jobs, initializer=_init_desc) as pool:
        results = list(tqdm(
            pool.imap(_desc_worker, smiles, chunksize=CHUNKSIZE),
            total=n, desc="descriptors"))

    X = np.full((n, m), np.nan, dtype=np.float64)
    bad = []
    for i, r in enumerate(results):
        if r is None:
            bad.append(i)          # imap preserves order -> i is the row index
        else:
            X[i] = r
    if bad:
        preview = bad[:10]
        print(f"  WARN: {len(bad)} molecules failed descriptors; rows left "
              f"NaN. First indices: {preview}")

    # +/-inf -> NaN, then drop NaN-bearing and zero-variance columns.
    X[~np.isfinite(X)] = np.nan
    col_has_nan = np.isnan(X).any(axis=0)
    var = np.zeros(m)
    var[~col_has_nan] = X[:, ~col_has_nan].var(axis=0)
    keep = (~col_has_nan) & (var > 0)

    Xk = X[:, keep].astype(np.float32)
    kept_names = [nm for nm, k in zip(names, keep) if k]
    print(f"  descriptors: {m} computed -> {Xk.shape[1]} survive "
          f"(dropped {int(col_has_nan.sum())} NaN, "
          f"{int(((~col_has_nan) & (var <= 0)).sum())} zero-variance)")
    return Xk, kept_names


def morgan_fingerprints(smiles, n_jobs=N_JOBS):
    """Build the Morgan fingerprint matrix (uint8 [n, 2048]) in parallel via
    the modern rdFingerprintGenerator API."""
    n = len(smiles)
    X = np.zeros((n, FP_SIZE), dtype=np.uint8)

    with Pool(n_jobs, initializer=_init_fp) as pool:
        results = list(tqdm(
            pool.imap(_fp_worker, smiles, chunksize=CHUNKSIZE),
            total=n, desc="fingerprints"))

    bad = []
    for i, r in enumerate(results):
        if r is None:
            bad.append(i)          # row already all-zeros
        else:
            X[i] = r
    if bad:
        print(f"  WARN: {len(bad)} molecules failed fingerprints; rows left "
              f"all-zero. First indices: {bad[:10]}")
    return X


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def _smiles_hash(smiles):
    h = hashlib.sha256()
    h.update("\n".join(map(str, smiles)).encode("utf-8"))
    return h.hexdigest()


def _meta(df):
    return {
        "qm9_csv_path": CSV_PATH,
        "n_rows": int(len(df)),
        "smiles_sha256": _smiles_hash(df["smiles"].tolist()),
        "fp_radius": FP_RADIUS,
        "fp_size": FP_SIZE,
    }


def _cache_valid(df):
    if not all(os.path.exists(p)
               for p in (DESC_NPY, DESC_COLS_JSON, FP_NPY, META_JSON)):
        return False
    try:
        with open(META_JSON) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    cur = _meta(df)
    return (meta.get("smiles_sha256") == cur["smiles_sha256"]
            and meta.get("n_rows") == cur["n_rows"]
            and meta.get("fp_size") == cur["fp_size"]
            and meta.get("fp_radius") == cur["fp_radius"])


def build_and_cache(df=None, n_jobs=N_JOBS):
    """Build descriptor + fingerprint matrices and cache them to disk.
    Returns (Xd, names, Xf)."""
    if df is None:
        df = load_qm9()
    smiles = df["smiles"].tolist()
    n = len(df)

    Xd, names = rdkit_descriptors(smiles, n_jobs=n_jobs)
    Xf = morgan_fingerprints(smiles, n_jobs=n_jobs)

    # Addition #2: features must align 1:1 with the load_qm9 DataFrame index.
    assert Xd.shape[0] == n, f"desc rows {Xd.shape[0]} != df len {n}"
    assert Xf.shape[0] == n, f"fp rows {Xf.shape[0]} != df len {n}"

    os.makedirs(FEATURE_DIR, exist_ok=True)
    np.save(DESC_NPY, Xd)
    with open(DESC_COLS_JSON, "w") as f:
        json.dump(names, f)
    np.save(FP_NPY, Xf)
    with open(META_JSON, "w") as f:
        json.dump(_meta(df), f, indent=2)
    return Xd, names, Xf


def load_features(kind, df=None):
    """Return cached features, building+caching them first if the cache is
    missing or stale.

      kind="desc"   -> (X float32 [n, k], names list[str])
      kind="fp"     -> X uint8 [n, 2048]
      kind="concat" -> X float32 [n, k+2048]  (built lazily from desc + fp)
    """
    if df is None:
        df = load_qm9()

    if kind == "concat":
        Xd, _ = load_features("desc", df)
        Xf = load_features("fp", df)
        return np.hstack([Xd, Xf.astype(np.float32)])

    if not _cache_valid(df):
        build_and_cache(df)

    if kind == "desc":
        with open(DESC_COLS_JSON) as f:
            names = json.load(f)
        return np.load(DESC_NPY), names
    if kind == "fp":
        return np.load(FP_NPY)
    raise ValueError(f"unknown feature kind: {kind!r}")


def _cache_size_bytes():
    return sum(os.path.getsize(p)
               for p in (DESC_NPY, DESC_COLS_JSON, FP_NPY, META_JSON)
               if os.path.exists(p))


# --------------------------------------------------------------------------- #
# Main: build everything, cache, and print sanity checks
# --------------------------------------------------------------------------- #
def main():
    df = load_qm9()
    print(f"\nBuilding features for {len(df)} molecules on {N_JOBS} cores...\n")

    Xd, names, Xf = build_and_cache(df)

    print(f"\nDescriptor matrix: shape={Xd.shape}, dtype={Xd.dtype}, "
          f"surviving columns={len(names)}")
    print(f"Fingerprint matrix: shape={Xf.shape}, dtype={Xf.dtype}")

    mean_bits = Xf.sum(axis=1, dtype=np.int64).mean()
    print(f"Mean bits set per fingerprint: {mean_bits:.2f}  "
          f"(METHODS.md sanity band: ~30-60)")

    Xc = load_features("concat", df)
    print(f"Concat matrix: shape={Xc.shape}, dtype={Xc.dtype} "
          f"(lazy: {Xd.shape[1]} desc + {Xf.shape[1]} fp)")

    size_mb = _cache_size_bytes() / (1024 ** 2)
    print(f"\nCache in {os.path.relpath(FEATURE_DIR, ROOT)}/  "
          f"(desc.npy + desc_columns.json + fp.npy + meta.json)")
    for p in (DESC_NPY, FP_NPY, DESC_COLS_JSON, META_JSON):
        print(f"  {os.path.basename(p):20s} "
              f"{os.path.getsize(p) / (1024 ** 2):8.2f} MB")
    print(f"  {'TOTAL':20s} {size_mb:8.2f} MB")


if __name__ == "__main__":
    main()
