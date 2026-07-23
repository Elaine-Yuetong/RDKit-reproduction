"""Property-space coverage map for Phase 4 generation.

Analysis only: no model training and no GPU. This reproduces the mentor-style
coverage comparison by binning the two Phase 3 DFT targets on a shared
equal-width grid.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from phase3.data import ALL_RECORDS_CSV, TARGET_COLUMNS, load_unique_labels
from phase3.train_eval import RESULTS_DIR

ESP_TARGET = "esp_vmin_mean_kcal_per_mol"
ZN_TARGET = "zn_e_bind_mean_kcal_per_mol"
SOURCE_GROUPS = [
    "agent",
    "PubChem random HT",
    "ECFP max-min",
    "descriptor-farthest",
]
RESULT_PATH = RESULTS_DIR / "phase4_coverage_map.json"


def source_group_for_dataset_id(dataset_id: str) -> str | None:
    """Map record-level dataset_id to the four mentor coverage categories."""
    if dataset_id.startswith("pubchem_random_ht1000"):
        return "PubChem random HT"
    if dataset_id.startswith("ecfp_maxmin_ht600"):
        return "ECFP max-min"
    if dataset_id.startswith("descriptor_farthest"):
        return "descriptor-farthest"
    if dataset_id.startswith(
        (
            "latest_40run_snapshot",
            "replicate_map15_native_deepseek",
            "current40_fig27_sparse_card",
            "old44_adaptive_council",
        )
    ):
        return "agent"
    return None


def build_source_group_lookup(records_csv: Path = ALL_RECORDS_CSV) -> dict[str, set[str]]:
    records = pd.read_csv(records_csv, usecols=["canonical_smiles", "dataset_id"])
    lookup: dict[str, set[str]] = defaultdict(set)
    unknown_dataset_ids = set()
    for row in records.itertuples(index=False):
        group = source_group_for_dataset_id(str(row.dataset_id))
        if group is None:
            unknown_dataset_ids.add(str(row.dataset_id))
            continue
        lookup[str(row.canonical_smiles)].add(group)
    if unknown_dataset_ids:
        raise ValueError(f"unmapped dataset_id values: {sorted(unknown_dataset_ids)}")
    return dict(lookup)


def attach_source_groups(df: pd.DataFrame) -> tuple[list[set[str]], dict]:
    lookup = build_source_group_lookup()
    source_sets: list[set[str]] = []
    missing_smiles = []
    for idx, smiles in enumerate(df["canonical_smiles"].astype(str)):
        groups = lookup.get(smiles, set())
        if not groups:
            missing_smiles.append((idx, smiles))
        source_sets.append(set(groups))

    if missing_smiles:
        raise ValueError(
            f"{len(missing_smiles)} unique molecules missing record-level source; "
            f"first examples: {missing_smiles[:10]}"
        )

    multi = [
        {"row_index": int(i), "canonical_smiles": str(df.loc[i, "canonical_smiles"]), "groups": sorted(groups)}
        for i, groups in enumerate(source_sets)
        if len(groups) > 1
    ]
    diagnostics = {
        "unique_molecules": int(len(df)),
        "multi_source_molecule_count": int(len(multi)),
        "multi_source_examples": multi[:20],
    }
    return source_sets, diagnostics


def make_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("target values contain NaN or Inf")
    if vmin == vmax:
        raise ValueError("cannot build equal-width bins for constant target")
    return np.linspace(vmin, vmax, n_bins + 1, dtype=np.float64)


def cell_counts(x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    return counts.astype(int)


def normalized_entropy(counts: np.ndarray, n_total_cells: int) -> tuple[float, float]:
    occupied = counts[counts > 0].astype(np.float64)
    if occupied.size == 0:
        return 0.0, 0.0
    probs = occupied / occupied.sum()
    entropy = float(-np.sum(probs * np.log(probs)))
    normalized = float(entropy / np.log(n_total_cells)) if n_total_cells > 1 else 0.0
    return entropy, normalized


def metrics_for_mask(
    df: pd.DataFrame,
    mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    n_total_cells: int,
) -> dict:
    x = df.loc[mask, ESP_TARGET].to_numpy(dtype=np.float64)
    y = df.loc[mask, ZN_TARGET].to_numpy(dtype=np.float64)
    counts = cell_counts(x, y, x_edges, y_edges)
    entropy, entropy_norm = normalized_entropy(counts, n_total_cells)
    return {
        "n_molecules": int(mask.sum()),
        "occupied_cells": int(np.count_nonzero(counts)),
        "shannon_entropy_natural_log": entropy,
        "normalized_entropy": entropy_norm,
        "cell_counts": counts.tolist(),
    }


def ordering_verdict(per_source: dict[str, dict]) -> tuple[bool, str]:
    ent = {source: per_source[source]["normalized_entropy"] for source in SOURCE_GROUPS}
    occ = {source: per_source[source]["occupied_cells"] for source in SOURCE_GROUPS}
    agent_highest_entropy = ent["agent"] == max(ent.values())
    pubchem_lowest_entropy = ent["PubChem random HT"] == min(ent.values())
    agent_highest_occupied = occ["agent"] == max(occ.values())
    verdict = bool(agent_highest_entropy and pubchem_lowest_entropy and agent_highest_occupied)
    if verdict:
        text = (
            "PASS: ordering reproduces the mentor trend "
            "(agent highest coverage/entropy, PubChem random HT lowest entropy)."
        )
    else:
        text = (
            "FAIL: ordering does not fully reproduce the mentor trend "
            f"(entropy ranking: {sorted(ent, key=ent.get, reverse=True)}; "
            f"occupied-cell ranking: {sorted(occ, key=occ.get, reverse=True)})."
        )
    return verdict, text


def print_table(per_source: dict[str, dict], normalization: str) -> None:
    print("Phase 4 property-space coverage map")
    print(f"Grid: x={ESP_TARGET}, y={ZN_TARGET}")
    print(f"Entropy normalization: {normalization}")
    print()
    print(f"{'source group':24s} {'n':>8s} {'occupied':>10s} {'H(ln)':>10s} {'H_norm':>10s}")
    print("-" * 68)
    for source in SOURCE_GROUPS:
        row = per_source[source]
        print(
            f"{source:24s} {row['n_molecules']:8d} {row['occupied_cells']:10d} "
            f"{row['shannon_entropy_natural_log']:10.4f} {row['normalized_entropy']:10.4f}"
        )


def run(n_bins: int) -> dict:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    df = load_unique_labels().reset_index(drop=True)
    missing_targets = sorted({ESP_TARGET, ZN_TARGET}.difference(df.columns))
    if missing_targets:
        raise ValueError(f"missing target columns from Phase 3 labels: {missing_targets}")

    source_sets, diagnostics = attach_source_groups(df)
    x = df[ESP_TARGET].to_numpy(dtype=np.float64)
    y = df[ZN_TARGET].to_numpy(dtype=np.float64)
    x_edges = make_edges(x, n_bins)
    y_edges = make_edges(y, n_bins)
    n_total_cells = int(n_bins * n_bins)
    normalization = f"Shannon H=-sum(p*ln(p)); normalized_entropy=H/ln({n_total_cells})"

    per_source = {}
    for source in SOURCE_GROUPS:
        mask = np.array([source in groups for groups in source_sets], dtype=bool)
        per_source[source] = metrics_for_mask(df, mask, x_edges, y_edges, n_total_cells)

    global_counts = cell_counts(x, y, x_edges, y_edges)
    verdict, verdict_text = ordering_verdict(per_source)
    print_table(per_source, normalization)
    print()
    print(f"multi_source_molecule_count={diagnostics['multi_source_molecule_count']}")
    if diagnostics["multi_source_molecule_count"]:
        print(f"multi_source_examples={diagnostics['multi_source_examples'][:5]}")
    print(f"global_occupied_cells={int(np.count_nonzero(global_counts))}/{n_total_cells}")
    print(f"ordering_verdict={verdict_text}")

    return {
        "n_bins": int(n_bins),
        "targets": {"x": ESP_TARGET, "y": ZN_TARGET},
        "n_molecules": int(len(df)),
        "source_groups": SOURCE_GROUPS,
        "bin_edges": {
            ESP_TARGET: [float(v) for v in x_edges],
            ZN_TARGET: [float(v) for v in y_edges],
        },
        "entropy_normalization": normalization,
        "source_group_diagnostics": diagnostics,
        "per_source": per_source,
        "global_cell_counts": global_counts.astype(int).tolist(),
        "global_occupied_cells": int(np.count_nonzero(global_counts)),
        "ordering_reproduces_mentor_trend": verdict,
        "ordering_verdict": verdict_text,
        "mentor_reference_note": (
            "Mentor-reported values used a related coverage figure: "
            "agent 34 cells / entropy 0.73, PubChem 17 / 0.47, ECFP 27 / 0.63. "
            "Exact values here can differ because bin edges and source deduping are explicit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 4 property-space coverage map.")
    parser.add_argument("--n_bins", type=int, default=7, help="Equal-width bins per axis.")
    args = parser.parse_args()

    results = run(args.n_bins)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"Saved metrics: {RESULT_PATH}")


if __name__ == "__main__":
    main()
