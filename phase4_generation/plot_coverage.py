"""Plot the Phase 4 property-space coverage heatmaps.

This reproduces the mentor Fig. 4 style as aggregate per-cell counts only:
no molecule identities are written into the figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from phase3.data import TARGET_COLUMNS, load_unique_labels
from phase4_generation.coverage_map import (
    ESP_TARGET,
    RESULT_PATH as COVERAGE_JSON,
    ZN_TARGET,
    cell_counts,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
OUT_PNG = FIG_DIR / "coverage_three_sources.png"
SOURCES = ["agent", "PubChem random HT", "ECFP max-min"]


def recompute_counts(source: str, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    """Fallback path if old JSON lacks per-source cell_counts."""
    df = load_unique_labels().reset_index(drop=True)
    missing = sorted({ESP_TARGET, ZN_TARGET, "source_kind", *TARGET_COLUMNS}.difference(df.columns))
    if missing:
        raise ValueError(f"load_unique_labels() missing required columns: {missing}")

    if source == "agent":
        mask = df["source_kind"].astype(str).to_numpy() == "agent"
    else:
        raise ValueError(
            f"Cannot recompute source={source!r} from source_kind alone; "
            "rerun phase4_generation.coverage_map.py to store per-source counts."
        )
    return cell_counts(
        df.loc[mask, ESP_TARGET].to_numpy(dtype=np.float64),
        df.loc[mask, ZN_TARGET].to_numpy(dtype=np.float64),
        x_edges,
        y_edges,
    )


def load_source_counts(data: dict, source: str, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    source_data = data["per_source"].get(source)
    if source_data is not None and "cell_counts" in source_data:
        return np.asarray(source_data["cell_counts"], dtype=int)
    print(f"WARNING: per-source cell_counts missing for {source}; attempting fallback recompute.")
    return recompute_counts(source, x_edges, y_edges)


def cell_labels(edges: np.ndarray) -> list[str]:
    return [f"{i + 1}" for i in range(len(edges) - 1)]


def plot() -> Path:
    data = json.loads(COVERAGE_JSON.read_text())
    x_edges = np.asarray(data["bin_edges"][ESP_TARGET], dtype=np.float64)
    y_edges = np.asarray(data["bin_edges"][ZN_TARGET], dtype=np.float64)

    counts_by_source = {
        source: load_source_counts(data, source, x_edges, y_edges) for source in SOURCES
    }
    vmax = max(int(counts.max()) for counts in counts_by_source.values())
    vmin = 0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
    for ax, source in zip(axes, SOURCES, strict=True):
        counts = counts_by_source[source]
        # histogram2d returns [x_bin, y_bin]; transpose for imshow's row=y, col=x.
        shown = counts.T
        image = ax.imshow(
            shown,
            origin="lower",
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        for y_i in range(shown.shape[0]):
            for x_i in range(shown.shape[1]):
                value = int(shown[y_i, x_i])
                text_color = "white" if value > vmax * 0.55 else "black"
                ax.text(x_i, y_i, str(value), ha="center", va="center", fontsize=8, color=text_color)

        source_stats = data["per_source"][source]
        ax.set_title(
            f"{source}\n"
            f"{source_stats['occupied_cells']} cells | entropy {source_stats['normalized_entropy']:.2f}",
            fontsize=11,
        )
        ax.set_xticks(np.arange(len(x_edges) - 1))
        ax.set_yticks(np.arange(len(y_edges) - 1))
        ax.set_xticklabels(cell_labels(x_edges))
        ax.set_yticklabels(cell_labels(y_edges))
        ax.set_xlabel("ESP bin (weak -> deep)")
        ax.grid(False)

    axes[0].set_ylabel("Zn binding bin (weak -> strong)")
    cbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    cbar.set_label("molecule count")
    fig.suptitle("Property-space coverage by source", fontsize=14)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return OUT_PNG


def main() -> None:
    out_path = plot()
    data = json.loads(COVERAGE_JSON.read_text())
    print(f"Saved coverage plot: {out_path}")
    print(
        "ordering_confirmation="
        f"{data['ordering_verdict']}"
    )


if __name__ == "__main__":
    main()
