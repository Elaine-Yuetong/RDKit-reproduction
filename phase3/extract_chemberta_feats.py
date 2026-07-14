"""Extract ChemBERTa molecule embeddings for the Phase 3 DFT molecules.

This script performs feature extraction only. It does not train models or touch
the private source data beyond reading the deduped label table through
phase3.data.load_unique_labels().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from phase3.data import DATA_DIR, TARGET_COLUMNS, load_unique_labels

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
MAX_LENGTH = 256
BATCH_SIZE = 64
OUT_DIR = DATA_DIR / "features_chemberta"
MEAN_NPY = OUT_DIR / "chemberta_mean.npy"
CLS_NPY = OUT_DIR / "chemberta_cls.npy"
LABELS_CSV = OUT_DIR / "labels_chemberta.csv"


def _count_truncated(tokenizer, smiles_batch: list[str]) -> int:
    raw = tokenizer(
        smiles_batch,
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    return int(sum(len(ids) > MAX_LENGTH for ids in raw["input_ids"]))


def _has_bad_values(array: np.ndarray) -> bool:
    return bool(np.isnan(array).any() or np.isinf(array).any())


def _all_zero_rows(array: np.ndarray) -> int:
    return int(np.all(array == 0, axis=1).sum())


def _summary(name: str, array: np.ndarray) -> None:
    print(
        f"{name}: shape={array.shape} dtype={array.dtype} "
        f"nan_or_inf={_has_bad_values(array)} "
        f"mean={float(array.mean()):.6f} std={float(array.std()):.6f} "
        f"min={float(array.min()):.6f} max={float(array.max()):.6f} "
        f"all_zero_rows={_all_zero_rows(array)}"
    )


def extract_features(batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, np.ndarray, int]:
    df = load_unique_labels().reset_index(drop=True)
    smiles = df["canonical_smiles"].astype(str).tolist()
    n_molecules = len(smiles)

    print(f"Loading ChemBERTa model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    hidden_size = int(model.config.hidden_size)
    print(f"hidden_size={hidden_size}")
    print("device=cpu")

    mean_features = np.empty((n_molecules, hidden_size), dtype=np.float32)
    cls_features = np.empty((n_molecules, hidden_size), dtype=np.float32)
    truncated_count = 0
    next_report = 500

    with torch.no_grad():
        for start in range(0, n_molecules, batch_size):
            end = min(start + batch_size, n_molecules)
            batch = smiles[start:end]
            truncated_count += _count_truncated(tokenizer, batch)
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            cls = hidden[:, 0, :]

            mean_features[start:end] = mean.cpu().numpy().astype(np.float32)
            cls_features[start:end] = cls.cpu().numpy().astype(np.float32)

            processed = end
            while processed >= next_report:
                print(f"processed {next_report}/{n_molecules}")
                next_report += 500

    if n_molecules and (next_report - 500) != n_molecules:
        print(f"processed {n_molecules}/{n_molecules}")

    return mean_features, cls_features, truncated_count


def save_outputs(mean_features: np.ndarray, cls_features: np.ndarray) -> None:
    df = load_unique_labels().reset_index(drop=True)
    labels = df[
        [
            "canonical_smiles",
            *TARGET_COLUMNS,
            "source_kind",
            "murcko_scaffold",
        ]
    ].copy()
    labels.insert(0, "row_id", np.arange(len(labels), dtype=int))

    if mean_features.shape[0] != len(labels) or cls_features.shape[0] != len(labels):
        raise ValueError(
            "feature row count does not match labels: "
            f"mean={mean_features.shape}, cls={cls_features.shape}, labels={len(labels)}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MEAN_NPY, mean_features.astype(np.float32, copy=False))
    np.save(CLS_NPY, cls_features.astype(np.float32, copy=False))
    labels.to_csv(LABELS_CSV, index=False)


def print_sanity_report(
    mean_features: np.ndarray,
    cls_features: np.ndarray,
    truncated_count: int,
) -> None:
    df = load_unique_labels().reset_index(drop=True)
    smiles = df["canonical_smiles"].astype(str).tolist()
    print("\nChemBERTa feature sanity report")
    print(f"n_molecules_processed={len(df)}")
    _summary("chemberta_mean", mean_features)
    _summary("chemberta_cls", cls_features)
    print(f"truncated_smiles_at_max_length_{MAX_LENGTH}={truncated_count}")
    print(f"saved_mean={MEAN_NPY}")
    print(f"saved_cls={CLS_NPY}")
    print(f"saved_labels={LABELS_CSV}")
    print("\nFirst 3 feature previews:")
    for i in range(min(3, len(smiles))):
        mean_head = np.array2string(mean_features[i, :5], precision=5, separator=", ")
        cls_head = np.array2string(cls_features[i, :5], precision=5, separator=", ")
        print(f"{i}: {smiles[i]} -> mean[:5]={mean_head} cls[:5]={cls_head}")


def main() -> None:
    mean_features, cls_features, truncated_count = extract_features()
    save_outputs(mean_features, cls_features)
    print_sanity_report(mean_features, cls_features, truncated_count)


if __name__ == "__main__":
    main()
