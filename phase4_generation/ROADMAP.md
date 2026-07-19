# Phase 4 Roadmap: Coverage-Aware Molecular Generation

## Goal

Phases 1-3 reproduced and stress-tested property-prediction baselines:
QM9 SchNet, RDKit plus XGBoost on the mentor DFT labels, and a ChemBERTa
feature attempt. Phase 4 is a different objective: use a small DFT budget to
achieve uniform coverage of property space for both prediction and molecular
generation.

The core problem is that molecules sampled to be diverse in structure space do
not necessarily cover property space uniformly. They often cluster near the
center of the observed distribution while leaving chemically important edge
regions empty. That is a problem for downstream generative models because they
cannot learn regions where the DFT dataset has no examples.

## Intended Loop

The full Phase 4 system is an active loop:

1. **Generator (planned: GFlowNet, GPU).** Proposes valid molecules while being
   rewarded for spreading across property space instead of collapsing to one
   high-scoring mode.
2. **Property predictor, the "eyes" (planned: PaiNN, GPU).** Estimates
   molecular properties quickly, without running DFT for every candidate.
3. **Coverage-aware reward (CPU first, later integrated with GPU generator).**
   Divides property space into cells, rewards candidates landing in empty or
   sparse cells, and penalizes candidates landing in already crowded cells.
4. **DFT-in-the-loop update (external DFT budget plus CPU/GPU retraining).**
   Selects the most valuable candidates for real DFT, adds those labels back
   into the dataset, retrains the predictor, and updates the coverage map.

The predictor is only an estimate and can be wrong, so real DFT remains the
source of truth for selected candidates. The loop is designed to spend DFT on
molecules that improve coverage, not merely on molecules that look easy or
central under the current model.

## Reuse From Phase 3

Phase 4 reuses the private in-house DFT dataset and metadata already handled in
`phase3/`, without moving or committing private data:

- `phase3.data.load_unique_labels()` provides the 3,663 unique molecules,
  averaged targets, `source_kind`, and Murcko scaffolds.
- Phase 3 element audits identify the exotic-element chemistry, especially
  S/P-rich molecules, where prediction was difficult.
- Phase 3 mentor split results establish the deployment setting: train on
  PubChem-derived baselines and test on agent-exploration campaigns, with zero
  canonical-SMILES overlap.
- Phase 3 negative ChemBERTa results are a useful constraint: simply reusing a
  pretrained SMILES language model did not close the S/P generalization gap.

No Phase 2 or Phase 3 code should be modified as part of Phase 4 setup unless a
later task explicitly asks for a shared interface.

## Build Order

1. **Coverage map (CPU, no GPU).** Reproduce the mentor's coverage figure by
   dividing property space into a grid and measuring how each source covers the
   cells. This is the first implementation step and reuses only the Phase 3
   data loader.
2. **PaiNN predictor, the "eyes" (GPU).** Upgrade from RDKit plus XGBoost to a
   3D equivariant predictor once cloud GPU access is available. This requires
   molecular structures and a GPU-capable PyTorch stack.
3. **Coverage-aware reward (CPU prototype, then generator integration).**
   Convert the coverage map into a reward function that scores candidates by
   how much they fill empty or sparse property-space cells.
4. **GFlowNet generator (GPU).** Train a generator to propose diverse,
   coverage-improving molecules rather than optimizing toward a single mode.
5. **DFT-in-the-loop active update (external DFT plus retraining).** Select a
   small batch of high-value candidates for real DFT, append the results, and
   repeat the predictor and coverage-map update.

## This Scaffold

This folder currently contains only the roadmap and package marker. The next
step is to add the CPU-only coverage-map implementation. No analysis code,
model code, training, or generation is part of this scaffold commit.
