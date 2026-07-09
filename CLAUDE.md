# Project: Molecular property surrogate models (Phase 1 baseline)

## Goal
Reproduce the classic descriptor-based baseline for molecular property
prediction on QM9, structured so the dataset/target can later be swapped
for our in-house DFT data (ESP, ion binding energy) with a one-line change.

## Constraints
- Runs on macOS, CPU/MPS only. No CUDA. Keep deps minimal.
- Python 3.11, single venv. Pin versions in requirements.txt.
- Core deps: rdkit, pandas, numpy, scikit-learn, xgboost, matplotlib.
- All pipeline logic in src/ as importable functions; notebooks call src/.
- METHODS.md is the scientific source of truth; consult it before
  implementing any featurization, split, or evaluation logic.

## Data
- QM9 via MoleculeNet CSV (DeepChem S3): 
  https://deepchem-data.s3-us-west-1.amazonaws.com/datasets/qm9.csv
- ~134k molecules, SMILES + quantum properties.
- Targets for Phase 1: homo, lumo, gap, mu (dipole), alpha.
- IMPORTANT: homo/lumo/gap are in Hartree -> convert to eV (x 27.2114)
  so results are comparable to published GNN MAEs.
- Drop rows where RDKit fails to parse the SMILES; log how many.

## Pipeline requirements
1. Splits: random 80/10/10 with fixed seed AND scaffold split
   (rdkit Murcko scaffolds). Report both — scaffold split is the honest one.
2. Features (build both, compare):
   a. ~200 RDKit 2D descriptors (Descriptors.CalcMolDescriptors),
      drop NaN/constant columns
   b. Morgan fingerprints, radius 2, 2048 bits
   c. concatenation of a+b
3. Models: XGBoost (main), RandomForest (sanity check).
   Light hyperparam search only (n_estimators, max_depth, lr).
4. Metrics: MAE and R^2 per target, saved to results/metrics.json,
   plus parity plots per target.
5. The target column and input CSV must be CLI args:
   `python -m src.train --csv data/qm9.csv --target gap`
   so swapping in our DFT dataset later requires zero code changes.

## Definition of done for Phase 1
- One command reproduces the full table: 5 targets x 3 feature sets x 2 splits.
- A short RESULTS.md comparing our MAE vs published values.