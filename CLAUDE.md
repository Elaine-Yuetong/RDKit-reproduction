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

## Phase 2

### Goal
Reproduce the learned-representation era on QM9 with SchNet in Colab, using
3D coordinates from `torch_geometric.datasets.QM9`, and compare gap MAE against
the audited Phase 1 baseline.

### Protocol
- Run Phase 2 on Colab GPU only; do not install torch/PyG into the local venv.
- Use `phase2/train_schnet.py` with PyTorch Geometric QM9 and target `gap`
  (PyG target index 4, already in eV).
- Standard-normalize the target on the training subset, early-stop on the
  frozen validation intersection, and report test MAE/R2 in eV.
- Start with `--train_subset 50000`; the exact full-paper number is not the
  goal for this first run.

### Success band
With 50k training molecules, expect gap MAE roughly 0.08-0.12 eV. The Phase 1
numbers to beat are 0.1364 eV on the random split and 0.2904 eV on the scaffold
split.

### Frozen-exam rule
The files in `data/splits/` are the portable exam definition. Phase 2 must
validate `data/splits/manifest.json`, match PyG QM9 molecules to those split
CSVs by canonical SMILES, and train/evaluate only on the matching intersections.
Never train on molecules matching the frozen validation or test SMILES.
