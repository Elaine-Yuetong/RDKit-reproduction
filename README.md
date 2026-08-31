# RDKit Reproduction: Molecular Property Prediction and Coverage Sampling

This repository tracks a staged reproduction project for AI-assisted molecular
materials discovery. Phases 1-3 reproduce property-prediction baselines; Phase 4
turns those predictors into a retrospective benchmark for coverage-aware
molecular selection.

Before changing code, read `CLAUDE.md`, `METHODS.md`, and `HISTORY.md`.
`METHODS.md` is the scientific source of truth. Use `./venv/bin/python` for
project commands, never system Python or conda. Do not re-split datasets:
frozen splits come from `src.data` with seed 42.

## Repository Map

- `src/`: Phase 1 QM9 RDKit/XGBoost baseline, frozen splits, features, training.
- `phase2/`: Colab SchNet reproduction on PyG QM9 3D coordinates.
- `phase3/`: Private DFT-label loading, RDKit/ChemBERTa feature tests, source-split evaluation.
- `phase4_generation/`: Coverage-map, 3D "eyes" predictor, and standardized coverage benchmark.
- `results/`: selected metrics JSON files. This directory is gitignored by default, so add result files intentionally.
- `benchmark_report_full.html`: human-facing Phase 4 result report generated from the full benchmark run.
- `benchmark_system_guide.html`: human-facing guide to the standardized benchmark system.
- `history.json`: chat history and project context. Treat it as context for terminology and intended workflow.

Private mentor data lives under `data/dft_real/` and is not public data. Do not
commit raw private data, structures, caches, or unreviewed generated files.

## Environment

Local CPU commands use the project venv:

```bash
./venv/bin/python -m src.data
./venv/bin/python -m src.featurize
```

Phase 2 and most Phase 4 model/benchmark runs need a GPU environment with
PyTorch Geometric and its compiled dependencies. The local venv is for CPU
RDKit/XGBoost/data-prep work unless that GPU stack is installed there.

## Phase 2: SchNet on QM9

Goal: reproduce the learned-representation era on QM9 using PyG SchNet and 3D
coordinates, then compare against the Phase 1 RDKit/XGBoost baseline.

Main script:

```bash
./venv/bin/python phase2/train_schnet.py \
  --splits_dir data/splits \
  --split random \
  --target gap \
  --train_subset 50000 \
  --epochs 100 \
  --batch_size 64 \
  --out_dir /content/drive/MyDrive/schnet_runs
```

Run this in Colab/GPU, following `phase2/README.md` for PyG installation. The
run matches PyG QM9 molecules to the frozen Phase 1 split by molecule identity,
normalizes the target on the training subset, and evaluates only on the frozen
test intersection.

Audited result:

- `results/phase2_schnet_gap_random.json`
- Gap MAE: 0.1156 eV with 50,000 training molecules.
- Baseline to beat: Phase 1 XGBoost concat gap MAE 0.1364 eV on the random split.

## Phase 3: Real DFT Property Prediction

Goal: test whether descriptor-era models transfer from baseline molecules to
the mentor's agent-exploration molecules.

Core dataset:

- `phase3.data.load_unique_labels()`
- 3,663 unique canonical SMILES.
- Targets: `esp_vmin_mean_kcal_per_mol` and `zn_e_bind_mean_kcal_per_mol`.
- Source split: train on `baseline`, test on `agent`.

Useful commands:

```bash
./venv/bin/python -m phase3.intake_audit
./venv/bin/python -m phase3.featurize_real
./venv/bin/python -m phase3.train_eval --target esp_vmin_mean_kcal_per_mol
./venv/bin/python -m phase3.train_eval --target zn_e_bind_mean_kcal_per_mol
./venv/bin/python -m phase3.task1_element_breakdown
./venv/bin/python -m phase3.task2_abc_compare
```

Audited takeaways:

| Target | Random 5-fold CV MAE | Source split MAE | Source split R2 |
|---|---:|---:|---:|
| ESP minimum | 7.35 kcal/mol | 18.74 kcal/mol | 0.18 |
| Zn binding | 18.94 kcal/mol | 23.37 kcal/mol | 0.09 |

The source split is much harder than random CV. The failure concentrates in
S/P-rich and other exotic-element agent molecules. ChemBERTa features did not
improve this transfer problem, so Phase 4 moves toward 3D predictors and
coverage-aware selection.

## Phase 4: Coverage-Aware Generation Benchmark

Phase 4 changes the objective. Instead of only asking "how accurate is the
property predictor?", it asks:

> Given a limited DFT budget, which selection strategy covers true ESP/Zn
> property space most uniformly?

This is a retrospective benchmark. The pool already has true DFT labels, but
pickers are not allowed to use them. True labels are revealed only during
evaluation. That makes the benchmark a fair simulation of "choose molecules now,
pay DFT later."

### Phase 4 Components

1. Coverage map: `phase4_generation/coverage_map.py`
   - CPU-only.
   - Loads the Phase 3 DFT labels.
   - Bins ESP/Zn property space into a 7 x 7 grid.
   - Measures source coverage and entropy.
   - Output: `results/phase4_coverage_map.json`.

2. PaiNN/PyG cache prep: `phase4_generation/prepare_painn_data.py`
   - CPU data preparation.
   - Resolves `.xyz` structures into plain Python records.
   - Output: `data/dft_real/painn_cache/painn_records.json.gz`.

3. Eyes predictor: `phase4_generation/train_painn.py`
   - GPU/PyG.
   - Trains a 3D model for ESP or Zn.
   - Supports `--model schnet` and `--model dimenetpp`.

4. Retrospective sampler: `phase4_generation/coverage_sampling_eval.py`
   - GPU/PyG.
   - Trains eyes on the seed set.
   - Uses eyes predictions to rank/select pool molecules.
   - Evaluates selected molecules using hidden true labels.

5. Standardized benchmark wrapper: `phase4_generation/benchmark/run_benchmark.py`
   - Config-driven entry point.
   - Runs both benchmark scenarios and all configured pickers.
   - Outputs CSV, JSON, and HTML.

### What Are "Eyes"?

"Eyes" means the learned property predictors used by the benchmark to estimate
ESP and Zn binding for unlabeled pool molecules. In the current benchmark config,
the eyes model is `dimenetpp`:

```yaml
eyes:
  model: dimenetpp
  lr: 1.0e-4
  grad_clip: 1.0
  epochs: 100
```

The eyes are not the final scientific truth. They are a cheap proxy for DFT.
They are trained only on the seed molecules, then used to predict pool
properties. The pool's true DFT labels stay hidden until evaluation.

### What Are `random` and `source`?

There are two different uses of the word "random":

- Scenario `random`: a random seed/pool split across the whole parsed DFT
  dataset. By default, `seed_frac=0.3`, so about 30 percent of molecules are
  treated as already known labels and the rest form the candidate pool.
- Picker `random`: a baseline picker that uniformly samples molecules from the
  pool without using structure or predicted properties.

There is also one scenario named `source`:

- Scenario `source`: the deployment-style split. The seed set is baseline
  molecules and the pool is agent-discovered molecules.
- In code, baseline records are `source_kind == "baseline"` and pool records are
  `source_kind == "agent"`.
- This is harder and more realistic because it tests transfer from PubChem-like
  baseline chemistry to the mentor's exploration campaigns.

If you see `sourse` in notes, treat it as a typo for `source`.

### Pickers Compared by the Benchmark

- `random`: uniformly sample from the pool.
- `structural_diversity`: Morgan fingerprint MaxMin selection in structure space.
- `uncertainty_distance`: select molecules far from the known seed set in predicted ESP/Zn space.
- `property_maxmin`: greedy farthest-point selection in predicted ESP/Zn space.
- `property_stratified`: round-robin sampling from predicted ESP/Zn grid cells.
- `eyes_coverage`: greedily fills the emptiest predicted ESP/Zn grid cells.

### Metrics

- `entropy`: normalized grid occupancy entropy. Higher is better.
- `occupied_cells`: number of occupied ESP/Zn grid cells. Higher is better.
- `mean_nn`: mean nearest-neighbor distance among selected molecules in standardized true ESP/Zn space. Higher is better.
- `min_nn`: minimum nearest-neighbor distance in standardized true ESP/Zn space. Higher is better.
- `coverage_radius`: maximum distance from any pool molecule to the nearest selected molecule. Lower is better.

### Fairness Rules

- Pickers never use pool ground-truth labels.
- Ground truth is revealed only inside evaluation.
- Eyes are trained only on the seed set, never on the pool.
- Seed labels may be used because they represent already-known DFT results.
- Pool values used by property-space pickers are eyes predictions.

## Phase 4 Benchmark Usage

Prepare the 3D cache first if it does not exist:

```bash
./venv/bin/python -m phase4_generation.prepare_painn_data
```

Build or refresh the property-space coverage map:

```bash
./venv/bin/python -m phase4_generation.coverage_map
```

Run a tiny GPU smoke gate:

```bash
./venv/bin/python -m phase4_generation.benchmark.run_benchmark \
  --config phase4_generation/benchmark/benchmark_config.yaml \
  --smoke
```

Run the full standardized benchmark on the GPU server:

```bash
./venv/bin/python -m phase4_generation.benchmark.run_benchmark \
  --config phase4_generation/benchmark/benchmark_config.yaml
```

Benchmark config:

```yaml
data:
  cache: data/dft_real/painn_cache/painn_records.json.gz
  min_atom_dist: 0.1

grid:
  coverage_map: results/phase4_coverage_map.json
  n_bins: 7

scenarios:
  - random
  - source

pickers:
  - random
  - structural_diversity
  - uncertainty_distance
  - property_maxmin
  - property_stratified
  - eyes_coverage

metrics:
  - entropy
  - occupied_cells
  - mean_nn
  - min_nn
  - coverage_radius

eval:
  budget: 200
  n_seeds: 3

eyes:
  model: dimenetpp
  lr: 1.0e-4
  grad_clip: 1.0
  epochs: 100
```

Expected outputs:

- `results/benchmark_results.csv`
- `results/benchmark_results.json`
- `phase4_generation/benchmark/report.html`

Full benchmark result captured in `history.json`:

| Scenario | Best entropy tier | Interpretation |
|---|---|---|
| `random` | `eyes_coverage` 0.8104, `property_stratified` 0.7976, `property_maxmin` 0.7900 | Property-space methods beat random and structural diversity. |
| `source` | `property_stratified` 0.8073, `eyes_coverage` 0.8066, `property_maxmin` 0.8000 | Coverage-aware and stratified methods form the top tier under deployment shift. |

Structural diversity was lowest on entropy in both scenarios
(`random`: 0.6107, `source`: 0.6901). In the `source` scenario,
`eyes_coverage` had the best coverage radius among the compared methods
(1.4296), meaning it reduced the worst uncovered distance most strongly.

## Using the Benchmark on New Data

To compare a new DFT dataset:

1. Convert it to the same cache record format produced by
   `phase4_generation.prepare_painn_data`: `row_id`, `canonical_smiles`, `z`,
   `pos`, `esp_vmin_mean`, `zn_e_bind_mean`, and `source_kind`.
2. Edit `data.cache` in `phase4_generation/benchmark/benchmark_config.yaml`.
3. If the property range changes, regenerate the coverage map and update
   `grid.coverage_map`.
4. Run the same benchmark command. The output table schema stays the same, so
   results are directly comparable.

To add a new picker, implement it in
`phase4_generation/coverage_sampling_eval.py`, add its exact name to
`PICKER_ORDER` and `run_all_pickers`, then include that name under `pickers:` in
`phase4_generation/benchmark/benchmark_config.yaml`.

## Result Reports

- `REPORT.md`: written progress report covering Phases 1-3 and the Phase 4 motivation.
- `RESULTS.md`: Phase 1 QM9 table and interpretation.
- `benchmark_report_full.html`: Phase 4 full result report for mentor discussion.
- `benchmark_system_guide.html`: Phase 4 system/usage guide for the standardized benchmark.
- `phase4_generation/benchmark/README.md`: narrower benchmark package README.

