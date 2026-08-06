# Phase 4 Coverage Benchmark

This package standardizes the Phase 4 retrospective sampling benchmark so new
DFT datasets can be compared directly. It wraps
`phase4_generation.coverage_sampling_eval` and reuses its eye training,
pickers, evaluation metrics, and aggregation code.

## What It Measures

The benchmark asks whether a fixed DFT budget picks molecules that cover the
true ESP/Zn property space uniformly. Pickers may use structure and the eyes'
predicted properties, but they never use pool ground-truth labels. True labels
are revealed only in evaluation.

## Pickers

- `random`: uniformly samples molecules from the pool.
- `structural_diversity`: Morgan fingerprint MaxMin selection in structure space.
- `uncertainty_distance`: picks molecules farthest from the known seed set in predicted ESP/Zn space.
- `property_maxmin`: greedy farthest-point selection in predicted ESP/Zn space.
- `property_stratified`: round-robin sampling from predicted ESP/Zn grid cells.
- `eyes_coverage`: greedily fills the emptiest predicted ESP/Zn grid cells.

## Metrics

- `entropy`: normalized grid occupancy entropy; higher is better.
- `occupied_cells`: number of occupied property-grid cells; higher is better.
- `mean_nn`: mean nearest-neighbor distance among selected molecules in standardized true ESP/Zn space; higher is better.
- `min_nn`: minimum nearest-neighbor distance among selected molecules in standardized true ESP/Zn space; higher is better.
- `coverage_radius`: maximum distance from any pool molecule to its nearest selected molecule in standardized true ESP/Zn space; lower is better.

## Run

Use the GPU/PyG environment:

```bash
python -m phase4_generation.benchmark.run_benchmark \
  --config phase4_generation/benchmark/benchmark_config.yaml
```

For a tiny server gate:

```bash
python -m phase4_generation.benchmark.run_benchmark \
  --config phase4_generation/benchmark/benchmark_config.yaml \
  --smoke
```

Outputs:

- `results/benchmark_results.csv`
- `results/benchmark_results.json`
- `phase4_generation/benchmark/report.html`

If PyYAML is missing, install it in the project environment:

```bash
./venv/bin/pip install pyyaml
```

## New Data

Prepare the new dataset into the same PaiNN-cache record format used by
`phase4_generation.prepare_painn_data`: one record per molecule with `row_id`,
`canonical_smiles`, `z`, `pos`, `esp_vmin_mean`, `zn_e_bind_mean`, and
`source_kind`. Then edit `data.cache` in `benchmark_config.yaml` to point at
the new cache path. If the property range changes, also regenerate the coverage
map and update `grid.coverage_map`.

## Add A Picker

Add the picker implementation in `phase4_generation.coverage_sampling_eval.py`,
wire it into `PICKER_ORDER` and `run_all_pickers`, and then include its exact
name in `benchmark_config.yaml`. The benchmark wrapper will pick it up without
copying the picker logic.
