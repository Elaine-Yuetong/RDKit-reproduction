# Progress Report: Molecular Property Surrogate Baseline (Phase 1 complete)
Phase 1 reproduces the classic descriptor-era baseline for molecular property prediction on QM9 (133,798 molecules, 40 audited model configurations).
Best HOMO-LUMO gap MAE: 0.136 eV (random split) / 0.290 eV (scaffold split), vs SchNet's published 0.063 eV -- the measured motivation for Phase 2 (3D graph neural networks).

## 1. What was built
We built a local, CPU-only QM9 surrogate-model baseline using RDKit features and classical regressors. The dataset is the deduped MoleculeNet QM9 CSV with 133,798 molecules after dropping 87 duplicate canonical SMILES. Evaluation uses two frozen splits: random 80/10/10 and Murcko scaffold 80/10/10. The audited Phase 1 grid covers 40 configurations: 5 targets, 3 representations, 2 splits for XGBoost, plus concat-only RandomForest sanity checks.

## 2. Headline table: gap test MAE (eV)
| Model / Representation | Random split | Scaffold split | Source |
|---|---:|---:|---|
| XGBoost + RDKit descriptors | 0.1430 | 0.2945 | ours |
| XGBoost + Morgan FP | 0.1934 | 0.3841 | ours |
| XGBoost + concat | 0.1364 | 0.2904 | ours |
| RandomForest + concat | 0.1569 | 0.3163 | ours |
| Plain 2D GNN (Morris 1-GNN), full QM9 | ~0.121 | -- | literature |
| SchNet (3D, ~110k train) | ~0.063 | -- | Schutt 2018 |
| GNN consensus band (SchNet/MPNN/MEGNet) | 0.06-0.09 | -- | literature |

Literature values are random-split, ~110k training molecules; our runs use 107,038 training molecules on the identical frozen splits reused by all Phase-1 models.

## 3. Findings
**Scaffold shift is the deployment warning.** The best gap model, XGBoost + concat, reaches 0.1364 eV on the random split but 0.2904 eV on the scaffold split. That is a 2.13x MAE increase when the test molecules come from unseen Murcko scaffolds, so the scaffold number is the honest estimate for novel additive chemotypes.

**SchNet is the Phase 2 motivation.** Our best random-split gap result is 2.2x worse than the SchNet reference value of ~0.063 eV; using the scaffold estimate, the distance is 4.6x. That gap is the representation-learning story: 3D learned representations should outperform fixed 2D descriptors when enough QM9-scale data and optimized coordinates are available.

**2D features miss some 3D physics.** For XGBoost + concat, dipole moment (`mu`, Debye) has R2 = 0.7512 on random and 0.6536 on scaffold, while polarizability (`alpha`) is much stronger at R2 = 0.9924 on random and 0.9417 on scaffold. This is consistent with the expectation that dipole is more geometry-dependent and less fully captured by 2D descriptors/fingerprints.

**Representation choice matters.** Alpha is the clearest case study: XGBoost + Morgan fingerprints alone gives 2.59 MAE (Bohr^3) on the random split, while RDKit descriptors give 0.4108 MAE (Bohr^3) and concat gives 0.3865 MAE (Bohr^3). The fingerprint representation misses the smooth size/shape descriptor signal that alpha needs. Binary fingerprint bits discard atom counts, so they cannot track an extensive, size-dependent property like polarizability.

## 4. Relevance to the additive project
The same Phase 1 pipeline is built to run on an in-house DFT table by changing the CLI inputs, for example `python -m src.train --csv dft_additives.csv --target binding_energy`. For a 10^2-10^3 molecule additive dataset, the recommended first model is descriptors + XGBoost with scaffold-split evaluation: it is data-efficient, does not require a 3D geometry workflow, and was the strongest or near-strongest Phase 1 model across targets. If the dataset is at the small end (a few hundred molecules), we will switch evaluation from a single held-out split to repeated cross-validation to keep error estimates reliable.

## 5. Next
Phase 2 assets are ready for a Colab SchNet reproduction using the frozen split artifacts in `data/splits/`. In parallel, the next practical in-house step is chemprop on the additive DFT data, because it works from SMILES, supports small datasets, and gives a learned 2D baseline between RDKit descriptors and 3D SchNet.
