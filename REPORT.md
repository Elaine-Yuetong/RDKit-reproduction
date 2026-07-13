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

## 5. Phase 2 (complete)
The learned-representation model was reproduced on Colab with SchNet, using PyG QM9 3D coordinates and the same frozen random-split test set. With 50,000 training molecules, SchNet reaches 0.116 eV gap MAE (R2 = 0.983), beating the Phase-1 hand-crafted-feature XGBoost concat baseline of 0.136 eV with less than half the training data. This quantifies the hand-crafted -> learned representations story from the Mena review with our own numbers. The remaining gap to the literature SchNet value of ~0.063 eV is expected because that reference used ~110k training molecules and longer schedules.

## Phase 3 Task 1: Property prediction under the mentor's train/test split
Per the mentor's instruction, the deployment split trains on the three PubChem-derived baseline sets and tests on the agent-exploration campaigns. This gives 2,140 baseline molecules for training and 1,523 agent molecules for testing. The two sets are chemically disjoint at the canonical-SMILES level, with zero overlap verified, so this is a genuine test of generalization to novel chemistry and mirrors the real additive-screening use case.

| Target | Random 5-fold CV MAE | Random CV R2 | Mentor split MAE | Mentor split R2 |
|---|---:|---:|---:|---:|
| ESP minimum (kcal/mol) | 7.35 | 0.82 | 18.74 | 0.18 |
| Zn binding (kcal/mol) | 18.94 | 0.25 | 23.37 | 0.09 |

Cross-source prediction is substantially harder than random cross-validation for both targets. The random split measures interpolation among molecules drawn from the same source mixture, while the mentor split asks whether PubChem-trained models transfer to the agent-discovered chemistry.

The key finding is that the cross-source difficulty is not uniform; it is concentrated in the exotic-element molecules the mentor cares about. For ESP, QM9-compatible agent molecules containing only C/N/O/F (n=454) are predicted reasonably well, with MAE = 8.34 kcal/mol and R2 = 0.77. Exotic-element agent molecules (n=1,069) degrade to MAE = 23.16 kcal/mol and R2 = -0.03, and the sulfur-containing subset (n=690) is worse still at MAE = 25.17 kcal/mol and R2 = -0.24. In plain terms, hand-crafted descriptors plus XGBoost still work across sources for QM9-like molecules, but essentially fail on the S/P-rich electrolyte chemistry where R2 is near zero or negative.

This quantifies the limit of the current descriptor-era approach and motivates Task 2: a learned 3D representation model that is trained on data containing S/P chemistry so it can represent the exotic-element regime directly. No Task 2 result is claimed yet; this section only establishes the baseline failure mode it needs to address.

## 6. Next
In parallel, the next practical in-house step is chemprop on the additive DFT data, because it works from SMILES, supports small datasets, and gives a learned 2D baseline between RDKit descriptors and 3D SchNet.
