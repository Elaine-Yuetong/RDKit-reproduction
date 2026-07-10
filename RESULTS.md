# RESULTS.md — Phase 1 Baseline vs. Literature

> Fill rule for Claude Code: replace every `TBD` with values from
> `results/metrics.json`. Do not edit the "Literature reference" rows.
> All energies in **eV** (Hartree→eV already applied in src/data.py).

## 1. Setup summary
- Dataset: QM9 (MoleculeNet CSV), rows after cleaning: **133,798** (dropped: **0 unparseable SMILES; 87 duplicate canonical SMILES**)
- Splits: random 80/10/10 (seed=42) and Murcko scaffold 80/10/10
- Models: XGBoost (main), RandomForest (sanity)
- Feature sets: (a) RDKit 2D descriptors, (b) Morgan FP r=2/2048, (c) a+b

## 2. Headline table — gap (eV), test MAE

| Model / Representation | Random split | Scaffold split | Source |
|---|---|---|---|
| XGBoost + RDKit descriptors | 0.1430 | 0.2945 | ours |
| XGBoost + Morgan FP | 0.1934 | 0.3841 | ours |
| XGBoost + concat | 0.1364 | 0.2904 | ours |
| RandomForest + concat | 0.1569 | 0.3163 | ours |
| — Literature reference — | | | |
| Plain 2D GNN (Morris 1-GNN), full QM9 | ~0.121 | — | lit. |
| SchNet (3D, ~110k train) | ~0.063 | — | Schütt 2018 |
| GNN consensus band (SchNet/MPNN/MEGNet) | 0.06–0.09 | — | lit. |

Expected zone for our rows: **0.25–0.5 eV** on random split; scaffold split worse.
If far outside this band, check (in order): Hartree→eV conversion, target
column mapping, train/test leakage in featurization.

## 3. Full grid — test MAE (eV for homo/lumo/gap; native units for mu, alpha)

| Target | Feat. set | XGB random | XGB scaffold | RF random | RF scaffold |
|---|---|---|---|---|---|
| homo | desc | 0.1027 | 0.1859 | -- | -- |
| homo | fp | 0.1381 | 0.2415 | -- | -- |
| homo | concat | 0.1016 | 0.1817 | 0.1220 | 0.2053 |
| lumo | desc | 0.1099 | 0.2424 | -- | -- |
| lumo | fp | 0.1509 | 0.3339 | -- | -- |
| lumo | concat | 0.1052 | 0.2450 | 0.1298 | 0.2817 |
| gap | desc | 0.1430 | 0.2945 | -- | -- |
| gap | fp | 0.1934 | 0.3841 | -- | -- |
| gap | concat | 0.1364 | 0.2904 | 0.1569 | 0.3163 |
| mu | desc | 0.5083 | 0.6160 | -- | -- |
| mu | fp | 0.5323 | 0.6550 | -- | -- |
| mu | concat | 0.4846 | 0.5964 | 0.5119 | 0.6039 |
| alpha | desc | 0.4108 | 0.9717 | -- | -- |
| alpha | fp | 2.5941 | 6.9978 | -- | -- |
| alpha | concat | 0.3865 | 0.9346 | 0.6041 | 1.2979 |

(R² values live in `results/metrics.json`; parity plots in `results/plots/`.)

## 4. Reference points for homo / lumo (eV, random-split literature)

| Model | HOMO | LUMO | Note |
|---|---|---|---|
| SchNet (~110k train) | ~0.041 | ~0.034 | 3D coords |
| Classical KRR + physics 3D reps (BoB/BAML/FCHL) | ~0.10–0.15 | ~0.10–0.15 | needs 3D |
| Our best (fill) | 0.1016 | 0.1052 | 2D only |

## 5. Interpretation (write 5–8 sentences after numbers are in)
The best gap model is XGBoost + concat: 0.1364 eV on the random split and 0.2904 eV on the scaffold split, a 2.13x MAE increase when the test molecules come from unseen Murcko scaffolds. That scaffold penalty is the deployment warning: for new additive chemotypes, the honest error estimate is much closer to the scaffold column than the random column. Concat wins for homo, gap, mu, and alpha on the random split; lumo is also best with concat on random, while descriptors are slightly better than concat on the scaffold split (0.2424 vs 0.2450), so the main pattern is that descriptors carry most of the signal and fingerprints alone are weakest, especially for alpha. The electronic targets are still harder under scaffold shift, and mu has the weakest XGBoost concat R2 among the main targets (0.7512 random, 0.6536 scaffold), which matches the expectation that dipole is 3D-dependent and under-served by 2D features. Our best random-split gap MAE is 2.17x SchNet's ~0.063 eV; using the scaffold estimate, the gap is 4.61x, which is the more relevant "hand-crafted -> learned representations" motivation for Phase 2. The 0.1364 eV random gap result is below the broad 0.25-0.5 eV text band above, but it matches the handoff's certified ~0.13-0.14 eV expectation after deduplication and the 4000-tree XGBoost ceiling, so I do not interpret it as leakage. For a 10^2-10^3 molecule in-house DFT dataset, I would deploy XGBoost + concat first because it is the best or near-best across targets, needs no 3D geometry workflow, and is more data-efficient than training a GNN from scratch. RandomForest + concat remains a useful sanity check, but it is consistently worse than XGBoost here.

## 6. Repro command
```bash
python -m src.run_all --csv data/qm9.csv \
  --targets homo lumo gap mu alpha \
  --features desc fp concat --splits random scaffold
```
