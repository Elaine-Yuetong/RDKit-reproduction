# RESULTS.md — Phase 1 Baseline vs. Literature

> Fill rule for Claude Code: replace every `TBD` with values from
> `results/metrics.json`. Do not edit the "Literature reference" rows.
> All energies in **eV** (Hartree→eV already applied in src/data.py).

## 1. Setup summary
- Dataset: QM9 (MoleculeNet CSV), rows after cleaning: **TBD** (dropped: **TBD**)
- Splits: random 80/10/10 (seed=42) and Murcko scaffold 80/10/10
- Models: XGBoost (main), RandomForest (sanity)
- Feature sets: (a) RDKit 2D descriptors, (b) Morgan FP r=2/2048, (c) a+b

## 2. Headline table — gap (eV), test MAE

| Model / Representation | Random split | Scaffold split | Source |
|---|---|---|---|
| XGBoost + RDKit descriptors | TBD | TBD | ours |
| XGBoost + Morgan FP | TBD | TBD | ours |
| XGBoost + concat | TBD | TBD | ours |
| RandomForest + concat | TBD | TBD | ours |
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
| homo | desc | TBD | TBD | TBD | TBD |
| homo | fp | TBD | TBD | TBD | TBD |
| homo | concat | TBD | TBD | TBD | TBD |
| lumo | desc | TBD | TBD | TBD | TBD |
| lumo | fp | TBD | TBD | TBD | TBD |
| lumo | concat | TBD | TBD | TBD | TBD |
| gap | desc | TBD | TBD | TBD | TBD |
| gap | fp | TBD | TBD | TBD | TBD |
| gap | concat | TBD | TBD | TBD | TBD |
| mu | desc | TBD | TBD | TBD | TBD |
| mu | fp | TBD | TBD | TBD | TBD |
| mu | concat | TBD | TBD | TBD | TBD |
| alpha | desc | TBD | TBD | TBD | TBD |
| alpha | fp | TBD | TBD | TBD | TBD |
| alpha | concat | TBD | TBD | TBD | TBD |

(R² values live in `results/metrics.json`; parity plots in `results/plots/`.)

## 4. Reference points for homo / lumo (eV, random-split literature)

| Model | HOMO | LUMO | Note |
|---|---|---|---|
| SchNet (~110k train) | ~0.041 | ~0.034 | 3D coords |
| Classical KRR + physics 3D reps (BoB/BAML/FCHL) | ~0.10–0.15 | ~0.10–0.15 | needs 3D |
| Our best (fill) | TBD | TBD | 2D only |

## 5. Interpretation (write 5–8 sentences after numbers are in)
Prompts to answer:
1. Random vs scaffold gap for our best model — how large, and what does it say
   about generalizing to unseen additive chemotypes?
2. Descriptors vs fingerprints vs concat — which wins per target, and does the
   pattern match intuition (electronic targets harder for 2D features)?
3. Our best gap MAE vs SchNet's 0.063 eV — the ~Nx ratio is the
   "hand-crafted → learned representations" story (Mena review, Sec. 2) and
   the motivation for Phase 2.
4. For a 10²–10³-molecule in-house DFT dataset, which of these models would we
   actually deploy first, and why?

## 6. Repro command
```bash
python -m src.run_all --csv data/qm9.csv \
  --targets homo lumo gap mu alpha \
  --features desc fp concat --splits random scaffold
```
