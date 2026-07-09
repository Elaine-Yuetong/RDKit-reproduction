# METHODS.md — Background & Reference: RDKit Baseline and SchNet

> Purpose: reference document for this repo. Explains *what* we are reproducing,
> *why* each method exists, and the *published numbers* our results must be
> compared against. Written for a data-science student entering molecular ML.
> Companion files: `CLAUDE.md` (build spec), `RESULTS.md` (our numbers vs. literature).

---

## 0. The one-paragraph story

Molecular property prediction went through two eras. In the **hand-crafted era**
(~2012–2017), you computed fixed numerical descriptors of a molecule (RDKit
descriptors, Morgan/ECFP fingerprints, Coulomb matrices) and fed them to a
classical regressor (KRR, random forest, gradient boosting). In the **learned-
representation era** (2017–), graph neural networks (MPNN, SchNet) learn the
features directly from the molecular graph or 3D geometry, and on the QM9
benchmark they cut errors by roughly 3–10×. Phase 1 of this repo reproduces the
first era; Phase 2 reproduces the second. The measured gap between them *is the
finding* — it is exactly the "hand-crafted → learned representations" narrative
in Section 2 of Mena et al., *Adv. Mater.* 2026 (our lab's assigned review).

---

## 1. The benchmark: QM9

- ~134k stable small organic molecules (up to 9 heavy atoms: C, N, O, F),
  enumerated from GDB-17, with properties computed at DFT (B3LYP/6-31G(2df,p)).
  Reference: Ramakrishnan et al., *Sci. Data* 1, 140022 (2014).
- The PyTorch Geometric distribution used in most modern papers contains
  **130,831 molecules** after removing ~3k "uncharacterized" ones — so if our
  cleaned CSV lands between ~130k and ~134k rows, we are consistent with the
  literature.
- Targets we use in Phase 1: `homo`, `lumo`, `gap`, `mu` (dipole, Debye),
  `alpha` (polarizability, Bohr³).
- **Units trap:** energies in the raw CSV are in **Hartree**. All published GNN
  MAEs for homo/lumo/gap are in **eV**. Conversion: 1 Ha = 27.2114 eV. In PyG,
  gap is target index 4 and the Hartree→eV conversion is standard practice.
  If our MAE looks ~27× too good, we forgot the conversion.
- Why QM9 for us: it is the dataset every classic paper reports on, so it is
  the only place we can *verify* a reproduction against published numbers
  before pointing the same pipeline at our in-house DFT data (ESP, ion binding
  energy).

---

## 2. Phase 1 — RDKit descriptor/fingerprint baseline (hand-crafted era)

### 2.1 What RDKit is
Open-source cheminformatics toolkit. For us it does four jobs:
1. **Parse & validate** SMILES → `Mol` objects (drop unparseable rows, log count).
2. **2D descriptors**: ~200 physicochemical/topological values per molecule
   (MolWt, LogP, TPSA, ring counts, charge descriptors, …) via
   `Descriptors.CalcMolDescriptors(mol)` (available in modern RDKit; returns a
   dict — drop NaN/constant columns afterwards).
3. **Morgan fingerprints** (= ECFP): circular substructure hashes. Use the
   modern generator API, not the deprecated one:
   ```python
   from rdkit.Chem import rdFingerprintGenerator
   gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
   fp = gen.GetFingerprintAsNumPy(mol)
   ```
   radius=2, 2048 bits ≈ "ECFP4", the de-facto standard.
4. **Murcko scaffolds** for the scaffold split
   (`rdkit.Chem.Scaffolds.MurckoScaffold`).

### 2.2 The models
- **XGBoost** (main) and **RandomForest** (sanity check) on the three feature
  sets: descriptors / fingerprints / concatenated.
- Light tuning only (n_estimators, max_depth, learning_rate). The point is a
  *representative* baseline, not a leaderboard entry.

### 2.3 The two splits and why both matter
- **Random split**: standard, comparable to most published QM9 numbers.
- **Scaffold split**: group molecules by Murcko scaffold so the test set
  contains *unseen chemotypes*. Always worse; it is the honest estimate of how
  the model will behave on genuinely new additive molecules — which is the
  deployment scenario for our mentor's project.

### 2.4 What "success" looks like (expected numbers)
For **gap** on a random split, fingerprint/descriptor + tree models on QM9
typically land in the **~0.25–0.5 eV MAE** range. Context from the literature:
- A plain 2D graph GNN (Morris 1-GNN, no 3D coordinates) reaches ~0.12 eV on
  full-QM9 gap — already better than any fixed-descriptor model.
- Classical KRR with physics-inspired 3D representations (BoB, BAML, FCHL)
  reached ~0.1–0.15 eV for HOMO/LUMO — better than fingerprints, but these
  need 3D geometries, which we deliberately avoid in Phase 1.
- 2D-input models are commonly reported as **2–5× worse in MAE** than
  3D-coordinate models on QM9.

**Sanity band:** gap MAE ≈ 0.25–0.5 eV → era reproduced correctly.
MAE ≈ 2 eV → bug (almost always the Hartree→eV conversion).
MAE ≈ 0.01 eV → leakage (check that features never see the target/test rows).

### 2.5 Why we still care about this baseline for the real project
Our in-house DFT dataset (ESP, ion binding energy for molecular additives) will
be *small* (likely 10²–10³ molecules). On small datasets, descriptor + gradient
boosting is frequently competitive with — or better than — GNNs, which are
data-hungry. Phase 1 is therefore not a warm-up; it is plausibly the production
model for the first iteration of the mentor's agent workflow.

---

## 3. Phase 2 — SchNet / MPNN (learned-representation era)

### 3.1 The two canonical papers
- **MPNN** — Gilmer et al., "Neural Message Passing for Quantum Chemistry,"
  ICML 2017 (arXiv:1704.01212). Unified framework: nodes (atoms) exchange
  learned "messages" along edges (bonds/distances) for T rounds, then a readout
  aggregates node states into a molecule-level prediction. Reported near-DFT
  accuracy on most QM9 targets; HOMO/LUMO/gap MAEs sit in the same
  0.04–0.07 eV band as SchNet below.
- **SchNet** — Schütt et al., *J. Chem. Phys.* 148, 241722 (2018)
  (arXiv:1706.08566). Key idea: **continuous-filter convolutions** — filters are
  generated from interatomic *distances* (via RBF expansion), so the model
  consumes raw 3D coordinates + atomic numbers, is rotation/translation
  invariant, and needs no bond graph at all.

### 3.2 Published SchNet numbers we compare against (QM9, ~110k train)
| Target | SchNet MAE | Unit |
|---|---|---|
| U0 (atomization energy) | ~0.31–0.32 kcal/mol (≈0.014 eV) | eV |
| HOMO | ~0.041 eV | eV |
| LUMO | ~0.034 eV | eV |
| gap | ~0.063 eV | eV |
GNN literature consensus band for gap (SchNet / MPNN / MEGNet / GCN variants):
**0.06–0.09 eV** — i.e., roughly **5–8× better** than our Phase 1 baseline.

Data-efficiency note relevant to our small-data future: on oligothiophene
datasets, SchNet reached < 0.1 eV MAE for gap-type targets with only ~5k
training molecules, and improvements beyond ~20k points were modest. This is
the argument for pretraining on QM9 → fine-tuning on our small DFT set
(transfer learning, Section 8 of the Mena review).

### 3.3 How we will run it (Phase 2 spec, requires Colab T4)
- Implementation: **PyTorch Geometric** — `torch_geometric.datasets.QM9`
  ships the 130,831-molecule dataset with 3D coordinates, and
  `torch_geometric.nn.models.SchNet` is a maintained reference implementation
  (there is also a `SchNet.from_qm9_pretrained` helper for sanity checks).
- Protocol: train on a 25–50k subset first (T4-friendly, a few hours), target =
  gap; standard-normalize the target; report test MAE in eV; then scale up if
  the trend matches the published learning curve.
- Success criterion: **match the trend, not the exact number.** With 50k
  training molecules expect gap MAE roughly 0.08–0.12 eV; the exact paper
  numbers used ~110k train + longer schedules.
- Important protocol difference vs Phase 1: SchNet uses QM9's **DFT-optimized
  3D coordinates**. Our in-house molecules will need a geometry step (RDKit
  ETKDG embedding + optional cheap optimization) before a SchNet-style model
  can be applied — one reason 2D models (Phase 1, chemprop) stay attractive
  for the additives project.

### 3.4 Where chemprop (D-MPNN) fits
Chemprop's directed MPNN is the industrial descendant of Gilmer's MPNN that
works from SMILES alone (2D, no coordinates), handles small datasets,
ensembling, and uncertainty out of the box — making it the most likely tool we
point at the mentor's ESP/binding-energy CSV in Phase 3. Reference: Yang et
al., *J. Chem. Inf. Model.* 59, 3370 (2019).

---

## 4. Repro checklist (both phases)

- [ ] Row count after cleaning ≈ 130–134k; dropped-SMILES count logged.
- [ ] homo/lumo/gap converted Hartree → eV (×27.2114); gap values ~0.5–12 eV, positive.
- [ ] Same splits (seed=42) reused across all models — never re-split per model.
- [ ] Report MAE + R² per target, per feature set, per split → `results/metrics.json`.
- [ ] Scaffold-split MAE reported next to random-split MAE in every table.
- [ ] Phase 2 only: normalize target, report eV, note training-set size next to
      every MAE (GNN numbers are meaningless without it).

## 5. Key references
1. Ramakrishnan et al., QM9 dataset, *Sci. Data* 1, 140022 (2014).
2. Gilmer et al., Neural Message Passing for Quantum Chemistry, ICML 2017, arXiv:1704.01212.
3. Schütt et al., SchNet — a deep learning architecture for molecules and materials, *J. Chem. Phys.* 148, 241722 (2018).
4. Rogers & Hahn, Extended-Connectivity Fingerprints, *J. Chem. Inf. Model.* 50, 742 (2010).
5. Faber et al., Prediction errors of molecular ML models lower than hybrid DFT, *JCTC* 13, 5255 (2017). (Descriptor-era benchmark on QM9.)
6. Yang et al., Analyzing Learned Molecular Representations (chemprop D-MPNN), *JCIM* 59, 3370 (2019).
7. Mena, Blaskovits, Lin, Andrienko, Organic Materials of Tomorrow, *Adv. Mater.* 2026, DOI 10.1002/adma.202523667 — Sections 2 (representations), 3 (GNNs), 8 (transfer learning) map directly onto this repo's phases.
