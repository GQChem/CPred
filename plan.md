# CPred Local Implementation Plan

## Context
CPred (Lo et al., 2012, NAR) predicts viable circular permutation (CP) cleavage sites in proteins. The original web server is dead and no open-source implementation exists. We will reimplement it locally in Python, using the algorithm details from the NAR paper and its companion PLoS ONE paper (Lo et al., 2012, doi:10.1371/journal.pone.0031791). Training data (Datasets S1-S5) are freely downloadable from the PLoS ONE supplementary materials.

## Project Structure

```
CPred/
├── pyproject.toml
├── requirements.txt
├── cpred/
│   ├── __init__.py
│   ├── cli.py                      # CLI entry point
│   ├── pipeline.py                 # PDB -> features -> prediction orchestrator
│   ├── io/
│   │   ├── __init__.py
│   │   ├── pdb_parser.py           # BioPython PDB loading -> ProteinStructure
│   │   └── output.py               # CSV/TSV/JSON formatters
│   ├── features/
│   │   ├── __init__.py
│   │   ├── sequence_propensity.py  # Cat A: AA/di/oligo propensities (±3 window)
│   │   ├── secondary_structure.py  # Cat B: DSSP, Ramachandran, kappa-alpha propensities
│   │   ├── tertiary_structure.py   # Cat C: RSA, CN, WCN, CM, depth, B-factor, H-bonds
│   │   ├── contact_network.py      # Closeness centrality, farness measures
│   │   ├── gnm.py                  # Gaussian Network Model fluctuation
│   │   ├── structural_codes.py     # Ramachandran code & kappa-alpha code assignment
│   │   ├── standardization.py      # Feature inversion + Z-score normalization
│   │   └── window.py               # ±3 residue window averaging
│   ├── propensity/
│   │   ├── __init__.py
│   │   ├── tables.py               # Pre-computed propensity lookup tables
│   │   └── scoring.py              # Propensity formula: Sp(i) = ((fe-fc)/fc) * (1-pval)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ann.py                  # 3-layer ANN (PyTorch): 46 -> 23 -> 1, sigmoid
│   │   ├── svm.py                  # SVM (sklearn, RBF kernel, probability=True)
│   │   ├── random_forest.py        # RF (sklearn, 500 trees, entropy criterion)
│   │   ├── hierarchical.py         # HI: tree-structured weighted feature averaging
│   │   └── ensemble.py             # Simple average of 4 model outputs
│   ├── training/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Parse supplementary datasets
│   │   ├── train.py                # Train all 4 models
│   │   └── evaluate.py             # 10-fold CV, AUC, sensitivity, specificity, MCC
│   └── data/
│       ├── propensity_tables/      # JSON lookup tables (shipped pre-computed)
│       ├── hi_tree.json            # HI tree structure and weights
│       └── trained_models/         # Serialized models (.pt, .pkl)
├── scripts/
│   ├── download_data.py            # Download PLoS ONE supplementary + PDB files
│   ├── build_propensity_tables.py  # Compute propensity tables from datasets
│   └── train_all_models.py         # Full training pipeline
└── tests/
    ├── test_features.py
    ├── test_models.py
    ├── test_pipeline.py
    └── fixtures/                   # Test PDB files (e.g., 1rx4 DHFR)
```

## Implementation Phases

### Phase 1: Infrastructure & Data Acquisition
1. **Create `pyproject.toml` and `requirements.txt`** — deps: biopython, numpy, scipy, scikit-learn, torch, networkx, pandas; external: mkdssp
2. **`scripts/download_data.py`** — download PLoS ONE supplementary files (Dataset S1-S5, Table S1-S3) + PDB structures for Dataset T and DHFR
3. **`cpred/io/pdb_parser.py`** — load PDB via BioPython, extract `ProteinStructure` object with CA/CB coords, sequence, atom sets

### Phase 2: Feature Extraction (Category C — Tertiary Structure)
4. **`cpred/features/tertiary_structure.py`** — compute per-residue: RSA (via DSSP), CN (CB within 6.4Å), WCN (1/d² sum over all CA pairs), CM (dist to centroid), B-factor (from PDB), H-bond count (from DSSP)
5. **`cpred/features/contact_network.py`** — build residue interaction graph (heavy atom dist <4Å), compute closeness centrality (NetworkX), compute farness measures (Fb, Fhpho, Fb+hpho, Fb*hpho using 1/d² weighting to buried/hydrophobic residue groups)
6. **`cpred/features/gnm.py`** — Kirchhoff matrix from CA coords (7Å cutoff), eigendecompose, MSF = Σ(u²ᵢₖ/λₖ) for non-trivial modes. Pure NumPy, optional ProDy.
7. **`cpred/features/structural_codes.py`** — Ramachandran codes: (phi,psi) → 23 letters via 36×36 grid. Kappa-alpha codes: CA bond angle + dihedral → 23 letters via 36×18 grid.
8. **`cpred/features/window.py`** — ±3 residue window averaging for all Cat C features

### Phase 3: Feature Extraction (Categories A & B — Propensities)
9. **`cpred/propensity/scoring.py`** — implement propensity formula: Sp(i) = ((fe(i)−fc(i))/fc(i)) × (1−p(i)), with permutation test for p-values
10. **`scripts/build_propensity_tables.py`** — compute propensity tables from nrCPsite_cpdb-40 (experimental) vs whole protein sequences (comparison) for: single AA, di-residue, oligo-residue, DSSP states, Ramachandran codes, kappa-alpha codes
11. **`cpred/features/sequence_propensity.py`** — look up AA propensities at each position in ±3 window
12. **`cpred/features/secondary_structure.py`** — look up DSSP/Ramachandran/kappa-alpha propensities at each position in ±3 window

### Phase 4: Feature Pipeline Assembly
13. **`cpred/features/standardization.py`** — invert features where low value = viable (CN, WCN, closeness, H-bonds, farness, depth); then Z-score normalize per protein: z = (x−μ)/σ
14. **`cpred/pipeline.py`** — orchestrate: parse PDB → extract all 46 features → window average Cat C → invert → Z-score → assemble feature matrix

### Phase 5: Machine Learning Models
15. **`cpred/models/random_forest.py`** — sklearn RF, 500 trees, entropy, predict_proba
16. **`cpred/models/svm.py`** — sklearn SVC, RBF kernel, probability=True, grid search C/gamma
17. **`cpred/models/ann.py`** — PyTorch: Linear(46,23) → Sigmoid → Linear(23,1) → Sigmoid, BCELoss, Adam
18. **`cpred/models/hierarchical.py`** — tree config in JSON, bottom-up weighted average, weights optimized via grid search on training AUC
19. **`cpred/models/ensemble.py`** — arithmetic mean of 4 model probability outputs

### Phase 6: Training, CLI, Testing
20. **`cpred/training/data_loader.py`** — parse Dataset T + DHFR into (features, labels) arrays
21. **`cpred/training/train.py`** — train all 4 models on Dataset T + DHFR combined
22. **`scripts/train_all_models.py`** — end-to-end: download data → compute features → build propensity tables → train → serialize models
23. **`cpred/cli.py`** — `cpred predict structure.pdb --chain A --threshold 0.5 -o results.tsv` and `cpred train --data-dir ./data --output-dir ./models`
24. **`cpred/io/output.py`** — format results as TSV/CSV/JSON with columns: residue_number, amino_acid, probability_score, predicted_viable

## Key Algorithm Details

- **CP site window**: ±3 residues around cleavage point (7 residues total)
- **Ensemble**: simple average of ANN + SVM + RF + HI → probability score ∈ [0,1]
- **Threshold**: ≥0.5 = viable (default); ≥0.85 = high confidence (precision ≈ 1.0)
- **Contact number cutoff**: 6.4Å (CB atoms, CA for Gly)
- **Closeness graph**: heavy atom distance <4Å
- **GNM cutoff**: 7.0Å (CA atoms)
- **Buried residues**: RSA < 10%
- **Hydrophobic residues**: A, V, I, L, M, F, W, P

## Target Performance (from paper Table 1)
| Metric | Training (10-fold CV) | DHFR (independent) |
|--------|----------------------|---------------------|
| AUC | 0.940 | 0.91 |
| Sensitivity | 0.889 | 0.71 |
| Specificity | 0.898 | 0.92 |

## Verification
1. Run on DHFR (PDB 1rx4) — compare probability profile to Figure 1b in paper
2. 10-fold CV on Dataset T — target AUC ≈ 0.94 (within 0.03)
3. Predict on nrCPDB-40 — target sensitivity ≈ 0.75
4. Unit tests for each feature extractor (verify value ranges and known properties)

## External Dependencies
- **mkdssp** (required) — secondary structure, RSA, H-bonds
- **MSMS** (optional) — residue depth; fallback: approximate as distance to nearest surface atom (RSA > 20%)
- **reduce** (optional) — add hydrogens; gracefully skip if absent
