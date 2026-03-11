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