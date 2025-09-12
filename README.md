# ML HER2 Inhibitor Discovery Pipeline
**MSc Project — [Krishna Prajapati]**

> An integrated computational pipeline using **XGBoost** and **LightGBM** to predict HER2 inhibitors from ChEMBL/PubChem/ZINC chemical libraries, followed by docking validation (HER2, PDB: 3PP0).

## Overview
This project demonstrates an ML-first approach to virtual screening for HER2-targeted compounds. Models are trained on curated HER2 bioactivity data and interpreted using SHAP. Top-ranked candidates are prepared for structural validation via AutoDock Vina.

## Key Achievements
- Curated HER2-specific bioactivity datasets (ChEMBL CHEMBL1824, PubChem, ZINC).  
- Generated molecular representations: **ECFP4**, **MACCS**, and RDKit/Mordred descriptors.  
- Trained and tuned **XGBoost** and **LightGBM** classifiers (scaffold split, cross-validation).  
- Applied **SHAP** for model interpretability and feature importance.  
- Produced top candidates for docking and benchmarked against Neratinib & Lapatinib.

## Results (summary)
- **LightGBM:** Accuracy = 1.000 | ROC-AUC = 1.000 | F1 = 1.000  
- **XGBoost:** Accuracy ≈ 0.9993 | ROC-AUC ≈ 0.99998 | F1 ≈ 0.9995  
- Feature importance: A small subset of descriptors strongly influences predictions (see `figures/feature_importance.png`).

> *See `notebooks/` for detailed code, evaluation, and SHAP analysis.*

## Repository Structure


