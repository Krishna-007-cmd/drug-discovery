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
      Model  Accuracy  ROC_AUC       F1  Precision  Recall
0   XGBoost     0.999  0.99996  0.99936    0.99873     1.0
1  LightGBM     1.000  1.00000  1.00000    1.00000     1.0
figures/feature_imp<img width="1091" height="490" alt="image" src="https://github.com/user-attachments/assets/98c15373-c6d0-49b3-b071-a79267234776" />
ortance.png.



