
# her2_drug_discovery.py
# ======================
# HER2 Drug Discovery using ML (XGBoost and LightGBM)
# Author: Krishna
# Description: This script downloads kinase inhibitors from ChEMBL, extracts molecular features,
# trains XGBoost & LightGBM models, evaluates them, and predicts top HER2 inhibitors.

# ==============================
# Install required libraries
# ==============================
# Run this section only once
# !pip install rdkit pandas scikit-learn xgboost lightgbm joblib matplotlib seaborn

# ==============================
# Import libraries
# ==============================
import pandas as pd
import numpy as np
import sqlite3
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
import joblib

# ==============================
# Download ChEMBL kinase inhibitors
# ==============================
# !wget ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35_sqlite.tar.gz
# !tar -xvzf chembl_35_sqlite.tar.gz

conn = sqlite3.connect("chembl_35/chembl_35_sqlite/chembl_35.db")
query = '''
SELECT DISTINCT
m.chembl_id as molecule_chembl_id,
cs.canonical_smiles,
a.standard_value,
a.standard_units
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary t ON ass.tid = t.tid
JOIN compound_structures cs ON a.molregno = cs.molregno
JOIN molecule_dictionary m ON a.molregno = m.molregno
WHERE a.standard_type = 'IC50'
AND a.standard_value <= 10000
AND a.standard_units = 'nM'
AND t.tax_id = 9606
AND t.pref_name LIKE '%kinase%';
'''
df = pd.read_sql_query(query, conn)
df.drop_duplicates(subset=["molecule_chembl_id"], inplace=True)
df.to_csv("kinase_inhibitor_library.csv", index=False)

# ==============================
# Descriptor calculation
# ==============================
def rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'MolWt': None, 'TPSA': None, 'LogP': None, 'HBD': None, 'HBA': None, 'RotB': None, 'NumRings': None}
    return {
        'MolWt': Descriptors.MolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotB': Descriptors.NumRotatableBonds(mol),
        'NumRings': rdMolDescriptors.CalcNumRings(mol)
    }

query = pd.read_csv("kinase_inhibitor_library.csv")
desc_list = [rdkit_descriptors(s) for s in query['canonical_smiles']]
desc_df = pd.DataFrame(desc_list)
query_data = pd.concat([query.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
query_data = query_data.dropna(subset=['MolWt'])

fps = []
for smi in query_data['canonical_smiles']:
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fps.append(list(map(int, fp.ToBitString())))
fp_df = pd.DataFrame(fps, columns=[f'fp_{i}' for i in range(2048)])
query_data = pd.concat([query_data.reset_index(drop=True), fp_df.reset_index(drop=True)], axis=1)
query_data.to_csv("query_processed.csv", index=False)

# ==============================
# FDA approved HER2 drugs
# ==============================
drug_list = [
("Neratinib","CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=CC(=C(C=C3)OCC4=CC=CC=N4)Cl)C#N)NC(=O)/C=C/CN(C)C"),
("Lapatinib","CS(=O(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl"),
("Tucatinib","CC1=C(C=CC(=C1)NC2=NC=NC3=C2C=C(C=C3)NC4=NC(CO4)(C)C)OC5=CC6=NC=NN6C=C5")
]
fda_df = pd.DataFrame(drug_list, columns=["DrugName", "canonical_smiles"])
fda_df.to_csv("fda_approved_her2_drugs.csv", index=False)

def get_ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

fda_df["FP"] = fda_df["canonical_smiles"].apply(get_ecfp4)
fda_df.to_pickle("fda_fp.pkl")

# ==============================
# Dataset expansion and splitting by scaffold
# ==============================
df = pd.read_csv("query_processed.csv")

def generate_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except:
        return None

df["scaffold"] = df["canonical_smiles"].apply(generate_scaffold)
scaffolds = df["scaffold"].dropna().unique()
np.random.shuffle(scaffolds)

train_cut, val_cut = int(0.7*len(scaffolds)), int(0.85*len(scaffolds))
train_df = df[df["scaffold"].isin(scaffolds[:train_cut])]
val_df = df[df["scaffold"].isin(scaffolds[train_cut:val_cut])]
test_df = df[df["scaffold"].isin(scaffolds[val_cut:])]

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# ==============================
# Train & evaluate models
# ==============================
feature_cols = [c for c in train_df.columns if c not in ["pIC50","canonical_smiles","scaffold","molecule_chembl_id","standard_units"]]
joblib.dump(feature_cols, "feature_columns.pkl")

X_train = train_df[feature_cols].astype(float)
y_train = (train_df["pIC50"] >= 6).astype(int)
X_val = val_df[feature_cols].astype(float)
y_val = (val_df["pIC50"] >= 6).astype(int)
X_test = test_df[feature_cols].astype(float)
y_test = (test_df["pIC50"] >= 6).astype(int)

def train_and_eval(model, name):
    if isinstance(model, xgb.XGBClassifier):
        model.set_params(early_stopping_rounds=20, eval_metric="logloss")
    if isinstance(model, lgb.LGBMClassifier):
        model.set_params(early_stopping_rounds=20, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "F1": f1_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds)
    }
    return metrics, model

xgb_model = xgb.XGBClassifier(use_label_encoder=False)
xgb_metrics, xgb_model = train_and_eval(xgb_model, "XGBoost")
lgbm_model = lgb.LGBMClassifier()
lgbm_metrics, lgbm_model = train_and_eval(lgbm_model, "LightGBM")
joblib.dump(xgb_model, "xgboost_her2.pkl")
joblib.dump(lgbm_model, "lightgbm_her2.pkl")

# ==============================
# Prediction + Tanimoto similarity
# ==============================
df = pd.read_csv("query_processed.csv")
fda_df = pd.read_pickle("fda_fp.pkl")
X = df[feature_cols].astype(float)

df["XGB_Prob"] = xgb_model.predict_proba(X)[:, 1]
df["LGBM_Prob"] = lgbm_model.predict_proba(X)[:, 1]
df["Avg_Prob"] = (df["XGB_Prob"] + df["LGBM_Prob"]) / 2
df["FP"] = df["canonical_smiles"].apply(get_ecfp4)

similarities = []
for fp in df["FP"]:
    if fp is None:
        similarities.append(0)
    else:
        max_sim = max((DataStructs.TanimotoSimilarity(fp, fda_fp) for fda_fp in fda_df["FP"] if fda_fp is not None), default=0)
        similarities.append(max_sim)
df["Max_Tanimoto_Sim"] = similarities

candidates = df.sort_values(by=["Avg_Prob", "Max_Tanimoto_Sim"], ascending=False).head(100)
candidates.to_csv("top100_candidates.csv", index=False)
