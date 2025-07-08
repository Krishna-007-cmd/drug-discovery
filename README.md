# drug-discovery
Integrated drug discovery against breast cancer using Machine learning and Autodock vina
Author - Krishna Prajapati
pipeline :
step 1 extreacted the data ligands from the Pubchem database using filters like molwt, Xlogp, H-bond Donner, H-bond accepter etc.
step 2 processing the dataset removing irrelavent columns
step 3 converting stuctural code like smiles into binaray  vector using morgan fingerprinting method of rdkit lib
step 4 creating the similarity score of all the ligands against fda approved drugs
step 5 final data set will form the split the data 80% trainng anf 20% testing 
step 6 model training (xgboost )
step 7 including SHAP
step 8 evaluating the accuaracy
step 9 
