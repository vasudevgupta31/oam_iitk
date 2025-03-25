# Create a custom configuration for your datasets
datasets_config = {
    "dataset1": {
        "file": "training_data/GIP.csv", 
        "smiles_col": "Ligand SMILES",
        "target_col": "EC50 (nM)",
        "model_file": "ec50_gip"
    },
    "dataset2": {
        "file": "training_data/GCGR.csv",
        "smiles_col": "Ligand SMILES",
        "target_col": "EC50 (nM)",
        "model_file": "ec50_gcgr"
    },
    "dataset3": {
        "file": "training_data/GLP-1R.csv",
        "smiles_col": "Ligand SMILES", 
        "target_col": "EC50 (nM)",
        "model_file": "ec50_glp1r"
    },
    "dataset4": {
        "file": "training_data/GIP.csv", 
        "smiles_col": "Ligand SMILES",
        "target_col": "Kd (nM)",
        "model_file": "kd_gip"
    },
    "dataset5": {
        "file": "training_data/GCGR.csv",
        "smiles_col": "Ligand SMILES",
        "target_col": "Kd (nM)",
        "model_file": "kd_gcgr"
    },
    "dataset6": {
        "file": "training_data/GLP-1R.csv",
        "smiles_col": "Ligand SMILES", 
        "target_col": "Kd (nM)",
        "model_file": "kd_glp1r"
    },
}
