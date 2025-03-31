import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from rdkit import DataStructs
from rdkit.Chem import PandasTools
import joblib

from configs.path_config import exp_output_path, exp_tsne_path, resources_path


def load_chembl_data():

    # Specify the paths to your CSV files
    chembl_text_path = os.path.join(exp_output_path, "molecules_totalabundance.txt")
    background_csv_path = os.path.join(resources_path, "bg_ligands.csv")

    # Reading text files
    # Load the text file manually
    smiles_list = []

    with open(chembl_text_path, 'r') as file:
        for line in file:
            # Assuming each line contains a SMILES string
            smiles_list.append(line.strip())

    # Create a DataFrame from the list (assuming SMILES is the only column needed)
    chembl_data = pd.DataFrame({
        'SMILES': smiles_list
    })
    # Reading CSV files
    # Load the CSV files into DataFrames
    background_data = pd.read_csv(background_csv_path)
    # Create a DataFrame for background smiles
    background_smiles_df = pd.DataFrame({
        'SMILES': background_data['CANONICAL_SMILES'].tolist()
    })
    return chembl_data, background_smiles_df


class FP:
    """
    Molecular fingerprint class, useful to pack features in pandas df

    Parameters
    ----------
    fp : np.array
        Features stored in numpy array
    names : list, np.array
        Names of the features
    """
    def __init__(self, fp, names):
        self.fp = fp
        self.names = names
    def __str__(self):
        return "%d bit FP" % len(self.fp)
    def __len__(self):
        return len(self.fp)
    
    
def get_cfps(mol, radius=2, nBits=1024, useFeatures=False, counts=False, dtype=np.float32):
    """Calculates circular (Morgan) fingerprint.
    http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius, default 2
    nBits : int
        Length of hashed fingerprint (without descriptors), default 1024
    useFeatures : bool
        To get feature fingerprints (FCFP) instead of normal ones (ECFP), defaults to False
    counts : bool
        If set to true it returns for each bit number of appearances of each substructure (counts). Defaults to false (fingerprint is binary)
    dtype : np.dtype
        Numpy data type for the array. Defaults to np.float32 because it is the default dtype for scikit-learn

    Returns
    -------
    ML.FP
        Fingerprint (feature) object
    """
    arr = np.zeros((1,), dtype)

    if counts is True:
        info = {}
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits, useFeatures=useFeatures)
        DataStructs.ConvertToNumpyArray(fp, arr)
    else:
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures), arr)
    return FP(arr, range(nBits))


def make_tsne(custom_paths=None):
    """
    Generate TSNE visualization for molecules.
    
    Parameters:
    -----------
    custom_paths : dict, optional
        A dictionary containing custom paths for the TSNE generation.
        Keys should include 'exp_output_path', 'exp_tsne_path', and 'resources_path'.
        If None, will use the default paths from path_config.
    """
    # Set paths if provided
    global exp_output_path, exp_tsne_path, resources_path
    
    if custom_paths is not None:
        exp_output_path = custom_paths.get('exp_output_path', exp_output_path)
        exp_tsne_path = custom_paths.get('exp_tsne_path', exp_tsne_path)
        resources_path = custom_paths.get('resources_path', resources_path)
    
    print(f"Using paths: \nOutput: {exp_output_path}\nTSNE: {exp_tsne_path}")
    
    # Load data
    chembl_data, background_smiles = load_chembl_data()
    
    # Make necessary directories
    os.makedirs(exp_tsne_path, exist_ok=True)
    
    print("Processing generated molecules...")
    #Calculate fingerprints
    PandasTools.AddMoleculeColumnToFrame(chembl_data, smilesCol='SMILES')
    chembl_data = chembl_data[~chembl_data['ROMol'].isnull()]
    chembl_data['FP'] = chembl_data['ROMol'].map(get_cfps)

    print("Processing background molecules...")
    #Calculate fingerprints
    background_smiles['SMILES'] = background_smiles['SMILES'].str.strip()
    background_smiles = background_smiles[~background_smiles['SMILES'].isna()]
    PandasTools.AddMoleculeColumnToFrame(background_smiles, smilesCol='SMILES')
    background_smiles = background_smiles[~background_smiles['ROMol'].isnull()]
    background_smiles['FP'] = background_smiles['ROMol'].map(get_cfps)
    combinedset=pd.concat([chembl_data, background_smiles], axis=0, ignore_index=True)

    print("Running TSNE for combined dataset with 3 components...")
    Z = np.array([z.fp for z in combinedset['FP']])
    model = TSNE(n_components=3, random_state=0, perplexity=30, n_iter=5000)
    tsne_combineddrugs = model.fit_transform(Z)

    print("Running TSNE for generated molecules only with 3 components...")
    Y = np.array([y.fp for y in chembl_data['FP']])
    perplexity = min(30, Y.shape[0] - 5)
    model = TSNE(n_components=3, random_state=0, perplexity=perplexity, n_iter=5000)
    tsne_newdrugs = model.fit_transform(Y)
    
    # Store all 3 components
    combinedset['TSNE_C1'] = tsne_combineddrugs.T[0]
    combinedset['TSNE_C2'] = tsne_combineddrugs.T[1]
    combinedset['TSNE_C3'] = tsne_combineddrugs.T[2]
    
    chembl_data['TSNE_C1'] = tsne_newdrugs.T[0]
    chembl_data['TSNE_C2'] = tsne_newdrugs.T[1]
    chembl_data['TSNE_C3'] = tsne_newdrugs.T[2]

    print("Saving TSNE results...")
    print(f"Combined shape: {combinedset.shape}")
    print(f"Generated molecules shape: {chembl_data.shape}")

    joblib.dump(combinedset, filename=os.path.join(exp_tsne_path, "combinedset"))
    joblib.dump(chembl_data, filename=os.path.join(exp_tsne_path, "chembl_data"))
    
    print("TSNE visualization complete!")
    return True
