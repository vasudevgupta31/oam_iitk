
import os
import time
import collections
import random
from multiprocessing import Pool

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

import funcs.helpers_data_io as hp
import funcs.helper_chem as hp_chem
import configs.fixed_params as FP


def load_data(data_path, min_len, max_len, verbose=False):
    """
    Optimized function to load a .txt file of SMILES,
    prune SMILES by length and check that they
    are convertible to RDKit mol format.

    Parameters:
    - data_path (str): Path to the dataset.
    - min_len (int): Minimum length of SMILES to be kept in the dataset.
    - max_len (int): Maximum length of SMILES to be kept in the dataset.

    Returns:
    - data (list): List of valid SMILES strings.
    - data_rdkit (list): List of valid RDKit molecule objects.
    """
    with open(data_path) as f:
        lines = f.read().splitlines()  # Read all lines at once (efficient I/O)

    # List comprehension for faster filtering and conversion
    valid_smiles = [
        sm for sm in lines if min_len <= len(sm) <= max_len
    ]

    # Vectorized molecule conversion using filter()
    data_rdkit = list(filter(None, (Chem.MolFromSmiles(sm) for sm in valid_smiles)))
    data = [Chem.MolToSmiles(mol) for mol in data_rdkit]  # Ensure canonical SMILES format

    if verbose:
        print(f'Size of dataset after filtering and RDKit conversion: {len(data)}')
    return data, data_rdkit


def randomSmiles(mol):
    """
    Generates a randomized SMILES representation of a molecule
    by shuffling atom indices while maintaining valid molecular structure.

    Parameters:
    - mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
    - str: Randomized SMILES string.
    """
    mol.SetProp("_canonicalRankingNumbers", "True")

    # Use NumPy for faster shuffling if available
    idxs = list(range(mol.GetNumAtoms()))
    random.shuffle(idxs)  

    # Vectorized atom property setting
    for atom, idx in zip(mol.GetAtoms(), idxs):
        atom.SetProp("_canonicalRankingNumber", str(idx))

    return Chem.MolToSmiles(mol, doRandom=True)  # Use built-in randomization


def smile_augmentation(smile, augmentation, min_len, max_len):
    """
    Generates a set of unique randomized SMILES representations for a given molecule.

    Parameters:
    - smile (str): Input SMILES string.
    - augmentation (int): Number of unique augmented SMILES to generate.
    - min_len (int): Minimum length of SMILES.
    - max_len (int): Maximum length of SMILES.

    Returns:
    - list: Unique augmented SMILES representations.
    """
    mol = Chem.MolFromSmiles(smile)
    if not mol:
        return []

    # Use a set for uniqueness and generate only as many as needed
    s = set()
    attempts = 0

    while len(s) < augmentation and attempts < augmentation * 2:  # Limit max attempts
        smiles = Chem.MolToSmiles(mol, doRandom=True)  # Built-in randomization
        if min_len <= len(smiles) <= max_len:
            s.add(smiles)
        attempts += 1  # Prevent infinite loops if unique SMILES are limited

    return list(s)


def augment_single(smile_args):
    """ Wrapper function for multiprocessing """
    smile, augmentation, min_len, max_len = smile_args
    return smile_augmentation(smile, augmentation, min_len, max_len)


def augment_dataset(data_ori, augmentation, min_len, max_len, verbose=False):
    """
    Optimized function to augment a dataset using multiprocessing.

    Parameters:
    - data_ori (list): List of SMILES strings to augment.
    - augmentation (int): Number of alternative SMILES to create.
    - min_len (int): Minimum length of alternative SMILES.
    - max_len (int): Maximum length of alternative SMILES.
    - verbose (bool): Whether to print progress.

    Returns:
    - list: Augmented SMILES representations of `data_ori`.
    """

    num_workers = min(os.cpu_count(), len(data_ori)) - 2  # Adjust workers based on data size and leave 2 for OS
    args_list = [(smile, augmentation, min_len, max_len) for smile in data_ori]  # Pack arguments

    # Use multiprocessing to speed up the augmentation
    with Pool(processes=num_workers) as pool:
        augmented_data = pool.map(augment_single, args_list)

    # Flatten the list efficiently
    all_alternative_smi = [sm for sublist in augmented_data for sm in sublist]

    if verbose:
        print(f'Data augmentation done; number of new SMILES: {len(all_alternative_smi)}')

    return all_alternative_smi
