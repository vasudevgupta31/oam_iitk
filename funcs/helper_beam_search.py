import os
import sys
import gc
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from rdkit import Chem
from rdkit import rdBase
import joblib

rdBase.DisableLog('rdApp.*')
from rdkit.Chem import Draw
from configs import fixed_params as FP
from funcs.helpers_data_io import write_in_file


def int_to_smile(array, indices_token, pad_char):
    """
    From an array of int, return a list of
    molecules in string smile format
    Note: remove the padding char
    """
    all_mols = []
    for seq in array:
        new_mol = [indices_token[str(int(x))] for x in seq]
        all_mols.append(''.join(new_mol).replace(pad_char, ''))
    return all_mols


def save_smiles(candidates, scores, indices_token, start_char, pad_char, end_char, save_path, name_file):
    """
    Save the valid SMILES, along with
    their score and a picture representation.
    """
    all_smi = []
    all_mols = []  # rdkit format
    all_scores = []

    # Check SMILES validity
    for x, s in zip(candidates, scores):
        # Convert sequence of indices to SMILES string
        ints = [indices_token[str(idx)] for idx in x]
        smiles = ''.join(ints).replace(start_char, '').replace(pad_char, '').replace(end_char, '')
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and len(smiles) > 1:
            all_smi.append(smiles)
            all_mols.append(mol)
            all_scores.append(s)
    
    d_smi_to_score = dict(zip(all_smi, all_scores))
    write_in_file(path_to_file=os.path.join(save_path, f'{name_file}_SMILES.txt'), data=all_smi)
    joblib.dump(value=d_smi_to_score, filename=os.path.join(save_path, f'{name_file}_smi_to_score.pkl'))


def beam_search_decoder_optimized(k, model, vocab_size, max_len, indices_token, token_indices, name_file,
                                  start_char, pad_char, end_char, save_path, verbose):
    """
    Memory-optimized beam search decoder that uses batched prediction and more efficient data structures.
    
    Args:
        k (int): Beam width
        model: Keras model for prediction
        vocab_size (int): Size of vocabulary
        max_len (int): Maximum sequence length
        indices_token (dict): Mapping from indices to tokens
        token_indices (dict): Mapping from tokens to indices
        name_file (str): Base name for output files
        start_char (str): Start character
        pad_char (str): Padding character
        end_char (str): End character
        save_path (str): Directory to save results
        verbose (bool): Whether to print progress
    """
    # Create a lookup dictionary for faster access
    print("Starting beam search with width:", k)
    sys.stdout.flush()
    
    # Prepare the seed token (start character)
    seed_idx = token_indices[start_char]
    
    # Setup for beam search
    max_len = max_len + 1  # Account for start char
    
    # Initialize with start token
    sequences = [[seed_idx]]  # List of sequences, each sequence is a list of token indices
    sequence_scores = [0.0]   # Log probabilities for each sequence
    
    # Setup progress bar
    pbar = tqdm(total=max_len, desc="Beam search progress")
    
    # Loop over sequence length
    for t in range(max_len):
        # Progress bar update
        pbar.update(1)
        
        # Store all candidate sequences for this timestep
        all_candidates = []
        
        # If no sequences remain, break
        if not sequences:
            break
            
        # Prepare batch input for model prediction
        batch_size = len(sequences)
        
        # Convert sequences to model input format (one-hot encoding at the sequence level)
        # This is more memory efficient than one-hot encoding the entire sequence history
        X_batch = np.zeros((batch_size, t+1, vocab_size))
        for i, seq in enumerate(sequences):
            for j, token_idx in enumerate(seq):
                X_batch[i, j, token_idx] = 1
        
        # Get predictions for all sequences in the batch
        predictions = model.predict(X_batch, verbose=0)
        
        # Process each sequence
        for i, (seq, score) in enumerate(zip(sequences, sequence_scores)):
            # Get log probabilities for next token
            log_probs = np.log(predictions[i, -1])
            
            # Get top k tokens
            top_indices = np.argsort(log_probs)[-k:]
            
            # Add each candidate to our list
            for idx in top_indices:
                candidate_seq = list(seq) + [idx]
                candidate_score = score + log_probs[idx]
                all_candidates.append((candidate_seq, candidate_score))
        
        # Select top k candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        sequences = [seq for seq, _ in all_candidates[:k]]
        sequence_scores = [score for _, score in all_candidates[:k]]
        
        # Clear memory
        del X_batch
        del predictions
        del all_candidates
        gc.collect()
    
    pbar.close()
    
    print(f'Number of candidates: {len(sequences)} out of {k}')
    sys.stdout.flush()
    
    # Save results
    save_smiles(
        candidates=sequences,
        scores=sequence_scores,
        indices_token=indices_token,
        start_char=start_char,
        pad_char=pad_char,
        end_char=end_char,
        save_path=save_path,
        name_file=name_file
    )
    
    return sequences, sequence_scores
