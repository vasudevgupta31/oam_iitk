# Copyright (c) 2019 ETH Zurich

import os
import time
import warnings

import joblib
from tqdm import tqdm
import numpy as np
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')
from keras.models import load_model
from pandas import read_csv

import configs.fixed_params as FP
from configs.path_config import exp_name, exp_models_path, exp_gen_samples_path, config_file, exp_beam_search_path

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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


def one_hot_encode(token_lists, n_chars):
    
    output = np.zeros((len(token_lists), len(token_lists[0]), n_chars))
    for i, token_list in enumerate(token_lists):
        for j, token in enumerate(token_list):
            output[i, j, int(token)] = 1
    return output
         
def sample(model, temp, start_char, end_char, max_len, indices_token, token_indices):

    n_chars = len(indices_token)

    seed_token = [token_indices[start_char]]
    generated = indices_token[str(seed_token[0])]
    
    while generated[-1] != end_char and len(generated) < max_len:
        x_seed = one_hot_encode([seed_token], n_chars)
        full_preds = model.predict(x_seed, verbose=0)[0]
        logits = full_preds[-1]
        
        probas, next_char_ind = get_token_proba(logits, temp)
                
        next_char = indices_token[str(next_char_ind)]
        generated += next_char
        seed_token += [next_char_ind]
            
    return generated

def get_token_proba(preds, temp):

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    
    probas = exp_preds / np.sum(exp_preds)
    char_ind = np.argmax(np.random.multinomial(1, probas, 1))
    
    return probas, char_ind


def softmax(preds):
    return np.exp(preds)/np.sum(np.exp(preds))


def perform_sampling():

    start = time.time()
    save_path = exp_gen_samples_path
    os.makedirs(save_path, exist_ok=True)

    # Parameters to sample novo smiles
    temp = float(config_file['SAMPLING']['temp'])
    n_sample = int(config_file['SAMPLING']['n_sample'])

    if n_sample>5000:
        warnings.warn('You will sample more than 5000 SMILES; this will take a while')
    
    last_n_epochs = int(config_file['SAMPLING']['last_n_epochs'])
    total_epochs = len(joblib.load(os.path.join(exp_models_path, 'history'))['loss'])
    start_epoch = total_epochs - last_n_epochs + 1
    end_epoch = total_epochs

    max_len = int(config_file['PROCESSING']['max_len'])
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES

    print('\nSTART SAMPLING')
    for epoch in range(start_epoch, end_epoch + 1):
        model_path = os.path.join(exp_models_path, f'epoch_{epoch:02d}.h5')
        model = load_model(model_path)
        print(f'Sampling from model saved at epoch {epoch} with temp {temp}')

        generated_smi = []
        counter=0
        start_sampling = time.time()
        for n in tqdm(range(n_sample)):
            gen_sample = sample(model, temp, start_char, end_char, max_len+1, indices_token, token_indices)
            generated_smi.append(gen_sample)

        # From 100 molecules to sample,
        # we indicate the current status
        # to the user
            if n_sample>=100:
                if len(generated_smi)%int(0.1*n_sample)==0:
                    counter+=10
                    delta_time = time.time()-start_sampling
                    start_sampling = start_sampling + delta_time
                    print(f'Status for model from epoch {epoch}: {counter}% of the molecules sampled in {delta_time:.2f} seconds')
        joblib.dump(value=generated_smi, filename=os.path.join(save_path, f"epoch_{epoch}"))

    end = time.time()
    print(f'SAMPLING DONE for model from epoch {epoch} in {end-start:.2f} seconds')  


def process_batch(model, batch_idx, num_batches, epoch, batch_size, n_sample, temp, start_char, 
               end_char, max_len, indices_token, token_indices, save_path):
    """
    Worker function to process a single batch of samples.
    
    Args:
        model: The loaded model to generate samples
        batch_idx: Current batch index
        num_batches: Total number of batches
        epoch: Current epoch
        batch_size: Size of each batch
        n_sample: Total number of samples
        temp: Temperature for sampling
        start_char, end_char, max_len, indices_token, token_indices: Parameters for the sample function
        save_path: Path to save the generated samples
        
    Returns:
        Number of samples processed
    """
    generated_smi = []
    counter = 0
    start_sampling = time.time()
    
    # Calculate how many samples to generate in this batch
    samples_in_batch = min(batch_size, n_sample - (batch_idx * batch_size))
    
    print(f'Starting batch {batch_idx+1}/{num_batches} for epoch {epoch} ({samples_in_batch} samples)')
    
    for n in tqdm(range(samples_in_batch)):
        gen_sample = sample(model, temp, start_char, end_char, max_len+1, indices_token, token_indices)
        generated_smi.append(gen_sample)

        # Display progress indicators for batches with many samples
        if samples_in_batch >= 100:
            if (n+1) % int(0.1 * samples_in_batch) == 0:
                counter += 10
                delta_time = time.time() - start_sampling
                start_sampling = time.time()
                print(f'Status for model from epoch {epoch}, batch {batch_idx+1}: {counter}% of the molecules sampled in {delta_time:.2f} seconds')
    
    # Save the batch with the batch number in the filename
    batch_filename = os.path.join(save_path, f"epoch_{epoch}_batch_{batch_idx+1}")
    joblib.dump(value=generated_smi, filename=batch_filename)
    print(f'Saved batch {batch_idx+1}/{num_batches} for epoch {epoch}')


# def perform_sampling_batches(batch_size=100):
#     start = time.time()
#     save_path = exp_gen_samples_path
#     os.makedirs(save_path, exist_ok=True)

#     # Parameters to sample novo smiles
#     temp = float(config_file['SAMPLING']['temp'])
#     n_sample = int(config_file['SAMPLING']['n_sample'])

#     if n_sample > 5000:
#         warnings.warn('You will sample more than 5000 SMILES; this will take a while')

#     last_n_epochs = int(config_file['SAMPLING']['last_n_epochs'])
#     total_epochs = len(joblib.load(os.path.join(exp_models_path, 'history'))['loss'])
#     start_epoch = total_epochs - last_n_epochs + 1
#     end_epoch = total_epochs

#     max_len = int(config_file['PROCESSING']['max_len'])
#     start_char = FP.PROCESSING_FIXED['start_char']
#     end_char = FP.PROCESSING_FIXED['end_char']
#     indices_token = FP.INDICES_TOKEN
#     token_indices = FP.TOKEN_INDICES

#     print('\nSTART SAMPLING')
    
#     for epoch in range(start_epoch, end_epoch + 1):
#         model_path = os.path.join(exp_models_path, f'epoch_{epoch:02d}.h5')
#         model = load_model(model_path)
#         print(f'Sampling from model saved at epoch {epoch} with temp {temp}')

#         # Calculate number of batches
#         num_batches = n_sample // batch_size
#         if n_sample % batch_size != 0:
#             num_batches += 1

#         # Process each batch sequentially
#         for batch_idx in range(num_batches):
#             process_batch(
#                 model=model, 
#                 batch_idx=batch_idx, 
#                 num_batches=num_batches, 
#                 epoch=epoch, 
#                 batch_size=batch_size, 
#                 n_sample=n_sample,
#                 temp=temp, 
#                 start_char=start_char, 
#                 end_char=end_char, 
#                 max_len=max_len, 
#                 indices_token=indices_token, 
#                 token_indices=token_indices,
#                 save_path=save_path
#             )

#     end = time.time()
#     print(f'SAMPLING DONE: processed across all epochs in {end-start:.2f} seconds')




import multiprocessing as mp
from functools import partial


def process_batch(batch_idx, num_batches, epoch, model_path, batch_size, n_sample, temp, start_char, 
                 end_char, max_len, indices_token, token_indices, save_path):
    """
    Worker function to process a single batch of samples.
    Modified to work with multiprocessing by loading the model inside the function.
    Each process works independently and saves its own results.
    
    Args:
        batch_idx: Current batch index
        num_batches: Total number of batches
        epoch: Current epoch
        model_path: Path to the model file to load
        batch_size: Size of each batch
        n_sample: Total number of samples
        temp: Temperature for sampling
        start_char, end_char, max_len, indices_token, token_indices: Parameters for the sample function
        save_path: Path to save the generated samples
    """
    # Load model inside the worker
    model = load_model(model_path)
    
    generated_smi = []
    counter = 0
    start_sampling = time.time()
    
    # Calculate how many samples to generate in this batch
    samples_in_batch = min(batch_size, n_sample - (batch_idx * batch_size))
    
    print(f'Starting batch {batch_idx+1}/{num_batches} for epoch {epoch} ({samples_in_batch} samples)')
    
    for n in tqdm(range(samples_in_batch)):
        gen_sample = sample(model, temp, start_char, end_char, max_len+1, indices_token, token_indices)
        generated_smi.append(gen_sample)

        # Display progress indicators for batches with many samples
        if samples_in_batch >= 100:
            if (n+1) % int(0.1 * samples_in_batch) == 0:
                counter += 10
                delta_time = time.time() - start_sampling
                start_sampling = time.time()
                print(f'Status for model from epoch {epoch}, batch {batch_idx+1}: {counter}% of the molecules sampled in {delta_time:.2f} seconds')
    
    # Save the batch with the batch number in the filename
    batch_filename = os.path.join(save_path, f"epoch_{epoch}_batch_{batch_idx+1}")
    joblib.dump(value=generated_smi, filename=batch_filename)
    print(f'Saved batch {batch_idx+1}/{num_batches} for epoch {epoch}')



def perform_sampling_batches(batch_size = 100, num_processes=None):
    start = time.time()
    save_path = exp_gen_samples_path
    os.makedirs(save_path, exist_ok=True)

    # Parameters to sample novo smiles
    temp = float(config_file['SAMPLING']['temp'])
    n_sample = int(config_file['SAMPLING']['n_sample'])
 
    # Determine number of processes to use (adjust based on your system)
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # Use up to 8 cores or however many are available
    print(f'Using {num_processes} processes for parallel sampling')

    if n_sample > 5000:
        warnings.warn('You will sample more than 5000 SMILES; this will take a while')

    max_len = int(config_file['PROCESSING']['max_len'])
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES

    print('\nSTART SAMPLING')
    similarity_df_path = os.path.join(exp_beam_search_path, "tanimoto_similarity_summary.csv")
    top_epochs = (read_csv(similarity_df_path)
                     .sort_values(by='avg_similarity', ascending=False)
                     .head(5)['epoch']
                     .values)

    for epoch in top_epochs:
        model_path = os.path.join(exp_models_path, f'epoch_{epoch:02d}.h5')
        print(f'Sampling from model saved at epoch {epoch} with temp {temp}')

        # Calculate number of batches
        num_batches = n_sample // batch_size
        if n_sample % batch_size != 0:
            num_batches += 1

        # Create a partial function with all the fixed parameters
        process_batch_partial = partial(
            process_batch,
            num_batches=num_batches,
            epoch=epoch,
            model_path=model_path,
            batch_size=batch_size,
            n_sample=n_sample,
            temp=temp,
            start_char=start_char,
            end_char=end_char,
            max_len=max_len,
            indices_token=indices_token,
            token_indices=token_indices,
            save_path=save_path
        )

        # Process batches in parallel - no result collection
        with mp.Pool(processes=num_processes) as pool:
            batch_indices = list(range(num_batches))
            # Apply the function to all batches and wait for all to complete
            pool.map(process_batch_partial, batch_indices)
        print(f'Completed all batches for epoch {epoch}')

    end = time.time()
    print(f'SAMPLING DONE: processed all samples across all epochs in {end-start:.2f} seconds')
