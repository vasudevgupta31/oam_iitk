import os
import time
import sys
import gc
import glob
import multiprocessing
from functools import partial
import concurrent.futures

import tensorflow as tf
from loguru import logger
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# Import the fixed beam search function
from funcs.helper_beam_search import beam_search_decoder_optimized
from configs.path_config import config_file, exp_name, exp_beam_search_path, exp_models_path, pretrained_model_path
import configs.fixed_params as FP



# Function to process a single model (for parallel execution)
def process_single_model(model_path, params):
    """Process a single model file for beam search.
    
    Args:
        model_path: Path to the model file
        params: Dictionary of parameters for beam search
    
    Returns:
        Tuple of (success, epoch_num)
    """
    # Extract parameters from the dictionary
    width = params['width']
    max_len = params['max_len']
    vocab_size = params['vocab_size']
    indices_token = params['indices_token']
    fixed_token_indices = params['fixed_token_indices']
    start_char = params['start_char']
    pad_char = params['pad_char']
    end_char = params['end_char']
    save_path = params['save_path']
    
    # Extract epoch number for naming
    epoch_num = os.path.basename(model_path).split("_")[1].split(".")[0]
    name_file = f"epoch_{epoch_num}"
    
    # Check if results already exist for this epoch
    output_file = os.path.join(save_path, f'{name_file}_smi_to_score.pkl')
    
    if os.path.isfile(output_file):
        logger.info(f'Results already exist for {name_file}, skipping')
        return True, epoch_num
    
    logger.info(f'Running beam search for model: {name_file}')
    sys.stdout.flush()
    
    # Create a new TensorFlow session for this process
    K.clear_session()
    
    try:
        # Load the model
        model = load_model(model_path)
        
        # Run optimized beam search
        beam_search_decoder_optimized(
            k=width, 
            model=model, 
            vocab_size=vocab_size, 
            max_len=max_len,
            indices_token=indices_token, 
            token_indices=fixed_token_indices,
            name_file=name_file,
            start_char=start_char, 
            pad_char=pad_char, 
            end_char=end_char, 
            save_path=save_path, 
            verbose=True
        )
        success = True
    except Exception as e:
        logger.error(f"Error during beam search for {name_file}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        success = False
    finally:
        # Clean up to prevent memory leaks
        if 'model' in locals():
            del model
        K.clear_session()
        gc.collect()
    
    return success, epoch_num


def beam_search():
    """
    Run beam search with optimized memory management and parallel processing.
    Uses epoch model files from experiment folder instead of pretrained model.
    """
    # Clear any existing TensorFlow session to start fresh
    K.clear_session()
    
    start = time.time()
    logger.info("Starting beam search for all epoch models with parallel processing")
    
    # Get parameters from config
    max_len = int(config_file['PROCESSING']['max_len'])
    width = int(config_file['BEAM']['width'])
    sys.stdout.flush()
    save_path = exp_beam_search_path
    
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # Generator parameters
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES
    vocab_size = len(indices_token)

    # Debug output for token mappings
    logger.info(f"Token mappings type check:")
    logger.info(f"- start_char: {start_char}, token_idx: {token_indices[start_char]}, type: {type(token_indices[start_char])}")
    
    # Make sure token_indices values are integers
    fixed_token_indices = {}
    for k, v in token_indices.items():
        try:
            fixed_token_indices[k] = int(v)
        except (ValueError, TypeError):
            logger.error(f"Invalid token index: {k} -> {v}, type: {type(v)}")
            fixed_token_indices[k] = 0  # Fallback to a safe value
    
    # Set memory growth for GPU to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    
    # Find all epoch model files in the models directory
    epoch_model_files = sorted(glob.glob(os.path.join(exp_models_path, "epoch_*.h5")))
    logger.info(f"Found {len(epoch_model_files)} epoch model files")
    
    if not epoch_model_files:
        logger.error(f"No epoch model files found in {exp_models_path}")
        return False
    
    # Create a dictionary of parameters to pass to the worker function
    params = {
        'width': width,
        'max_len': max_len,
        'vocab_size': vocab_size,
        'indices_token': indices_token,
        'fixed_token_indices': fixed_token_indices,
        'start_char': start_char,
        'pad_char': pad_char,
        'end_char': end_char,
        'save_path': save_path
    }
    
    # Determine the number of workers
    # Use half of available CPUs to avoid overloading the system
    max_workers = max(1, min(len(epoch_model_files), multiprocessing.cpu_count() // 2))
    logger.info(f"Running with {max_workers} parallel workers")

    # Process models in parallel using ThreadPoolExecutor
    # (ProcessPoolExecutor might have issues with TensorFlow)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the process_single_model function to each model path
        future_to_model = {
            executor.submit(process_single_model, model_path, params): model_path
            for model_path in epoch_model_files
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model_path = future_to_model[future]
            try:
                success, epoch_num = future.result()
                results.append((success, epoch_num))
                logger.info(f"Completed beam search for epoch {epoch_num}")
            except Exception as exc:
                logger.error(f"Error processing {model_path}: {exc}")

    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r[0])
    logger.info(f"Successfully processed {successful} out of {total} epoch models")

    end = time.time()
    logger.info(f'ALL BEAM SEARCHES COMPLETED in {end - start:.2f} seconds')
    sys.stdout.flush()
    return True
