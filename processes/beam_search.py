import os
import time
import sys
import gc

import tensorflow as tf
from loguru import logger
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# Import the fixed beam search function
from funcs.helper_beam_search import beam_search_decoder_optimized
from configs.path_config import config_file, exp_name, exp_beam_search_path, pretrained_model_path
import configs.fixed_params as FP


def beam_search():
    """
    Run beam search with optimized memory management and performance.
    """
    # Clear any existing TensorFlow session to start fresh
    K.clear_session()
    
    start = time.time()
    logger.info("Starting beam search with memory optimization")
    
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

    filename = os.path.basename(pretrained_model_path)
    name_file = filename.replace('.h5', '').split('_')[0]
    output_file = os.path.join(exp_beam_search_path, f'{name_file}_smi_to_score.pkl')
    
    if not os.path.isfile(output_file):
        logger.info(f'Running beam search for model: {filename}')
        sys.stdout.flush()
        
        # Load model inside a clean TensorFlow session
        K.clear_session()
        model = load_model(pretrained_model_path)
        
        # Run optimized beam search with fixed token_indices
        try:
            beam_search_decoder_optimized(
                k=width, 
                model=model, 
                vocab_size=vocab_size, 
                max_len=max_len,
                indices_token=indices_token, 
                token_indices=fixed_token_indices,  # Use fixed version
                name_file=name_file,
                start_char=start_char, 
                pad_char=pad_char, 
                end_char=end_char, 
                save_path=save_path, 
                verbose=True
            )
        except Exception as e:
            logger.error(f"Error during beam search: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Clean up to prevent memory leaks
        del model
        K.clear_session()
        gc.collect()
    else:
        logger.info(f'Results already exist at {output_file}, skipping beam search')
    
    end = time.time()
    logger.info(f'BEAM SEARCH COMPLETED in {end - start:.2f} seconds')
    sys.stdout.flush()
    
    return True
