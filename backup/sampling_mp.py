import os
import time
import warnings
import joblib
from loguru import logger
import multiprocessing as mp
from functools import partial
import gc


from funcs.helpers_training import load_model
from backup.helpers_sampling import sample
from configs.path_config import (exp_name, 
                                 config_file, 
                                 exp_memory_path, 
                                 exp_gen_samples_path, 
                                 exp_models_path)
import configs.fixed_params as FP


def sample_epoch(epoch, config_file, exp_gen_samples_path, exp_models_path, 
                 pad_char, start_char, end_char, indices_token, token_indices,
                 use_gpu=False):
    """
    Process a single epoch for SMILES sampling.
    This function will be called by each process.
    """
    import os
    import time
    import joblib
    import logging
    import gc
    import tensorflow as tf
    
    # Force CPU usage to avoid GPU memory issues in multiprocessing
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # If using GPU, configure memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Don't allocate all memory at once
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
                )
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    
    # Make TensorFlow quieter
    tf.get_logger().setLevel('ERROR')
    
    from tensorflow.keras.models import load_model
    from tensorflow.keras import backend as K

    # Set up logger for this process
    logger = logging.getLogger(f"sampling_epoch_{epoch}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    # Get parameters from config
    temp = float(config_file['SAMPLING']['temp'])
    n_sample = int(config_file['SAMPLING']['n_sample'])
    max_len = int(config_file['PROCESSING']['max_len'])

    save_path = os.path.join(exp_gen_samples_path, f'{epoch}_{temp}')
    if not os.path.isfile(save_path):
        model_path = os.path.join(exp_models_path, f'epoch_{epoch:02d}.h5')
        model = load_model(model_path)
        logger.info("=========================================================================")
        logger.info(f'Sampling {n_sample} samples from model at epoch {epoch} with temp {temp}')
        logger.info("=========================================================================")

        start_sampling = time.time()
        
        # Create directory for batch files if it doesn't exist
        batch_dir = save_path + '_batches'
        os.makedirs(batch_dir, exist_ok=True)
        
        # Track which batches were created
        completed_batches = []
        
        # Create a metadata file to store information about all batches
        metadata_path = os.path.join(batch_dir, 'metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"epoch: {epoch}\n")
            f.write(f"temperature: {temp}\n")
            f.write(f"total_samples: {n_sample}\n")
            f.write("---\n")
        
        # Process in smaller batches to manage memory better
        batch_size = min(100, n_sample)  # Process in batches of 100 or fewer
        
        for batch_start in range(0, n_sample, batch_size):
            batch_end = min(batch_start + batch_size, n_sample)
            batch_samples = []
            
            for i in range(batch_start, batch_end):
                print(f"Sampled - sample {i} from epoch - {epoch}")
                smi = sample(model, temp, start_char, end_char, max_len + 1, indices_token, token_indices)
                batch_samples.append(smi)

            # Save this batch as a separate file
            batch_id = len(completed_batches) + 1
            batch_filename = f"batch_{batch_id:03d}.pkl"
            batch_path = os.path.join(batch_dir, batch_filename)
            
            try:
                joblib.dump(value=batch_samples, filename=batch_path)
                completed_batches.append(batch_path)

                # Update metadata
                with open(metadata_path, 'a') as f:
                    f.write(f"batch_{batch_id:03d}: {len(batch_samples)} samples, range {batch_start}-{batch_end-1}\n")
                
                logger.info(f"Saved batch {batch_id} with {len(batch_samples)} samples to {batch_path}")
            except Exception as e:
                logger.error(f"Error saving batch {batch_id}: {e}")
            
            # Clear batch from memory
            del batch_samples
            
            # Clear some memory after each batch
            gc.collect()
            if tf.__version__.startswith('2'):
                # TF 2.x specific cleanup
                tf.keras.backend.clear_session()
        
        # Write completion flag to metadata
        with open(metadata_path, 'a') as f:
            f.write(f"---\ncompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"total_batches: {len(completed_batches)}\n")
        
        # Create a simple json file that indicates this epoch is complete
        completion_marker = save_path + '.complete'
        with open(completion_marker, 'w') as f:
            f.write(f"Sampling completed with {len(completed_batches)} batches at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info(f"All batches saved to {batch_dir}")
        logger.info(f"To combine batches later, use the batch files in {batch_dir}")

        # Clean up - explicitly delete model and clear session
        del model
        K.clear_session()
        gc.collect()

        delta_time = time.time() - start_sampling
        logger.info(f'SAMPLING DONE for model from epoch {epoch} in {delta_time:.2f} seconds')

        return epoch, delta_time
    else:
        logger.info(f'Results for epoch {epoch} already exist, skipping')
        return epoch, 0


def generate_samples_multiprocessing(use_gpu=False):
    """
    Performs SMILES sequence sampling using a trained model with multiprocessing.

    This function loads pre-trained models from different epochs and generates 
    SMILES sequences using temperature-based sampling in parallel. The generated 
    sequences are saved to a specified directory.
    
    Args:
        use_gpu (bool): Whether to use GPU in worker processes. Default is False,
                        which forces CPU usage to avoid memory conflicts.
    
    Returns:
        None: Saves the generated SMILES sequences to files.
    """
    start = time.time()

    # Parameters for SMILES sampling
    n_sample = int(config_file['SAMPLING']['n_sample'])
    if n_sample > 5000:
        warnings.warn('You will sample more than 5000 SMILES; this will take a while')
    
    # Get specific values from FP module that we need instead of passing the whole module
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES

    # Determine which epochs to process
    last_n_epochs = int(config_file['SAMPLING']['last_n_epochs'])
    total_epochs = len(joblib.load(os.path.join(exp_models_path, 'history'))['loss'])
    start_epoch = total_epochs - last_n_epochs + 1
    end_epoch = total_epochs

    # Process fewer epochs concurrently to reduce memory pressure
    # Using fewer processes is generally better for GPU memory management
    num_cpus = min(3, mp.cpu_count() - 1)  # Reduced from 5 to 3

    # With GPU, limit processes more aggressively to avoid memory issues
    if use_gpu:
        num_cpus = 1  # Running only 1 process at a time when using GPU

    # Limit to the number of epochs we need to process
    num_processes = min(mp.cpu_count() - 2, last_n_epochs)

    logger.info(f'\nSTART SAMPLING with {num_processes} parallel processes using {"GPU" if use_gpu else "CPU"}')

    # Use spawn context instead of fork to avoid pickling issues
    mp_context = mp.get_context("spawn")

    # Create a pool of workers
    with mp_context.Pool(processes=num_processes) as pool:
        # Prepare the partial function with fixed arguments, passing individual values instead of modules
        worker_func = partial(
            sample_epoch, 
            config_file=config_file,
            exp_gen_samples_path=exp_gen_samples_path,
            exp_models_path=exp_models_path,
            pad_char=pad_char,
            start_char=start_char,
            end_char=end_char,
            indices_token=indices_token,
            token_indices=token_indices,
            use_gpu=use_gpu
        )

        # Map the function to all epochs, but limit concurrent execution
        epochs = range(start_epoch, end_epoch + 1)
        
        # Use imap instead of map to process epochs one at a time
        # This helps control memory usage better
        results = list(pool.imap(worker_func, epochs))

    # Log results
    for epoch, delta_time in results:
        if delta_time > 0:
            logger.info(f'Epoch {epoch} completed in {delta_time:.2f} seconds')

    end = time.time()
    logger.info(f'Total Sampling Time: {end - start:.2f} seconds')
    
    # Final garbage collection
    gc.collect()
