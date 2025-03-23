import os
import ast

import joblib
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model

from configs.path_config import (config_file, 
                                 exp_memory_path, 
                                 pretrained_model_path, 
                                 exp_models_path)
from configs.fixed_params import (PROCESSING_FIXED, 
                                  TOKEN_INDICES, 
                                  INDICES_TOKEN)
from funcs.data_generator import DataGenerator
from funcs.helpers_training import create_model_checkpoint, SeqModel



def train_network():
    """
    Trains a neural network using a sequential model.

    This function initializes neural network parameters from a configuration file,
    sets up data generators for training and validation, applies a learning rate
    reduction strategy, and performs model training with optional transfer learning
    from a pre-trained model.

    Steps:
    1. Creates required directories.
    2. Loads neural network hyperparameters from the configuration file.
    3. Initializes data generators for training and validation.
    4. Configures model checkpoints and learning rate adjustments.
    5. Builds the model architecture using `SeqModel`.
    6. Loads pre-trained model weights for transfer learning.
    7. Trains the model using the training and validation generators.
    8. Saves the loss history after training.

    Returns:
        None
    """

    # Neural net parameters
    patience_lr = int(config_file['MODEL']['patience_lr'])
    batch_size = int(config_file['MODEL']['batch_size'])
    epochs = int(config_file['MODEL']['epochs'])
    period = int(config_file['MODEL']['period'])
    n_workers = int(config_file['MODEL']['n_workers'])
    min_lr = float(config_file['MODEL']['min_lr'])
    factor = float(config_file['MODEL']['factor'])

    max_len_model = int(config_file['PROCESSING']['max_len']) + 2 # for start and end token
    pad_char = PROCESSING_FIXED['pad_char']
    start_char = PROCESSING_FIXED['start_char']
    end_char = PROCESSING_FIXED['end_char']
    indices_token = INDICES_TOKEN
    token_indices  =  TOKEN_INDICES
    vocab_size = len(indices_token)


    # Loss
    monitor = 'val_loss'
    lr_reduction = ReduceLROnPlateau(monitor=monitor, 
                                     patience=patience_lr, 
                                     verbose=0, 
                                     factor=factor, 
                                     min_lr=min_lr)
    
    # Define early stopping with momentum
    early_stopping = EarlyStopping(
                                    monitor='val_loss',          
                                    patience=5,                 
                                    min_delta=0.001,             
                                    restore_best_weights=True,   
                                    mode='min',                  
                                    verbose=1                    
                                )

    # Data Generators
    tr_generator = DataGenerator(list_IDs=joblib.load(os.path.join(exp_memory_path, 'idx_tr')),
                                batch_size=batch_size, 
                                max_len_model=max_len_model,
                                path_data=os.path.join(exp_memory_path, "full_datalist.txt"),
                                n_chars=vocab_size,
                                indices_token=indices_token,
                                token_indices=token_indices,
                                pad_char=pad_char,
                                start_char=start_char,
                                end_char=end_char,
                                shuffle=True)

    val_generator = DataGenerator(list_IDs=joblib.load(os.path.join(exp_memory_path, 'idx_val')),
                                batch_size=batch_size, 
                                max_len_model=max_len_model,
                                path_data=os.path.join(exp_memory_path, "full_datalist.txt"),
                                n_chars=vocab_size,
                                indices_token=indices_token,
                                token_indices=token_indices,
                                pad_char=pad_char,
                                start_char=start_char,
                                end_char=end_char,
                                shuffle=True)

    checkpointer = create_model_checkpoint(period=period, save_path=exp_models_path)
    layers = ast.literal_eval(config_file['MODEL']['neurons'])
    dropouts = ast.literal_eval(config_file['MODEL']['dropouts'])
    trainables = ast.literal_eval(config_file['MODEL']['trainables'])
    lr = float(config_file['MODEL']['lr'])

    # Init Model architecture as a seq model
    seqmodel = SeqModel(n_chars=vocab_size, 
                        max_length=max_len_model, 
                        layers=layers, 
                        dropouts=dropouts, 
                        trainables=trainables, 
                        lr=lr, 
                        verbose=True)

    # Transfer Learning
    pre_model = load_model(pretrained_model_path)
    pre_weights = pre_model.get_weights()
    seqmodel.model.set_weights(pre_weights)

    # Train
    history = seqmodel.model.fit(x=tr_generator,
                                validation_data=val_generator,
                                use_multiprocessing=False,
                                epochs=epochs,
                                callbacks=[checkpointer, lr_reduction],
                                workers=n_workers)
    # total_epochs_run = len(history.history['loss'])    
    # Save the loss history
    joblib.dump(value=history.history, filename=os.path.join(exp_models_path, 'history'))
