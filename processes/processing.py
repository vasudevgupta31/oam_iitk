# Copyright (c) 2021 ETH Zurich
import os
import random

import joblib
import numpy as np
from loguru import logger  # Import logger

import funcs.helpers_data_io as hp
from funcs.helpers_data_processing import load_data, augment_dataset


def preprocessing(split, 
                  input_data_file, 
                  augmentation, 
                  min_len, 
                  max_len, 
                  output_save_dir, 
                  verbose=True):

    logger.info("Starting data preprocessing.")
    logger.info(f"Parameters: split={split}, augmentation={augmentation}, min_len={min_len}, max_len={max_len}")

    data_ori, _ = load_data(data_path=input_data_file, 
                            min_len=min_len, 
                            max_len=max_len, 
                            verbose=verbose)

    logger.info(f"Loaded {len(data_ori)} original data samples.")

    if verbose: logger.info("Start data processing")
    
    # Define index for the tr-val split and shuffle them
    all_idx = np.arange(len(data_ori))
    idx_split = int(split * len(all_idx))
    np.random.shuffle(all_idx)

    if idx_split == 0:
        idx_tr_canon = [0]
        idx_val_canon = [0]
    elif split == 1.0:
        idx_tr_canon = all_idx
        idx_val_canon = all_idx
    else:
        idx_tr_canon = all_idx[:idx_split]
        idx_val_canon = all_idx[idx_split:]

    assert len(idx_tr_canon) != 0
    assert len(idx_val_canon) != 0

    logger.info(f"Size of the training set after split: {len(idx_tr_canon)}")
    logger.info(f"Size of the validation set after split: {len(idx_val_canon)}")

    d = dict(enumerate(data_ori))
    data_tr = [d.get(item) for item in idx_tr_canon]
    data_val = [d.get(item) for item in idx_val_canon]

    hp.write_in_file(os.path.join(output_save_dir, 'data_tr.txt'), data_tr)
    hp.write_in_file(os.path.join(output_save_dir, 'data_val.txt'), data_val)

    if augmentation > 0:
        logger.info(f"Starting data augmentation with {augmentation}-fold.")
        
        tr_aug = augment_dataset(data_tr, augmentation, min_len, max_len, verbose=False)
        val_aug = augment_dataset(data_val, augmentation, min_len, max_len, verbose=False)

        full_training_set = list(set(data_tr + tr_aug))
        random.shuffle(full_training_set)
        full_validation_set = list(set(data_val + val_aug))
        random.shuffle(full_validation_set)
        full_datalist = full_training_set + full_validation_set

        logger.info(f"Size of the training set after augmentation: {len(full_training_set)}")
        logger.info(f"Size of the validation set after augmentation: {len(full_validation_set)}")

        idx_tr = np.arange(len(full_training_set))
        idx_val = np.arange(len(full_training_set), len(full_training_set) + len(full_validation_set))

        hp.write_in_file(path_to_file=os.path.join(output_save_dir, 'full_datalist.txt'), data=full_datalist)
        joblib.dump(value=list(idx_tr), filename=os.path.join(output_save_dir, 'idx_tr'), compress=3)
        joblib.dump(value=list(idx_val), filename=os.path.join(output_save_dir, 'idx_val'), compress=3)
    else:
        hp.write_in_file(path_to_file=os.path.join(output_save_dir, 'data_ori.txt'), data=data_ori)
        joblib.dump(value=list(idx_tr_canon), filename=os.path.join(output_save_dir, 'idx_tr'), compress=3)
        joblib.dump(value=list(idx_val_canon), filename=os.path.join(output_save_dir, 'idx_val'), compress=3)
    logger.info("Data preprocessing completed successfully.")
