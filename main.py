"""
HGM - Hierarchical Generative Model Execution Pipeline

This script orchestrates the end-to-end execution of the HGM pipeline, including:
1. Data Processing: Prepares and processes input SMILES data.
2. Model Training: Trains a deep learning model on processed data.
3. Sampling: Generates molecular structures from the trained model.
4. Analysis: Evaluates and filters generated molecules.

Logging:
- Logs are saved in the `logs_path` directory with a timestamped filename.
- Log rotation is enabled at 10MB with retention for 10 days.

Usage:
Run this script as the main entry point to execute the pipeline:
```bash
python main.py
```
"""

import os
import shutil
import datetime
from loguru import logger

from configs.path_config import (config_file, 
                                 exp_memory_path, 
                                 input_file_path, 
                                 clean_experiment_memory, 
                                 logs_path)
from processes.processing import preprocessing
from processes.training import train_network
from processes.beam_search import beam_search
from processes.sampling_org import perform_sampling, perform_sampling_batches
from processes.novo_analysis_org import perform_novo_analysis


# Set up logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_path, f"{timestamp}.log")
logger.add(log_file, rotation="10MB", level="INFO")


def main():

    # # 0. Check Memory/interim path
    # logger.info("==================> Checking and cleaning experiment memory...")
    # clean_experiment_memory()
    # logger.info("Memory check completed.")
    # logger.info("======================================================")
    # shutil.copy("config.ini", exp_memory_path)
    init_time = datetime.datetime.now()
    
    # # # 1. Data Processing
    # logger.info("==================> Starting data processing...")

    # start_time = datetime.datetime.now()
    # preprocessing(split=float(config_file['PROCESSING']['split']), 
    #             input_data_file=input_file_path, 
    #             augmentation=int(config_file['PROCESSING']['augmentation']), 
    #             min_len=int(config_file['PROCESSING']['min_len']), 
    #             max_len=int(config_file['PROCESSING']['max_len']), 
    #             output_save_dir=exp_memory_path, 
    #             verbose=True)
    # processing_time = datetime.datetime.now() - start_time
    # logger.info(f"Data processing completed successfully in {processing_time}.")
    # logger.info("======================================================")

    # # 2. Training
    # logger.info("==================> Starting network training...")
    # start_time = datetime.datetime.now()
    # train_network()
    # training_time = datetime.datetime.now() - start_time
    # logger.info(f"Network training completed successfully in {training_time}")
    # logger.info("======================================================")

    # # # 3. Beam Search
    # logger.info("==================> Starting beam search...")
    # start_time = datetime.datetime.now()
    # beam_search()
    # beam_search_time = datetime.datetime.now() - start_time
    # logger.info(f"Beam search completed successfully in {beam_search_time}")
    # logger.info("======================================================")

    # 4. Sampling
    logger.info("==================> Generating samples from trained network.")   
    start_time = datetime.datetime.now()
    perform_sampling_batches(batch_size=250, num_processes=4)  # Batch multi processing is done at samples_level within an epoch so 1000 samples can be done parallely with 4 num processes at once with a btach size of 250
    sampling_time = datetime.datetime.now() - start_time
    logger.info(f"Sample generation completed in {sampling_time}.")
    logger.info("======================================================")

    # 5. Analysis
    logger.info("==================> Performing Novo analysis on generated samples.")
    start_time = datetime.datetime.now()
    perform_novo_analysis()
    analysis_time = datetime.datetime.now() - start_time
    logger.info(f"Novo analysis completed successfully in {analysis_time}.")
    logger.info("======================================================")

    # # Log Total Execution Time
    total_execution_time = datetime.datetime.now() - init_time
    logger.info(f"🚀 Pipeline execution completed in {total_execution_time}.")
    logger.info("======================================================")


if __name__ == '__main__':
    main()
