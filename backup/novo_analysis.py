import os
import time
import joblib
from loguru import logger
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

from configs import fixed_params as FP
from configs.path_config import (exp_name, 
                                 config_file, 
                                 exp_gen_samples_path,
                                 exp_output_path)


def novo_analysis():
    """
    Analyzes interim SMILES samples, validates molecules, and generates final filtered results.
    Processes all epochs and batches within each epoch.
    """
    start = time.time()

    # Get back the experiment parameters
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    min_len = int(config_file['PROCESSING']['min_len'])
    max_len = int(config_file['PROCESSING']['max_len'])
    temp = float(config_file['SAMPLING']['temp'])

    logger.info('\nSTART NOVO ANALYSIS')

    # Path to generated data
    path_gen = exp_gen_samples_path
    d_abundance = {}
    t0 = time.time()

    # Get all epoch directories
    epoch_dirs = [d for d in sorted(os.listdir(path_gen)) if os.path.isdir(os.path.join(path_gen, d))]

    for epoch_dir in epoch_dirs:
        epoch_path = os.path.join(path_gen, epoch_dir)
        logger.info(f'Processing Epoch: {epoch_dir}')

        epoch_valids = []
        epoch_n_valid = 0
        epoch_d_mol_count = {}

        # Process all batch files within this epoch
        batch_files = [f for f in sorted(os.listdir(epoch_path)) if os.path.isfile(os.path.join(epoch_path, f)) and f.startswith('batch_')]

        for batch_file in batch_files:
            batch_path = os.path.join(epoch_path, batch_file)
            logger.info(f'  Processing Batch: {batch_file}')

            try:
                data = joblib.load(batch_path)

                for gen_smile in data:
                
                    if len(gen_smile)!=0 and isinstance(gen_smile, str):
                        gen_smile = gen_smile.replace(pad_char,'')
                        gen_smile = gen_smile.replace(end_char,'')
                        gen_smile = gen_smile.replace(start_char,'')
                        print(gen_smile)
                        mol = Chem.MolFromSmiles(gen_smile)
                        # print(f"Debug2", mol)
                        # if mol is not None:
                        #     cans = Chem.MolToSmiles(mol)
                        #     if len(cans)>=1:
                        #         n_valid+=1
                        #         valids.append(cans)
                        #         if cans in d_mol_count:
                        #             d_mol_count[cans] += 1
                        #         else:
                        #             d_mol_count[cans] = 1
                        #         if cans in d_abundance:
                        #             d_abundance[cans] += 1
                        #         else:
                        #             d_abundance[cans] = 1
            except Exception as e:
                logger.error(f"Error processing batch {batch_file}: {str(e)}")

        # Save epoch-specific results
        sorted_epoch_d_mol_count = sorted(epoch_d_mol_count.items(), key=lambda x: x[1], reverse=True)
        logger.info(f'Generated {epoch_n_valid} valid SMILES in Epoch {epoch_dir}')
        
        # Save epoch abundance
        with open(os.path.join(exp_output_path, f'abundance_epoch_{epoch_dir}.txt'), 'w+') as f:
            for smi, count in sorted_epoch_d_mol_count:
                f.write(f'{smi} \t {count}\n')
        
        # Save epoch molecules
        novo_name = os.path.join(exp_output_path, f'molecules_epoch_{epoch_dir}')
        with open(f'{novo_name}.txt', 'w+') as f:
            for item in epoch_valids:
                f.write("%s\n" % item)
    
    # Save global abundance across all epochs
    sorted_d_abundance = sorted(d_abundance.items(), key=lambda x: x[1], reverse=True)
    novo_name = os.path.join(exp_output_path, 'molecules')
    with open(f'{novo_name}_totalabundance_{temp}.txt', 'w+') as f:
        for smi, count in sorted_d_abundance:
            f.write(f'{smi} \t {count}\n')
    
    logger.info(f"Total unique molecules: {len(d_abundance)}")
    logger.info(f"Total processing time: {time.time() - start:.2f} seconds")
