import os
import time
import csv
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

from loguru import logger
import joblib
import configs.fixed_params  as FP
from configs.path_config import exp_gen_samples_path, exp_output_path


def perform_novo_analysis():

    # get back the experiment parameters
    #mode = config['EXPERIMENTS']['mode']
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    
    logger.info('\nSTART NOVO ANALYSIS')
            
    # Path to the generated data
    path_gen = exp_gen_samples_path    
    
    # Path to save the novo analysis
    save_path = exp_output_path
    os.makedirs(save_path, exist_ok=True)
   
    d_abundance = {}
    t0 = time.time()
    
    for filename in sorted(os.listdir(path_gen)):
        name = filename
        data = joblib.load(os.path.join(path_gen, filename))

        valids = []
        n_valid = 0
        d_mol_count = {}
        
        for gen_smile in data:
            
            if len(gen_smile)!=0 and isinstance(gen_smile, str):
                gen_smile = gen_smile.replace(pad_char,'')
                gen_smile = gen_smile.replace(end_char,'')
                gen_smile = gen_smile.replace(start_char,'')
                
                mol = Chem.MolFromSmiles(gen_smile)
                if mol is not None:
                    cans = Chem.MolToSmiles(mol)
                    if len(cans)>=1:
                        n_valid+=1
                        valids.append(cans)
                        if cans in d_mol_count:
                            d_mol_count[cans] += 1
                        else:
                            d_mol_count[cans] = 1
                        if cans in d_abundance:
                            d_abundance[cans] += 1
                        else:
                            d_abundance[cans] = 1
        # save abundance of the generated molecules
        sorted_d_mol_count = sorted(d_mol_count.items(), key=lambda x: x[1],reverse = True)
        logger.info(f'Generated {n_valid} SMILES in Epoch {name}')

        # # we save the novo molecules also as .txt
        # novo_name = f'molecules_{name}'
        # with open(os.path.join(save_path, f'{novo_name}_abundance.txt'), 'w+') as f:
        #     for smi, count in sorted_d_mol_count:
        #         f.write(f'{smi} \t {count}\n')

        # with open(f'{novo_name}.txt', 'w+') as f:
        #     for item in valids:
        #         f.write("%s\n" % item)

    # SAVE ALL EPOCH FILES GENERATIONS IN ONE
    sorted_d_abundance = sorted(d_abundance.items(), key=lambda x: x[1],reverse = True)
    with open(os.path.join(save_path, f'molecules_totalabundance.txt'), 'w+') as f:
        for smi, count in sorted_d_abundance:
            f.write(f'{smi} \t {count}\n')
            
    # Write to CSV file
    csv_path = os.path.join(save_path, 'molecules_totalabundance.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        
        # Write the header row
        csv_writer.writerow(['SMILES', 'Count'])
        
        # Write all data rows
        for smi, count in sorted_d_abundance:
            csv_writer.writerow([smi, count])

