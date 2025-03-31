import os
import configparser
import shutil
from loguru import logger

root_path = os.getcwd()
ini_file_path = 'config.ini'
config_file = configparser.ConfigParser()
config_file.read(ini_file_path)

logs_path = os.path.join(root_path, 'logs')
os.makedirs(logs_path, exist_ok=True)

resources_path = os.path.join(root_path, 'resources')
os.makedirs(resources_path, exist_ok=True)

# Pre-Trained Model
pretrained_model_path = os.path.join('pretrained', config_file['MODEL']['pretrained_model'])
if not os.path.isfile(pretrained_model_path):
    msg = "Looks like the pretrained model file is missing in /pretrained  Please add the file and make sure the name of the added file matches name in config.ini"
    logger.error(msg)
    raise ValueError(msg)

# All child paths should be relative to the input file
input_file = config_file["INPUT"]["NameData"]
input_file_path = os.path.join("input_data", input_file)

# TO store multiruns and interim variables/states
memory_path = os.path.join(root_path, "memory")
os.makedirs(memory_path, exist_ok=True)

# Experiment name and Override
exp_name = config_file["INPUT"]["experiment_name"]
exp_memory_path = os.path.join(memory_path, exp_name)
os.makedirs(exp_memory_path, exist_ok=True)
overide_experiment = True if config_file["INPUT"]["override_experiment"].lower().startswith('y') else False

# Models path
exp_models_path = os.path.join(exp_memory_path, "models")
os.makedirs(exp_models_path, exist_ok=True)

# Gen-Samples path
exp_gen_samples_path = os.path.join(exp_memory_path, 'generated_samples')
os.makedirs(exp_gen_samples_path, exist_ok=True)

# Beam search path
exp_beam_search_path = os.path.join(exp_memory_path, 'beam_search')
os.makedirs(exp_beam_search_path, exist_ok=True)

# Experiment results path
exp_output_path = os.path.join(exp_memory_path, 'output')
os.makedirs(exp_output_path, exist_ok=True)

# Experiments T-Sne 
exp_tsne_path = os.path.join(exp_memory_path, 'tsne')
os.makedirs(exp_tsne_path, exist_ok=True)

# BPP models path
bpp_models_path = os.path.join(root_path, 'bpp', 'models')
os.makedirs(bpp_models_path, exist_ok=True)



def clean_experiment_memory():
    if exp_name in os.listdir(memory_path):
        if not overide_experiment:
            msg = f"Experiment named: `{exp_name}` already exists and override is set to `N`. In the confi.ini file, please change the experiment name to make a new experiment or change override to `Y` to redo with the same name."
            logger.error(msg)
            raise ValueError(msg)
        else:
            try:
                shutil.rmtree(exp_memory_path)
                shutil.rmtree(exp_models_path)
                shutil.rmtree(exp_gen_samples_path)
                shutil.rmtree(exp_output_path)
            except:
                pass
            
            os.makedirs(exp_memory_path, exist_ok=True)
            os.makedirs(exp_models_path, exist_ok=True)
            os.makedirs(exp_gen_samples_path, exist_ok=True)
            os.makedirs(exp_output_path, exist_ok=True)
