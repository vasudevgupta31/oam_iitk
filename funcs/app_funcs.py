import configparser
import os
import time
import re
from datetime import datetime
import streamlit as st
import pandas as pd
import glob
import joblib



def load_backend_docs():
    """Load the backend architecture documentation from the README file"""
    with open("README.md", "r") as f:
        content = f.read()
        
    # Extract the Backend Architecture section
    if "## **Backend Architecture & Implementation**" in content:
        start_idx = content.find("## **Backend Architecture & Implementation**")
        # Find the next major section header which would be the end of our section
        next_sections = ["## **Installation**", "## **Usage**", "## **Configuration**"]
        end_indices = [content.find(section, start_idx) for section in next_sections if content.find(section, start_idx) != -1]
        
        if end_indices:
            # Get the closest next section
            end_idx = min(end_indices)
        else:
            # If no next section found, take everything to the end
            end_idx = len(content)
            
        # Extract the section
        backend_docs = content[start_idx:end_idx].strip()
        return backend_docs
    return "Backend documentation not found in README.md"



def load_config(config_path="config.ini"):
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path)
    else:
        # Create default sections if file doesn't exist
        config["PROCESSING"] = {}
        config["MODEL"] = {}
        config["INPUT"] = {}
        config["BEAM"] = {}
        config["SAMPLING"] = {}
        config["Communication"] = {}
    return config


# Function to ensure input_data directory exists
def ensure_input_data_dir(input_data_dir="input_data"):
    os.makedirs(input_data_dir, exist_ok=True)
    return input_data_dir

# Function to save uploaded file
def save_uploaded_file(uploaded_file, input_data_dir="input_data"):
    file_path = os.path.join(input_data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to save config
def save_config(config, config_path="config.ini"):
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    return True

# Function to validate email
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

# Function to update logs
def update_logs(message, log_placeholder=None):
    timestamp = time.strftime('%H:%M:%S')
    log_entry = f"{timestamp} - {message}"
    st.session_state.logs.append(log_entry)
    
    if log_placeholder:
        log_placeholder.code('\n'.join(st.session_state.logs), language='bash')
    
    # Also print to console for debugging
    print(log_entry)


def get_experiment_folders(base_dir="memory"):
        if not os.path.exists(base_dir):
            return []

        experiment_folders = []
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path):
                # Check if this is an experiment folder (should have certain subdirectories)
                if any([os.path.exists(os.path.join(folder_path, subdir)) for subdir in 
                    ["models", "generated_samples"]]):
                    experiment_folders.append(folder)
        
        return sorted(experiment_folders, reverse=True)  # Latest first


# Function to get experiment metadata
def get_experiment_metadata(experiment_name, base_dir="memory"):
    ini_file_path = os.path.join(base_dir, experiment_name ,'config.ini')
    config_file = configparser.ConfigParser()
    config_file.read(ini_file_path)
    metadata = {section: dict(config_file[section]) for section in config_file.sections()}
    exp_dir = os.path.join(base_dir, experiment_name)

    # Check created date (use folder creation time)
    if os.path.exists(exp_dir):
        created_time = os.path.getctime(exp_dir)
        metadata["created"] = datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')

    # Check if model exists
    model_dir = os.path.join(exp_dir, "models")
    model_files = os.listdir(model_dir)
    if 'history' in model_files:
        metadata["model_status"] = "Trained"

    # Check if sampling done
    sampling_dir = os.path.join(exp_dir, "generated_samples")
    sampling_files = sorted(set([file.split('_batch')[0] for file in os.listdir(sampling_dir)]))
    if len(sampling_files) == int(metadata['SAMPLING']['last_n_epochs']):
        metadata["sampling_status"] = "Completed"
    else:
        metadata["sampling_status"] = "Incomplete"

    # Check if results exist
    results_dir = os.path.join(exp_dir, "output")
    if os.path.exists(results_dir):
        try:
            csv_output = pd.read_csv(os.path.join(results_dir, "molecules_totalabundance.csv"))
        except FileNotFoundError:
            csv_output = pd.DataFrame()
            metadata["novo_analysis_status"] = "Incomplete"
        else:
            metadata["novo_analysis_status"] = "Completed"
        finally:
            metadata["num_samples"] = csv_output.shape[0]
    return metadata



# Function to load training history
def load_training_history(experiment_name, base_dir="memory"):
    history_path = os.path.join(base_dir, experiment_name, "models", "history")
    if os.path.exists(history_path):
        try:
            return joblib.load(history_path)
        except:
            return None
    return None

# Function to load sample results
def load_sample_results(experiment_name, base_dir="memory"):
    results_path = os.path.join(base_dir, experiment_name, "output" ,"molecules_totalabundance_bpp.csv")
    if os.path.exists(results_path):
        try:
            return pd.read_csv(results_path)
        except:
            return None
    return None

# Function to load Tanimoto similarity data
def load_similarity_data(experiment_name, base_dir="memory"):
    # Check for the summary CSV file
    summary_path = os.path.join(base_dir, experiment_name, "beam_search", "similarity_summary.csv")
    if not os.path.exists(summary_path):
        # Also check for tanimoto_similarity_summary.csv (the default name)
        summary_path = os.path.join(base_dir, experiment_name, "beam_search", "tanimoto_similarity_summary.csv")
        if not os.path.exists(summary_path):
            return None
    
    # Check for the plots
    plots = {}
    main_plot_path = os.path.join(base_dir, experiment_name, "beam_search", "tanimoto_similarity_plots.png")
    if os.path.exists(main_plot_path):
        plots["main"] = main_plot_path
    
    # Look for the best epoch histogram
    best_epoch_plots = glob.glob(os.path.join(base_dir, experiment_name, "beam_search", "best_epoch_*_similarity_histogram.png"))
    if best_epoch_plots:
        plots["best_epoch"] = best_epoch_plots[0]
    
    # Try to load the summary CSV
    try:
        summary_df = pd.read_csv(summary_path)
        return {
            "summary": summary_df,
            "plots": plots
        }
    except Exception as e:
        print(f"Error loading similarity data: {str(e)}")
        return None
    
# Function to load TSNE data
def load_tsne_data(experiment_name, base_dir="memory"):
    combinedset_path = os.path.join(base_dir, experiment_name, "tsne", "combinedset")
    chembl_data_path = os.path.join(base_dir, experiment_name, "tsne", "chembl_data")
    
    if os.path.exists(combinedset_path) and os.path.exists(chembl_data_path):
        try:
            import joblib
            combinedset = joblib.load(combinedset_path)
            chembl_data = joblib.load(chembl_data_path)
            return {"combinedset": combinedset, "chembl_data": chembl_data}
        except Exception as e:
            print(f"Error loading TSNE data: {str(e)}")
            return None
    return None
