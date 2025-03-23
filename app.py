import streamlit as st
import configparser
import os
import subprocess
import pandas as pd
import time
import joblib
import shutil
import sys
from pathlib import Path
from PIL import Image
import re
import glob
import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# Add the current directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="HGM Pipeline Manager",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Alternative approach using columns for left padding
logo_path = "resources/iitklogo.png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    
    col1, col2 = st.sidebar.columns([1, 50])  # Adjust these values to control padding
    with col2:
        st.image(logo, width=200)
    st.sidebar.divider()

st.markdown("<h1 style='margin-top: 0; padding-top: 0; margin-left: 0px; color: #3366cc;'>IIT Kanpur: Hit Generative Model Pipeline</h1>", unsafe_allow_html=True)


# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #1E88E5;
    }
    .info-text {
        font-size: 0.9rem;
        color: #555;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .separator {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #eee;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .log-container {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        max-height: 300px;
        overflow-y: auto;
        font-family: monospace;
    }
    .sidebar .css-1d391kg {
    padding-top: 1rem;
}
.nav-item {
    margin-bottom: 0.5rem;
    padding: 0.5rem;
    border-radius: 0.3rem;
    cursor: pointer;
}
.nav-item-active {
    background-color: #e6f3ff;
    font-weight: bold;
}
.metrics-card {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: bold;
    color: #1E88E5;
}
.metric-label {
    font-size: 0.9rem;
    color: #666;
}
.experiment-card {
    background-color: #fff;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 15px;
    transition: transform 0.2s;
}
.experiment-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.dropdown-container {
    background-color: #f9f9f9;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
    .nav-container {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    .sidebar-header {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 10px;
        color: #1E88E5;
    }
    .sidebar-divider {
        margin: 15px 0;
        border-top: 1px solid #eee;
    }
    .current-page {
        padding: 8px;
        background-color: #f0f2f6;
        border-radius: 4px;
        text-align: center;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🧪 Create Experiment"

st.sidebar.markdown(f"<div class='current-page' style='font-size: 1.4rem; font-weight: 700;'>Navigation</div>", unsafe_allow_html=True)
# Create buttons with container
st.sidebar.markdown("<div class='nav-container'>", unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)


with col1:
    create_exp_button = st.button("🧪 **Create**", key="nav_create_exp", use_container_width=True)
with col2:
    browse_exp_button = st.button("📊 **Browse**", key="nav_browse_exp", use_container_width=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Handle navigation
if create_exp_button:
    st.session_state.current_page = "🧪 Create Experiment"
elif browse_exp_button:
    st.session_state.current_page = "📊 Browse Experiments"

# Display current page with nice styling
st.sidebar.divider()
st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='current-page' style='font-size: 1rem; font-weight: 700;'>Currently on page: </div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='current-page' style='font-size: 1.4rem; font-weight: 700;'>{st.session_state.current_page}</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.divider()


# App title
if not os.path.exists(logo_path):
    st.markdown("<div class='main-header'>HGM Pipeline Manager</div>", unsafe_allow_html=True)

# Create a session state object to store state across reruns
if 'sampling_tab_use_multiprocessing' not in st.session_state:
    st.session_state.sampling_tab_use_multiprocessing = True

if 'logs' not in st.session_state:
    st.session_state.logs = []

# Define functions
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


# Function to get experiment folders
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
def get_experiment_metadata(experiment_name, base_dir="output"):
    metadata = {
        "name": experiment_name,
        "created": "Unknown",
        "status": "Unknown",
        "model_file": None,
        "results_file": None,
        "num_samples": 0
    }

    exp_dir = os.path.join(base_dir, experiment_name)
    
    # Check created date (use folder creation time)
    if os.path.exists(exp_dir):
        created_time = os.path.getctime(exp_dir)
        metadata["created"] = datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')

    # Check if model exists
    model_dir = os.path.join(exp_dir, "models")
    if os.path.exists(model_dir):
        model_files = glob.glob(os.path.join(model_dir, "*.h5"))
        if model_files:
            metadata["model_file"] = os.path.basename(model_files[-1])  # Get the latest
            metadata["status"] = "Trained"

    # Check if results exist
    results_dir = os.path.join(exp_dir, "results")
    if os.path.exists(results_dir):
        csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
        if csv_files:
            metadata["results_file"] = os.path.basename(csv_files[-1])
            metadata["status"] = "Completed"
    
    # Check samples
    samples_dir = os.path.join(exp_dir, "generated_samples")
    if os.path.exists(samples_dir):
        sample_files = glob.glob(os.path.join(samples_dir, "*.csv"))
        metadata["num_samples"] = len(sample_files)

    return metadata

# Function to load training history
def load_training_history(experiment_name, base_dir="memory"):
    history_path = os.path.join(base_dir, experiment_name, "models", "history")
    if os.path.exists(history_path):
        try:
            history = joblib.load(history_path)
        except:
            return None
    return None


# Function to load sample results
def load_sample_results(experiment_name, base_dir="output_data"):
    results_path = os.path.join(base_dir, experiment_name, "final.csv")
    if os.path.exists(results_path):
        try:
            return pd.read_csv(results_path)
        except:
            return None
    return None


# Ensure input_data directory exists
input_data_dir = ensure_input_data_dir()

# Display the selected page
if st.session_state.current_page == "🧪 Create Experiment":

    st.markdown("""
This application guides you through configuring and running the Hit Generative Model (HGM) pipeline.
Start by uploading your input file and setting an experiment name, then configure all parameters, and execute the pipeline with a single click.
""")


    # Load config file
    config = load_config()

    # Create tabs for different sections
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "**:red[Setup]**", "**Processing**", "**Model**", "**Input**", "**Beam Search**", "**Sampling**", "**:red[Communication]**", "**:rainbow[Pipeline Execution]**"
    ])

    # Setup Section - First tab for uploading files and setting experiment name
    with tab0:
        st.markdown("<div class='section-header'>Experiment Setup</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Upload your input file and set up your experiment</div>", 
                    unsafe_allow_html=True)

        # File upload
        uploaded_file = st.file_uploader("Upload Input File (SMILES data)", type=["txt"])
        
        if uploaded_file is not None:
            # Save the file
            file_path = save_uploaded_file(uploaded_file, input_data_dir)
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # Update the config with the new filename
            config["INPUT"]["NameData"] = uploaded_file.name
            
            # Display file preview
            try:
                if uploaded_file.name.endswith(('.txt', '.csv', '.smi')):
                    file_contents = uploaded_file.read().decode("utf-8")
                    if len(file_contents) > 1000:
                        st.text_area("File Preview (first 1000 characters):", file_contents[:1000] + "...", height=200)
                    else:
                        st.text_area("File Preview:", file_contents, height=200)
            except UnicodeDecodeError:
                st.warning("Unable to preview file contents. The file may contain binary data.")
            
        
        # Experiment name
        experiment_name = st.text_input(
            "Experiment Name", 
            value=config.get("INPUT", "experiment_name", fallback="experiment1"),
            help="Name for this experiment run (used for folder naming)"
        )
        
        # Update config with experiment name
        config["INPUT"]["experiment_name"] = experiment_name
        
        # Override experiment checkbox (in tab0)
        override_experiment = st.checkbox(
            "Override Existing Experiment", 
            value=config.get("INPUT", "override_experiment", fallback="Y").upper() == "Y",
            key="override_experiment_checkbox_tab0",
            help="Whether to overwrite an existing experiment with the same name"
        )
        
        # Update config
        config["INPUT"]["override_experiment"] = "Y" if override_experiment else "N"
        
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px;'>
            <p><strong>Next Steps:</strong></p>
            <p>1. After uploading your file and setting the experiment name, proceed to the other tabs to configure parameters.</p>
            <p>2. Finally, go to the "Pipeline Execution" tab to run the pipeline.</p>
        </div>
        """, unsafe_allow_html=True)

    # Processing Section
    with tab1:
        st.markdown("<div class='section-header'>Processing Configuration</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Configure data processing parameters for SMILES sequences</div>", 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            split = st.slider(
                "Training/Validation Split Ratio", 
                min_value=0.5, 
                max_value=1.0, 
                value=float(config.get("PROCESSING", "split", fallback="1.0")),
                step=0.05,
                help="Portion of data to use for training (1.0 means no validation split)"
            )
            
            min_len = st.number_input(
                "Minimum Sequence Length", 
                min_value=1, 
                value=int(config.get("PROCESSING", "min_len", fallback="1")),
                help="Minimum length of SMILES to include in training"
            )
        
        with col2:
            max_len = st.number_input(
                "Maximum Sequence Length", 
                min_value=50, 
                max_value=500,
                value=int(config.get("PROCESSING", "max_len", fallback="240")),
                help="Maximum length of SMILES to include in training"
            )
            
            augmentation = st.number_input(
                "Augmentation Factor", 
                min_value=1, 
                max_value=100,
                value=int(config.get("PROCESSING", "augmentation", fallback="10")),
                help="Number of augmented variants to generate for each molecule"
            )
        
        # Update config
        config["PROCESSING"]["split"] = str(split)
        config["PROCESSING"]["min_len"] = str(min_len)
        config["PROCESSING"]["max_len"] = str(max_len)
        config["PROCESSING"]["augmentation"] = str(augmentation)

    # Model Section
    with tab2:
        st.markdown("<div class='section-header'>Model Configuration</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Configure neural network parameters for training</div>", 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            pretrained_model = st.text_input(
                "Pretrained Model Path",
                value=config.get("MODEL", "pretrained_model", fallback="c24_augmentationx10_minlen1_maxlen140.h5"),
                help="Path to a pretrained model file (optional)"
            )
            
            epochs = st.number_input(
                "Number of Epochs", 
                min_value=1, 
                max_value=1000,
                value=int(config.get("MODEL", "epochs", fallback="100")),
                help="Number of training epochs"
            )
            
            lr = st.number_input(
                "Learning Rate", 
                min_value=0.00001, 
                max_value=0.01,
                value=float(config.get("MODEL", "lr", fallback="0.0005")),
                format="%.5f",
                step=0.0001,
                help="Initial learning rate"
            )
            
            batch_size = st.number_input(
                "Batch Size", 
                min_value=8, 
                max_value=256,
                value=int(config.get("MODEL", "batch_size", fallback="32")),
                step=8,
                help="Training batch size"
            )
        
        with col2:
            neurons_str = config.get("MODEL", "neurons", fallback="[1024, 256]")
            neurons_default = eval(neurons_str) if neurons_str.startswith('[') else [1024, 256]
            
            neuron_layer1 = st.number_input(
                "Neurons in Layer 1", 
                min_value=64, 
                max_value=2048,
                value=neurons_default[0],
                step=64
            )
            
            neuron_layer2 = st.number_input(
                "Neurons in Layer 2", 
                min_value=64, 
                max_value=1024,
                value=neurons_default[1],
                step=64
            )
            
            dropouts_str = config.get("MODEL", "dropouts", fallback="[0.40, 0.40]")
            dropouts_default = eval(dropouts_str) if dropouts_str.startswith('[') else [0.40, 0.40]
            
            dropout_layer1 = st.slider(
                "Dropout Rate Layer 1", 
                min_value=0.0, 
                max_value=0.8,
                value=dropouts_default[0],
                step=0.05
            )
            
            dropout_layer2 = st.slider(
                "Dropout Rate Layer 2", 
                min_value=0.0, 
                max_value=0.8,
                value=dropouts_default[1],
                step=0.05
            )
            
            trainables_str = config.get("MODEL", "trainables", fallback="[False, True]")
            trainables_default = eval(trainables_str) if trainables_str.startswith('[') else [False, True]
            
            trainable_layer1 = st.checkbox(
                "Trainable Layer 1", 
                value=trainables_default[0],
                key="trainable_layer1_checkbox"
            )
            
            trainable_layer2 = st.checkbox(
                "Trainable Layer 2", 
                value=trainables_default[1],
                key="trainable_layer2_checkbox"
            )
        
        # Advanced model parameters
        with st.expander("Advanced Model Parameters"):
            col3, col4 = st.columns(2)
            
            with col3:
                patience_lr = st.number_input(
                    "Learning Rate Patience", 
                    min_value=1, 
                    max_value=20,
                    value=int(config.get("MODEL", "patience_lr", fallback="3")),
                    help="Number of epochs with no improvement after which learning rate will be reduced"
                )
                
                factor = st.number_input(
                    "Learning Rate Reduction Factor", 
                    min_value=0.1, 
                    max_value=0.9,
                    value=float(config.get("MODEL", "factor", fallback="0.5")),
                    format="%.2f",
                    step=0.05,
                    help="Factor by which the learning rate will be reduced"
                )
            
            with col4:
                min_lr = st.number_input(
                    "Minimum Learning Rate", 
                    min_value=0.000001, 
                    max_value=0.001,
                    value=float(config.get("MODEL", "min_lr", fallback="0.00001")),
                    format="%.6f",
                    step=0.00001,
                    help="Lower bound on the learning rate"
                )
                
                period = st.number_input(
                    "Checkpoint Period", 
                    min_value=1, 
                    max_value=10,
                    value=int(config.get("MODEL", "period", fallback="1")),
                    help="Interval (epochs) between checkpoints"
                )
                
                n_workers = st.number_input(
                    "Number of Workers", 
                    min_value=1, 
                    max_value=16,
                    value=int(config.get("MODEL", "n_workers", fallback="5")),
                    help="Number of parallel workers for data loading"
                )
        
        # Update config
        config["MODEL"]["pretrained_model"] = pretrained_model
        config["MODEL"]["epochs"] = str(epochs)
        config["MODEL"]["lr"] = str(lr)
        config["MODEL"]["neurons"] = f"[{neuron_layer1}, {neuron_layer2}]"
        config["MODEL"]["dropouts"] = f"[{dropout_layer1}, {dropout_layer2}]"
        config["MODEL"]["trainables"] = f"[{trainable_layer1}, {trainable_layer2}]"
        config["MODEL"]["patience_lr"] = str(patience_lr)
        config["MODEL"]["factor"] = str(factor)
        config["MODEL"]["min_lr"] = str(min_lr)
        config["MODEL"]["period"] = str(period)
        config["MODEL"]["batch_size"] = str(batch_size)
        config["MODEL"]["n_workers"] = str(n_workers)

    # Input Section
    with tab3:
        st.markdown("<div class='section-header'>Input Configuration</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>View and modify input settings</div>", 
                    unsafe_allow_html=True)
        
        # Display current input settings
        st.info(f"""
        **Current Settings:**
        - Input Data File: {config.get("INPUT", "NameData", fallback="No file selected")}
        - Experiment Name: {config.get("INPUT", "experiment_name", fallback="No experiment name")}
        - Override Existing: {"Yes" if config.get("INPUT", "override_experiment", fallback="Y").upper() == "Y" else "No"}
        
        You can change these settings in the Setup tab.
        """)
        
        # List available input files
        st.markdown("### Available Input Files")
        input_files = [f for f in os.listdir(input_data_dir) if os.path.isfile(os.path.join(input_data_dir, f))]
        
        if input_files:
            selected_file = st.selectbox(
                "Select from available input files", 
                input_files,
                index=input_files.index(config.get("INPUT", "NameData", fallback=input_files[0])) if config.get("INPUT", "NameData", fallback="") in input_files else 0
            )
            
            # Update config with selected file
            if st.button("Use Selected File"):
                config["INPUT"]["NameData"] = selected_file
                st.success(f"Now using {selected_file} as input file")
        else:
            st.warning("No input files found. Please upload a file in the Setup tab.")
            
        # Additional input parameters section (if needed)
        with st.expander("Advanced Input Parameters"):
            st.markdown("Add any additional input parameters here if needed for your pipeline.")
            # Example:
            # descriptor_type = st.selectbox("Molecular Descriptor Type", ["ECFP", "MACCS", "Morgan"])
            # config["INPUT"]["descriptor_type"] = descriptor_type

    # Beam Search Section
    with tab4:
        st.markdown("<div class='section-header'>Beam Search Configuration</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Configure beam search parameters for molecular generation</div>", 
                    unsafe_allow_html=True)

        width = st.number_input(
            "Beam Width", 
            min_value=1, 
            max_value=200,
            value=int(config.get("BEAM", "width", fallback="50")),
            help="Width of the beam search"
        )

        from_epoch = st.number_input(
            "Start From Epoch", 
            min_value=1, 
            max_value=100,
            value=int(config.get("BEAM", "from_epoch", fallback="1")),
            help="Epoch from which to start beam search"
        )
        
        # Update config
        config["BEAM"]["width"] = str(width)
        config["BEAM"]["from_epoch"] = str(from_epoch)

    # Sampling Section
    with tab5:
        st.markdown("<div class='section-header'>Sampling Configuration</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Configure parameters for molecule sampling</div>", 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            temp = st.slider(
                "Temperature", 
                min_value=0.1, 
                max_value=2.0,
                value=float(config.get("SAMPLING", "temp", fallback="0.2")),
                step=0.1,
                help="Temperature for sampling (higher = more diversity)"
            )
            
            n_sample = st.number_input(
                "Number of Samples", 
                min_value=10, 
                max_value=10000,
                value=int(config.get("SAMPLING", "n_sample", fallback="50")),
                help="Number of molecules to sample per epoch"
            )
        
        with col2:
            # Check if we should use last_n_epochs or start/end_epoch
            last_n_epochs = st.number_input(
                "Last N Epochs", 
                min_value=1, 
                max_value=100,
                value=int(config.get("SAMPLING", "last_n_epochs", fallback="5")),
                help="Number of most recent epochs to sample from"
            )
            # Update config
            config["SAMPLING"]["last_n_epochs"] = str(last_n_epochs)

        # Store multiprocessing choice in session state to avoid duplicate widgets
        st.session_state.sampling_tab_use_multiprocessing = st.checkbox(
            "Use Multiprocessing for Sampling", 
            value=st.session_state.sampling_tab_use_multiprocessing,
            key="use_multiprocessing_sampling_tab",
            help="Enable parallel processing for faster sampling (recommended)"
        )
        
        # Update config
        config["SAMPLING"]["temp"] = str(temp)
        config["SAMPLING"]["n_sample"] = str(n_sample)

    # Communication Section
    with tab6:
        st.markdown("<div class='section-header'>Communication Settings</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Configure email notifications for experiment completion</div>", 
                    unsafe_allow_html=True)
        
        # Email notification
        email = st.text_input(
            "Email Address for Notifications", 
            value=config.get("Communication", "email", fallback=""),
            help="Enter your email address to receive notifications when experiments complete"
        )
        
        # Validate email
        if email and not is_valid_email(email):
            st.warning("Please enter a valid email address.")
        
        # Update config
        if email:
            if "Communication" not in config:
                config["Communication"] = {}
            config["Communication"]["email"] = email
        
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px;'>
            <p><strong>Email Notifications:</strong></p>
            <p>When provided, an email notification will be sent upon successful completion of the pipeline.</p>
            <p>The email will include basic information about the experiment and results.</p>
        </div>
        """, unsafe_allow_html=True)

    # Pipeline Execution Section
    with tab7:
        st.markdown("<div class='section-header'>Pipeline Execution</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Save configuration and run the HGM pipeline</div>", 
                    unsafe_allow_html=True)
        
        # Configuration summary
        st.markdown("### Configuration Summary")
        
        # Check if input file is set
        input_file = config.get("INPUT", "NameData", fallback=None)
        experiment_name = config.get("INPUT", "experiment_name", fallback=None)

        if not input_file:
            st.error("❌ No input file selected. Please upload or select a file in the Setup tab.")
        else:
            st.success(f"✓ Input file: {input_file}")
        
        if not experiment_name:
            st.warning("⚠️ No experiment name set. Please specify an experiment name in the Setup tab.")
        else:
            st.success(f"✓ Experiment name: {experiment_name}")
        
        # Email status
        if "Communication" in config and "email" in config["Communication"] and config["Communication"]["email"]:
            st.success(f"✓ Email notifications will be sent to: {config['Communication']['email']}")
        else:
            st.info("ℹ️ No email address provided. You won't receive notifications upon experiment completion.")
        
        # Save configuration button
        if st.button("Save Configuration"):
            if save_config(config):
                st.success("Configuration saved successfully to config.ini!")
            else:
                st.error("Failed to save configuration.")

        # Disable the run button if no input file or experiment name
        run_disabled = not input_file or not experiment_name
        
        if st.button("Run Pipeline", type="primary", disabled=run_disabled):

            from app_link import run_pipeline_in_background

            # First save the configuration
            save_config(config)

            # Get the experiment name
            experiment_name = config["INPUT"]["experiment_name"]
            
            # Check if experiment is already running
            status_file = os.path.join("output", experiment_name, "status.json")
            
            if os.path.exists(status_file):
                with open(status_file, "r") as f:
                    status = json.load(f)
                
                if status["status"] == "running":
                    st.warning(f"Experiment '{experiment_name}' is already running. View status in the Browse Experiments page.")
                    st.session_state.current_page = "Browse Experiments"
                else:
                    # Start the background job
                    status_file = run_pipeline_in_background(experiment_name)
                    st.success(f"Started pipeline for experiment '{experiment_name}' in the background.")
                    st.info("You can close your browser and the process will continue running.")
                    st.info("Check the status in the 'Browse Experiments' page later.")
                    
                    # Switch to browse page to see progress
                    st.session_state.current_page = "Browse Experiments" 
            else:
                # Start the background job
                status_file = run_pipeline_in_background(experiment_name)
                st.success(f"Started pipeline for experiment '{experiment_name}' in the background.")
                st.info("You can close your browser and the process will continue running.")
                st.info("Check the status in the 'Browse Experiments' page later.")
                
                # Switch to browse page to see progress
                st.session_state.current_page = "Browse Experiments"
        
        # if st.button("Run Pipeline", type="primary", disabled=run_disabled):
        #     # First save the configuration
        #     save_config(config)
        #     start_time = time.time()
        #     cmd = ["python", "main.py"]

        #     # Create a placeholder for the progress bar
        #     progress_placeholder = st.empty()
            
        #     # Create a placeholder for logs
        #     log_placeholder = st.empty()
            
        #     # Reset logs for a fresh run
        #     st.session_state.logs = []
            
        #     # Start pipeline execution
        #     progress_bar = progress_placeholder.progress(0.0)

        #     update_logs(f"Starting HGM Pipeline execution for experiment: {experiment_name}", log_placeholder)
        #     update_logs(f"Using input file: {input_file}", log_placeholder)
        #     update_logs("Configuration saved to config.ini", log_placeholder)

        #     try:
        #         # Update progress to 10% to indicate pipeline has started
        #         progress_bar.progress(0.1)
                
        #         # Prepare command with arguments
        #         cmd = ["python", "main.py"]

        #         # Run the command
        #         update_logs(f"Executing command: {' '.join(cmd)}", log_placeholder)
                
        #         # Execute command and capture output
        #         process = subprocess.Popen(
        #             cmd,
        #             stdout=subprocess.PIPE,
        #             stderr=subprocess.STDOUT,
        #             text=True,
        #             bufsize=1,
        #             universal_newlines=True
        #         )

        #         # Stream output in real-time
        #         progress_value = 0.1
        #         progress_step = 0.9 / 4  # Divide remaining progress among 4 stages

        #         while True:
        #             line = process.stdout.readline()
        #             if not line and process.poll() is not None:
        #                 break
        #             if line:
        #                 update_logs(line.strip(), log_placeholder)
                        
        #                 # Update progress based on pipeline stages
        #                 if "Data processing completed successfully" in line:
        #                     progress_value += progress_step
        #                     progress_bar.progress(progress_value)
        #                 elif "Network training completed successfully" in line:
        #                     progress_value += progress_step
        #                     progress_bar.progress(progress_value)
        #                 elif "Sample generation completed" in line:
        #                     progress_value += progress_step
        #                     progress_bar.progress(progress_value)
        #                 elif "Novo analysis completed successfully" in line:
        #                     progress_value += progress_step
        #                     progress_bar.progress(progress_value)

        #         # Check return code
        #         return_code = process.poll()
        #         if return_code == 0:
        #             # Set progress to 100% when complete
        #             progress_bar.progress(1.0)
                    
        #             # Log completion message
        #             update_logs("🚀 Pipeline execution completed successfully!", log_placeholder)
        #             update_logs(f"Results are available in the output directory for experiment: {experiment_name}", log_placeholder)
                    
        #             # Display a success message
        #             st.markdown(
        #                 "<div class='success-message'>Pipeline execution completed successfully! Results are available in the output directory.</div>", 
        #                 unsafe_allow_html=True
        #             )
        #             # Send email notification if email is configured
        #             if "Communication" in config and "email" in config["Communication"] and config["Communication"]["email"]:
        #                 email = config["Communication"]["email"]
                        
        #                 # Collect some basic results to include in the email
        #                 results = {
        #                     "Execution Time": f"{time.time() - start_time:.2f} seconds",
        #                     "Output Directory": f"output/{experiment_name}"
        #                 }
                        
        #                 # Try to get more detailed results if available
        #                 try:
        #                     # Check if any results are available
        #                     result_files = glob.glob(f"output/{experiment_name}/results/*.csv")
        #                     if result_files:
        #                         results["Results Files"] = len(result_files)
        #                 except Exception:
        #                     pass

        #                 from funcs.email import send_email_notification
        #                 # Send the email
        #                 if send_email_notification(email, experiment_name, "Completed Successfully", results):
        #                     update_logs(f"✉️ Email notification sent to {email}", log_placeholder)
        #                 else:
        #                     update_logs(f"⚠️ Failed to send email notification to {email}")
        #         else:
        #             # Log error message
        #             update_logs(f"❌ Pipeline execution failed with return code: {return_code}", log_placeholder)
                    
        #             # Display an error message
        #             st.error(f"Pipeline execution failed with return code: {return_code}")
                
        #     except Exception as e:
        #         # Update progress to indicate failure
        #         progress_bar.progress(1.0)
                
        #         # Log error message
        #         update_logs(f"❌ Error during pipeline execution: {str(e)}", log_placeholder)
                
        #         # Display an error message
        #         st.error(f"An error occurred during pipeline execution: {str(e)}")
                
        # elif run_disabled:
        #     st.warning("Please upload an input file and set an experiment name before running the pipeline.")

else:  # Browse Experiments page
    # New code for the experiment browser page
    st.write("View and analyze past experiments and their results")
    
    # Get experiment folders
    experiment_folders = get_experiment_folders()

    if not experiment_folders:
        st.warning("No experiments found. Run an experiment from the 'Create Experiment' page first.")
    else:
        # Create experiment browser
        st.markdown("### Available Experiments")
        
        # Quick metrics
        total_experiments = len(experiment_folders)
        completed_experiments = sum(1 for exp in experiment_folders if 
                                   get_experiment_metadata(exp)["status"] == "Completed")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="metrics-card">
                    <div class="metric-value">{total_experiments}</div>
                    <div class="metric-label">Total Experiments</div>
                </div>
                """, 
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metrics-card">
                    <div class="metric-value">{completed_experiments}</div>
                    <div class="metric-label">Completed Experiments</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Experiment selection
        selected_experiment = st.selectbox(
            "Select Experiment", 
            experiment_folders,
            format_func=lambda x: f"{x} ({get_experiment_metadata(x)['status']})"
        )

        if selected_experiment:
            # Get metadata
            metadata = get_experiment_metadata(selected_experiment)
            
            # Display experiment details
            st.markdown("### Experiment Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Name:** {metadata['name']}")
                st.markdown(f"**Created:** {metadata['created']}")
                st.markdown(f"**Status:** {metadata['status']}")
            
            with col2:
                st.markdown(f"**Model File:** {metadata['model_file'] or 'None'}")
                st.markdown(f"**Results File:** {metadata['results_file'] or 'None'}")
                st.markdown(f"**Number of Samples:** {metadata['num_samples']}")
            
            # Training history (if available)
            history = load_training_history(selected_experiment)
            if history:
                st.markdown("### Training History")
                
                # Create figure for training plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot training and validation loss
                epochs = range(1, len(history['loss']) + 1)
                ax.plot(epochs, history['loss'], 'b-', label='Training Loss')
                
                if 'val_loss' in history:
                    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
                
                ax.set_title('Training and Validation Loss')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                
                st.pyplot(fig)

            # Sample results (if available)
            results = load_sample_results(selected_experiment)
            if results is not None:
                st.markdown("### Generated Molecules")

                # Display summary statistics
                st.markdown("#### Summary Statistics")
                
                # Show a sample of data
                st.dataframe(results.head(10))
                
                # Download button for full results
                results_path = os.path.join("output", selected_experiment, "results", "metrics.csv")
                if os.path.exists(results_path):
                    with open(results_path, "rb") as file:
                        st.download_button(
                            label="Download Full Results CSV",
                            data=file,
                            file_name=f"{selected_experiment}_results.csv",
                            mime="text/csv"
                        )





# Footer
st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

# Create columns for footer layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 10px;'>
            HGM Pipeline Manager | Hit Generative Model for Molecule Generation
        </div>
        """, 
        unsafe_allow_html=True
    )
