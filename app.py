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
import matplotlib.cm as cm
import matplotlib.colors as colors
from datetime import datetime
import numpy as np

from funcs.stauth import check_password
from funcs.app_funcs import (load_backend_docs,
                             load_config, 
                             ensure_input_data_dir, 
                             save_uploaded_file, 
                             save_config, 
                             is_valid_email, 
                             update_logs,
                             get_experiment_folders,
                             get_experiment_metadata,
                             load_training_history,
                             load_sample_results,
                             load_similarity_data,
                             load_tsne_data)
from resources.app_styling import app_style, app_style2

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


st.markdown(app_style, unsafe_allow_html=True)
st.markdown(app_style2, unsafe_allow_html=True)


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

# Add Backend Documentation button separately
backend_docs_button = st.sidebar.button("🧠 **Backend Documentation**", key="nav_backend_docs", use_container_width=True)

# Handle navigation
if create_exp_button:
    st.session_state.current_page = "🧪 Create Experiment"
elif browse_exp_button:
    st.session_state.current_page = "📊 Browse Experiments"
elif backend_docs_button:
    st.session_state.current_page = "🧠 Backend Documentation"

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

if not check_password():
    st.stop()

# Ensure input_data directory exists
input_data_dir = ensure_input_data_dir()

# Display the selected page
if st.session_state.current_page == "🧠 Backend Documentation":
    st.markdown("## 🧠 Backend Documentation")
    st.markdown("This page provides detailed information about the architecture and implementation of the HGM pipeline backend.")
    
    # Create tabs for different documentation views
    doc_tabs = st.tabs(["**Overview**", "**Detailed Process Flow**"])
    
    with doc_tabs[0]:
        # Load backend documentation from README.md
        backend_docs = load_backend_docs()
        
        # Display the documentation
        st.markdown(backend_docs)
        
        # Add a download button for the documentation
        with open("README.md", "r") as f:
            readme_content = f.read()
            
        st.download_button(
            label="Download Complete Documentation",
            data=readme_content,
            file_name="HGM_documentation.md",
            mime="text/markdown"
        )
    
    with doc_tabs[1]:
        # Load the detailed process flow HTML
        process_flow_path = "resources/process_flow.html"
        
        if os.path.exists(process_flow_path):
            with open(process_flow_path, "r") as f:
                process_flow_html = f.read()
            
            # Display the HTML content
            st.components.v1.html(process_flow_html, height=800, scrolling=True)
            
            # Add a download button for the process flow documentation
            with open(process_flow_path, "r") as f:
                process_flow_content = f.read()
                
            st.download_button(
                label="Download Process Flow Documentation",
                data=process_flow_content,
                file_name="HGM_process_flow.html",
                mime="text/html"
            )
        else:
            st.error(f"Process flow documentation file not found at {process_flow_path}")
            
        # Add explanations about the visualization
        from funcs.app_docs import about_flow
        st.markdown(about_flow)

elif st.session_state.current_page == "🧪 Create Experiment":

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

else:  # Browse Experiments page
    # New code for the experiment browser page
    st.write("View and analyze past experiments and their results")
    experiment_folders = get_experiment_folders(base_dir="memory")

    if not experiment_folders:
        st.warning("No experiments found. Run an experiment from the 'Create Experiment' page first.")
    else:
        # Create experiment browser
        st.markdown("### Available Experiments")

        # Quick metrics
        total_experiments = len(experiment_folders)
        completed_experiments = sum(1 for exp in experiment_folders if 
                                   get_experiment_metadata(exp, base_dir='memory')["novo_analysis_status"] == "Completed")

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
            experiment_folders
        )

        if selected_experiment:
            
            with st.container(border=True):
                # Get metadata
                metadata = get_experiment_metadata(selected_experiment, base_dir='memory')

                # Display experiment details
                st.markdown("### Experiment Details")
                
                # First, show status overview
                st.markdown("#### Experiment Status")
                status_cols = st.columns(3)
                
                with status_cols[0]:
                    model_status = metadata.get("model_status", "Not Started")
                    status_color = "#4CAF50" if model_status == "Trained" else "#FF9800"
                    st.markdown(f"""
                    <div style="
                        background-color: {status_color};
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                        color: white;
                        text-align: center;
                    ">
                        <p style="font-size: 16px; margin-bottom: 5px;"><strong>Model Training</strong></p>
                        <p style="font-size: 18px;">{model_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with status_cols[1]:
                    sampling_status = metadata.get("sampling_status", "Not Started")
                    status_color = "#4CAF50" if sampling_status == "Completed" else "#FF9800"
                    st.markdown(f"""
                    <div style="
                        background-color: {status_color};
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                        color: white;
                        text-align: center;
                    ">
                        <p style="font-size: 16px; margin-bottom: 5px;"><strong>Molecule Sampling</strong></p>
                        <p style="font-size: 18px;">{sampling_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with status_cols[2]:
                    analysis_status = metadata.get("novo_analysis_status", "Not Started")
                    status_color = "#4CAF50" if analysis_status == "Completed" else "#FF9800"
                    st.markdown(f"""
                    <div style="
                        background-color: {status_color};
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                        color: white;
                        text-align: center;
                    ">
                        <p style="font-size: 16px; margin-bottom: 5px;"><strong>Novo Analysis</strong></p>
                        <p style="font-size: 18px;">{analysis_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display experiment creation date and sample count
                info_cols = st.columns(2)
                
                with info_cols[0]:
                    st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    ">
                        <p style="font-size: 16px; color: #0e1117; margin-bottom: 5px;"><strong>Created</strong></p>
                        <p style="font-size: 14px; color: #4b5563;">{metadata.get('created', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with info_cols[1]:
                    st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 10px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    ">
                        <p style="font-size: 16px; color: #0e1117; margin-bottom: 5px;"><strong>Generated Samples</strong></p>
                        <p style="font-size: 14px; color: #4b5563;">{metadata.get('num_samples', 0)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            with st.container(border=True):
                # Now display the config items in a nice placard style
                st.markdown("#### Configuration Parameters")
            
                # Function to show data file content in a scrollable table
                def show_data_content(experiment_name, filename, base_dir="memory"):
                    file_path = os.path.join(base_dir, experiment_name, filename)
                    if os.path.exists(file_path):
                        try:
                            # Read the entire file
                            with open(file_path, 'r') as f:
                                lines = [line.strip() for line in f.readlines()]
                            
                            # Count total lines
                            total_lines = len(lines)
                            
                            # Create a nicely formatted display
                            st.markdown(f"**{filename}** ({total_lines} entries)")
                            
                            # Create a DataFrame for display
                            df = pd.DataFrame({
                                "Entry #": range(1, total_lines + 1),
                                "SMILES": lines
                            })
                            
                            # Display the dataframe with a height limit
                            st.dataframe(df, use_container_width=True, height=400)
                            
                            # Add a download button
                            with open(file_path, 'rb') as f:
                                st.download_button(
                                    label=f"Download {filename}",
                                    data=f,
                                    file_name=filename,
                                    mime="text/plain"
                                )
                        except Exception as e:
                            st.error(f"Error loading {filename}: {str(e)}")
                    else:
                        st.warning(f"{filename} file not found")
                
                # Create tabs for different config sections
                config_sections = [s for s in metadata if s not in ['created', 'model_status', 'sampling_status', 'novo_analysis_status', 'num_samples']]
                # Add a Data Samples tab
                all_tabs = config_sections + ["Data Samples"]
                config_tabs = st.tabs(all_tabs)
                
                for i, section in enumerate(all_tabs):
                    with config_tabs[i]:
                        if section == "Data Samples":
                            
                            # Create tabs for training and validation data
                            data_tab1, data_tab2 = st.tabs(["Training Data", "Validation Data"])
                            
                            with data_tab1:
                                show_data_content(selected_experiment, "data_tr.txt")
                            
                            with data_tab2:
                                show_data_content(selected_experiment, "data_val.txt")
                        
                        elif isinstance(metadata.get(section), dict):
                            # Create a grid layout for placards
                            param_cols = st.columns(3)
                            
                            # Display each parameter as a placard
                            for idx, (param, value) in enumerate(metadata[section].items()):
                                col_idx = idx % 3
                                with param_cols[col_idx]:
                                    st.markdown(f"""
                                    <div style="
                                        background-color: #f0f2f6;
                                        border-radius: 10px;
                                        padding: 15px;
                                        margin-bottom: 10px;
                                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                                    ">
                                        <p style="font-size: 16px; color: #0e1117; margin-bottom: 5px;"><strong>{param}</strong></p>
                                        <p style="font-size: 14px; color: #4b5563;">{value}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.write(f"{section}: {metadata[section]}")
            with st.container(border=True):
                # Training history (if available)
                history = load_training_history(selected_experiment)
                if history:
                    st.markdown("### Training History")
                    
                    # Create a dataframe for the history data
                    history_df = pd.DataFrame({
                        'epoch': range(1, len(history['loss']) + 1),
                        'Training Loss': history['loss']
                    })
                    
                    if 'val_loss' in history:
                        history_df['Validation Loss'] = history['val_loss']
                    
                    # Create a Plotly figure
                    import plotly.express as px
                    
                    fig = px.line(
                        history_df, 
                        x='epoch', 
                        y=['Training Loss', 'Validation Loss'] if 'val_loss' in history else ['Training Loss'],
                        labels={'value': 'Loss', 'variable': 'Type'},
                        color_discrete_sequence=['#2E86C1', '#E74C3C'],  # Blue for training, Red for validation
                        title='Training and Validation Loss'
                    )

                    # Customize the figure
                    fig.update_layout(
                        xaxis_title='Epochs',
                        yaxis_title='Loss (categorical crossentropy)',
                        hovermode='x unified',
                        legend_title_text='',
                        height=700,
                        template='plotly_white',
                        grid=dict(rows=1, columns=1, pattern='independent'),
                    )
                    
                    # Add grid lines
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Optionally, display the dataframe with the raw values
                    with st.expander("**View Training Losses**"):
                        st.dataframe(history_df)

            # Create tabs for molecule exploration
            st.markdown("### Molecule Explorer")
            molecule_tabs = st.tabs(["Generated Molecules", 
                                     "Beam Search Molecules", 
                                     "TSNE Visualization"])
            
            # Tab 1: Generated Molecules (from sampling)
            with molecule_tabs[0]:
                # Sample results (if available)
                results = load_sample_results(selected_experiment)
                if results is not None:
                    # Display summary statistics
                    st.markdown("#### Summary Statistics")
                    
                    # Show sample data
                    st.dataframe(results)
                    
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
                    
                    # Show molecule visualization
                    if 'SMILES' in results.columns and len(results) > 0:
                        selected_index = st.selectbox(
                            "Select molecule to view:",
                            options=range(len(results)),
                            format_func=lambda i: f"Molecule {i+1}"
                        )
                        
                        selected_smiles = results['SMILES'].iloc[selected_index]
                        st.code(selected_smiles, language=None)
                        
                        # Show "View Structure" button
                        if st.button("View Structure", key="view_structure"):
                            # Check if rdkit is available
                            try:
                                from rdkit import Chem
                                from rdkit.Chem import Draw
                                
                                mol = Chem.MolFromSmiles(selected_smiles)
                                if mol:
                                    # Generate the image
                                    img = Draw.MolToImage(mol, size=(1260, 400))
                                    
                                    # Display the image
                                    st.image(img, caption="Molecular Structure")
                                    
                                    # Add some basic molecular properties
                                    st.markdown("#### Properties")
                                    from rdkit.Chem import Descriptors, Lipinski

                                    props = {
                                        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
                                        "LogP": round(Descriptors.MolLogP(mol), 2),
                                        "H-Bond Donors": Lipinski.NumHDonors(mol),
                                        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
                                        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol)
                                    }
                                    
                                    prop_cols = st.columns(3)

                                    # Display each property as a placard
                                    for idx, (name, value) in enumerate(props.items()):
                                        col_idx = idx % 3
                                        with prop_cols[col_idx]:
                                            st.markdown(f"""
                                            <div style="
                                                background-color: #f0f2f6;
                                                border-radius: 10px;
                                                padding: 15px;
                                                margin-bottom: 10px;
                                                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                                            ">
                                                <p style="font-size: 16px; color: #0e1117; margin-bottom: 5px;"><strong>{name}</strong></p>
                                                <p style="font-size: 14px; color: #4b5563;">{value}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    
                                    # Add download buttons
                                    download_col1, download_col2 = st.columns(2)
                                    import io
                                    from PIL import Image
                                    import base64
                                    from reportlab.lib.pagesizes import letter
                                    from reportlab.pdfgen import canvas
                                    from reportlab.lib.utils import ImageReader
                                    
                                    # Function to create PNG
                                    def get_png_download():
                                        buf = io.BytesIO()
                                        img.save(buf, format='PNG')
                                        byte_im = buf.getvalue()
                                        return byte_im
                                    
                                    # Function to create PDF with structure and properties
                                    def get_pdf_download():
                                        buffer = io.BytesIO()
                                        c = canvas.Canvas(buffer, pagesize=letter)
                                        width, height = letter
                                        
                                        # Add title
                                        c.setFont("Helvetica-Bold", 16)
                                        c.drawString(72, height - 72, f"Molecule: {selected_index + 1}")
                                        
                                        # Add SMILES
                                        c.setFont("Helvetica", 10)
                                        c.drawString(72, height - 100, "SMILES:")
                                        c.drawString(72, height - 115, selected_smiles)
                                        
                                        # Add the molecule image
                                        img_buf = io.BytesIO()
                                        img.save(img_buf, format='PNG')
                                        img_buf.seek(0)
                                        img_reader = ImageReader(img_buf)
                                        c.drawImage(img_reader, 72, height - 400, width=450, height=250, preserveAspectRatio=True)
                                        
                                        # Add properties
                                        c.setFont("Helvetica-Bold", 14)
                                        c.drawString(72, height - 420, "Properties:")
                                        
                                        c.setFont("Helvetica", 12)
                                        y_pos = height - 450
                                        for name, value in props.items():
                                            c.drawString(72, y_pos, f"{name}: {value}")
                                            y_pos -= 20
                                        
                                        c.save()
                                        buffer.seek(0)
                                        return buffer
                                    
                                    # Add PNG download button
                                    with download_col1:
                                        st.download_button(
                                            label="Download PNG",
                                            data=get_png_download(),
                                            file_name=f"molecule_{selected_index + 1}.png",
                                            mime="image/png"
                                        )
                                    
                                    # Add PDF download button
                                    with download_col2:
                                        st.download_button(
                                            label="Download PDF Report",
                                            data=get_pdf_download(),
                                            file_name=f"molecule_{selected_index + 1}_report.pdf",
                                            mime="application/pdf"
                                        )
                                else:
                                    st.error("Could not generate molecular structure. Invalid SMILES string.")
                            except ImportError:
                                st.error("RDKit is required to display molecular structures. Please install it with `pip install rdkit`.")
                else:
                    st.warning("No generated molecules results found.")
            
            # Tab 2: Beam Search Molecules section (now only showing Tanimoto Similarity Analysis)
            with molecule_tabs[1]:
                # Tanimoto Similarity Analysis
                st.markdown("### Tanimoto Similarity Analysis")
                st.markdown("""
                This section visualizes the Tanimoto similarity measures for molecules generated 
                by the beam search algorithm during different training epochs. Tanimoto similarity 
                quantifies the structural similarity between generated molecules and reference compounds.
                """)
                
                # Load similarity data using the existing function
                similarity_data = load_similarity_data(selected_experiment)
                
                if similarity_data is not None and 'summary' in similarity_data:
                    summary_df = similarity_data['summary']
                    
                    # Create a sidebar to select metrics to display
                    similarity_metrics = st.multiselect(
                        "Select similarity metrics to display:",
                        options=['avg_similarity', 'median_similarity', 'min_similarity', 'max_similarity', 'std_similarity'],
                        default=['avg_similarity', 'max_similarity', 'min_similarity'],
                        help="Choose which similarity metrics to show on the chart"
                    )
                    
                    if not similarity_metrics:
                        st.warning("Please select at least one metric to display")
                    else:
                        # Create a line chart using Plotly
                        import plotly.express as px
                        
                        # Filter out rows with missing values 
                        filtered_df = summary_df.dropna(subset=similarity_metrics)
                        
                        # Create labels mapping for better display
                        metric_labels = {
                            'avg_similarity': 'Average Similarity',
                            'median_similarity': 'Median Similarity',
                            'min_similarity': 'Minimum Similarity',
                            'max_similarity': 'Maximum Similarity',
                            'std_similarity': 'Standard Deviation'
                        }
                        
                        # Create the line chart
                        fig = px.line(
                            filtered_df,
                            x='epoch',
                            y=similarity_metrics,
                            labels={
                                'epoch': 'Training Epoch',
                                'value': 'Tanimoto Similarity',
                                'variable': 'Metric'
                            },
                            title='Tanimoto Similarity Measures Across Training Epochs',
                            color_discrete_sequence=px.colors.qualitative.Bold,
                            height=500
                        )
                        
                        # Update the legend labels
                        for trace in fig.data:
                            trace.name = metric_labels.get(trace.name, trace.name)
                        
                        # Update the layout for better visualization
                        fig.update_layout(
                            xaxis_title="Training Epoch",
                            yaxis_title="Tanimoto Similarity",
                            legend_title="Metric",
                            hovermode="x unified",
                            template="plotly_white"
                        )
                        
                        # Add grid lines
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create a secondary visualization - bar chart for valid SMILES per epoch
                        st.markdown("#### Valid SMILES Generated per Epoch")
                        
                        # Create the bar chart for valid SMILES
                        bar_fig = px.bar(
                            filtered_df,
                            x='epoch',
                            y='valid_smiles',
                            labels={
                                'epoch': 'Training Epoch',
                                'valid_smiles': 'Number of Valid SMILES'
                            },
                            title='Valid SMILES Generated per Training Epoch',
                            color_discrete_sequence=['#3366cc'],
                            height=400
                        )
                        
                        # Update the layout
                        bar_fig.update_layout(
                            xaxis_title="Training Epoch",
                            yaxis_title="Count",
                            hovermode="x unified",
                            template="plotly_white"
                        )
                        
                        # Display the chart
                        st.plotly_chart(bar_fig, use_container_width=True)
                        
                        # Display the raw data table with heatmap styling
                        st.markdown("#### Similarity Data Table")
                        st.markdown("""
                        <div style="background-color: #e7f9e7; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #2e7d32;">
                        <b>Note:</b> The top 5 epochs by average similarity are used for further processing in the model pipeline.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create a styled version of the dataframe with numeric values for heatmap
                        numeric_df = filtered_df.copy()
                        st.write(filtered_df)
                        
                        # Get top 5 epochs by average similarity for highlighting
                        if 'avg_similarity' in numeric_df.columns:
                            top_avg_epochs = numeric_df.dropna(subset=['avg_similarity'])
                            top_avg_epochs = top_avg_epochs.sort_values('avg_similarity', ascending=False).head(5)['epoch'].tolist()
                        else:
                            top_avg_epochs = []

                        
                        # Create a separate highlighted table showing just the top 5 epochs 
                        if top_avg_epochs:
                            st.markdown("#### Top 5 Epochs by Average Similarity")
                            st.markdown("These epochs are used for further processing in the model pipeline.")
                            
                            # Filter for just the top epochs
                            top_df = numeric_df[numeric_df['epoch'].isin(top_avg_epochs)].sort_values('avg_similarity', ascending=False).reset_index(drop=True)
                            st.write(top_df)
                        
                        # Add download button for the similarity data
                        st.download_button(
                            label="Download Similarity Data CSV",
                            data=filtered_df.to_csv(index=False).encode(),
                            file_name=f"{selected_experiment}_tanimoto_similarity.csv",
                            mime="text/csv"
                        )
                        
                        # Check if we have plots to display
                        if 'plots' in similarity_data and similarity_data['plots']:
                            st.markdown("#### Similarity Plots")
                            for plot_name, plot_path in similarity_data['plots'].items():
                                if os.path.exists(plot_path):
                                    try:
                                        st.image(plot_path, caption=f"{plot_name.replace('_', ' ').title()} Plot")
                                    except Exception as e:
                                        st.error(f"Error displaying plot: {str(e)}")
                else:
                    # Check both possible similarity file paths
                    similarity_file1 = os.path.join("memory", selected_experiment, "beam_search", "similarity_summary.csv")
                    similarity_file2 = os.path.join("memory", selected_experiment, "beam_search", "tanimoto_similarity_summary.csv")
                    
                    if not os.path.exists(similarity_file1) and not os.path.exists(similarity_file2):
                        st.info("No Tanimoto similarity data found for this experiment.")
                        
                        # Add option to generate similarity data
                        if st.button("Generate Tanimoto Similarity Analysis"):
                            try:
                                st.info("Generating Tanimoto similarity analysis. This may take a minute...")
                                progress_bar = st.progress(0)
                                
                                # Import the similarity calculation function
                                sys.path.append(os.getcwd())
                                from processes.tanimoto_similarity import calculate_similarities
                                
                                # Set up paths
                                beam_search_dir = os.path.join("memory", selected_experiment, "beam_search")
                                
                                # Run the similarity calculation
                                progress_bar.progress(50)
                                calculate_similarities(beam_search_dir)
                                progress_bar.progress(100)
                                
                                st.success("Tanimoto similarity analysis generated successfully! Refresh the page to view.")
                                st.button("Refresh Page")
                                
                            except Exception as e:
                                st.error(f"Error generating Tanimoto similarity analysis: {str(e)}")
                                st.info("Please check that the required files exist in the beam_search directory.")
                    else:
                        st.error("Error loading similarity data. The file exists but could not be processed correctly.")
            
            # Tab 3: TSNE Visualization
            with molecule_tabs[2]:
                # Load TSNE data
                tsne_data = load_tsne_data(selected_experiment)
                
                if tsne_data:
                    st.markdown("#### TSNE Visualization")
                    
                    # Add an explanation of the TSNE technique
                    explanation_text = """
                    This visualization shows the molecular space using t-SNE dimensionality reduction technique. 
                    t-SNE (t-Distributed Stochastic Neighbor Embedding) projects high-dimensional molecular fingerprints 
                    into a lower-dimensional space while preserving local relationships between molecules.
                    
                    - Similar molecules appear closer together in this visualization
                    - Clusters indicate groups of structurally related molecules
                    - Distance between points indicates molecular similarity
                    """
                    st.markdown(explanation_text)

                    try:
                        import plotly.express as px
                        import plotly.graph_objects as go
                        
                        combinedset = tsne_data["combinedset"]
                        chembl_data = tsne_data["chembl_data"]

                        # Create visualization options including 3D
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            viz_type = st.radio(
                                "Select data view:",
                                ["Combined View (Generated + Background)"]
                            )
                        
                        with col2:
                            plot_dim = st.radio(
                                "Select plot dimension:",
                                ["2D Plot", "3D Plot"],
                                help="3D plot provides an additional dimension for better visualization of chemical space"
                            )
                        
                        if viz_type == "Combined View (Generated + Background)":
                            # Create a column for molecule source and add molecule numbers for the generated ones
                            combinedset['Source'] = ['Generated' if i < len(chembl_data) else 'Background' 
                                                    for i in range(len(combinedset))]
                            
                            # Add molecule numbers for easier selection in tooltips
                            combinedset['Molecule_Number'] = [-1] * len(combinedset)  # Default value for background
                            for i in range(len(chembl_data)):
                                combinedset.iloc[i, combinedset.columns.get_loc('Molecule_Number')] = i + 1
                            
                            # Create marker sizes - larger for generated molecules
                            combinedset['marker_size'] = [8 if x == 'Generated' else 4 for x in combinedset['Source']]

                            # Create opacity - higher for generated molecules
                            combinedset['opacity'] = [1.0 if x == 'Generated' else 0.4 for x in combinedset['Source']]
                            
                            # Create tooltip text
                            combinedset['hover_text'] = combinedset.apply(
                                lambda row: f"Molecule #{int(row['Molecule_Number'])}<br>SMILES: {row['SMILES'][:30]}..." 
                                if row['Source'] == 'Generated' else "Background Molecule", 
                                axis=1
                            )

                            # Choose between 2D and 3D visualization
                            if plot_dim == "2D Plot":
                                # Create the combined 2D scatter plot
                                fig = px.scatter(
                                    combinedset, 
                                    x='TSNE_C1', 
                                    y='TSNE_C2',
                                    color='Source',
                                    size='marker_size',
                                    size_max=10,
                                    opacity=combinedset['opacity'],
                                    color_discrete_map={'Generated': '#1E88E5', 'Background': '#d3d3d3'},
                                    title='t-SNE 2D Visualization of Chemical Space',
                                    hover_name='hover_text',
                                    custom_data=['Molecule_Number', 'Source']
                                )
                            else:
                                # Create 3D scatter plot if TSNE_C3 exists
                                if 'TSNE_C3' in combinedset.columns:
                                    # For 3D plots, we need to handle opacity differently
                                    # Create separate dataframes for generated and background molecules
                                    generated_df = combinedset[combinedset['Source'] == 'Generated'].copy()
                                    background_df = combinedset[combinedset['Source'] == 'Background'].copy()

                                    # Create the 3D scatter plot with two traces
                                    fig = go.Figure()
                                    
                                    # Add generated molecules with full opacity
                                    fig.add_trace(go.Scatter3d(
                                        x=generated_df['TSNE_C1'],
                                        y=generated_df['TSNE_C2'],
                                        z=generated_df['TSNE_C3'],
                                        mode='markers',
                                        marker=dict(
                                            size=6,
                                            color='#1E88E5',
                                            opacity=1.0,
                                            line=dict(width=1, color='black')
                                        ),
                                        text=generated_df['hover_text'],
                                        hoverinfo='text',
                                        name='Generated'
                                    ))
                                    
                                    # Add background molecules with lower opacity
                                    fig.add_trace(go.Scatter3d(
                                        x=background_df['TSNE_C1'],
                                        y=background_df['TSNE_C2'],
                                        z=background_df['TSNE_C3'],
                                        mode='markers',
                                        marker=dict(
                                            size=3,
                                            color='#d3d3d3',
                                            opacity=0.4
                                        ),
                                        text=background_df['hover_text'],
                                        hoverinfo='text',
                                        name='Background'
                                    ))

                                    # Set the title
                                    fig.update_layout(
                                        title='t-SNE 3D Visualization of Chemical Space',
                                    )
                                    # Set better 3D camera and options
                                    fig.update_layout(
                                        scene=dict(
                                            xaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
                                            yaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
                                            zaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
                                            camera=dict(
                                                eye=dict(x=1.5, y=1.5, z=1.2)
                                            )
                                        ),
                                        scene_aspectmode='data'
                                    )
                                    
                                    # Add instructions for 3D interaction
                                    st.info("""
                                    **3D Navigation Tips:**
                                    - **Rotate**: Click and drag to rotate the 3D view
                                    - **Zoom**: Use scroll wheel to zoom in/out
                                    - **Pan**: Right-click and drag to move the view
                                    - **Reset**: Double-click to reset the camera
                                    """)
                                    
                                    # Add a note explaining the benefits of 3D visualization
                                    st.markdown("""
                                    💡 **Why 3D?** The third dimension provides better separation between generated molecules and background molecules,
                                    helping to visualize which regions of chemical space are being explored by the generative model.
                                    """)
                                else:
                                    st.warning("3D data not available. Please regenerate the TSNE visualization.")
                                    # Fallback to 2D
                                    fig = px.scatter(
                                        combinedset, 
                                        x='TSNE_C1', 
                                        y='TSNE_C2',
                                        color='Source',
                                        size='marker_size',
                                        size_max=10,
                                        opacity=combinedset['opacity'],
                                        color_discrete_map={'Generated': '#1E88E5', 'Background': '#d3d3d3'},
                                        title='t-SNE 2D Visualization of Chemical Space (3D not available)',
                                        hover_name='hover_text',
                                        custom_data=['Molecule_Number', 'Source']
                                    )

                            # Update traces to emphasize generated molecules
                            fig.update_traces(
                                marker=dict(
                                    line=dict(width=1, color='#000000'),
                                ),
                                selector=dict(name='Generated')
                            )
                            
                            fig.update_layout(
                                height=700,
                                template='plotly_white',
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="right",
                                    x=0.99
                                ),
                                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                            )
                            
                            # Add molecule index numbers for easier selection
                            # Instead of highlighting regions, we'll add better tooltips
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        else:
                            # Create scatter plot for just the generated molecules with enhanced visuals
                            # Add color intensity based on TSNE coordinates to provide more visual information
                            chembl_data['distance'] = np.sqrt(chembl_data['TSNE_C1']**2 + chembl_data['TSNE_C2']**2)
                            chembl_data['marker_size'] = 12  # Larger markers for better visibility
                            
                            # Add molecule numbers (1-indexed) for the dataset
                            chembl_data['Molecule_Number'] = range(1, len(chembl_data) + 1)
                            
                            # Create custom hover text with molecule numbers for easier selection
                            chembl_data['hover_text'] = chembl_data.apply(
                                lambda row: f"Molecule #{int(row['Molecule_Number'])}<br>SMILES: {row['SMILES'][:30]}...", 
                                axis=1
                            )
                            
                            # Choose between 2D and 3D visualization for generated molecules only
                            if plot_dim == "2D Plot":
                                # Create a detailed 2D visualization with enhanced tooltips
                                fig = px.scatter(
                                    chembl_data, 
                                    x='TSNE_C1', 
                                    y='TSNE_C2',
                                    color='distance',
                                    size=[12] * len(chembl_data),  # Uniform size for all points
                                    size_max=15,
                                    color_continuous_scale='viridis',
                                    title='t-SNE 2D Visualization of Generated Molecules',
                                    hover_name='hover_text',
                                    custom_data=['Molecule_Number'],  # Include molecule number for callbacks
                                    labels={'distance': 'Distance from Origin'}
                                )
                            else:
                                # Create 3D scatter plot if TSNE_C3 exists
                                if 'TSNE_C3' in chembl_data.columns:
                                    # Calculate 3D distance for coloring
                                    chembl_data['distance_3d'] = np.sqrt(
                                        chembl_data['TSNE_C1']**2 + 
                                        chembl_data['TSNE_C2']**2 + 
                                        chembl_data['TSNE_C3']**2
                                    )
                                    
                                    # For 3D scatter plot, use go.Figure directly for better customization
                                    # Create normalized colors based on distance
                                    min_dist = chembl_data['distance_3d'].min()
                                    max_dist = chembl_data['distance_3d'].max()
                                    norm_distances = (chembl_data['distance_3d'] - min_dist) / (max_dist - min_dist)

                                    # Create a custom colormap
                                    import matplotlib.cm as cm
                                    import matplotlib.colors as colors

                                    # Get colors from viridis colormap
                                    viridis = cm.get_cmap('viridis')
                                    norm = colors.Normalize(vmin=min_dist, vmax=max_dist)
                                    
                                    # Create color values for each point
                                    colorscale = [[0, 'rgb(68,1,84)'], 
                                                 [0.25, 'rgb(59,82,139)'],
                                                 [0.5, 'rgb(33,144,141)'],
                                                 [0.75, 'rgb(93,201,99)'],
                                                 [1, 'rgb(253,231,37)']]
                                    
                                    # Create the 3D scatter plot
                                    fig = go.Figure(data=[go.Scatter3d(
                                        x=chembl_data['TSNE_C1'],
                                        y=chembl_data['TSNE_C2'],
                                        z=chembl_data['TSNE_C3'],
                                        mode='markers',
                                        marker=dict(
                                            size=8,
                                            color=chembl_data['distance_3d'],
                                            colorscale=colorscale,
                                            opacity=1.0,
                                            colorbar=dict(
                                                title='Distance from Origin',
                                                titleside='right'
                                            ),
                                            line=dict(width=1, color='black')
                                        ),
                                        text=chembl_data['hover_text'],
                                        hoverinfo='text'
                                    )])
                                    
                                    # Set the title and other layout options
                                    fig.update_layout(
                                        title='t-SNE 3D Visualization of Generated Molecules',
                                        scene=dict(
                                            xaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
                                            yaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
                                            zaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
                                            camera=dict(
                                                eye=dict(x=1.5, y=1.5, z=1.2)
                                            )
                                        ),
                                        scene_aspectmode='data'
                                    )
                                    
                                    # Instructions for 3D navigation follow
                                    
                                    # Add instructions for 3D interaction
                                    st.info("""
                                    **3D Navigation Tips:**
                                    - **Rotate**: Click and drag to rotate the 3D view
                                    - **Zoom**: Use scroll wheel to zoom in/out
                                    - **Pan**: Right-click and drag to move the view
                                    - **Reset**: Double-click to reset the camera
                                    """)
                                    
                                    # Add a note explaining the benefits of 3D visualization
                                    if viz_type == "Combined View (Generated + Background)":
                                        st.markdown("""
                                        💡 **Why 3D?** The third dimension provides better separation between molecule clusters, 
                                        helping to visualize chemical space relationships that might be hidden in the 2D projection.
                                        """)
                                    else:
                                        st.markdown("""
                                        💡 **Why 3D?** The third dimension helps reveal structural patterns and similarity clusters
                                        among generated molecules that might be compressed or overlapping in 2D view.
                                        """)
                                else:
                                    st.warning("3D data not available. Please regenerate the TSNE visualization.")
                                    # Fallback to 2D
                                    fig = px.scatter(
                                        chembl_data, 
                                        x='TSNE_C1', 
                                        y='TSNE_C2',
                                        color='distance',
                                        size=[12] * len(chembl_data),
                                        size_max=15,
                                        color_continuous_scale='viridis',
                                        title='t-SNE 2D Visualization of Generated Molecules (3D not available)',
                                        hover_name='hover_text',
                                        custom_data=['Molecule_Number'],
                                        labels={'distance': 'Distance from Origin'}
                                    )
                            
                            # Add black outline to make points more distinct
                            fig.update_traces(
                                marker=dict(
                                    line=dict(width=1, color='black'),
                                )
                            )
                            
                            # Enhance the layout
                            fig.update_layout(
                                height=700,
                                template='plotly_white',
                                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                                coloraxis_colorbar=dict(
                                    title='Distance',
                                    tickvals=[chembl_data['distance'].min(), 
                                              (chembl_data['distance'].min() + chembl_data['distance'].max())/2, 
                                              chembl_data['distance'].max()],
                                    ticktext=['Low', 'Medium', 'High'],
                                ),
                                # Add grid for better readability
                                xaxis=dict(
                                    showgrid=True,
                                    gridcolor='rgba(200, 200, 200, 0.4)'
                                ),
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor='rgba(200, 200, 200, 0.4)'
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Add interactive molecule selection and viewing
                        st.markdown("#### Interactive Molecule Viewer")
                        st.info("Hover over points in the TSNE plot to see molecule numbers. Use these numbers to select molecules from the dropdown below to view their structures.")
                        
                        # Create a dropdown for selecting molecules with better labeling
                        selected_index = st.selectbox(
                            "Select a molecule to view:",
                            options=range(len(chembl_data)),
                            format_func=lambda i: f"Molecule #{i+1} - {chembl_data['SMILES'].iloc[i][:20]}..."
                        )
                        
                        if selected_index is not None:
                            selected_smiles = chembl_data['SMILES'].iloc[selected_index]
                            st.code(selected_smiles, language=None)
                            
                            # Show "View Structure" button
                            if st.button("View Selected Structure", key="view_tsne_structure"):
                                try:
                                    from rdkit import Chem
                                    from rdkit.Chem import Draw
                                    
                                    mol = Chem.MolFromSmiles(selected_smiles)
                                    if mol:
                                        # Generate the image
                                        img = Draw.MolToImage(mol, size=(600, 400))
                                        
                                        # Display the image
                                        st.image(img, caption=f"Structure of Selected Molecule {selected_index + 1}")
                                        
                                        # Add some basic molecular properties
                                        st.markdown("##### Properties")
                                        from rdkit.Chem import Descriptors, Lipinski
                                        
                                        props = {
                                            "Molecular Weight": round(Descriptors.MolWt(mol), 2),
                                            "LogP": round(Descriptors.MolLogP(mol), 2),
                                            "H-Bond Donors": Lipinski.NumHDonors(mol),
                                            "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
                                            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
                                            "TSNE Coordinates": f"({round(chembl_data['TSNE_C1'].iloc[selected_index], 2)}, {round(chembl_data['TSNE_C2'].iloc[selected_index], 2)})"
                                        }
                                        
                                        # Display properties in a nice grid layout
                                        prop_cols = st.columns(3)
                                        for idx, (name, value) in enumerate(props.items()):
                                            col_idx = idx % 3
                                            with prop_cols[col_idx]:
                                                st.metric(label=name, value=value)
                                        
                                        # Add nearest neighbors analysis - helps understand the local chemical space
                                        st.markdown("##### Nearest Neighbors in Chemical Space")
                                        try:
                                            # Calculate distances from this molecule to all other generated molecules
                                            current_point = np.array([chembl_data['TSNE_C1'].iloc[selected_index], 
                                                                      chembl_data['TSNE_C2'].iloc[selected_index]])
                                            
                                            # Calculate Euclidean distances to all other points
                                            distances = []
                                            for i in range(len(chembl_data)):
                                                if i != selected_index:  # Skip self
                                                    point = np.array([chembl_data['TSNE_C1'].iloc[i], 
                                                                     chembl_data['TSNE_C2'].iloc[i]])
                                                    dist = np.linalg.norm(current_point - point)
                                                    distances.append((i, dist))
                                            
                                            # Sort by distance and get the 5 closest neighbors
                                            distances.sort(key=lambda x: x[1])
                                            closest_neighbors = distances[:5]
                                            
                                            # Create a dataframe for display
                                            neighbors_df = pd.DataFrame({
                                                "Molecule": [f"Molecule {i+1}" for i, _ in closest_neighbors],
                                                "Distance": [round(dist, 3) for _, dist in closest_neighbors],
                                                "SMILES": [chembl_data['SMILES'].iloc[i] for i, _ in closest_neighbors]
                                            })
                                            
                                            # Display the dataframe
                                            st.dataframe(neighbors_df)
                                            
                                            # Allow comparing with a neighbor
                                            if st.button("Compare with First Neighbor", key="compare_neighbor"):
                                                neighbor_idx = closest_neighbors[0][0]
                                                neighbor_smiles = chembl_data['SMILES'].iloc[neighbor_idx]
                                                
                                                # Create molecules
                                                mol1 = Chem.MolFromSmiles(selected_smiles)
                                                mol2 = Chem.MolFromSmiles(neighbor_smiles)
                                                
                                                if mol1 and mol2:
                                                    # Generate side-by-side images
                                                    img = Draw.MolsToGridImage(
                                                        [mol1, mol2], 
                                                        molsPerRow=2,
                                                        subImgSize=(300, 300),
                                                        legends=[f"Selected (#{selected_index+1})", 
                                                                f"Neighbor (#{neighbor_idx+1})"]
                                                    )
                                                    st.image(img, caption="Comparison of Selected Molecule with Nearest Neighbor")
                                        except Exception as e:
                                            st.warning(f"Could not calculate nearest neighbors: {str(e)}")
                                    else:
                                        st.error("Could not generate molecular structure. Invalid SMILES string.")
                                except ImportError:
                                    st.error("RDKit is required to display molecular structures. Please install it with `pip install rdkit`.")
                            
                        # Optional 3D visualization (if available)
                        with st.expander("**View Data Details**"):
                            st.markdown("#### TSNE Dataset Information")
                            
                            # Print shape information
                            st.write(f"Generated molecules dataset shape: {chembl_data.shape}")
                            st.write(f"Combined dataset shape: {combinedset.shape}")
                            
                            # Display the first few rows of each dataset
                            st.markdown("##### Generated Molecules Sample:")
                            st.dataframe(chembl_data.drop(columns=['FP']) if 'FP' in chembl_data.columns else chembl_data, height=200)
                            
                            st.markdown("##### Combined Dataset Sample:")
                            st.dataframe(combinedset.drop(columns=['FP']) if 'FP' in combinedset.columns else combinedset, height=200)
                            
                    except Exception as e:
                        st.error(f"Error creating TSNE visualization: {str(e)}")
                        st.info("Please check the format of the TSNE data files or run the TSNE analysis again.")
                else:
                    # Show option to run TSNE analysis
                    st.info("No TSNE data found for this experiment.")
                    
                    if st.button("Generate TSNE Visualization"):
                        try:
                            # Run the TSNE generation script
                            st.info("Generating TSNE visualization. This may take a minute...")
                            progress_bar = st.progress(0)
                            
                            # Import the make_tsne function from processes.tsne
                            sys.path.append(os.getcwd())
                            from processes.tsne import make_tsne
                            
                            # Redirect stdout to capture progress
                            import io
                            from contextlib import redirect_stdout
                            
                            with io.StringIO() as buf, redirect_stdout(buf):
                                # Set up custom paths for the TSNE generation
                                from configs.path_config import resources_path, memory_path
                                
                                # Define the experiment-specific paths
                                exp_memory_path = os.path.join(memory_path, selected_experiment)
                                exp_output_path = os.path.join(exp_memory_path, 'output')
                                exp_tsne_path = os.path.join(exp_memory_path, 'tsne')
                                
                                # Ensure the directories exist
                                os.makedirs(exp_tsne_path, exist_ok=True)
                                
                                # Create a custom paths dictionary
                                custom_paths = {
                                    'exp_output_path': exp_output_path,
                                    'exp_tsne_path': exp_tsne_path,
                                    'resources_path': resources_path
                                }
                                
                                # Run the TSNE generation with the custom paths
                                progress_bar.progress(30)
                                make_tsne(custom_paths=custom_paths)
                                progress_bar.progress(100)
                            
                            st.success("TSNE visualization generated successfully! Refresh the page to view.")
                            st.button("Refresh Page")
                            
                        except Exception as e:
                            st.error(f"Error generating TSNE visualization: {str(e)}")
                            st.info("Please check that the required files exist in the output directory.")


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


# Style the centered logout button
st.sidebar.markdown("""
<style>
div[data-testid="stButton"][data-baseweb="button"]:has(button:contains("Logout")) {
    display: flex;
    justify-content: center;
}
div[data-testid="stButton"][data-baseweb="button"]:has(button:contains("Logout")) button {
    background-color: #f63366;
    color: white;
    font-weight: bold;
    width: auto;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1.5rem;
    margin-bottom: 1rem;
    transition: background-color 0.3s;
}
div[data-testid="stButton"][data-baseweb="button"]:has(button:contains("Logout")) button:hover {
    background-color: #d01a4b;
}
</style>
""", unsafe_allow_html=True)

# Logout function
def logout():
    # Clear the session state completely to avoid "password incorrect" message
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# Create the logout button
if st.sidebar.button("Logout", key="logout_button", on_click=logout):
    pass  # The action happens in the on_click function
