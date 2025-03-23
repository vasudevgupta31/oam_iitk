# **HGM - Hierarchical Generative Model**

![IIT Kanpur Logo](resources/iitklogo.png)

## **Overview**
HGM is a Hierarchical Generative Model designed for molecular generation and optimization using SMILES sequences. This project integrates deep learning techniques for data processing, training, beam search, and sampling, enabling efficient molecular design for drug discovery and materials science applications.

The repository includes both a command-line pipeline and a user-friendly Streamlit application for model configuration, training, and visualization.

## **Project Structure**
```
hgm/
│── configs/              # Configuration files for model and pipeline settings
│── input_data/           # Raw input files for training
│── output_data/          # Generated resultant molecules per experiment
│── pretrained/           # Pretrained model files (.h5) for transfer learning
│── funcs/                # Utility functions for data processing and model handling
│── processes/            # Process abstractions for different pipeline routines
│── memory/               # Generated interim files, models etc. for experiments
│── resources/            # UI resources and static files
│── app.py                # Streamlit application entry point
│── app_link.py           # Connects the Streamlit app with the backend pipeline
│── main.py               # Entry point for executing the pipeline via CLI
│── README.md             # Project documentation
```

## **Installation**

### **Prerequisites**
- Python 3.9 or newer
- CUDA compatible GPU (optional but recommended for faster training)

### **Setup**
1. **Clone the repository**:
   ```bash
   git clone git@github.com:vasudevgupta31/oam_iitk.git
   cd hgm
   ```

2. **Create and activate a virtual environment using conda** (recommended):
   ```bash
   conda env create -f environment.yml
   conda activate oam_hgm
   ```
   
   Alternatively, you can use pip with the requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pretrained model** (optional):
   - Download the pretrained model from [this link](https://drive.google.com/file/d/1hyMgwQnU9V7u5cKER9dSS_0pwJneWluj/view?usp=drive_link)
   - Place the downloaded .h5 file in the `/pretrained` directory

## **Usage**

### **Command Line Interface**
You can run the pipeline directly from the command line for batch processing:

1. Modify `config.ini` to set your parameters:
   - Input file location (should be placed within `input_data/`)
   - Pretrained model name
   - Training hyperparameters
   - Other processing options

2. Run the pipeline:
   ```bash
   conda activate oam_hgm
   python main.py
   ```

3. Results will be available in `/output_data` under your experiment name

### **Streamlit Web Application**

The Streamlit application provides an intuitive GUI for configuring, running, and monitoring experiments.

1. **Start the Streamlit app**:
   ```bash
   conda activate oam_hgm
   streamlit run app.py
   If want to tunnel for other people without hosting <!-- ngrok http http://localhost:8501 -->
   ```

2. **Access the web interface** in your browser at `http://localhost:8501`

### **Web Application Features**

The Streamlit application is divided into several key sections:

#### **1. Navigation**
- **Create Experiment**: Configure and launch new experiments
- **Browse Experiments**: Review and analyze completed experiments

#### **2. Create Experiment Tab**
This section is organized into multiple tabs for step-by-step configuration:

##### **Setup Tab**
- Upload SMILES input files
- Name your experiment
- Configure override options

##### **Processing Tab**
- Configure train/validation split ratio
- Set minimum and maximum sequence lengths
- Adjust data augmentation factors

##### **Model Tab**
- Select pretrained model
- Configure neural network architecture
- Adjust training parameters (epochs, learning rate, batch size)
- Fine-tune dropout rates and trainable layers

##### **Input Tab**
- View and select from available input files
- Verify current input settings

##### **Beam Search Tab**
- Configure beam width for molecular generation
- Set the starting epoch for beam search

##### **Sampling Tab**
- Adjust temperature for molecule sampling
- Set number of samples to generate
- Configure sampling from last N epochs
- Enable multiprocessing for faster sampling

##### **Communication Tab**
- Configure email notifications for experiment completion

##### **Pipeline Execution Tab**
- Review configuration summary
- Save configuration settings
- Run the pipeline

#### **3. Browse Experiments Tab**
- View all completed and in-progress experiments
- Analyze training history with interactive charts
- Explore generated molecules with 2D structure visualization
- Download experiment results in CSV format
- View molecular properties and export molecule data

## **Configuration Options**

### **Key Configuration Parameters**

#### **Processing**
- `split`: Train/validation split ratio (0.5-1.0)
- `min_len`: Minimum SMILES sequence length
- `max_len`: Maximum SMILES sequence length
- `augmentation`: Data augmentation factor

#### **Model**
- `pretrained_model`: Path to pretrained .h5 file
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `neurons`: Neural network layer sizes
- `dropouts`: Dropout rates for regularization
- `batch_size`: Training batch size

#### **Beam Search**
- `width`: Beam width for molecule generation
- `from_epoch`: Starting epoch for beam search

#### **Sampling**
- `temp`: Temperature for sampling (higher = more diversity)
- `n_sample`: Number of molecules to sample
- `last_n_epochs`: Number of recent epochs to sample from

## **Pipeline Steps**

The HGM pipeline consists of several sequential processes:

1. **Data Processing**: Prepares and augments SMILES data
2. **Model Training**: Trains a neural network on processed data
3. **Beam Search**: Performs beam search for molecular generation
4. **Sampling**: Generates molecular structures through probabilistic sampling
5. **Novo Analysis**: Evaluates and filters generated molecules

## **Email Notifications**

The system supports email notifications for experiment completion. To enable this feature:

1. Create a `.env` file in the project root with:
   ```
   EMAIL_USER=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password
   ```
2. Enter your email address in the Communication tab of the Streamlit app
3. You'll receive a notification when your experiment completes

## **Logging & Debugging**

- Detailed logs are stored in the `logs/` directory
- The Streamlit app displays real-time progress during experiment execution
- Check the `memory/{experiment_name}/status.json` file for current status

## **Advanced Usage**

### **Custom Token Vocabulary**
To modify the token vocabulary:
1. Edit `configs/fixed_params.py`
2. Update the `INDICES_TOKEN` and `TOKEN_INDICES` dictionaries

### **Custom Neural Network Architecture**
Advanced users can modify the neural network architecture by:
1. Editing `funcs/helpers_training.py`
2. Adjusting the `SeqModel` class implementation

## **Troubleshooting**

- **Missing pretrained model**: Ensure you've downloaded the pretrained model file to the `/pretrained` directory
- **CUDA errors**: Check your GPU drivers and CUDA installation
- **Memory errors**: Reduce batch size or model complexity for larger datasets
- **Invalid SMILES**: Ensure your input file contains valid SMILES strings

## **Contributing**
Contributions are welcome! Please feel free to submit a Pull Request.

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.

## **Contact**
For queries or support, reach out to:
- `akshaykakar@gmail.com`
- `guptavasudelhi@gmail.com`
