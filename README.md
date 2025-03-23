 # **HGM - Hierarchical Generative Model**
 
 ## **Overview**
 HGM is a Hierarchical Generative Model designed for molecular generation and optimization using SMILES sequences. This project integrates deep learning techniques for data processing, training, beam search, and sampling, ensuring efficient molecular design.
 
 ## **Project Structure**
 ```
 hgm/
 │── configs/              # Configuration files for model and pipeline settings
 │── input_data/           # Raw input files for training
 │── output_data/          # Generated resultant molecules per experiment
 │── pretrained/           # Pretrained model files (.h5) for transfer learning
 │── funcs/                # Utility functions for data processing and model handling
 │── processes/            # processes abstraction for different routines in 
 │── memory/               # Generated interim files, models etc for the experiment
 │── README.md             # Project documentation
 │── pyproject.toml        # Poetry configuration file for dependency management
 │── main.py               # Entry point for executing the full pipeline
 ```
 
 ## **Installation**
 1. **Clone the repository**:
    ```bash
    git clone https://github.com/vasudevgupta31/iitk_oam_hgm.git
    cd hgm
    ```
 2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    ```
 3. **Install dependencies**:
 Local deployment is better with conda 
 ```conda env create -f environment.yml```
 

 
 ## **Usage**
 Modify `configs/config.ini` to set up:
 - input_file (File should be placed within `input_data`)
 - pretrained model (h5 file should be placed in /pretrained and the name should be given in config.ini file) (you can download from - https://drive.google.com/file/d/1hyMgwQnU9V7u5cKER9dSS_0pwJneWluj/view?usp=drive_link)
 - Trianing hyperparameters
 - other training options.

 1. Add an input file (my_input.txt) to /input_data
 2. Set `NameData` value to the added file in config.ini[INPUT] example: `my_input_file.txt`
 3. Set `experiment_name` in config.ini[INPUT] example: `my_experiment`
 4. Set `override` to `Y` in config.ini[INPUT] if the same input file should overwrite the previous runs
```
> conda activate oam_hgm
> python main.py
```

# You will see results in /output_data under the folder `my_experiment`
 
 
 ## **Logging & Debugging**
 The project utilizes `loguru` for logging. Logs are stored in `results/logs/` and provide insights into execution.
  
 ## **Contact**
 For queries or support, reach out at 
 `akshaykakkar.email@example.com`.
 `guptavasudelhi@gmail.com`
 `lokesh@domain.com`
