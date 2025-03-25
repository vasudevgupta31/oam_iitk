# CLAUDE.md - Project Guidelines

## Project Structure
- Feature engineering (feature_engineering.py): Molecular transformers and pipelines
- Data preprocessing (preprocess.py): Molecular data handling and cleaning
- Dataset configuration (dataset_config.py): Dataset paths and metadata

## Commands
```bash
# Run a specific dataset pipeline 
python -c "import feature_engineering as fe, preprocess as pp; from dataset_config import datasets_config; [commands]"

# Run linting 
flake8 *.py

# Type checking
mypy --ignore-missing-imports *.py
```

## Code Style Guidelines
- **Imports**: Group standard libs first, then third-party (numpy/pandas), then local
- **Type Hints**: Use for all function parameters and return values (Optional, List, Dict, etc.)
- **Docstrings**: Use Google style docstrings with Parameters and Returns sections
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Error Handling**: Use try/except blocks for specific exceptions, document failure modes
- **Logging**: Print statements with verbosity levels
- **Constants/Config**: Use dataset_config.py for configurable values