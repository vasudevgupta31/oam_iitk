import pandas as pd
import numpy as np
from rdkit import Chem
import os
from typing import List, Union, Optional, Dict, Tuple, Any


class MolecularDataPreprocessor:
    """
    Class for preprocessing molecular datasets with SMILES strings and target values.
    Handles file loading, data cleaning, and dataset preparation.
    """
    
    def __init__(self, 
                 smiles_col: str = 'SMILES',
                 target_col: Optional[str] = None,
                 id_col: Optional[str] = None,
                 validate_smiles: bool = True,
                 verbose: int = 1):
        """
        Initialize the preprocessor with column names and options
        
        Parameters:
        -----------
        smiles_col : str
            Name of the SMILES column
        target_col : str or None
            Name of the target column (None for inference-only datasets)
        id_col : str or None
            Name of the molecule ID column (if any)
        validate_smiles : bool
            Whether to validate SMILES with RDKit and drop invalid ones
        verbose : int
            Verbosity level (0=quiet, 1=basic info, 2=detailed info)
        """
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.id_col = id_col
        self.validate_smiles = validate_smiles
        self.verbose = verbose
    
    def load_data(self, 
                  file_path: str, 
                  file_format: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load data from a file (auto-detect format if not specified)
        
        Parameters:
        -----------
        file_path : str
            Path to the file to load
        file_format : str or None
            Format of the file ('csv', 'tsv', 'excel', etc.)
            If None, auto-detect from file extension
        **kwargs : 
            Additional arguments to pass to the pandas read function
            
        Returns:
        --------
        pandas.DataFrame
            Loaded dataset
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect format from extension if not specified
        if file_format is None:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext in ['.csv', '.txt']:
                file_format = 'csv'
            elif ext in ['.tsv']:
                file_format = 'tsv'
            elif ext in ['.xlsx', '.xls']:
                file_format = 'excel'
            elif ext in ['.pkl', '.pickle']:
                file_format = 'pickle'
            elif ext in ['.parquet']:
                file_format = 'parquet'
            elif ext in ['.feather']:
                file_format = 'feather'
            else:
                raise ValueError(f"Unsupported file extension: {ext}. Please specify file_format.")
        
        # Load data based on format
        if file_format == 'csv':
            # Default to comma delimiter if not specified
            if 'sep' not in kwargs:
                kwargs['sep'] = ','
            data = pd.read_csv(file_path, **kwargs)
        elif file_format == 'tsv':
            # Default to tab delimiter
            if 'sep' not in kwargs:
                kwargs['sep'] = '\t'
            data = pd.read_csv(file_path, **kwargs)
        elif file_format == 'excel':
            data = pd.read_excel(file_path, **kwargs)
        elif file_format == 'pickle':
            data = pd.read_pickle(file_path, **kwargs)
        elif file_format == 'parquet':
            data = pd.read_parquet(file_path, **kwargs)
        elif file_format == 'feather':
            data = pd.read_feather(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        if self.verbose >= 1:
            print(f"Loaded {len(data)} rows from {file_path}")

        # Check if required columns exist
        if self.smiles_col not in data.columns:
            raise ValueError(f"SMILES column '{self.smiles_col}' not found in the dataset")
        
        if self.target_col is not None and self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in the dataset")
        
        if self.id_col is not None and self.id_col not in data.columns:
            if self.verbose >= 1:
                print(f"Warning: ID column '{self.id_col}' not found in the dataset")
            
        return data
    
    def _preprocess_data(self, 
                         data: pd.DataFrame, 
                         smiles_col: str, 
                         target_col: Optional[str] = None,
                         verbose: int = 1) -> pd.DataFrame:
        """
        Preprocess the dataset by cleaning target variable and handling missing values
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset to preprocess
        smiles_col : str
            Name of the SMILES column
        target_col : str or None
            Name of the target column (None for inference-only datasets)
        verbose : int
            Verbosity level
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed dataset
        """
        original_row_count = len(data)
        
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        # Check for missing SMILES
        missing_smiles = data[smiles_col].isna().sum()
        if missing_smiles > 0:
            if verbose >= 1:
                print(f"Found {missing_smiles} rows with missing SMILES. Dropping these rows.")
            data = data.dropna(subset=[smiles_col])
        
        # Convert SMILES to string type if not already
        data[smiles_col] = data[smiles_col].astype(str)
        
        # Check for invalid SMILES (completely empty or just whitespace)
        invalid_smiles = data[data[smiles_col].str.strip() == ""].shape[0]
        if invalid_smiles > 0:
            if verbose >= 1:
                print(f"Found {invalid_smiles} rows with empty SMILES. Dropping these rows.")
            data = data[data[smiles_col].str.strip() != ""]
        
        # Validate SMILES with RDKit if requested
        if self.validate_smiles:
            valid_molecules = []
            for idx, smiles in enumerate(data[smiles_col]):
                mol = Chem.MolFromSmiles(smiles)
                valid_molecules.append(mol is not None)
                
                # Print progress for large datasets
                if verbose >= 2 and (idx + 1) % 10000 == 0:
                    print(f"Validated {idx + 1}/{len(data)} SMILES")
            
            data['_valid_smiles'] = valid_molecules
            invalid_count = (~data['_valid_smiles']).sum()
            
            if invalid_count > 0:
                if verbose >= 1:
                    print(f"Found {invalid_count} invalid SMILES strings. Dropping these rows.")
                data = data[data['_valid_smiles']]
                data = data.drop(columns=['_valid_smiles'])
        
        # Handle target variable
        if target_col is not None and target_col in data.columns:
            # Try to convert target to numeric, coercing errors to NaN
            original_target_type = data[target_col].dtype
            data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
            
            # Count how many conversions failed (became NaN)
            conversion_failures = data[target_col].isna().sum()
            if conversion_failures > 0 and verbose >= 1:
                print(f"Converted target from {original_target_type} to numeric. "
                      f"{conversion_failures} values couldn't be converted and became NaN.")
            
            # Drop rows with missing target values
            missing_target = data[target_col].isna().sum()
            if missing_target > 0:
                if verbose >= 1:
                    print(f"Dropping {missing_target} rows with missing target values.")
                data = data.dropna(subset=[target_col])
        
        # Report total rows removed
        final_row_count = len(data)
        rows_removed = original_row_count - final_row_count
        if rows_removed > 0 and verbose >= 1:
            print(f"Preprocessing removed {rows_removed} rows ({rows_removed/original_row_count:.1%}). "
                  f"{final_row_count} rows remaining.")
        
        return data
    
    def prepare_dataset(self, 
                        data: pd.DataFrame, 
                        additional_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare dataset for model training or inference
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset to prepare
        additional_columns : list of str or None
            Additional columns to keep in the output dataset
            
        Returns:
        --------
        tuple
            - pandas.DataFrame with SMILES and additional columns
            - pandas.Series with target values (or None if no target column)
        """
        data = self._preprocess_data(data, self.smiles_col, self.target_col, self.verbose)
        
        # Select columns to keep
        keep_columns = [self.smiles_col]
        
        if self.id_col is not None and self.id_col in data.columns:
            keep_columns.append(self.id_col)

        if additional_columns is not None:
            for col in additional_columns:
                if col in data.columns and col not in keep_columns:
                    keep_columns.append(col)
        
        # Extract target if it exists
        target = None
        if self.target_col is not None and self.target_col in data.columns:
            target = data[self.target_col]
            target.clip(upper=target.quantile(0.95))

        # Return the processed DataFrame and target
        return data[keep_columns], target

    def process_file(self, 
                     file_path: str, 
                     file_format: Optional[str] = None,
                     additional_columns: Optional[List[str]] = None,
                     **kwargs) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Load and process a file in a single step

        Parameters:
        -----------
        file_path : str
            Path to the file to load
        file_format : str or None
            Format of the file ('csv', 'tsv', 'excel', etc.)
            If None, auto-detect from file extension
        additional_columns : list of str or None
            Additional columns to keep in the output dataset
        **kwargs : 
            Additional arguments to pass to the pandas read function
            
        Returns:
        --------
        tuple
            - pandas.DataFrame with SMILES and additional columns
            - pandas.Series with target values (or None if no target column)
        """
        data = self.load_data(file_path, file_format, **kwargs)
        return self.prepare_dataset(data, additional_columns)


# Utility function for simple usage
def preprocess_molecular_data(file_path: str, 
                              smiles_col: str = 'SMILES',
                              target_col: Optional[str] = None,
                              id_col: Optional[str] = None,
                              validate_smiles: bool = True,
                              additional_columns: Optional[List[str]] = None,
                              verbose: int = 1,
                              **kwargs) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Utility function to preprocess molecular data in a single call
    
    Parameters:
    -----------
    file_path : str
        Path to the file to load
    smiles_col : str
        Name of the SMILES column
    target_col : str or None
        Name of the target column (None for inference-only datasets)
    id_col : str or None
        Name of the molecule ID column (if any)
    validate_smiles : bool
        Whether to validate SMILES with RDKit and drop invalid ones
    additional_columns : list of str or None
        Additional columns to keep in the output dataset
    verbose : int
        Verbosity level (0=quiet, 1=basic info, 2=detailed info)
    **kwargs : 
        Additional arguments to pass to the pandas read function
        
    Returns:
    --------
    tuple
        - pandas.DataFrame with SMILES and additional columns
        - pandas.Series with target values (or None if no target column)
    """
    preprocessor = MolecularDataPreprocessor(
        smiles_col=smiles_col,
        target_col=target_col,
        id_col=id_col,
        validate_smiles=validate_smiles,
        verbose=verbose
    )
    
    return preprocessor.process_file(
        file_path=file_path,
        additional_columns=additional_columns,
        **kwargs
    )
