import os
import pandas as pd
import numpy as np
from joblib import load
import glob
import sys
from typing import List, Optional
from rdkit import Chem
from rdkit import DataStructs

# Disable RDKit warnings and logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Add parent directory to path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try both import paths to handle different usage scenarios
try:
    from bpp.dataset_config import datasets_config
    from bpp.feature_engineering import LogTargetTransformer, SMILESTransformer, MorganFingerprintTransformer
except ImportError:
    try:
        from dataset_config import datasets_config
        from feature_engineering import LogTargetTransformer, SMILESTransformer, MorganFingerprintTransformer
    except ImportError:
        raise ImportError("Could not import required modules from either bpp package or local imports")

# Import the shared transformer for consistent class definitions
try:
    from bpp.transformers import SafeLogTransformer
except ImportError:
    try:
        from transformers import SafeLogTransformer
    except ImportError:
        # Fallback if we can't import from the shared module
        class SafeLogTransformer(LogTargetTransformer):
            """Modified log transformer to handle zeros - redefining here for model loading"""
            def __init__(self, base=10, min_value=0.0001):
                super().__init__(base=base)
                self.min_value = min_value
                
            def transform(self, y):
                """Apply log transformation with safety for zeros/negatives"""
                if isinstance(y, pd.Series):
                    y = y.values
                # Replace zeros and negative values with minimum value
                y_safe = np.maximum(y, self.min_value)
                return np.log(y_safe) / np.log(self.base)


def predict_from_bpp(
    df: pd.DataFrame,
    smiles_col: str = 'SMILES',
    model_names: Optional[List[str]] = None,
    models_dir: str = 'models',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Predict EC50 values for a DataFrame with SMILES strings using all available models
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing SMILES strings
    smiles_col : str
        Name of the column containing SMILES strings
    model_names : list of str, optional
        Specific models to use. If None, uses all available models
    models_dir : str
        Directory containing model files
    prefix : str
        Prefix for prediction column names
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional columns for predictions from each model
    """
    # Verify the smiles column exists
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES colfrom rdkit import Chem'{smiles_col}' not found in DataFrame")
    
    # Make a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()

    # Find all .joblib files that don't end with _transformer.joblib
    model_files  = [os.path.basename(f).replace('.joblib', '') 
                for f in glob.glob(f"{models_dir}/*.joblib") 
                if not f.endswith('_transformer.joblib') and  
                not os.path.basename(f).startswith('kd_')]
                
    model_names = model_files

    if not model_names:
        raise ValueError(f"No models found in {models_dir} directory")

    # Track successful predictions
    successful_models = []

    # Process each model
    for model_name in model_names:
        # Determine the model file name
        if model_name in datasets_config:
            model_file = datasets_config[model_name]['model_file']
        else:
            model_file = model_name
        
        # Check if model files exist
        model_path = f"{models_dir}/{model_file}.joblib"
        transformer_path = f"{models_dir}/{model_file}_transformer.joblib"

        if not os.path.exists(model_path) or not os.path.exists(transformer_path):
            if verbose:
                print(f"Warning: Model files for {model_name} not found")
            continue

        if verbose:
            print(f"Making predictions with model: {model_name}")

        try:
            # Load model and transformer
            pipeline = load(model_path)
            target_transformer = load(transformer_path)

            # Create a temporary DataFrame with the right column name expected by the model
            # Our pipeline always expects 'SMILES' column
            temp_df = pd.DataFrame({
                'SMILES': df[smiles_col].values
            })

            # Make predictions
            predictions_log = pipeline.predict(temp_df)
            
            # Transform predictions back to original scale
            predictions = target_transformer.inverse_transform(predictions_log)
            
            # ADD TANIMOTO SIMILARITY
            base_smiles = 'CC[C@H](C)[C@@H](C(=O)N[C@@H](C)C(=O)N[C@@H](CC1=CNC2=CC=CC=C21)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)O)NC(=O)[C@H](CC3=CC=CC=C3)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCCCNC(=O)COCCOCCNC(=O)COCCOCCNC(=O)CC[C@H](C(=O)O)NC(=O)CCCCCCCCCCCCCCCCC(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CCC(=O)N)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC4=CC=C(C=C4)O)NC(=O)[C@H](CO)NC(=O)[C@H](CO)NC(=O)[C@H](C(C)C)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CO)NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CC5=CC=CC=C5)NC(=O)[C@H]([C@@H](C)O)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)C(C)(C)NC(=O)[C@H](CC6=CN=CN6)N'

            # Convert SMILES to molecule object
            base_mol = Chem.MolFromSmiles(base_smiles)

            if base_mol is None:
                print("Error: Could not parse base molecule SMILES")
            else:
                # Generate fingerprint from molecule object
                base_fp = Chem.RDKFingerprint(base_mol)

                # Calculate similarities
                result_df['tanimoto_similarity_semaglutide'] = result_df[smiles_col].apply(
                    lambda x: DataStructs.TanimotoSimilarity(
                        Chem.RDKFingerprint(Chem.MolFromSmiles(x)), base_fp
                    ) if Chem.MolFromSmiles(x) else 0
                )

            # Add predictions to result DataFrame
            result_df[model_name] = predictions
            successful_models.append(model_name)

        except Exception as e:
            if verbose:
                print(f"Error predicting with model {model_name}: {str(e)}")
            continue
    
    if verbose:
        if successful_models:
            print(f"Successfully applied {len(successful_models)} models: {', '.join(successful_models)}")
        else:
            print("No successful predictions were made")
    
    return result_df


# def main():
#     """Command-line interface for making predictions"""
#     import argparse
#     import sys
    
#     parser = argparse.ArgumentParser(description="Predict EC50 values for molecules")
#     parser.add_argument("--csv", "-c", required=True, help="CSV file with SMILES column")
#     parser.add_argument("--smiles-col", default="SMILES", help="Column name for SMILES in CSV (default: 'SMILES')")
#     parser.add_argument("--model", "-m", help="Specific model to use (defaults to all models)")
#     parser.add_argument("--output", "-o", required=True, help="Output file for results (CSV format)")
    
#     args = parser.parse_args()
    
#     try:
#         # Load the CSV file
#         print(f"Loading data from {args.csv}...")
#         df = pd.read_csv(args.csv)
        
#         if args.smiles_col not in df.columns:
#             print(f"Error: SMILES column '{args.smiles_col}' not found in CSV.")
#             print(f"Available columns: {', '.join(df.columns)}")
#             return 1
        
#         # Make predictions
#         model_names = [args.model] if args.model else None
#         result_df = predict_ec50_dataframe(
#             df=df, 
#             smiles_col=args.smiles_col, 
#             model_names=model_names
#         )
        
#         # Save results
#         result_df.to_csv(args.output, index=False)
#         print(f"Results saved to {args.output}")
        
#         # Print a summary
#         ec50_cols = [col for col in result_df.columns if col.startswith('EC50_')]
#         print(f"\nPredictions summary:")
#         print(f"- Processed {len(df)} compounds")
#         print(f"- Added {len(ec50_cols)} EC50 prediction columns")
#         print(f"- Prediction columns: {', '.join(ec50_cols)}")
        
#         return 0
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return 1


# if __name__ == "__main__":
#     # Run the CLI
#     import sys
#     sys.exit(main())