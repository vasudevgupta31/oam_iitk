# Disable RDKit warnings first thing
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Suppress RDKit output
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from joblib import dump
import csv
import time
from tqdm import tqdm

import sys
import os

# Add parent directory to path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try both import paths to handle different usage scenarios
try:
    from bpp.dataset_config import datasets_config
    from bpp.preprocess import MolecularDataPreprocessor
    from bpp.feature_engineering import create_molecular_pipeline, LogTargetTransformer
except ImportError:
    from dataset_config import datasets_config
    from preprocess import MolecularDataPreprocessor
    from feature_engineering import create_molecular_pipeline, LogTargetTransformer

# Create models directory if not exists
os.makedirs('models', exist_ok=True)

# Use lower n_jobs value to avoid pickling issues with RDKit
N_JOBS = 1

# Modified log transformer to handle zeros
class SafeLogTransformer(LogTargetTransformer):
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

def train_model(dataset_name, config, verbose=1):
    """Train a model for a specific dataset configuration"""
    if verbose >= 1:
        print(f"\nProcessing dataset: {dataset_name}")
    
    # Extract configuration parameters
    file_path = config['file']
    smiles_col = config['smiles_col']
    target_col = config['target_col']
    model_file = config['model_file']

    # Check if the file exists
    if not os.path.exists(file_path):
        if verbose >= 1:
            print(f"Error: File {file_path} does not exist")
        return False
    
    # Setup data preprocessor
    preprocessor = MolecularDataPreprocessor(
        smiles_col=smiles_col,
        target_col=target_col,
        validate_smiles=True,
        verbose=verbose
    )

    # Load and preprocess the data
    if verbose >= 1:
        print(f"Loading data from {file_path}...")

    try:
        data = preprocessor.load_data(file_path)
        X_data, y_target = preprocessor.prepare_dataset(data)

        if len(X_data) == 0 or y_target is None or len(y_target) == 0:
            if verbose >= 1:
                print(f"Error: No valid data for {dataset_name}")
            return False
            
        # Ensure the DataFrame has the right column name for the pipeline
        # This is critical - the feature pipeline expects 'SMILES' column
        if smiles_col != 'SMILES':
            X_data = X_data.rename(columns={smiles_col: 'SMILES'})

    except Exception as e:
        if verbose >= 1:
            print(f"Error preprocessing data: {str(e)}")
        return False

    if verbose >= 1:
        print(f"Data loaded: {len(X_data)} samples")
    
    # Create molecular feature extraction pipeline with single thread to avoid pickling errors
    feature_pipeline = create_molecular_pipeline(
        fingerprint_radius=2,
        fingerprint_bits=2048,
        n_jobs=N_JOBS,  # Use single thread to avoid pickling errors
        scale_features=True,
        use_descriptors=False  # Disable descriptors to avoid pickling errors
    )
    
    # Create a log transformer for the target values (EC50)
    target_transformer = SafeLogTransformer(base=10, min_value=0.0001)
    
    # Transform the target values (usually EC50 values benefit from log transformation)
    y_transformed = target_transformer.transform(y_target)
    
    if verbose >= 1:
        print(f"Target range: {y_target.min():.4f} - {y_target.max():.4f}")
        print(f"Log-transformed target range: {y_transformed.min():.4f} - {y_transformed.max():.4f}")
    
    # Create a full ML pipeline with feature extraction and model
    full_pipeline = Pipeline([
        ('features', feature_pipeline),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=N_JOBS,  # Use single thread to avoid pickling errors
            random_state=42
        ))
    ])

    # Perform hyperparameter tuning with Optuna before fitting
    n_trials = 50

    print(f"Performing hyperparameter optimization with {n_trials} trials...")
    
    try:
        import optuna
        from sklearn.model_selection import cross_val_score, KFold
        
        def objective(trial):
            # Hyperparameters for RandomForest
            n_estimators = trial.suggest_int('n_estimators', 50, 501, 25)
            max_depth = trial.suggest_int('max_depth', 10, 200, 5)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)

            # Create a tuned model pipeline
            tuned_pipeline = Pipeline([
                ('features', feature_pipeline),
                ('model', RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=N_JOBS,
                    random_state=42
                ))
            ])
            
            # Cross-validate
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(
                tuned_pipeline, 
                X_data, 
                y_transformed, 
                cv=cv, 
                scoring='neg_mean_squared_error',
                n_jobs=1
            )

            # Return the negative RMSE (to be maximized)
            return np.mean(scores)

        # Create optuna study
        study = optuna.create_study(direction='maximize')

        # Run optimization
        start_opt_time = time.time()
        study.optimize(objective, n_trials=n_trials)
        opt_time = time.time() - start_opt_time
        
        if verbose >= 1:
            print(f"Optimization completed in {opt_time:.2f} seconds")
            print(f"Best trial: {study.best_trial.number}")
            print(f"Best score: {study.best_value:.4f}")
            print("\nBest hyperparameters:")
            for param, value in study.best_params.items():
                print(f"- {param}: {value}")
        
        # Update pipeline with best parameters
        best_params = study.best_params
        full_pipeline = Pipeline([
            ('features', feature_pipeline),
            ('model', RandomForestRegressor(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', None),
                min_samples_split=best_params.get('min_samples_split', 2),
                min_samples_leaf=best_params.get('min_samples_leaf', 1),
                n_jobs=N_JOBS,
                random_state=42
            ))
        ])

        # Save optimization results
        opt_results_path = f"models/{model_file}_optimization.csv"
        study.trials_dataframe().to_csv(opt_results_path, index=False)
        if verbose >= 1:
            print(f"Optimization results saved to {opt_results_path}")

    except ImportError:
        if verbose >= 1:
            print("Optuna not installed, skipping optimization")
    
    # Fit the pipeline on the data
    if verbose >= 1:
        print(f"Training model for {dataset_name}...")
        start_time = time.time()

    try:
        full_pipeline.fit(X_data, y_transformed)

        if verbose >= 1:
            elapsed = time.time() - start_time
            print(f"Training completed in {elapsed:.2f} seconds")

        # Save the trained pipeline and target transformer
        model_path = f"models/{model_file}.joblib"
        transformer_path = f"models/{model_file}_transformer.joblib"

        dump(full_pipeline, model_path)
        dump(target_transformer, transformer_path)

        if verbose >= 1:
            print(f"Model saved to {model_path}")
            print(f"Target transformer saved to {transformer_path}")

        return True

    except Exception as e:
        if verbose >= 1:
            print(f"Error training model: {str(e)}")
        return False


if __name__ == "__main__":

    # Train all datasets
    successful = 0
    for dataset_name, config in tqdm(datasets_config.items(), desc="Training models"):
        if train_model(dataset_name, config, verbose=1):
            successful += 1
    
    print(f"\nTraining complete: {successful}/{len(datasets_config)} models successfully trained")
