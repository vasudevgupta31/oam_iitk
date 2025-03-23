import os
import numpy as np
import pandas as pd
import joblib

from feature_engineering import create_molecular_pipeline
from transformations import LogTargetTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def create_molecular_prediction_pipeline(
        regressor=None,
        fingerprint_radius=2, 
        fingerprint_bits=2048,
        descriptor_subset=None,
        log_transform_target=True,
        log_base=np.e,
        n_jobs=-1, 
        scale_features=True):
    """
    Create a complete pipeline including feature generation and prediction
    
    Parameters:
    -----------
    regressor : estimator, default=None
        The regression model to use (defaults to RandomForestRegressor if None)
    fingerprint_radius : int
        Radius for Morgan fingerprint generation
    fingerprint_bits : int
        Number of bits in the fingerprint
    descriptor_subset : list or None
        Subset of RDKit descriptors to use (if None, all descriptors are used)
    log_transform_target : bool
        Whether to apply log transformation to target values
    log_base : float
        Base for the logarithm (default is natural log with base e)
    n_jobs : int
        Number of parallel jobs to run (-1 for all available cores)
    scale_features : bool
        Whether to scale the features with StandardScaler
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Complete pipeline for feature generation and prediction
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # Use default regressor if none provided
    if regressor is None:
        regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=n_jobs
        )
    
    # Create feature generation pipeline
    feature_pipeline = create_molecular_pipeline(
        fingerprint_radius=fingerprint_radius,
        fingerprint_bits=fingerprint_bits,
        descriptor_subset=descriptor_subset,
        n_jobs=n_jobs,
        scale_features=scale_features
    )
    
    # If log transform is requested, wrap the regressor in TransformedTargetRegressor
    if log_transform_target:
        from sklearn.compose import TransformedTargetRegressor
        final_regressor = TransformedTargetRegressor(
            regressor=regressor,
            transformer=LogTargetTransformer(base=log_base)
        )
    else:
        final_regressor = regressor
    
    # Complete pipeline with feature generation and prediction
    complete_pipeline = Pipeline([
        ('features', feature_pipeline),
        ('regressor', final_regressor)
    ])
    
    return complete_pipeline
