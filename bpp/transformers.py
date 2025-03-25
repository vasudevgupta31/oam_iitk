"""
Shared transformers module to ensure consistent class definitions
across train and predict operations
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Import LogTargetTransformer with flexible paths
try:
    from bpp.feature_engineering import LogTargetTransformer
except ImportError:
    try:
        from feature_engineering import LogTargetTransformer
    except ImportError:
        raise ImportError("Could not import LogTargetTransformer")


class SafeLogTransformer(LogTargetTransformer):
    """Modified log transformer to handle zeros/negative values"""
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