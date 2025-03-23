import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Target transformer for log transformation of y values
class LogTargetTransformer(BaseEstimator, TransformerMixin):
    """Transform target values with log and inverse transform predictions to original scale"""
    
    def __init__(self, base=np.e):
        self.base = base
    
    def fit(self, y, X=None):
        return self
    
    def transform(self, y):
        """Apply log transformation to y values"""
        if isinstance(y, pd.Series):
            y = y.values
        return np.log(y) / np.log(self.base)
    
    def inverse_transform(self, y_pred):
        """Reverse log transformation to get predictions in original scale"""
        return np.power(self.base, y_pred)
