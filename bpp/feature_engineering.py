import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

class SMILESTransformer(BaseEstimator, TransformerMixin):
    """Base class for SMILES transformers with parallel processing capabilities"""
    
    def __init__(self, n_jobs=-1, verbose=0):
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def _process_smiles(self, smiles):
        """Process a single SMILES string - to be implemented by child classes"""
        raise NotImplementedError
    
    def fit(self, X, y=None):
        """Nothing to fit for SMILES transformers"""
        return self
    
    def transform(self, X):
        """Transform SMILES strings in parallel"""
        if isinstance(X, pd.DataFrame) and 'SMILES' in X.columns:
            smiles_list = X['SMILES'].values
        elif isinstance(X, pd.Series):
            smiles_list = X.values
        else:
            smiles_list = X
            
        # Use parallel processing with joblib
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._process_smiles)(smile) for smile in smiles_list
        )
        
        return np.array(results)


class MorganFingerprintTransformer(SMILESTransformer):
    """Transformer to convert SMILES to Morgan fingerprints"""
    
    def __init__(self, radius=2, n_bits=2048, n_jobs=-1, verbose=0):
        super().__init__(n_jobs=n_jobs, verbose=verbose)
        self.radius = radius
        self.n_bits = n_bits
        
    def _process_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Use MorganGenerator - the recommended approach (avoids deprecation warnings)
                from rdkit.Chem.AllChem import MorganGenerator
                fp = np.zeros((0,), dtype=np.int8)
                MorganGenerator.GetMorganFingerprintBitVect(mol, radius=self.radius, nBits=self.n_bits, bitInfo=None, useChirality=False, useBondTypes=True, useFeatures=False, vec=fp)
                return np.array(fp)
            else:
                return np.zeros(self.n_bits)
        except Exception:
            # Fallback to older method if MorganGenerator not available
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                return np.array(fp)
            else:
                return np.zeros(self.n_bits)


class RDKitDescriptorTransformer(SMILESTransformer):
    """Transformer to calculate RDKit descriptors from SMILES"""
    
    def __init__(self, descriptors=None, n_jobs=-1, verbose=0):
        super().__init__(n_jobs=n_jobs, verbose=verbose)
        
        # Store descriptor names and functions separately to avoid lambda pickling issues
        if descriptors is None:
            # Only use built-in descriptors (avoid custom lambdas)
            safe_descriptors = []
            for desc_name, desc_func in Descriptors.descList:
                # Skip descriptors that are lambda functions (can't be pickled)
                if "<lambda>" not in str(desc_func):
                    safe_descriptors.append((desc_name, desc_func))
            self.descriptors = safe_descriptors
        else:
            self.descriptors = descriptors
            
        # Store descriptor count for initializing empty arrays
        self.n_descriptors = len(self.descriptors)
    
    def _process_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                return np.array([desc_func(mol) for _, desc_func in self.descriptors])
            except Exception:
                return np.zeros(self.n_descriptors)
        else:
            return np.zeros(self.n_descriptors)


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



def create_molecular_pipeline(fingerprint_radius=2, 
                             fingerprint_bits=2048,
                             descriptor_subset=None,
                             n_jobs=-1, 
                             scale_features=True,
                             use_descriptors=True):
    """
    Create an optimized pipeline for molecular feature extraction
    
    Parameters:
    -----------
    fingerprint_radius : int
        Radius for Morgan fingerprint generation
    fingerprint_bits : int
        Number of bits in the fingerprint
    descriptor_subset : list or None
        Subset of RDKit descriptors to use (if None, all descriptors are used)
    n_jobs : int
        Number of parallel jobs to run (-1 for all available cores)
    scale_features : bool
        Whether to scale the features with StandardScaler
    use_descriptors : bool
        Whether to include RDKit descriptors (set to False to avoid pickling issues)
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Ready-to-use sklearn pipeline for feature generation
    """
    # Create feature generation components
    fingerprint_transformer = MorganFingerprintTransformer(
        radius=fingerprint_radius,
        n_bits=fingerprint_bits,
        n_jobs=n_jobs
    )
    
    # Create preprocessing for fingerprints (optional scaling)
    fingerprint_pipeline = Pipeline([
        ('fingerprints', fingerprint_transformer),
        ('scaler', StandardScaler() if scale_features else 'passthrough')
    ])
    
    # If we're including descriptors
    if use_descriptors:
        # Define descriptor subset if provided
        if descriptor_subset is not None:
            # Ensure we only use pickable descriptors (no lambdas)
            descriptors = []
            for desc in Descriptors.descList:
                if desc[0] in descriptor_subset and "<lambda>" not in str(desc[1]):
                    descriptors.append(desc)
        else:
            descriptors = None
        
        descriptor_transformer = RDKitDescriptorTransformer(
            descriptors=descriptors,
            n_jobs=n_jobs
        )
        
        # Create preprocessing for descriptors (handle NaNs and scale)
        descriptor_pipeline = Pipeline([
            ('descriptors', descriptor_transformer),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler() if scale_features else 'passthrough')
        ])
        
        # Complete pipeline with feature union
        complete_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('fingerprint_pipe', fingerprint_pipeline),
                ('descriptor_pipe', descriptor_pipeline)
            ]))
        ])
    else:
        # Just use fingerprints without descriptors
        complete_pipeline = fingerprint_pipeline
    
    return complete_pipeline
