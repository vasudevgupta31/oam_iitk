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
        if descriptors is None:
            self.descriptors = Descriptors.descList
        else:
            self.descriptors = descriptors
    
    def _process_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                return np.array([desc[1](mol) for desc in self.descriptors])
            except:
                return np.zeros(len(self.descriptors))
        else:
            return np.zeros(len(self.descriptors))


def create_molecular_pipeline(fingerprint_radius=2, 
                             fingerprint_bits=2048,
                             descriptor_subset=None,
                             n_jobs=-1, 
                             scale_features=True):
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
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Ready-to-use sklearn pipeline for feature generation
    """
    # Define descriptor subset if provided
    if descriptor_subset is not None:
        descriptors = [(desc[0], desc[1]) for desc in Descriptors.descList 
                      if desc[0] in descriptor_subset]
    else:
        descriptors = None
    
    # Create feature generation components
    fingerprint_transformer = MorganFingerprintTransformer(
        radius=fingerprint_radius,
        n_bits=fingerprint_bits,
        n_jobs=n_jobs
    )
    
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
    
    # Create preprocessing for fingerprints (optional scaling)
    fingerprint_pipeline = Pipeline([
        ('fingerprints', fingerprint_transformer),
        ('scaler', StandardScaler() if scale_features else 'passthrough')
    ])
    
    # Complete pipeline with feature union
    complete_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('fingerprint_pipe', fingerprint_pipeline),
            ('descriptor_pipe', descriptor_pipeline)
        ]))
    ])
    
    return complete_pipeline
