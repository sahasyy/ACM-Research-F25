import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.metrics import accuracy_score, classification_report

def load_genetic_data(file_path: str, format: str = 'csv') -> Tuple[np.ndarray, np.ndarray]:
    if format == 'csv':
        data = pd.read_csv(file_path)
        # Assume last column is population labels
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y
    else:
        raise NotImplementedError(f"Format {format} not yet implemented")

def evaluate_clustering(X_transformed: np.ndarray, y_true: np.ndarray, 
                       y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'adjusted_rand_score': adjusted_rand_score(y_true, y_pred),
        'silhouette_score': silhouette_score(X_transformed, y_true)
    }
    
    return metrics

def prepare_genetic_data(X: np.ndarray, missing_threshold: float = 0.1,
                        maf_threshold: float = 0.05) -> np.ndarray:
    
    # Handle missing data (assuming -1 or NaN represents missing)
    missing_rate = np.mean(X == -1, axis=0)
    valid_snps = missing_rate < missing_threshold
    
    X_filtered = X[:, valid_snps]
    
    # Filter by MAF
    allele_freq = np.mean(X_filtered, axis=0) / 2.0
    maf = np.minimum(allele_freq, 1 - allele_freq)
    maf_filter = maf >= maf_threshold
    
    X_filtered = X_filtered[:, maf_filter]
    
    return X_filtered
