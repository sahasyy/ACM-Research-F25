
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh, inv, pinv
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, Tuple, Dict, Any
import warnings

class KLFDAPC:
    

    
    def __init__(
        self,
        n_components: int = 20,
        n_reduced_features: int = 3,
        kernel: str = 'rbf',
        sigma: float = 1.0,
        knn: int = 5,
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        metric: str = 'weighted'
    ):
       
        self.n_components = n_components
        self.n_reduced_features = n_reduced_features
        self.kernel = kernel
        self.sigma = sigma
        self.knn = knn
        self.gamma = gamma if gamma is not None else 1.0 / (2 * sigma**2)
        self.degree = degree
        self.coef0 = coef0
        self.metric = metric
        
        # Initialize components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        
        # Store fitted data
        self.X_pca = None
        self.X_kernel = None
        self.y = None
        self.classes = None
        self.transformation_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        
        if self.kernel == 'rbf':
            return rbf_kernel(X, gamma=self.gamma)
        elif self.kernel == 'polynomial':
            return polynomial_kernel(X, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'linear':
            return linear_kernel(X)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_local_affinity_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        n_samples = X.shape[0]
        A = np.zeros((n_samples, n_samples))
        
        # Find k-nearest neighbors for each sample
        nbrs = NearestNeighbors(n_neighbors=self.knn + 1, metric='euclidean')
        nbrs.fit(X)
        _, indices = nbrs.kneighbors(X)
        
        for i in range(n_samples):
            for j in indices[i, 1:]:  # Exclude self (index 0)
                if y[i] == y[j]:  # Same class
                    A[i, j] = 1.0 / self.knn
                    A[j, i] = 1.0 / self.knn
        
        return A
    
    def _compute_scatter_matrices(self, K: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        n_samples = K.shape[0]
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Compute local affinity matrix
        A = self._compute_local_affinity_matrix(K, y)
        
        # Compute class membership matrix
        class_mask = np.zeros((n_samples, n_classes))
        for i, cls in enumerate(classes):
            class_mask[y == cls, i] = 1
        
        # Compute class probabilities
        class_probs = np.sum(class_mask, axis=0) / n_samples
        
        # Initialize scatter matrices
        Sb = np.zeros((n_samples, n_samples))
        Sw = np.zeros((n_samples, n_samples))
        
        # Compute within-class scatter matrix
        for i in range(n_samples):
            for j in range(n_samples):
                if y[i] == y[j]:
                    weight = A[i, j]
                else:
                    weight = 0
                
                diff = K[:, i] - K[:, j]
                Sw += weight * np.outer(diff, diff)
        
        # Compute between-class scatter matrix
        for i in range(n_samples):
            for j in range(n_samples):
                if y[i] != y[j]:
                    # Local scaling
                    weight = (1.0 / n_samples) - A[i, j]
                else:
                    weight = 0
                
                diff = K[:, i] - K[:, j]
                Sb += weight * np.outer(diff, diff)
        
        return Sb, Sw
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KLFDAPC':
        
        # Store original data
        self.y = y
        self.classes = np.unique(y)
        
        # Step 1: Standardize and apply PCA
        X_scaled = self.scaler.fit_transform(X)
        self.X_pca = self.pca.fit_transform(X_scaled)
        
        # Step 2: Normalize PCA features
        self.X_pca = self._normalize_features(self.X_pca)
        
        # Step 3: Compute kernel matrix
        self.X_kernel = self._compute_kernel_matrix(self.X_pca)
        
        # Step 4: Compute scatter matrices
        Sb, Sw = self._compute_scatter_matrices(self.X_kernel, y)
        
        # Step 5: Solve generalized eigenvalue problem
        try:
            # Add small regularization to avoid singularity
            Sw_reg = Sw + 1e-10 * np.eye(Sw.shape[0])
            eigenvalues, eigenvectors = eigh(Sb, Sw_reg)
            
            # Sort by eigenvalues in descending order
            idx = np.argsort(eigenvalues)[::-1]
            self.eigenvalues = eigenvalues[idx]
            self.eigenvectors = eigenvectors[:, idx]
            
            # Select top reduced features
            self.transformation_matrix = self.eigenvectors[:, :self.n_reduced_features]
            
        except np.linalg.LinAlgError:
            warnings.warn("Eigenvalue decomposition failed. Using pseudo-inverse.")
            try:
                Sw_pinv = pinv(Sw)
                eigenvalues, eigenvectors = eigh(Sb @ Sw_pinv)
                
                idx = np.argsort(eigenvalues)[::-1]
                self.eigenvalues = eigenvalues[idx]
                self.eigenvectors = eigenvectors[:, idx]
                self.transformation_matrix = self.eigenvectors[:, :self.n_reduced_features]
                
            except Exception as e:
                raise RuntimeError(f"Failed to compute KLFDAPC transformation: {e}")
        
        return self
    
    def transform(self, X: Optional[np.ndarray] = None) -> np.ndarray:
       
        if self.transformation_matrix is None:
            raise ValueError("Model must be fitted before transform")
        
        if X is None:
            # Transform fitted data
            return self.X_kernel @ self.transformation_matrix
        else:
            # Transform new data
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)
            X_pca_norm = self._normalize_features(X_pca)
            
            # Compute kernel with training data
            K_new = self._compute_kernel_matrix_new(X_pca_norm, self.X_pca)
            
            return K_new @ self.transformation_matrix
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
       
        return self.fit(X, y).transform()
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
      
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        
        return (X - X_min) / X_range
    
    def _compute_kernel_matrix_new(self, X_new: np.ndarray, X_train: np.ndarray) -> np.ndarray:
       
        if self.kernel == 'rbf':
            return rbf_kernel(X_new, X_train, gamma=self.gamma)
        elif self.kernel == 'polynomial':
            return polynomial_kernel(X_new, X_train, degree=self.degree, 
                                   gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'linear':
            return linear_kernel(X_new, X_train)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def plot_results(self, X_transformed: Optional[np.ndarray] = None, 
                    y: Optional[np.ndarray] = None, 
                    title: str = "KLFDAPC Results") -> plt.Figure:
      
        if X_transformed is None:
            X_transformed = self.transform()
        if y is None:
            y = self.y
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot first two dimensions
        scatter = axes[0].scatter(X_transformed[:, 0], X_transformed[:, 1], 
                                c=y, cmap='tab10', alpha=0.7)
        axes[0].set_xlabel('KLFDAPC 1')
        axes[0].set_ylabel('KLFDAPC 2')
        axes[0].set_title(f'{title} - Dimensions 1 vs 2')
        axes[0].grid(True, alpha=0.3)
        
        # Add legend for populations
        classes = np.unique(y)
        for i, cls in enumerate(classes):
            mask = y == cls
            axes[0].scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                          label=f'Pop {cls}', alpha=0.7)
        axes[0].legend()
        
        # Plot eigenvalues
        if self.eigenvalues is not None:
            axes[1].plot(range(1, len(self.eigenvalues[:10]) + 1), 
                        self.eigenvalues[:10], 'bo-')
            axes[1].set_xlabel('Component')
            axes[1].set_ylabel('Eigenvalue')
            axes[1].set_title('Eigenvalues (Top 10)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_feature_importance(self) -> np.ndarray:
        
        if self.transformation_matrix is None:
            raise ValueError("Model must be fitted first")
        
        # Compute importance as the sum of absolute values across components
        importance = np.sum(np.abs(self.transformation_matrix), axis=1)
        return importance / np.sum(importance)  # Normalize
    
    def predict_population(self, X_transformed: np.ndarray) -> np.ndarray:
        
        if self.y is None:
            raise ValueError("Model must be fitted first")
        
        # Compute centroids for each population
        centroids = {}
        X_train_transformed = self.transform()
        
        for pop in self.classes:
            mask = self.y == pop
            centroids[pop] = np.mean(X_train_transformed[mask], axis=0)
        
        # Predict based on nearest centroid
        predictions = []
        for sample in X_transformed:
            distances = {}
            for pop, centroid in centroids.items():
                distances[pop] = np.linalg.norm(sample - centroid)
            
            predicted_pop = min(distances, key=distances.get)
            predictions.append(predicted_pop)
        
        return np.array(predictions)

# Utility functions for data simulation and analysis

def simulate_genetic_data(n_samples: int = 500, n_snps: int = 1000, 
                         n_populations: int = 3, 
                         population_structure: str = 'island') -> Tuple[np.ndarray, np.ndarray]:
    
    np.random.seed(42)
    
    # Create population labels
    samples_per_pop = n_samples // n_populations
    y = np.repeat(range(n_populations), samples_per_pop)
    
    # Initialize genetic data
    X = np.zeros((n_samples, n_snps))
    
    if population_structure == 'island':
        # Island model: populations are well-differentiated
        for pop in range(n_populations):
            start_idx = pop * samples_per_pop
            end_idx = start_idx + samples_per_pop
            
            # Each population has different allele frequencies
            pop_freq = 0.2 + 0.6 * pop / (n_populations - 1)
            X[start_idx:end_idx] = np.random.binomial(2, pop_freq, 
                                                    (samples_per_pop, n_snps))
    
    elif population_structure == 'stepping_stone':
        # Stepping stone model: gradual change between adjacent populations
        for pop in range(n_populations):
            start_idx = pop * samples_per_pop
            end_idx = start_idx + samples_per_pop
            
            # Gradual frequency change
            for snp in range(n_snps):
                base_freq = 0.3 + 0.4 * snp / n_snps
                pop_freq = base_freq + 0.2 * pop / n_populations
                pop_freq = np.clip(pop_freq, 0.05, 0.95)
                
                X[start_idx:end_idx, snp] = np.random.binomial(2, pop_freq, 
                                                              samples_per_pop)
    
    else:  # hierarchical
        # Hierarchical model: nested population structure
        for pop in range(n_populations):
            start_idx = pop * samples_per_pop
            end_idx = start_idx + samples_per_pop
            
            # Two levels of structure
            major_group = pop // 2
            minor_group = pop % 2
            
            base_freq = 0.3 + 0.3 * major_group
            pop_freq = base_freq + 0.1 * minor_group
            
            X[start_idx:end_idx] = np.random.binomial(2, pop_freq, 
                                                    (samples_per_pop, n_snps))
    
    return X, y

def compare_methods(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    
    results = {}
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    results['PCA'] = X_pca
    
    # LDA
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(StandardScaler().fit_transform(X), y)
    results['LDA'] = X_lda
    
    # KLFDAPC
    klfdapc = KLFDAPC(n_components=20, n_reduced_features=2, sigma=1.0)
    X_klfdapc = klfdapc.fit_transform(X, y)
    results['KLFDAPC'] = X_klfdapc
    
    return results

# Example usage and testing
if __name__ == "__main__":
    print("KLFDAPC Python Implementation")
    print("=" * 50)
    
    # Simulate data
    print("Simulating genetic data...")
    X, y = simulate_genetic_data(n_samples=300, n_snps=500, 
                                n_populations=3, 
                                population_structure='island')
    
    print(f"Data shape: {X.shape}")
    print(f"Populations: {np.unique(y)}")
    
    # Fit KLFDAPC
    print("\nFitting KLFDAPC...")
    klfdapc = KLFDAPC(n_components=20, n_reduced_features=3, 
                     kernel='rbf', sigma=1.0, knn=5)
    
    X_transformed = klfdapc.fit_transform(X, y)
    print(f"Transformed data shape: {X_transformed.shape}")
    
    # Plot results
    print("\nPlotting results...")
    fig = klfdapc.plot_results()
    plt.show()
    
    # Compare methods
    print("\nComparing with other methods...")
    comparison_results = compare_methods(X, y)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = ['PCA', 'LDA', 'KLFDAPC']
    
    for i, method in enumerate(methods):
        X_method = comparison_results[method]
        scatter = axes[i].scatter(X_method[:, 0], X_method[:, 1], 
                                c=y, cmap='tab10', alpha=0.7)
        axes[i].set_title(f'{method}')
        axes[i].set_xlabel(f'{method} 1')
        axes[i].set_ylabel(f'{method} 2')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
