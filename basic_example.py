
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from klfdapc import KLFDAPC, simulate_genetic_data

def main():
    print("KLFDAPC Basic Example")
    print("=" * 30)
    
    # Generate simulated data
    print("Generating simulated genetic data...")
    X, y = simulate_genetic_data(n_samples=300, n_snps=1000, 
                                n_populations=4, 
                                population_structure='island')
    
    print(f"Data shape: {X.shape}")
    print(f"Populations: {np.unique(y)}")
    print(f"Samples per population: {np.bincount(y)}")
    
    # Initialize and fit KLFDAPC
    print("\nFitting KLFDAPC model...")
    klfdapc = KLFDAPC(
        n_components=20,
        n_reduced_features=3,
        kernel='rbf',
        sigma=2.0,
        knn=7
    )
    
    # Fit and transform
    X_transformed = klfdapc.fit_transform(X, y)
    print(f"Transformed data shape: {X_transformed.shape}")
    
    # Plot results
    print("Plotting results...")
    fig = klfdapc.plot_results(title="KLFDAPC Analysis - Basic Example")
    plt.show()
    
    # Evaluate performance
    predictions = klfdapc.predict_population(X_transformed)
    accuracy = np.mean(predictions == y)
    print(f"\nPopulation prediction accuracy: {accuracy:.3f}")
    
    # Feature importance
    importance = klfdapc.get_feature_importance()
    print(f"Top 5 most important features: {np.argsort(importance)[-5:]}")

if __name__ == "__main__":
    main()