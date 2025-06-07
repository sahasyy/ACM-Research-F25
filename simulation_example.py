
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from klfdapc import KLFDAPC, simulate_genetic_data
from klfdapc.utils import evaluate_clustering

def parameter_sensitivity_analysis():
   
    print("Parameter Sensitivity Analysis")
    print("-" * 30)
    
    # Generate base data
    X, y = simulate_genetic_data(n_samples=300, n_snps=800, 
                               n_populations=3, 
                               population_structure='island')
    
    # Test different sigma values
    sigma_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    
    for sigma in sigma_values:
        print(f"Testing sigma = {sigma}")
        
        klfdapc = KLFDAPC(n_components=15, n_reduced_features=2, 
                         kernel='rbf', sigma=sigma, knn=5)
        
        X_transformed = klfdapc.fit_transform(X, y)
        predictions = klfdapc.predict_population(X_transformed)
        
        metrics = evaluate_clustering(X_transformed, y, predictions)
        metrics['sigma'] = sigma
        results.append(metrics)
    
    # Plot results
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(df_results['sigma'], df_results['accuracy'], 'bo-')
    axes[0].set_xlabel('Sigma')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Sigma')
    axes[0].grid(True)
    
    axes[1].plot(df_results['sigma'], df_results['adjusted_rand_score'], 'ro-')
    axes[1].set_xlabel('Sigma')
    axes[1].set_ylabel('Adjusted Rand Score')
    axes[1].set_title('ARI vs Sigma')
    axes[1].grid(True)
    
    axes[2].plot(df_results['sigma'], df_results['silhouette_score'], 'go-')
    axes[2].set_xlabel('Sigma')
    axes[2].set_ylabel('Silhouette Score')
    axes[2].set_title('Silhouette vs Sigma')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df_results

def main():
    print("KLFDAPC Simulation Study")
    print("=" * 30)
    
    # Run parameter sensitivity analysis
    results = parameter_sensitivity_analysis()
    
    # Find optimal parameters
    best_idx = results['silhouette_score'].idxmax()
    best_params = results.iloc[best_idx]
    
    print(f"\nBest parameters (by silhouette score):")
    print(f"Sigma: {best_params['sigma']}")
    print(f"Accuracy: {best_params['accuracy']:.3f}")
    print(f"ARI: {best_params['adjusted_rand_score']:.3f}")
    print(f"Silhouette: {best_params['silhouette_score']:.3f}")

if __name__ == "__main__":
    main()