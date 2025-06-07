
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from klfdapc import simulate_genetic_data, compare_methods
from klfdapc.visualization import plot_comparison

def main():
    print("KLFDAPC Method Comparison")
    print("=" * 30)
    
    # Test different population structures
    structures = ['island', 'stepping_stone', 'hierarchical']
    
    for structure in structures:
        print(f"\nTesting {structure} model...")
        
        # Generate data
        X, y = simulate_genetic_data(n_samples=200, n_snps=500, 
                                   n_populations=3, 
                                   population_structure=structure)
        
        # Compare methods
        results = compare_methods(X, y)
        
        # Plot comparison
        fig = plot_comparison(results, y, figsize=(18, 5))
        fig.suptitle(f'Method Comparison - {structure.title()} Model', fontsize=16)
        plt.show()
        
        # Evaluate silhouette scores
        from sklearn.metrics import silhouette_score
        
        print("Silhouette Scores:")
        for method, X_transformed in results.items():
            score = silhouette_score(X_transformed, y)
            print(f"  {method}: {score:.3f}")

if __name__ == "__main__":
    main()
