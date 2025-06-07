import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, Tuple
import pandas as pd

def plot_comparison(results: Dict[str, np.ndarray], y: np.ndarray, 
                   figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for i, (method, X_transformed) in enumerate(results.items()):
        scatter = axes[i].scatter(X_transformed[:, 0], X_transformed[:, 1], 
                                c=y, cmap='tab10', alpha=0.7)
        axes[i].set_title(f'{method}')
        axes[i].set_xlabel(f'{method} Component 1')
        axes[i].set_ylabel(f'{method} Component 2')
        axes[i].grid(True, alpha=0.3)
        
        # Add population labels
        for pop in np.unique(y):
            mask = y == pop
            axes[i].scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                          label=f'Pop {pop}', alpha=0.7)
        
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def plot_eigenvalues(eigenvalues: np.ndarray, n_components: int = 10) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = range(1, min(len(eigenvalues), n_components) + 1)
    ax.plot(components, eigenvalues[:len(components)], 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Component Number')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('KLFDAPC Eigenvalues')
    ax.grid(True, alpha=0.3)
    
    # Add cumulative variance explained
    ax2 = ax.twinx()
    cumvar = np.cumsum(eigenvalues[:len(components)]) / np.sum(eigenvalues)
    ax2.plot(components, cumvar, 'r^-', linewidth=2, markersize=6, alpha=0.7)
    ax2.set_ylabel('Cumulative Proportion', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    return fig

def plot_geographic_projection(X_transformed: np.ndarray, coordinates: np.ndarray, 
                              y: np.ndarray) -> plt.Figure:
   
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Genetic space
    scatter = axes[0, 0].scatter(X_transformed[:, 0], X_transformed[:, 1], 
                               c=y, cmap='tab10', alpha=0.7)
    axes[0, 0].set_xlabel('KLFDAPC 1')
    axes[0, 0].set_ylabel('KLFDAPC 2')
    axes[0, 0].set_title('Genetic Structure')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Geographic space
    axes[0, 1].scatter(coordinates[:, 0], coordinates[:, 1], 
                      c=y, cmap='tab10', alpha=0.7)
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    axes[0, 1].set_title('Geographic Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation plots
    axes[1, 0].scatter(X_transformed[:, 0], coordinates[:, 1], 
                      c=y, cmap='tab10', alpha=0.7)
    axes[1, 0].set_xlabel('KLFDAPC 1')
    axes[1, 0].set_ylabel('Latitude')
    axes[1, 0].set_title('KLFDAPC 1 vs Latitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(X_transformed[:, 1], coordinates[:, 0], 
                      c=y, cmap='tab10', alpha=0.7)
    axes[1, 1].set_xlabel('KLFDAPC 2')
    axes[1, 1].set_ylabel('Longitude')
    axes[1, 1].set_title('KLFDAPC 2 vs Longitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_population_structure(X_transformed: np.ndarray, y: np.ndarray, 
                             population_names: Optional[Dict] = None) -> plt.Figure:
   
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Main scatter plot
    unique_pops = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pops)))
    
    for i, pop in enumerate(unique_pops):
        mask = y == pop
        pop_name = population_names.get(pop, f'Pop {pop}') if population_names else f'Pop {pop}'
        axes[0, 0].scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                          c=[colors[i]], label=pop_name, alpha=0.7, s=50)
    
    axes[0, 0].set_xlabel('KLFDAPC 1')
    axes[0, 0].set_ylabel('KLFDAPC 2')
    axes[0, 0].set_title('Population Structure')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Population centroids
    centroids = []
    for pop in unique_pops:
        mask = y == pop
        centroid = np.mean(X_transformed[mask, :2], axis=0)
        centroids.append(centroid)
        axes[0, 1].scatter(centroid[0], centroid[1], c=[colors[pop]], 
                          s=200, marker='s', edgecolor='black', linewidth=2)
        axes[0, 1].text(centroid[0], centroid[1], f'P{pop}', 
                       ha='center', va='center', fontweight='bold')
    
    axes[0, 1].set_xlabel('KLFDAPC 1')
    axes[0, 1].set_ylabel('KLFDAPC 2')
    axes[0, 1].set_title('Population Centroids')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution plots
    for i, pop in enumerate(unique_pops):
        mask = y == pop
        axes[1, 0].hist(X_transformed[mask, 0], alpha=0.5, 
                       label=f'Pop {pop}', color=colors[i], bins=20)
    
    axes[1, 0].set_xlabel('KLFDAPC 1')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('KLFDAPC 1 Distribution by Population')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for i, pop in enumerate(unique_pops):
        mask = y == pop
        axes[1, 1].hist(X_transformed[mask, 1], alpha=0.5, 
                       label=f'Pop {pop}', color=colors[i], bins=20)
    
    axes[1, 1].set_xlabel('KLFDAPC 2')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('KLFDAPC 2 Distribution by Population')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
