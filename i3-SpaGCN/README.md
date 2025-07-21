# SpaGCN: Integrating gene expression, spatial location and histology to identify spatial domains and spatially variable genes by graph convolutional network


## Motivation

Understanding spatial domains and gene expression patterns in tissue is critical for studying disease microenvironments, especially in cancer and immunology. Traditional clustering methods using only gene expression (e.g. k-means, Louvain) fail to integrate spatial coordinates or histology context, leading to biologically incoherent regions.
SpaGCN (Graph Convolutional Networks for Spatial Gene Expression) addresses this by combining gene expression, spatial location, and histological features into a unified graph-based model for detecting spatial domains and spatially variable genes (SVGs). It works well across multiple spatial transcriptomics datasets such as 10x Visium and Slide-seq.

## Novelty

- Integrate histology, gene expression, and spatial coordinates 
- Custom graph topology based on spot adjacency and spatial distance
- Capable of unsupervised spatial domain detection with domain-specific gene outputs
- Considers not only transcript count, but also tissue coordinates and image-derived morphology, in clustering and SVG calling

## Methods

- Graph constructed using radius-based neighbor connections among spatial transcriptomic spots
- Each node is initialized with gene expression features and optionally low-level RGB histology features
- Graph Convolutional Network (GCN) propagates information across nearby nodes
- Final soft clustering yields spatial domains
- Performs differential expression analysis per domain to identify SVGs

### Model Evaluation:
- Quantitative metrics: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
- Visual evaluation: Overlays of predicted spatial domains on tissue image
- SVGs evaluated using p-values and biological coherence
- Benchmarked against stLearn, BayesSpace, Louvain, KMeans

## Advantages/Disadvantages 

### Advantages
- Interpretable spatial domains and gene enrichment patterns
- Applicable across various spatial platforms (10x Visium, Slide-seq, MERFISH)
- Human-interpretable tissue maps for downstream analysis

### Limitations
- Histology features are weak (uses raw RGB averages)
- Sensitive to input quality: performance drops on noisy or low-depth datasets
- Not adaptable to tissue heterogeneity or complex structures

