![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)


# Papers Read

Deep learning integrates gene expression and clinical data to predict type 2 diabetes onset
Single-cell transcriptomics of human pancreatic islets in health and type 2 diabetes
Machine learning-based early detection of type 2 diabetes using EHR and genomics data
Paper 1 Chosen

# "Deep learning integrates gene expression and clinical data to predict type 2 diabetes onset"

## Summary of Paper

The paper "Deep learning integrates gene expression and clinical data to predict type 2 diabetes onset" explores the use of multi-input neural networks to combine transcriptomic (gene expression) data from pancreatic β-cells and patient clinical metrics such as BMI, age, glucose levels, and insulin resistance. The model effectively predicts the likelihood of developing type 2 diabetes (T2D) and identifies key biological markers contributing to β-cell dysfunction. This integrative AI approach offers promise for early intervention and personalized treatment planning.

## Justification for the Approach

Type 2 diabetes is a complex, multifactorial disease influenced by both genetic and metabolic factors. Traditional diagnostic tools often rely solely on clinical cutoffs like A1C or fasting glucose, which may miss early warning signs. This paper’s integrative approach combines molecular-level signals (gene expression in pancreatic islets) with patient-level data to model disease onset more holistically. Deep learning models are ideal for handling the nonlinear interactions between omics and clinical features, enabling more accurate and individualized predictions.

# Evaluation of Strengths and Weaknesses

## Strengths:

Integrates multiple data types (gene expression + clinical features), improving prediction accuracy.
Highlights the role of β-cell gene expression in disease progression.
Enables personalized risk profiling for early T2D detection.
Uses interpretable layers and SHAP values to identify high-impact genes and factors.

## Weaknesses:

Requires access to high-quality gene expression data, which may not always be available.
Interpretability still limited compared to simpler models (e.g., logistic regression).
Deep models may overfit small-sample omics datasets without proper regularization.
Variability in gene expression across patients adds noise and complexity to training.

## Some Novelties Noticed in the Paper

Multi-modal input model: Jointly trains on two distinct data types, learning shared representations that improve prediction.
β-cell focus: Connects predictions to real cellular mechanisms in the pancreas.
Use of dropout and batch normalization: Ensures robust learning even with noisy omics features.
Model interpretation via SHAP: Provides feature importance at the level of individual genes and clinical variables.
