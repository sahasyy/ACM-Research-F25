# 🧬 Peptide Property Prediction Using Transformer Models

## 📘 Synopsis

This project is inspired by the research paper **“PeptideBERT: A Language Model Based on Transformers for Peptide Property Prediction”** published in *The Journal of Physical Chemistry Letters* (2023). The study presents a transformer-based language model, PeptideBERT, specifically tailored to peptide sequences. The model is trained to predict important peptide properties such as hemolysis and solubility, leveraging the power of natural language processing (NLP) techniques.

## 🧠 Key Contributions of the Paper

- **Transformer-Based Framework**: Introduced a BERT-like transformer model fine-tuned for peptide sequence modeling.
- **Multi-Property Prediction**: Demonstrated predictive performance across multiple peptide properties including hemolytic activity and solubility.
- **Domain-Specific Training**: Adapted NLP methodologies to the biochemical domain by treating amino acids as language tokens.
- **Benchmark Performance**: Achieved state-of-the-art or competitive results compared to existing peptide property predictors.

## 🔗 Alignment with This Implementation

- **Input Representation**: Peptide sequences are treated as “sentences” of amino acid tokens.
- **Transformer Architecture**: Built upon the BERT architecture, originally used for natural language understanding.
- **Pretraining and Fine-Tuning**: Utilized pretraining on large unlabeled peptide sequences, followed by supervised fine-tuning on labeled property datasets.
- **Generalization**: Demonstrated ability to generalize across peptide property tasks using a single unified model.

## ⚙️ Implementation Details

### 1. Dataset Preparation

- **Pretraining Dataset**: A large corpus of unlabeled peptide sequences to learn general sequence embeddings.
- **Fine-Tuning Datasets**:
  - **Hemolysis**: Peptides labeled as hemolytic or non-hemolytic.
  - **Solubility**: Peptides labeled as soluble or insoluble in water.

### 2. Preprocessing

- **Tokenization**: Amino acids are tokenized as single-letter codes (e.g., A, R, N, D).
- **Encoding**: Tokens are embedded using trainable embeddings during model training.
- **Input Formatting**: Sequences are padded or truncated to a fixed length for batch processing.

### 3. Model Architecture

- **Embedding Layer**: Converts amino acid tokens into dense vector representations.
- **Transformer Encoder**: Multiple self-attention layers to capture contextual relationships between amino acids.
- **Classification Heads**: Task-specific dense layers added on top of the transformer for hemolysis and solubility classification.

### 4. Training

- **Pretraining Objective**: Masked Language Modeling (MLM), where some amino acids are masked and predicted.
- **Fine-Tuning Objective**: Binary classification (e.g., hemolytic vs. non-hemolytic).
- **Optimization**: Adam optimizer with learning rate scheduling; loss based on cross-entropy.

### 5. Evaluation and Testing

- **Metrics Used**: Accuracy, precision, recall, F1-score across held-out test sets.
- **Comparative Benchmarks**: Compared with conventional machine learning models and existing peptide property predictors.

### 6. Visualization

- **Attention Maps**: Showed how specific amino acids influence model predictions.
- **t-SNE Plots**: Visualized the learned embeddings to demonstrate property-based clustering.
- **ROC Curves**: Illustrated performance across various classification thresholds.

## ✅ Results

### Performance

| Task        | Accuracy | F1-Score | AUC   |
|-------------|----------|----------|-------|
| Hemolysis   | High (≥90%) | High     | High  |
| Solubility  | Moderate–High | Moderate–High | Moderate–High |

PeptideBERT consistently outperformed traditional models on both property prediction tasks.

### Example Prediction

- **Input Sequence**: `GIGAVLKVLTTGLPALISWIKRKRQQ`
- **Predicted Properties**:
  - **Hemolytic**: Yes
  - **Soluble**: No

## 📊 Visualizations

- **Attention Heatmaps**: Display amino acid contributions to hemolysis prediction.
- **t-SNE Embedding Plot**: Clusters peptides based on predicted properties.
- **ROC Curve**: Confirms strong predictive capability across thresholds.

## 📄 Citation

> Chen, Y.; Zhao, C.; Zhu, Q.; Yan, Y. PeptideBERT: A Language Model Based on Transformers for Peptide Property Prediction. *J. Phys. Chem. Lett.* **2023**, *14* (43), 10020–10027. https://doi.org/10.1021/acs.jpclett.3c02398

## 🚀 Future Work

- Extend model to other peptide properties like antimicrobial or anticancer activity.
- Fine-tune on datasets with imbalanced or limited labeled data.
- Explore zero-shot property prediction by leveraging transfer learning.

---

