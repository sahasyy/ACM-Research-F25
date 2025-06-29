
# Malicious URL Detection Using Machine Learning

## Requirements
To install the necessary packages for this project, use the provided `requirements.txt` file. You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Papers Read
- Malicious URL Detection Using Machine Learning — Cho Do Xuan, Hoa Dinh Nguyen, Tisenko Victor Nikolaevich (IJACSA, 2020)
- Detecting Malicious URLs Using Lexical Analysis and Machine Learning
- An Empirical Study on Machine Learning Algorithms for Malicious URL Detection

## Paper Chosen
**"Malicious URL Detection Using Machine Learning"**

## Summary of Paper
The paper presents a method for detecting malicious URLs using machine learning models. It explores lexical features of URLs such as length, special characters, and the presence of IP addresses. Using these features in combination with character-level n-gram TF-IDF vectors, the study applies various classifiers—particularly Random Forest—to successfully distinguish malicious URLs from benign ones.

## Justification for the Approach
The paper emphasizes using lightweight, content-based features (rather than querying third-party services) to enable faster, scalable, and real-time detection. This approach is suitable for early filtering of harmful links in firewalls, browsers, and security systems. Random Forest is chosen due to its robustness, feature interpretability, and resistance to overfitting.

## Evaluation of Strengths and Weaknesses
### Strengths:
- Uses lexical features, requiring no access to WHOIS, DNS, or blacklist services.
- Fast and scalable for real-time URL detection.
- Shows strong performance using TF-IDF vectorization + Random Forest.
- Applicable to large-scale web security environments.

### Weaknesses:
- Accuracy depends on quality and diversity of training data.
- Cannot detect sophisticated phishing attempts relying on webpage content or redirection chains.
- Static features (like character counts) may be insufficient for zero-day threats.

## Some Novelties Noticed in the Paper
- Combination of handcrafted lexical features with character-level TF-IDF.
- Avoidance of slow, API-heavy detection pipelines by relying purely on URL text.
- Demonstrates that even simple models can perform competitively with appropriate preprocessing.
- Highlights practical utility for firewall or proxy-based URL inspection tools.



# Paper 2
# Sentiment Classification Using BiLSTM with 2D Max Pooling

## Requirements

To install the necessary packages for this project, run:

```bash
pip install torch==2.0.1 torchtext==0.15.2 matplotlib seaborn
```

## Papers Read

- *Text Classification Improved by Integrating Bidirectional LSTM with Two-Dimensional Max Pooling* — Zhou et al., 2016
- *A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification* — Zhang & Wallace, 2017
- *Recurrent Neural Network Architectures for Sentence Classification* — Liu et al., 2016

## Paper Chosen

**Text Classification Improved by Integrating Bidirectional LSTM with Two-Dimensional Max Pooling** — Zhou et al., 2016

## Summary of Paper

The paper proposes enhancing LSTM models for text classification by incorporating Bidirectional LSTM with 2D Max Pooling. BiLSTM captures dependencies in both directions of a text sequence, while 2D max pooling aggregates the most informative features. This produces robust sentence representations for tasks such as sentiment analysis.

## Justification for the Approach

The IMDB dataset provides a benchmark for binary sentiment classification. The BiLSTM with 2D Max Pooling architecture is suitable because:

- **Context Awareness:** Captures both past and future context.
- **Effective Feature Aggregation:** 2D Max Pooling focuses on the most significant features.
- **Simplicity:** Lightweight compared to transformer models.
- **Reproducibility:** Easy to implement with publicly available datasets.

## Strengths and Weaknesses

### Strengths
- Captures sequential context effectively.
- 2D Max Pooling simplifies feature selection.
- Computationally efficient compared to large models.
- Suitable for low-resource settings.

### Weaknesses
- May not match transformer performance.
- Hyperparameter tuning required.
- Pre-trained embeddings could improve results.

## Notable Aspects

- Introduces 2D Max Pooling for text, inspired by vision tasks.
- Simple architectural changes yield performance improvements.
- Competitive results without complex models.

## Potential Extensions

- Incorporate pre-trained embeddings (GloVe, FastText).
- Compare with transformer baselines (e.g., DistilBERT).
- Apply to multi-class classification tasks.
- Explore pooling strategy variants.

---

## Visualizations

The notebook includes:

- Training Loss vs Epochs
- Validation Accuracy vs Epochs

---

## Dataset

The [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) is used for binary sentiment classification.

