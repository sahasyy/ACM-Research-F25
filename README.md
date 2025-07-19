![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

## Papers Read

1. A Structured Self-Attentive Sentence Embedding
2. High-performance brain-to-text communication via handwriting

## Paper 1 Chosen

**"A Structured Self-Attentive Sentence Embedding"**

## Summary of Paper

This paper proposes a novel structured self-attentive sentence embedding model that replaces traditional vector-based embeddings with a matrix-based representation, where each row captures different aspects of sentence semantics. The method uses a self-attention mechanism on top of a bidirectional LSTM to compute multiple weighted sums of hidden states, forming a 2D matrix embedding. A penalization term is introduced to encourage diversity among attention vectors. The model is interpretable, as attention weights indicate which parts of the sentence contribute to each row. Empirical evaluations on author profiling, sentiment analysis, and textual entailment demonstrate significant improvements over standard models.

## Justification for the Approach

The authors argue that traditional pooling or final hidden state representations compress sentence semantics too aggressively and lack interpretability. By replacing a single vector with a matrix of multiple attention-based summaries, the model can better capture multiple semantic components in a sentence. The self-attention mechanism allows the model to focus directly on relevant parts of the input without needing auxiliary input sources. Furthermore, the penalization term addresses redundancy, improving the diversity and expressiveness of attention vectors. This structured embedding approach thus supports better downstream task performance and interpretability.

## Evaluation of Strengths and Weaknesses

**Strengths:**
- Produces interpretable sentence embeddings with visualizable attention distributions.
- Shows consistent performance improvements across diverse NLP tasks (author profiling, sentiment, entailment).
- Avoids reliance on parse trees or linguistic structures.

**Weaknesses:**
- The model cannot be trained in an unsupervised fashion, limiting pretraining flexibility.
- The matrix embedding increases parameter size, requiring pruning tricks to reduce model size.
- Training stability may require careful hyperparameter tuning (e.g., penalization weight).

## Some Novelties Noticed in the Paper

- **Matrix sentence embedding:** Instead of a single vector, the model outputs a 2D matrix where each row captures a distinct aspect of sentence meaning.
- **Penalization term:** Encourages diversity among attention vectors and sharper focus within each.
- **Model pruning for size efficiency:** Introduces a 2D-structured hidden layer that drastically reduces parameters in large FC layers.
