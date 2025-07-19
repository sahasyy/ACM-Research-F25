![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

## Papers Read

1. Sequence Transduction with Recurrent Neural Networks
2. Hierarchical Multiscale Recurrent Neural Networks
3. Fundamentals of Recurrent Neural Network (RNN)
and Long Short-Term Memory (LSTM) Network

## Paper 2 Chosen

**"Hierarchical Multiscale Recurrent Neural Networks"**

## Summary of Paper

This paper presents Hierarchical Multiscale Recurrent Neural Networks (HM-RNNs), a novel RNN architecture capable of learning both hierarchical and temporal representations by dynamically discovering latent structures in sequential data. Unlike traditional RNNs or fixed-timescale multiscale models, HM-RNNs use boundary detectors and operations (COPY, UPDATE, FLUSH) to adaptively adjust update frequencies across layers. Experiments on character-level language modeling and handwriting sequence generation demonstrate state-of-the-art or competitive results and reveal interpretable hierarchical structures learned without explicit supervision.

## Justification for the Approach

 The authors argue that temporal sequences such as language and handwriting often possess latent hierarchical boundaries (e.g., words, phrases, strokes), which traditional RNNs are ill-equipped to capture. Fixed-timescale multiscale models impose rigid boundaries, failing to adapt to variable-length segments. By introducing learnable boundary detectors and adaptive operations, HM-RNNs better model long-term dependencies, improve computational efficiency, and yield internal representations that align with natural data hierarchies—all without requiring labeled boundaries.

## Evaluation of Strengths and Weaknesses

**Strengths:**
- Learns hierarchical structure without explicit boundary labels.
- Introduces interpretable operations (COPY, UPDATE, FLUSH) and gating for temporal abstraction.
- Demonstrates strong empirical performance on both discrete and real-valued sequence tasks.

**Weaknesses:**
- Relies on biased gradient estimators (e.g., straight-through estimator) for discrete variable training.
- Sensitive to training heuristics like slope annealing, which require careful tuning.
- Interpretation of higher-level boundaries can be ambiguous or task-specific.

## Some Novelties Noticed in the Paper

- **Dynamic boundary detection:**  Learns when to pass summarized representations to upper layers via binary boundary units.
- **Hierarchical update scheme:** Uses three distinct operations (COPY, UPDATE, FLUSH) to efficiently manage memory and computation.
- **Slope annealing trick:** Gradually increases sigmoid slope to better approximate discrete boundary functions during training.
