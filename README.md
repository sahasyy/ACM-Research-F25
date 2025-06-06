# Early Detection of Network Attacks Using Deep Learning (Paper Implementation)

## 📌 Project Summary

This repository contains an implementation of the model proposed in the paper:  
**"Early Detection of Network Attacks Using Deep Learning"**  
📄 [arXiv:2201.11628](https://arxiv.org/pdf/2201.11628)

The paper proposes a novel deep learning-based Intrusion Detection System (IDS) capable of **detecting attacks early**, before they cause significant system damage. The system analyzes sequences of network packets using a compact and efficient neural network architecture and introduces a novel evaluation metric called **earliness** to assess how quickly an attack is detected.

---

## 🧠 Model Architecture

The paper introduces a lightweight **1D Convolutional Neural Network (CNN)** architecture tailored for analyzing raw packet-based network traffic.
For the implementation, I tried an LSTM
---

## 📊 Dataset

### 🗃️ CICIDS2017

- Provides realistic benign and malicious network traffic flows.
- Stratified sampling is used to preserve class distribution in train/test splits.
- The data is highly imbalanced (e.g., ~1,292x more normal flows than SQL injection).

#### 🧪 Train/Test Split (70:30) in the paper:

| Class         | Train Samples | Test Samples |
|---------------|---------------|--------------|
| Normal        | 18,990        | 8,139        |
| Brute Force   | 1,055         | 452          |
| XSS           | 456           | 196          |
| SQL Injection | 15            | 6            |

---

## 🧪 Evaluation Metrics

- **Accuracy**, **Precision**, **Recall**, **F1-score**
- **Balanced Accuracy** (primary metric used in the paper)
- **Earliness**: A custom metric measuring how *soon* an attack is detected during its progression (novel contribution)

> A perfect IDS would achieve Recall = 1.0 at FPR = 0.0 — identifying all attacks without false alarms — though this remains impractical in real-world conditions.

---

**install:** scapy, dataset (removed bc it was too large to push)

**preliminary results** 
![resutls](pics/Screenshot 2025-06-06 at 4.36.02 PM.png)
![results](pics/Screenshot 2025-06-06 at 4.03.51 PM.png)

