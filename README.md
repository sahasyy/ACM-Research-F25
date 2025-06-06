
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
