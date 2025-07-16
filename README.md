![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Summer 2025 Paper Implementations

# Paper 1: SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression

**Overview and Motivation.** Leading compression techniques for LLMs are based on SVD, or Singular Value Decomposition, and avoid constraints like hardware dependency and the need for retraining. However, SVD-based techniques suffer from performance drop-offs when compression must reduce a large number of parameters. ASVD fails to directly relate singular values to the model compression loss, so truncating smaller singular values may cause great compression loss. Additionally, SVD-based compression techniques do not update remaining parameters in the compressed model to compensate for the large number of truncated parameters.

SVD-LLM is a post-training compression method that uses Truncation-Aware Data Whitening to create a direct relationship between singular values and compression loss to identify the optimal singular values to truncate to minimize loss. SVD-LLM also implements a Layer-Wise Closed-Form Model Parameter Update strategy to compressively update compressed weights one layer at a time, compensating for accuracy drop-off for models with high compression ratios.

**Novelty.** Truncation-Awareness is proposed to reduce loss, as a standard SVD compression involves truncating the smallest singular value but does not relate the singular values to compression loss. Truncation-Aware Data Whitening forces the whitening activation S-1X to be an orthogonal matrix where each channel is independent. Then, SVD is applied to WS, where W is the weight matrix of the original LLM, and the smallest singular values are truncated. Using this strategy, truncating the smallest singular value leads to the smallest loss, since the loss when one singular value is truncated equals the singular value itself. When two singular values are truncated, the loss equals the square root of the sum of their squares.

In ASVD (Activation-aware SVD), an optimization algorithm is applied to minimize ||WX - W’X||F, where W is the original weight matrix of, X is the original activation matrix, and W’ is the weight matrix of the compressed LLM. Compressing the LLM creates a new activation matrix, X’, which differs more from X as more singular values are truncated. Layer-Wise Closed-Form Update is proposed to update W’ to minimize || WX - W’X’ ||F. To optimize each layer i, the previously updated layer i-1 is used to create the activation X’i-1. To preserve the low-rank structure of W’i, matrix Ui is updated to its closed-form solution U’i, while Trunc.(Σ)i and Vi are preserved. U’i minimizes ||WiX’i-1 - W’iX’i-1||F.

**Combination with Other Compression Methods.** SVD-LLM improves the performance of quantization and parameter pruning methods. Using the GPTQ quantization method, GPTQ-4bit with SVD-LLM has an 18% lower perplexity, or measure of how a model predicts samples, than GPTQ-3bit alone and also has a smaller memory footprint (2.1 GB compared to 2.8 GB). Additionally, combining SVD-LLM with LLM-Pruner and a 30% compression ratio to compress LLaMA-7B achieves 13% lower perplexity than LLM-Prumer with a 40% compression ratio. Additionally, SVD-LLM increases token generation speed for all compression ratios.

**Results Summary.** SVD-LLM outperforms normal SVD, FWSVD, and ASVD for all compression ratios, with greater performance increases shown for higher compression ratios. SVD-LLM is also more stable across various LLM’s, avoids out of memory errors for large models, and achieves a lower perplexity for all compression ratios when combined with LoRA fine-tuning compared to ASVD + LoRA.

# Paper 2: Dynamic Compressing Prompts for Efficient Inference of Large Language Models

## Overview.
Dynamic Compressing Prompts (LLM-DCP) is a task-agnostic technique that preserves the meaning of prompts by using a Markov Decision Process (MDP) for compression. Redundancy is reduced iteratively, and tokens are sequentially removed. Hierarchical Prompt Compression (HPC) is proposed to slowly increase the compression difficulty while training a DCP-Agent.

## Motivation.
### White-box Prompt Compression
* Works at the token embedding level, changing a model’s parameters, structure, and transformer self-attention mechanism

### Black-Box Prompt Compression
#### Strengths
* Circumvents the need for source code access by working at the natural language level
#### Weaknesses
* Usually fine-tined for a specific task (like solving math problems), which makes it hard to use the same model for other tasks
* Overlook the sequential nature of prompts
* Rely on black-box LLMs for training, making them costly and impractical

## Novelty.
LLM-DPC uses a reward function to train the DCP-Agent which doesn’t need access to the LLM source code or a black-box LLM. HPC gradually increases compression difficulty to balance efficiency with information preservation. LLM-DPC follows a Markov Decision Process to mitigate losses in LLM performance.

## Advantages/Disadvantages.
### Advantages
* No need for a black-box LLM or LLM source code to train the DCP-Agent
* Uses a pre-trained small language model (SLM) since training is at token-level
* Replay buffer used to store trajectories and support multiple updates per episode.

### Disadvantages
* Reward system is prone to fluctuation during training
* Semantic compression not captured
* Compression policy is dataset/prompt-dependent (may not transfer well to other datasets)

## Implementation.
Implement Algorithm 1 from the LLM-DCP paper.
![Algorithm 1](llm-dcp-algorithm.png)

### Key Ideas
* Compression difficulty increases over time
* Critic aids the agent in learning by calculating advantage
* Weighted advantage by weight to improve training quality

# Paper 3: LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression
## Overview.
LLMLingua-2 is a task-agnostic prompt compression method that uses a transformer architecture to predict the likelihood each word in the original prompt will be preserved in the compressed one, then chooses the words with the highest probability. LLMLingua-2 outperforms existing compression methods like Selective-Context and LLMLingua, both with in-of-domain and out-of-domain prompts and across various target LLM’s, by utilizing bidirectional context when selecting tokens to preserve.

## Motivation.
Casual LLM’s are unidirectional, and only capture partial context. LLMLingua-2 leverages bidirectional context for more effective prompt compression.
### Abstractive Prompt Compression
* Rephrase original prompts into compressed ones using an autoregressive process
* Slow and prone to hallucinations
### Extractive Prompt Compression
* Geared towards summarization and not as detailed as abstractive prompt compression

LLMLingua-2 aims to create an extractive prompt compression technique that keeps core information, avoiding slow-downs and hallucinations seen in abstractive methods.

## Novelty.
LLMLingua-2 treats compression like a binary classification task, marking individual tokens as preserve or discard. The compression metric is the predicted probability of each token being preserved. A transformer encoder is used for feature extraction, allowing LLMLingua-2 to take advantage of bidirectional context. The extractive approach ensures an accurate representation of the prompt, compared to an abstractive one. LLMLingua-2 is 3-6x as fast as existing methods and can improve overall latency by 1.6-2.9x using compression ratios 2-5x as large.

## Advantages/Disadvantages.
### Advantages
* Take agnosticism allows for general use of LLMLingua-2
* Outperforms Selective-Context, LLMLingua, and LLMLingua-2-small for QA (question and answer) and summarization tasks using the LLM Mistral-7B, coming closer to matching the original prompt’s performance. LLMLingua-2 also has 1.6-2.9x lower latency and reduced GPU memory costs by up to a factor of 8
* Maintains the most informative words as compression ratio increases due to bidirectional context-aware feature extraction

### Disadvantages
* LLMLingua-2 still falls short of task-aware methods, such as LongLLMlingua
* Compression dataset was derived from MeetingBank, which consists of meeting transcripts, and may limit generalizability
* For out-of-domain prompts, LongBench and ZeroSCROLLS, although LLMLingua-2 does well (having the highest average exact match ratio for ZeroSCROLLS and only getting beaten by LLMLingua’s average on LongBench), there is still a steep loss in exact match results compared to the in-domain task (for instance, LLMLingua-2 has 76.22% and 30.18% exact match over MeetingBank for QA and summarization respectively, but only 25-26% for summarization over LongBench while QA wasn’t compared).

Ultimately, LLMLingua-2 outperforms other task-agnostic tools, even with datasets it wasn’t trained on, and even after expanding the training dataset, the authors observed limited performance improvement. This suggests that patterns of language are similar across datasets, and that LLMLingua-2 can learn these patterns in its training domain then transfer them, making the case for task-agnostic compression techniques. Additionally, its lower BERTScores for summarization tasks highlights the challenge of task-agnostic analysis and summarization, compared to QA, where we can observe higher exact match ratio.

## Methodology.
### Dataset
GPT-4 is used to compress prompts, focusing on token reduction (fewer tokens than the original), informativeness (retaining essential information), and faithfulness (avoid hallucinations). GPT-4 is prompted to compress the text as short as possible and keep as much information as possible. To ensure effectiveness for various prompt lengths and styles, no compression ratio is specified.

GPT-4 compresses long prompts at a much higher compression ratio, so the authors split the longer prompt into chunks no longer than 512 tokens and terminating with a period, which were then individually compressed.

Note that LLMLingua-2 uses exact match from HuggingFace as an evaluation metric for QA, which “Returns the rate at which the input predicted strings exactly match their references, ignoring any strings input as part of the regexes_to_ignore list” (HuggingFace, https://huggingface.co/spaces/evaluate-metric/exact_match). A higher rate is better. For the summarization task, BERTScore from HuggingFace is used, which “computes a similarity score for each token in the candidate sentence with each token in the reference sentence” (HuggingFace, https://huggingface.co/spaces/evaluate-metric/bertscore). A higher score is better.

### Quality Control
**Variation Rate** measures the proportion of words in the compressed prompt that aren’t in the original. More variation corresponds with a higher likelihood of hallucinations, so the prompts with the highest 5% of variation rates are excluded from the dataset.

**Alignment Gap** measures the quality of the automatically annotated labels. It’s the difference between the hitting rate and the matching rate, where the hitting rate is the proportion of words in the compressed prompt that appear in the original, and the matching rate is the proportion of words in the original prompt that correspond to a word in the compressed.

### Compression
To compress a prompt, LLMLingua-2 finds the target number of tokens for the compressed prompt, computed by N’ = TN, where N is the number of words in the original prompt, and T is the quotient of words in the compressed prompt and words in the original. The transformer classification model then predicts the probability of each word, xi, to be preserved. Lastly, the top N’ words with the highest probability of being preserved are chosen, maintaining their original order.



# Downloading Depencies
Run pip install -r requirements.txt to install necessary libraries.
