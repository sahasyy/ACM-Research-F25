# LLaDA Supervised Fine-tuning for Code Completion
# This script demonstrates how to fine-tune the GSAI-ML/LLaDA-8B-Instruct model
# on a code completion dataset using the principles of diffusion models.
# VERSION 3: Implemented gradient accumulation to handle CUDA Out of Memory errors.

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
import re

# --- 1. CONFIGURATION & SETUP ---
# ---------------------------------

# Model and Tokenizer Configuration
MODEL_NAME = 'GSAI-ML/LLaDA-8B-Instruct'
DATASET_NAME = 'bigcode/the-stack-smol'
DATASET_CONFIG = 'data/python' # Using the Python subset

# LLaDA Specific Tokens and IDs (as per the documentation)
MASK_TOKEN_STR = "[MASK]" 
MASK_ID = 126336
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

# SFT Formatting (emulating the user/assistant dialogue structure)
PROMPT_TEMPLATE = "<BOS><start_id>user<end_id>\n{code_prefix}<eot_id><start_id>assistant<end_id>\n"
# The {code_completion} will be appended after this template.

# Training Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 5e-5
# Reduce batch size to 1 to minimize memory per forward/backward pass
BATCH_SIZE = 1 
# Accumulate gradients over 4 steps to simulate a larger batch size (1 * 4 = 4)
GRADIENT_ACCUMULATION_STEPS = 4 
MAX_SEQ_LENGTH = 1024 
TRAIN_STEPS = 200 

print(f"Using device: {DEVICE}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print("--- Configuration Loaded ---")

# --- 2. LOAD MODEL & TOKENIZER ---
# ---------------------------------

print("Loading model and tokenizer...")
# trust_remote_code is necessary for LLaDA
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 # Use bfloat16 for memory efficiency
).to(DEVICE)

# Set padding token if it's not already set. 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print("--- Model and Tokenizer Loaded ---")

# --- 3. DATASET PREPARATION ---
# ------------------------------

def create_completion_split(example):
    """
    Splits a code document into a prefix (prompt) and a suffix (completion).
    """
    content = example['content']
    if not content or len(content) < 50:
        return {'prefix': None, 'suffix': None}
        
    split_point = len(content) // 2
    split_point = content.find('\n', split_point) + 1
    if split_point <= 0:
        split_point = len(content) // 2

    return {
        'prefix': content[:split_point],
        'suffix': content[split_point:]
    }

def preprocess_function(examples):
    """
    The core function to format the dataset into LLaDA's expected SFT structure.
    """
    prompts = []
    full_texts = []
    
    for prefix, suffix in zip(examples['prefix'], examples['suffix']):
        if prefix and suffix:
            prompt_text = PROMPT_TEMPLATE.format(code_prefix=prefix)
            prompts.append(prompt_text)
            full_texts.append(prompt_text + suffix + tokenizer.eos_token)

    model_inputs = tokenizer(
        full_texts,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    prompt_token_ids = tokenizer(prompts, add_special_tokens=False)['input_ids']
    model_inputs['prompt_lengths'] = [len(p) for p in prompt_token_ids]
    
    return model_inputs


print("Loading and preprocessing dataset...")
raw_dataset = load_dataset(DATASET_NAME, data_dir=DATASET_CONFIG, split='train', streaming=True).take(5000)
dataset_splits = raw_dataset.map(create_completion_split, remove_columns=list(next(iter(raw_dataset)).keys()))
tokenized_dataset = dataset_splits.map(
    preprocess_function, 
    batched=True,
    batch_size=10,
    remove_columns=['prefix', 'suffix']
)
dataloader = DataLoader(tokenized_dataset.with_format("torch"), batch_size=BATCH_SIZE)

print("--- Dataset Ready ---")

# --- 4. LLaDA FORWARD PROCESS & LOSS ---
# ----------------------------------------

def forward_process(input_ids, eps=1e-3):
    """
    Applies random masking to the input sequences.
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, MASK_ID, input_ids)
    return noisy_batch, masked_indices, p_mask


# --- 5. TRAINING LOOP ---
# ------------------------

print("Starting fine-tuning...")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
# The number of training steps is now the number of optimizer steps, not batches
num_optimizer_steps = TRAIN_STEPS // GRADIENT_ACCUMULATION_STEPS
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_optimizer_steps
)

model.train()
progress_bar = tqdm(range(num_optimizer_steps))
step_count = 0

# The outer loop now corresponds to total batches processed
for i, batch in enumerate(dataloader):
    if step_count >= num_optimizer_steps:
        break

    input_ids = batch['input_ids'].to(DEVICE)
    prompt_lengths = batch['prompt_lengths'].to(DEVICE)
    attention_mask = batch['attention_mask'].to(DEVICE)

    # --- Core SFT Logic ---
    noisy_batch, masked_indices, p_mask = forward_process(input_ids)
    
    token_positions = torch.arange(noisy_batch.shape[1], device=DEVICE).expand_as(noisy_batch)
    prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
    
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    masked_indices[prompt_mask] = False

    outputs = model(input_ids=noisy_batch, attention_mask=attention_mask)
    logits = outputs.logits

    if masked_indices.sum() == 0:
        continue

    response_logits = logits[masked_indices]
    response_labels = input_ids[masked_indices]
    
    loss = F.cross_entropy(response_logits, response_labels, reduction='none')
    loss = loss / p_mask[masked_indices]
    
    # We normalize by BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS in total.
    # So here, we divide by GRADIENT_ACCUMULATION_STEPS.
    final_loss = loss.sum() / (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    
    # Backpropagate the scaled loss
    final_loss.backward()
    
    # --- Gradient Accumulation Step ---
    # Check if we have accumulated enough gradients
    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        optimizer.step()    # Update weights
        lr_scheduler.step() # Update learning rate
        optimizer.zero_grad() # Clear gradients for the next accumulation cycle
        
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{final_loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"}) # Show avg loss
        step_count += 1

print("--- Fine-tuning Complete ---")

# --- 6. SAVING THE MODEL (Optional) ---
# --------------------------------------
output_dir = "./llada-8b-code-completion-finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
