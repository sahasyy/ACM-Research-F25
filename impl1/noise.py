import torch
import torch.nn as nn
from transformers import AutoTokenizer

class LogLinearNoise(nn.Module):
  """
  Log Linear noise schedule. Total noise is -log(1 - (1 - eps) * t),
  so the sigma will be (1 - eps) * t.
  This ensures that 1 - 1/e^(n(t)) interpolates between 0 and ~1
  as t varies from 0 to 1.
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def total_noise(self, t):
    """
    Calculates the total noise (sigma) at a given timestep t.
    Args:
        t (torch.Tensor): A tensor of timesteps, where t is in [0, 1].
    Returns:
        torch.Tensor: The corresponding sigma values.
    """
    return -torch.log1p(-(1 - self.eps) * t)

def add_noise_iteratively(text: str, num_steps: int) -> list[str]:
    """
    Applies noise to a text iteratively, from a clean state to a fully noised state.

    This function simulates the forward diffusion process by incrementally increasing
    the amount of noise (masking) applied to the input text over a specified
    number of steps.

    Args:
        text: The initial, coherent text string.
        num_steps: The total number of steps in the noising process.

    Returns:
        A list of strings, where each string represents the state of the text
        at each iterative step of the noising process.
    """
    # 1. Initialize Tokenizer and Noise Schedule
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    noise_schedule = LogLinearNoise()

    # Add a mask token if it doesn't exist, as this model uses masking for noise.
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    mask_token_id = tokenizer.mask_token_id

    original_tokens = tokenizer(text, return_tensors="pt")["input_ids"]

    iterative_steps = []

    # 3. Loop through each step from 0 to num_steps
    for step in range(num_steps + 1):
        # Calculate the timestep `t` corresponding to the current step.
        # `t` scales from 0.0 to 1.0.
        t = torch.tensor([step / num_steps], dtype=torch.float32)

        # Get the noise level (sigma) from the schedule for timestep `t`.
        sigma = noise_schedule.total_noise(t)

        # Calculate the probability of a token being masked.
        # As `sigma` increases, `move_chance` approaches 1.
        move_chance = 1 - torch.exp(-sigma)

        # 4. Apply noise (masking) based on `move_chance`.
        # This logic is a direct implementation of the q_xt function from diffusion.py.
        
        # Generate random numbers for each token.
        rand_numbers = torch.rand_like(original_tokens, dtype=torch.float32)
        
        # Create a boolean mask where `True` indicates a token should be masked.
        should_mask = rand_numbers < move_chance
        
        # Replace tokens with the mask token ID where `should_mask` is True.
        noised_tokens = torch.where(should_mask, mask_token_id, original_tokens)

        # 5. Decode the noised tokens back to a string and store the result.
        decoded_text = tokenizer.decode(noised_tokens.squeeze(0))
        iterative_steps.append(f"Step {step:03d} (t={t.item():.2f}, move_chance={move_chance.item():.2f}): {decoded_text}")

    return iterative_steps

if __name__ == '__main__':
    with open("gptoutput.txt", "r", encoding="utf-8") as f:
        input_text = f.read()
    steps = 10
    
    noised_sequence = add_noise_iteratively(input_text, steps)
    
    full = []
    for line in noised_sequence:
        print(line)
        full.append(line + "\n")
    with open("noised_output.txt", "w", encoding="utf-8") as f:
        f.writelines(full)
