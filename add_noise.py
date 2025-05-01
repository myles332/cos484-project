import os
import torch
from random import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR = "./llama2-noisy"
NOISE_STDDEV = 0.0001  # standard deviation of Gaussian noise
CACHE_DIR = os.getenv("HF_HOME")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16
)

# Add noise to parameters
with torch.no_grad():
    for param in model.parameters():
        if random() < 0.1:
            param.add_(torch.randn_like(param) * NOISE_STDDEV)

# Save model with modified weights
model.save_pretrained(OUTPUT_DIR)

print(f"Noisy model saved to {OUTPUT_DIR}")

