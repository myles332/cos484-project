import os
import torch
import numpy as np
from random import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
NOISE_STDDEV = 0.001  # standard deviation of Gaussian noise
CACHE_DIR = os.getenv("HF_HOME")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16
)

for i in np.arange(0.1, 1.0, 0.1):
    OUTPUT_DIR = f"./llama2-noisy-{i}"
    # Add noise to parameters
    with torch.no_grad():
        for param in model.parameters():
            if random() < i:
                param.add_(torch.randn_like(param) * NOISE_STDDEV)

    # Save model with modified weights
    model.save_pretrained(OUTPUT_DIR)

    print(f"Noisy model saved to {OUTPUT_DIR}")

