import os
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR = "./llama2-noisy"
NOISE_STDDEV = 0.01  # standard deviation of Gaussian noise
CACHE_DIR = os.getenv("HF_HOME")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16
)

# Add noise to parameters
for name, param in tqdm(model.named_parameters()):
    if param.requires_grad:
        noise = torch.randn_like(param) * NOISE_STDDEV
        param.data += noise

# Save model with modified weights
model.save_pretrained(OUTPUT_DIR)

print(f"Noisy model saved to {OUTPUT_DIR}")

