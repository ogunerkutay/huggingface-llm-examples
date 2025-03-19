# FILE: hf_transformers_pipeline_gemma3_1b_it.py
"""
This script uses the Hugging Face Transformers library to load and run the 'gemma-3-1b-it' model using pipeline.
"""

import os
import time
import torch
from transformers import pipeline
from huggingface_hub import scan_cache_dir, login

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    print("HUGGINGFACE_HUB_TOKEN environment variable is not set.")

start_time = time.time()
torch.manual_seed(100)

model_name = 'google/gemma-3-1b-it'
print(f"Model name: {model_name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

pipe = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

param_size = sum(p.numel() for p in pipe.model.parameters())

cache_info = scan_cache_dir()
model_cache_info = next((item for item in cache_info.repos if model_name in item.repo_id), None)

file_size = model_cache_info.size_on_disk if model_cache_info else 0

print(f"Parameter size: {param_size}")
print(f"File size: {file_size}")

# Example prompt
prompt = "Explain quantum computing in simple terms."
print(f"Prompt: {prompt}")

with torch.inference_mode(): # Set the model to inference mode, better than torch.no_grad() for inference
    # Use autocast for mixed precision during the model generation
    with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
        response_start_time = time.time()
        response = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95)
        response_time = time.time() - response_start_time

print("Generated Response:", response)
print(f"Response generation time: {response_time:.2f} seconds")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")