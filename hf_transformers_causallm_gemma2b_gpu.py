# FILE: hf_transformers_causallm_gemma2b_gpu.py
"""
This script runs the 'gemma-2b' model on a single/multi GPU.
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, scan_cache_dir

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    print("HUGGINGFACE_HUB_TOKEN environment variable is not set.")

start_time = time.time()
torch.manual_seed(100)

model_name = 'google/gemma-2b'
print(f"Model name: {model_name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute model layers across devices CPU and CUDA
    )

model = model.eval()  # Set the model to evaluation mode for inference

param_size = sum(p.numel() for p in model.parameters())

cache_info = scan_cache_dir()
model_cache_info = next((item for item in cache_info.repos if model_name in item.repo_id), None)

file_size = model_cache_info.size_on_disk if model_cache_info else 0

print(f"Parameter size: {param_size}")
print(f"File size: {file_size}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What is the capital of France?"
print(f"Prompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode(): # Set the model to inference mode, better than torch.no_grad() for inference
    # Use autocast for mixed precision during the model generation
    with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
        response_start_time = time.time()
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_time = time.time() - response_start_time

print("Generated Response:", response)
print(f"Response generation time: {response_time:.2f} seconds")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")