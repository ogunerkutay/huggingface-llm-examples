# FILE: hf_transformers_causallm_gemma2b_cpu.py
"""
This script runs the 'gemma-2b' model on a CPU.
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login  # Import login function from Hugging Face Hub
from huggingface_hub import scan_cache_dir # Import function to scan the cache directory

# Log in to Hugging Face Hub using environment variable
# The model 'google/gemma-2b' is in a gated repository, which means access is restricted.
# You need to be authenticated to access it. Ensure you have set the HUGGINGFACE_HUB_TOKEN environment variable with your API token.
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    print("HUGGINGFACE_HUB_TOKEN environment variable is not set.")

# Start the stopwatch
start_time = time.time()

# Set a manual seed for reproducibility
torch.manual_seed(100)

model_name = 'google/gemma-2b'
print(f"Model name: {model_name}")

device = torch.device("cpu")
print(f"Device: {device}")

# Load model directly
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.eval().to(device)  # Set model to evaluation mode and move it to the appropriate device GPU or CPU

# Calculate the number of parameters
param_size = sum(p.numel() for p in model.parameters())

# Get cache information
cache_info = scan_cache_dir()
model_cache_info = next((item for item in cache_info.repos if model_name in item.repo_id), None)

file_size = model_cache_info.size_on_disk if model_cache_info else 0

print(f"Parameter size: {param_size}")
print(f"File size: {file_size}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  

with torch.inference_mode(): # Set the model to inference mode, better than torch.no_grad() for inference
    response_start_time = time.time()
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_time = time.time() - response_start_time

print("Generated Response:", response)

elapsed_time = time.time() - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")