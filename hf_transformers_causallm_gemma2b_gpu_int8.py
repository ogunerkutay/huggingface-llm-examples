# FILE: hf_transformers_causallm_gemma2b_gpu_int8.py
"""
This script runs the 'gemma-2b' model on a GPU using 8-bit precision (int8).
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

# Configure BitsAndBytes quantization for 8-bit precision (LLM.int8)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,                     # Enable 8-bit quantization to reduce memory usage while keeping good precision
    llm_int8_has_fp16_weight=True,         # Keep some weights in FP16 for higher accuracy
    llm_int8_skip_modules=["lm_head"],     # Skip quantization for output layers (if needed, useful for model accuracy)
    llm_int8_threshold=6.0,                # Threshold for mixed precision quantization (default is 6.0)
    llm_int8_enable_fp32_cpu_offload=True  # Enable FP32 offloading to CPU for better memory efficiency
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute model layers across devices CPU and CUDA
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16 # Use bfloat16 for faster computation
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
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

response_start_time = time.time()
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_time = time.time() - response_start_time

print("Generated Response:", response)

elapsed_time = time.time() - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")