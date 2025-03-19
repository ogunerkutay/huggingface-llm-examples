# FILE: hf_transformers_pipeline_gemma2b.py
"""
This script uses the Hugging Face Transformers library to load and run the 'gemma-2b' model.
"""

# Import necessary libraries
import os  # Library for interacting with the operating system
import time  # Library for time-related functions
import torch  # Import torch to check for CUDA support
from transformers import pipeline # Import pipeline from Hugging Face
from huggingface_hub import scan_cache_dir # Import function to scan the cache directory
from huggingface_hub import login # Import function to login to the Hugging Face Hub

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    print("HUGGINGFACE_HUB_TOKEN environment variable is not set.")

# Start the stopwatch
start_time = time.time()

# Set a manual seed for reproducibility
torch.manual_seed(100)

# Define the model name
model_name = 'google/gemma-2b'
print(f"Model name: {model_name}")

# Check if CUDA is available and choose the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Use a pipeline as a high-level helper
pipe = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto",  # Automatically distribute model layers across devices CPU and CUDA
    )

# Calculate the number of parameters
param_size = sum(p.numel() for p in pipe.model.parameters())

# Get cache information
cache_info = scan_cache_dir()
model_cache_info = next((item for item in cache_info.repos if model_name in item.repo_id), None)

# Calculate the file size of the model
if model_cache_info:
    file_size = model_cache_info.size_on_disk
else:
    file_size = 0

# Print the parameter size and file size
print(f"Parameter size: {param_size}")
print(f"File size: {file_size}")

# Prompt for the model to process
prompt = "What is the capital of France?"
with torch.inference_mode(): # Set the model to inference mode, better than torch.no_grad() for inference
    # Measure response time
    response_start_time = time.time()
    # Generate response using the pipeline
    response = pipe(prompt, max_length=50, truncation=True)
    response_time = time.time() - response_start_time

# Print the generated response
print("Generated Response:", response)

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")