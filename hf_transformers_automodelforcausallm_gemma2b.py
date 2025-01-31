# FILE: hf_transformers_pipeline_gemma2b.py
"""
This script uses the Hugging Face Transformers library to load and run the 'gemma-2b' model.
"""

# Import necessary libraries
import os # Library for interacting with the operating system
import time  # Library for time-related functions
import torch  # Library for tensor computations and GPU support
from transformers import AutoTokenizer, AutoModelForCausalLM # Import AutoTokenizer and AutoModelForCausalLM classes
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

# Define the model name
model_name = 'google/gemma-2b'
print(f"Model name: {model_name}")

# Check if CUDA is available and choose the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model directly
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.eval().to(device)  # Set model to evaluation mode and move it to the appropriate device GPU or CPU

# Calculate the number of parameters
param_size = sum(p.numel() for p in model.parameters())

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

# Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt for the model to process
prompt = "What is the capital of France?"

# Tokenize the prompt to convert it into a format the model can understand (input token ids)
inputs = tokenizer(prompt, return_tensors="pt")  # 'pt' stands for PyTorch tensors

# Move inputs to the same device as the model (GPU if available)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Measure response time
response_start_time = time.time()
# Generate text using the model
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_time = time.time() - response_start_time

# Print the generated response
print("Generated Response:", response)

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")