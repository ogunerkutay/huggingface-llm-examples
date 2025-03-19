# FILE: llama_cpp_gemma2b.py
"""
This script uses the llama_cpp library to load and run a LLaMA GGUF model named 'gemma-2b'.
The model generates text based on the provided input prompt.
"""

# Import necessary libraries
import os # Library for interacting with the operating system
import time  # Library for time-related functions
from llama_cpp import Llama  # Import LLaMA library for GGUF models
from huggingface_hub import login  # Import login function from Hugging Face Hub
from huggingface_hub import scan_cache_dir # Import function to scan the cache directory

# Log in to Hugging Face Hub using environment variable
# The model 'google/gemma-2b' is in a gated repository, which means access is restricted.
# You need to be authenticated to access it. Ensure you have set the HUGGINGFACE_HUB_TOKEN environment variable with your API token.
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

# Start the stopwatch
start_time = time.time()

# Define the model name
model_name = 'google/gemma-2b'
print(f"Model name: {model_name}")

device = "cpu"  # Set the device to CPU
print(f"Device: {device}")

# Load the pretrained LLaMA GGUF model
model = Llama.from_pretrained(
    repo_id = model_name,  # The repository ID of the model on Hugging Face Hub
    filename ="gemma-2b.gguf",  # The filename of the GGUF model file
    n_ctx=0,  # The context length for the model (0 means default context length)
    temperature=0.01,  # The temperature for sampling (lower values make the output more deterministic)
    top_p=1,  # The cumulative probability for nucleus sampling (1 means no nucleus sampling)
    verbose=False  # Whether to print verbose logs during model loading
)

# Retrieve the parameter size
param_size = 0

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

# Measure response time
response_start_time = time.time()
# Generate text based on the input prompt
response = model(prompt, max_tokens=512, echo=False)
response_time = time.time() - response_start_time

# Print the generated response
print("Generated Response:", response)
print(f"Response generation time: {response_time:.2f} seconds")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")