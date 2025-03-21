# FILE: llama_gguf_minicpmo.py
"""
This script loads and runs a LLaMA GGUF model optimized for CPU usage and measures the execution time.
Unlike other models that may require GPU support, LLaMA GGUF models are designed to work efficiently on CPU.
It uses the 'time' module to measure the total execution time and the response generation time.
The model used here is "MiniCPM-o-2_6-gguf", specifically a 7.6B parameter model.
The input is processed through a simple chat-like interface, with a system and user message.
The model generates a response based on the provided input text.
The output is printed as the model's generated response along with the execution times.
For more details on LLaMA GGUF models and their usage, refer to the official llama-cpp documentation:
https://github.com/abetlen/llama-cpp-python
"""

# Import necessary libraries
import time  # Library for time-related functions
from llama_cpp import Llama # Import LLaMA library for GGUF models
from huggingface_hub import scan_cache_dir # Import function to scan the cache directory

# Start the stopwatch
start_time = time.time()

# Define the model name
model_name = 'openbmb/MiniCPM-o-2_6-gguf'
print(f"Model name: {model_name}")

device = "cpu"  # Set the device to CPU
print(f"Device: {device}")

# Load the pretrained LLaMA GGUF model
model = Llama.from_pretrained(
    repo_id= model_name,  # The repository ID of the model on Hugging Face Hub
    filename="Model-7.6B-Q5_K_M.gguf",  # The filename of the GGUF model file
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

""" messages = [
    {"role": "system", "content": "You are a helpful assistant."},  # Optional system message
    {"role": "user", "content": prompt},
] """

# Measure response time
response_start_time = time.time()
#response = model.create_chat_completion(messages=messages)
response = model(prompt, max_tokens=512, echo=False)
response_time = time.time() - response_start_time

# Print the generated response
#print("Generated Response:", response['choices'][0]['message']['content'])
print("Generated Response:", response['choices'][0]['text'].split('\n')[0].strip())
print(f"Response generation time: {response_time:.2f} seconds")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")