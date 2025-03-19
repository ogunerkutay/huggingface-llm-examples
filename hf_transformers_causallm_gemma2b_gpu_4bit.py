# FILE: hf_transformers_causallm_gemma2b_gpu_4bit.py
"""
This script runs the 'gemma-2b' model on a GPU using 4-bit precision.
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

"""
Explanation of Key Parameters:

1. **load_in_4bit=True**:
   - This enables **4-bit quantization** of the model weights. Quantization reduces the precision of the model's weights (from float32 to 4-bit), dramatically reducing the memory footprint and enabling the model to fit into GPU memory more easily.
   - **Trade-off**: There might be a slight loss in accuracy, but it improves the speed and memory efficiency.

2. **bnb_4bit_use_double_quant=True**:
   - **Double quantization** is a technique that applies an additional round of quantization after the first, allowing for even further compression.
   - This reduces the model's memory usage even more at the cost of some performance degradation. It's especially useful when memory is a bottleneck.

3. **bnb_4bit_quant_type="nf4"**:
   - Specifies the **Normal Float 4 (NF4)** quantization format, which is optimized for deep learning tasks. NF4 balances the trade-offs between memory reduction and computation efficiency. NF4 is often chosen because it preserves important model behavior even in highly compressed formats.

4. **bnb_4bit_compute_dtype=torch.float16**:
   - This tells the model to use **float16 precision** for computations. **Float16** is a mixed-precision format that reduces memory usage and accelerates computation on GPUs (especially for inference tasks).
   - Using **float16** is a good practice when aiming for faster processing times (higher TPS) as it improves both memory and compute efficiency, especially on modern GPUs that are optimized for float16.

5. **low_cpu_mem_usage=True**:
   - This setting ensures that the model will use less CPU memory, even when loading the model or running it in inference mode.
   - It's useful in situations where the machine might have limited RAM or if you are working with extremely large models.
"""

# Configure BitsAndBytes quantization for 4-bit precision and computation settings
bnb_quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit quantization to reduce model size and memory usage
    bnb_4bit_use_double_quant=True,         # Use double quantization for further compression, reducing memory usage
    bnb_4bit_quant_type="nf4",              # Use Normal Float 4 (NF4) format, optimized for deep learning tasks
    bnb_4bit_compute_dtype=torch.bfloat16,  # Specify computation to be done in bfloat16 precision for better memory efficiency and speed
    low_cpu_mem_usage=True                  # Explicitly set low_cpu_mem_usage to True for efficient CPU memory usage
)

# Load the model with the specified quantization configuration and the torch_dtype set to bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute model layers across devices CPU and CUDA
    quantization_config=bnb_quantization_config,  # Apply the BitsAndBytes quantization configuration
    torch_dtype=torch.bfloat16  # Set the dtype for model parameters and activations to bfloat16
)

model = model.eval()  # Set the model to evaluation mode for inference

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