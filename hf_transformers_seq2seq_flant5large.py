# FILE: hf_transformers_seq2seq_flant5large.py
"""
This script uses Hugging Face's AutoModelForSeq2SeqLM and AutoTokenizer to load and run various pretrained sequence-to-sequence models.
AutoModelForSeq2SeqLM is a flexible interface in the Hugging Face 'transformers' library that allows loading and running a variety of Seq2Seq models.
These models are designed for tasks such as text generation, translation, summarization, and question answering.

In this example, we will load a pretrained Seq2Seq model, tokenize input text, generate output, and print the generated text.
"""

# Import necessary libraries
import time  # Library for time-related functions
import torch  # Import torch to check for CUDA support
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer # Import AutoModelForSeq2SeqLM and AutoTokenizer from Hugging Face
from huggingface_hub import scan_cache_dir # Import function to scan the cache directory

# Start the stopwatch
start_time = time.time()

# Set a manual seed for reproducibility
torch.manual_seed(100)

# Define the model name
model_name = "google/flan-t5-large"
print(f"Model name: {model_name}")

# Check if CUDA is available and choose the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load the model and move it to the device (GPU if available)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute model layers across devices CPU and CUDA
    )

model = model.eval()  # Set the model to evaluation mode for inference

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

with torch.inference_mode(): # Set the model to inference mode, better than torch.no_grad() for inference
    # Use autocast for mixed precision during the model generation
    with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
        # Measure response time
        response_start_time = time.time()
        # Generate model output using the `generate()` method to get human-readable text
        model_output = model.generate(**inputs, max_length=50)
        # Decode the generated output to human-readable text
        response = tokenizer.decode(model_output[0], skip_special_tokens=True)
        response_time = time.time() - response_start_time

# Print the generated response
print("Generated Response:", response)
print(f"Response generation time: {response_time:.2f} seconds")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")