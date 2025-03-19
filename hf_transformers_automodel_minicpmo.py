# FILE: hf_transformers_automodel_minicpmo.py
"""
This script uses the Hugging Face `transformers` library to load and run a vision-language model for image-based question answering tasks.
The model used here is `MiniCPM-o-2_6`, which is designed for multimodal tasks involving both images and text.
In this example, we will load a pretrained vision-language model, process an input image, generate responses to questions about the image, and print the generated responses.
"""

# Import necessary libraries
import time  # Library for time-related functions
import torch  # Library for tensor computations and GPU support
from transformers import AutoModel, AutoTokenizer  # Hugging Face libraries for model and tokenizer
from huggingface_hub import scan_cache_dir # Import function to scan the cache directory
from PIL import Image  # Library for image processing

# Start the stopwatch
start_time = time.time()

# Set a manual seed for reproducibility
torch.manual_seed(100)

# Define the model name
model_name = 'openbmb/MiniCPM-o-2_6'
print(f"Model name: {model_name}")

# Check if CUDA is available and choose the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load the pretrained vision-language model from Hugging Face model hub
model = AutoModel.from_pretrained(
    model_name,
    device=device,  # Choose the device (CPU or GPU) for running the model
    attn_implementation='sdpa', # Use the 'sdpa' attention implementation for faster inference
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    )  # Load model with specific attention implementation and data type

model = model.eval().to(device)  # Set the model to evaluation mode for inference

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
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load the input image from a local file
image = Image.open(r"C:\sise.jpeg").convert('RGB')  # Convert image to RGB format

# First round chat
question = "What is in the image?"  # Define the question to ask about the image
msgs = [{'role': 'user', 'content': [image, question]}]  # Create a message with the image and question

with torch.inference_mode(): # Set the model to inference mode, better than torch.no_grad() for inference
    # Use autocast for mixed precision during the model generation
    with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
        # Measure response time
        response_start_time = time.time()
        # Generate an answer using the model's chat method
        response = model.chat(
            msgs=msgs,  # Pass the messages to the model
            tokenizer=tokenizer  # Use the tokenizer for processing
        )
        response_time = time.time() - response_start_time

# Print the generated response
print("Generated Response:", response)

""" # Second round chat, pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": [response]})  # Add the model's previous answer to the conversation history
msgs.append({"role": "user", "content": ["What should I pay attention to when traveling here?"]})  # Add a new question

# Measure response time for second round
response_start_time = time.time()
# Generate another answer using the model's chat method
answer = model.chat(
    msgs=msgs,  # Pass the updated messages to the model
    tokenizer=tokenizer  # Use the tokenizer for processing
)
response_time += time.time() - response_start_time
print(answer)  # Print the generated answer """

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")