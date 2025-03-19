# FILE: lmdeploy_image_description.py
"""
This script uses the lmdeploy library to load and run a vision-language model for image description tasks.
The lmdeploy library provides tools for deploying large models efficiently, particularly in vision-language applications.
In this example, we will load a pretrained vision-language model, process an input image, generate a description, and print the generated description.
"""

# Import necessary libraries
import time  # Library for time-related functions
import torch  # Import torch to check for CUDA support
from lmdeploy import pipeline, TurbomindEngineConfig  # Import pipeline and configuration for the engine
from lmdeploy.vl import load_image  # Import function to load images
from huggingface_hub import scan_cache_dir # Import function to scan the cache directory

# Start the stopwatch
start_time = time.time()

# Set a manual seed for reproducibility
torch.manual_seed(100)

# Define the model name and configuration
model_name = 'OpenGVLab/InternVL2-1B'  # Vision-language model for image description
print(f"Model name: {model_name}")

# Check if CUDA is available and choose the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Set the backend configuration for the model
config = TurbomindEngineConfig(session_len=8192)

# Create a vision-language model pipeline for the model with specific backend configuration
vl_pipeline = pipeline(
    model_name,
    device_map="auto",  # Automatically distribute model layers across devices CPU and CUDA
    backend_config=config
    )  # Initialize pipeline with model and configuration

# Calculate the number of parameters
param_size = sum(p.numel() for p in vl_pipeline.parameters())

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

# Load the input image from a URL
#image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')  # Load image from URL

# Load the input image from a local file
image_path = r"C:\sise.jpeg"
image = load_image(image_path)

with torch.inference_mode(): # Set the model to inference mode, better than torch.no_grad() for inference
    # Measure response time
    response_start_time = time.time()
    # Generate a description for the input image using the pipeline
    response = vl_pipeline(('describe this image', image))  # Generate description for the image
    response_time = time.time() - response_start_time

# Print the generated description
print(f"Image Description: {response}")  # Print the generated text response

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")