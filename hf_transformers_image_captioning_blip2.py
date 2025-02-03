# FILE: hf_transformers_image_captioning_blip2.py
'''
This script uses Hugging Face's Blip2Processor and Blip2ForConditionalGeneration to load and run the BLIP-2 model.
The BLIP-2 model is designed for image captioning tasks, where the model generates a textual description of an input image.
In this example, we will load a pretrained BLIP-2 model, process an input image, generate a caption, and print the generated caption.
'''

# Import necessary libraries
import time  # Library for time-related functions
import torch  # Import torch to check for CUDA support
from PIL import Image  # Library for image processing
from transformers import Blip2Processor, Blip2ForConditionalGeneration  # Hugging Face libraries for BLIP-2 model
from huggingface_hub import scan_cache_dir # Import function to scan the cache directory

# Start the stopwatch
start_time = time.time()

# Set a manual seed for reproducibility
torch.manual_seed(100)

# Define the model name
model_name = "Salesforce/blip2-opt-2.7b"
print(f"Model name: {model_name}")

# Check if CUDA is available and choose the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load the BLIP-2 processor and model from the Hugging Face model hub
processor = Blip2Processor.from_pretrained(model_name)  # Load the processor for the image captioning model
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto"  # Automatically distribute model layers across devices CPU and CUDA
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

# Load the input image
image = Image.open(r"C:\sise.jpeg")

# Process the image and convert it into a format the model can understand (input tensors)
inputs = processor(image, return_tensors="pt").to(device)  # Move inputs to the same device as the model (GPU if available)

# Measure response time
response_start_time = time.time()
# Generate a caption for the input image using the model's generate() method
outputs = model.generate(**inputs, max_new_tokens=100)
# Decode the generated output to human-readable text
caption = processor.decode(outputs[0], skip_special_tokens=True)
response_time = time.time() - response_start_time

# Print the final generated caption
print(f"Image Caption: {caption}")

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")