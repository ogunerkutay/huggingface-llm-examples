# FILE: hf_transformers_image_captioning_blip2.py
'''
This script uses Hugging Face's Blip2Processor and Blip2ForConditionalGeneration to load and run the BLIP-2 model.
The BLIP-2 model is designed for image captioning tasks, where the model generates a textual description of an input image.
In this example, we will load a pretrained BLIP-2 model, process an input image, generate a caption, and print the generated caption.
'''

# Import necessary libraries
import time  # Library for time-related functions
from PIL import Image  # Library for image processing
from transformers import Blip2Processor, Blip2ForConditionalGeneration  # Hugging Face libraries for BLIP-2 model
import torch  # Library for tensor computations and GPU support

# Start the stopwatch
start_time = time.time()

# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set a manual seed for reproducibility
torch.manual_seed(100)

# Define the model name
model_name = "Salesforce/blip2-opt-2.7b"

# Load the BLIP-2 processor and model from the Hugging Face model hub
processor = Blip2Processor.from_pretrained(model_name)  # Load the processor
model = Blip2ForConditionalGeneration.from_pretrained(model_name).eval().to(device)  # Set model to evaluation mode and move it to the appropriate device GPU or CPU

# Load the input image
image = Image.open(r"C:\sise.jpeg")

# Process the image and convert it into a format the model can understand (input tensors)
inputs = processor(image, return_tensors="pt").to(device)  # Move inputs to the same device as the model (GPU if available)

# Measure response time
response_start_time = time.time()
# Generate a caption for the input image using the model's generate() method
caption_ids = model.generate(**inputs, max_new_tokens=100)
response_time = time.time() - response_start_time

# Decode the generated caption IDs back into human-readable text
caption = processor.decode(caption_ids[0], skip_special_tokens=True)

# Print the final generated caption
print(f"Image Caption: {caption}")

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")