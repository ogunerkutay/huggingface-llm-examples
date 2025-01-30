# FILE: lmdeploy_image_description.py
"""
This script uses the lmdeploy library to load and run a vision-language model for image description tasks.
The lmdeploy library provides tools for deploying large models efficiently, particularly in vision-language applications.
In this example, we will load a pretrained vision-language model, process an input image, generate a description, and print the generated description.
"""

# Import necessary libraries
import time  # Library for time-related functions
from lmdeploy import pipeline, TurbomindEngineConfig  # Import pipeline and configuration for the engine
from lmdeploy.vl import load_image  # Import function to load images

# Start the stopwatch
start_time = time.time()

# Define the model name and configuration
model_name = 'OpenGVLab/InternVL2-1B'  # Vision-language model for image description
config = TurbomindEngineConfig(session_len=8192)

# Create a vision-language model pipeline for the model with specific backend configuration
vl_pipeline = pipeline(model_name, backend_config=config)  # Initialize pipeline with model and configuration

# Load the input image from a URL
#image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')  # Load image from URL

# Load the input image from a local file
image_path = r"C:\sise.jpeg"
image = load_image(image_path)

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