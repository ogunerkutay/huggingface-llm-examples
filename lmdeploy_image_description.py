# FILE: lmdeploy_image_description.py
"""
This script uses the lmdeploy library to load and run a vision-language model for image description tasks.
The lmdeploy library provides tools for deploying large models efficiently, particularly in vision-language applications.
In this example, we will load a pretrained vision-language model, process an input image, generate a description, and print the generated description.
"""

# Import necessary libraries from lmdeploy
from lmdeploy import pipeline, TurbomindEngineConfig  # Import pipeline and configuration for the engine
from lmdeploy.vl import load_image  # Import function to load images

# Specify the model to be used
model_name = 'OpenGVLab/InternVL2-1B'  # Vision-language model for image description

# Load the input image from a URL
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')  # Load image from URL

# Create a pipeline for the model with specific backend configuration
pipe = pipeline(model_name, backend_config=TurbomindEngineConfig(session_len=8192))  # Initialize pipeline with model and configuration

# Generate a description for the input image using the pipeline
response = pipe(('describe this image', image))  # Generate description for the image

# Print the generated description
print(response.text)  # Print the generated text response