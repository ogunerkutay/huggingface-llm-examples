# FILE: hf_transformers_seq2seq_t5.py
"""
This script uses Hugging Face's AutoModelForSeq2SeqLM and AutoTokenizer to load and run various pretrained sequence-to-sequence models.
AutoModelForSeq2SeqLM is a flexible interface in the Hugging Face 'transformers' library that allows loading and running a variety of Seq2Seq models.
These models are designed for tasks such as text generation, translation, summarization, and question answering.

In this example, we will load a pretrained Seq2Seq model, tokenize input text, generate output, and print the generated text.
"""

# Import necessary libraries
import time  # Library for time-related functions
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer # Import AutoModelForSeq2SeqLM and AutoTokenizer from Hugging Face
import torch  # Import torch to check for CUDA support
import torch  # Import torch to check for CUDA support

# Start the stopwatch
start_time = time.time()

# Check if CUDA is available and choose the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set a manual seed for reproducibility
torch.manual_seed(100)

# Load google/flan-t5-large model and tokenizer
model_name = "google/flan-t5-large"  # Example model; you can replace it with any other Seq2Seq model

# Load the model and move it to the device (GPU if available)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()  # Move model to GPU or CPU and set to evaluation mode

# Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input text to be processed by the model
input_text = "What is the capital of France?"

# Tokenize the input text to convert it into a format the model can understand (input_ids)
inputs = tokenizer(input_text, return_tensors="pt")  # 'pt' stands for PyTorch tensors

# Move inputs to the same device as the model (GPU if available)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Measure response time
response_start_time = time.time()
# Generate model output using the `generate()` method to get human-readable text
generated_ids = model.generate(**inputs, max_length=50)
response_time = time.time() - response_start_time

# Decode the generated IDs back into human-readable text
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Print the final generated text
print(generated_text)

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")