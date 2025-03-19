# FILE: hf_transformers_embed_bert.py
"""
This script uses Hugging Face's AutoModel and AutoTokenizer to load and run various pretrained models.
AutoModel is a flexible interface in the Hugging Face 'transformers' library that allows loading and running a wide variety of models.
It supports both CPU and GPU backends, providing flexibility to run models on different hardware platforms.
The models can be used for many NLP tasks such as classification, token classification, question answering, and more.
AutoModel supports different types of transformer models like BERT, GPT-2, T5, and others.

However, AutoModel is not designed for model formats like GGUF, which is specific to LLaMA and requires specialized libraries like llama_cpp.
The AutoModel class works with models from Hugging Face's model hub, such as 'google/flan-t5-large' for Seq2Seq tasks.
The 'google/flan-t5-large' model is a text-to-text transformer, commonly used for text generation, summarization, and question answering.

Below, we demonstrate how to use AutoModel and AutoTokenizer with a model like BERT, which is typically used for classification tasks.
Unlike Seq2Seq models (e.g., T5, GPT), which are used for tasks like text generation, BERT-based models are commonly used for tasks such as:
- text classification (e.g., sentiment analysis)
- token classification (e.g., Named Entity Recognition - NER)

The current model used here is "bert-base-uncased", a transformer-based model trained on English text.
The input text is tokenized and passed to the model to generate outputs such as embeddings or predictions.
For tasks like sentiment analysis, the model can be fine-tuned to classify text into categories such as positive, negative, or neutral.
For more details on AutoModel and AutoTokenizer, refer to the Hugging Face documentation:
https://huggingface.co/transformers/model_doc/auto.html
"""

# Import necessary libraries
import time  # Library for time-related functions
import torch  # Import torch to check for CUDA support
from transformers import AutoModel, AutoTokenizer # Import AutoModel and AutoTokenizer from Hugging Face
from huggingface_hub import scan_cache_dir # Import function to scan the cache directory

# Start the stopwatch
start_time = time.time()

# Set a manual seed for reproducibility
torch.manual_seed(100)

# Load a BERT model compatible with AutoModel for classification tasks
model_name = "bert-base-uncased"  # BERT model for classification or embedding extraction
print(f"Model name: {model_name}")

# Check if CUDA is available and choose the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load the model and move it to the device (GPU if available)
model = AutoModel.from_pretrained(
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

# Print raw tokens (token IDs)
print("Raw token IDs:", inputs['input_ids'])

# Decode the tokens to human-readable form
decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
print("Human-readable text:", decoded_text)

# Move inputs to the same device as the model (GPU if available)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.inference_mode(): # Set the model to inference mode, better than torch.no_grad() for inference
    # Use autocast for mixed precision during the model generation
    with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
        # Measure response time
        response_start_time = time.time()
        # Generate model output (e.g., embeddings or hidden states)
        outputs = model(**inputs)
        response_time = time.time() - response_start_time

# Print the output (e.g., model's hidden states or embeddings)
print("Model Output:", outputs)

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")