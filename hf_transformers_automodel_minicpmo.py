# FILE: hf_transformers_automodel_minicpmo.py
"""
This script uses the Hugging Face `transformers` library to load and run a vision-language model for image-based question answering tasks.
The model used here is `MiniCPM-o-2_6`, which is designed for multimodal tasks involving both images and text.
In this example, we will load a pretrained vision-language model, process an input image, generate responses to questions about the image, and print the generated responses.
"""

# Import necessary libraries
import torch  # Library for tensor computations and GPU support
from PIL import Image  # Library for image processing
from transformers import AutoModel, AutoTokenizer  # Hugging Face libraries for model and tokenizer

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set a manual seed for reproducibility
torch.manual_seed(100)

# Define the model name
model_name = 'openbmb/MiniCPM-o-2_6'

# Load the pretrained vision-language model from Hugging Face model hub
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)  # Load model with specific attention implementation and data type
model = model.eval().to(device)  # Set model to evaluation mode and move it to the appropriate device GPU or CPU

# Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load the input image from a local file
image = Image.open(r"C:\sise.jpeg").convert('RGB')  # Convert image to RGB format

# First round chat
question = "What is the landform in the picture?"  # Define the question to ask about the image
msgs = [{'role': 'user', 'content': [image, question]}]  # Create a message with the image and question

# Generate an answer using the model's chat method
answer = model.chat(
    msgs=msgs,  # Pass the messages to the model
    tokenizer=tokenizer  # Use the tokenizer for processing
)
print(answer)  # Print the generated answer

# Second round chat, pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": [answer]})  # Add the model's previous answer to the conversation history
msgs.append({"role": "user", "content": ["What should I pay attention to when traveling here?"]})  # Add a new question

# Generate another answer using the model's chat method
answer = model.chat(
    msgs=msgs,  # Pass the updated messages to the model
    tokenizer=tokenizer  # Use the tokenizer for processing
)
print(answer)  # Print the generated answer