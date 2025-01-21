# FILE: huggingface_causallm_gpt2.py
"""
This script uses Hugging Face's AutoModelForCausalLM and AutoTokenizer to load and run various pretrained causal language models.
AutoModelForCausalLM is a flexible interface in the Hugging Face 'transformers' library that allows loading and running a variety of causal language models.
These models are designed for tasks such as text generation, where the model predicts the next token in a sequence based on the previous tokens.

Unlike models designed for classification or sequence-to-sequence tasks, causal language models focus on generating coherent and contextually relevant text.
They are typically used in applications such as chatbots, story generation, and other creative writing tasks.

In this example, we will load a pretrained causal language model, tokenize input text, generate output, and print the generated response.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pretrained causal language model and its tokenizer
model_name = "gpt2"  # Example model; you can replace it with any other causal language model
model = AutoModelForCausalLM.from_pretrained(model_name) # Load the causal language model
tokenizer = AutoTokenizer.from_pretrained(model_name) # Tokenizer for the model

# Input text for the model to process
input_text = "Once upon a time in a land far away,"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate model output
outputs = model.generate(**inputs, max_length=50)

# Decode the generated output to human-readable text
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated response
print("Generated Response:", response)