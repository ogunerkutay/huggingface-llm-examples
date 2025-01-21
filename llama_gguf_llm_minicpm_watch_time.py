# FILE: Llama_watch_time.py
"""
This script loads and runs a LLaMA GGUF model optimized for CPU usage and measures the execution time.
It uses the 'time' module to measure the total execution time and the response generation time.
The model used here is "MiniCPM-o-2_6-gguf", specifically a 7.6B parameter model.
The input is processed through a simple chat-like interface, with a system and user message.
The model generates a response based on the provided input text.
The output is printed as the model's generated response along with the execution times.
"""

import time
from llama_cpp import Llama

# Start the stopwatch
start_time = time.time()

llm = Llama.from_pretrained(
    repo_id="openbmb/MiniCPM-o-2_6-gguf",
    filename="Model-7.6B-Q5_K_M.gguf",
    n_ctx=0,
    temperature=0.01,
    top_p=1,
    verbose=False
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},  # Optional system message
    {"role": "user", "content": "What is the capital of France?"},
]

# Measure response time
response_start_time = time.time()
response = llm.create_chat_completion(messages=messages)
response_time = time.time() - response_start_time

print(response['choices'][0]['message']['content'])

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")
