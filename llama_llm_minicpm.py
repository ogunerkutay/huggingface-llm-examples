# FILE: Llama_gguf_cpu.py
"""
This script loads and runs LLaMA GGUF models optimized for CPU usage.
Unlike other models that may require GPU support, LLaMA GGUF models are designed to work efficiently on CPU.
The model used here is "MiniCPM-o-2_6-gguf", specifically a 7.6B parameter model.
The input is processed through a simple chat-like interface, with a system and user message.
The model generates a response based on the provided input text.
The output is printed as the model's generated response.
For more details on LLaMA GGUF models and their usage, refer to the official llama-cpp documentation:
https://github.com/abetlen/llama-cpp-python
"""

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="openbmb/MiniCPM-o-2_6-gguf",
	filename="Model-7.6B-Q5_K_M.gguf",
    n_ctx=0,
    temperature=0.01,
    top_p=1,
    verbose = False
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},  # Optional system message
    {"role": "user", "content": "What is the capital of France?"},
]

response = llm.create_chat_completion(messages=messages)
print(response['choices'][0]['message']['content'])