# Contents of /README.md

# HuggingFace LLM Benchmark

## Overview
This project is designed to facilitate the use of various large language models for natural language processing tasks. It includes scripts for loading and running different models, and measuring their performances.

## Project Structure
The project is organized into the following directories and files:

- **CODE/**: Contains Python scripts for running various models and checking system capabilities.
  - `cache_analyzer.py`: Analyzes the Hugging Face Hub cache directory and provides a summary of the cached models and their sizes.
  - `check_cuda_availability.py`: Checks the availability of CUDA (GPU support) using PyTorch.
  - `hf_transformers_automodel_minicpmo.py`: Loads and runs the MiniCPM-o-2_6 model for multimodal tasks.
  - `hf_transformers_causallm_gemma2b_*.py`: Loads and runs Gemma2B model using the AutoModelForCausalLM class with different configurations (GPU, CPU, quantization).
  - `hf_transformers_causallm_gpt2.py`: Loads and runs the GPT-2 model using the AutoModelForCausalLM class.
  - `hf_transformers_embed_bert.py`: Loads and runs a BERT model for embedding extraction using Hugging Face's AutoModel.
  - `hf_transformers_image_captioning_blip2.py`: Loads and runs the BLIP-2 model for image captioning tasks.
  - `hf_transformers_multimodalcausallm_januspro1b.py`: Loads and runs the Janus-Pro-1B model for multimodal causal language tasks.
  - `hf_transformers_pipeline_gemma2b.py`: Loads and runs the Gemma-2B model using the Hugging Face pipeline.
  - `hf_transformers_seq2seq_flant5large.py`: Loads the Google Flan-T5 model for sequence-to-sequence tasks.
  - `llama_gguf_gemma2b.py`: Loads and runs the Gemma-2B model using the LLaMA GGUF format.
  - `llama_gguf_minicpmo.py`: Loads and runs a LLaMA GGUF model optimized for CPU usage.
  - `lmdeploy_image_description.py`: Loads and runs the InternVL2-1B model for image description tasks.
  - `run_all_models.py`: Automates the execution of multiple model scripts, captures their performance metrics, and writes these metrics to an Excel file.

## Model Interaction Methods
In this project, different methods are used to interact with models from the Hugging Face library and other libraries. Here are the differences between `model.generate`, `model.chat`, and `model(prompt)`:

1. **[model.generate]**:
   - This method is used with Hugging Face models that support text generation, such as those loaded with `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`.
   - It generates text based on the input tokens and can be used for tasks like text completion, story generation, and more.
   - Example usage: 
     ```python
     outputs = model.generate(**inputs, max_length=50)
     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
     ```

2. **[model.chat]**:
   - This method is specific to certain models that support a chat-like interface, often used for multimodal tasks involving both text and images.
   - It allows for more complex interactions, such as maintaining a conversation history and handling multiple rounds of dialogue.
   - Example usage:
     ```python
     response = model.chat(
         msgs=msgs,  # Pass the messages to the model
         tokenizer=tokenizer  # Use the tokenizer for processing
     )
     ```

3. **[model(prompt)]**:
   - This method is used with models that support direct text input and output, such as those from the `llama_cpp` library.
   - It takes a text prompt and generates a response directly, often used for simpler text generation tasks.
   - Example usage:
     ```python
     response = model(prompt, max_tokens=512, echo=False)
     ```

### Difference Between Structured Messages and [model(prompt)]
- **Structured Messages**:
  - Using structured messages with roles (e.g., system, user, assistant) provides more context and can guide the model's responses more effectively.
  - It is useful for maintaining conversation history and providing specific instructions to the model.
  - Example usage:
    ```python
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = model.create_chat_completion(messages=messages)
    ```

Each method is tailored to different types of models and tasks. `model.generate is` commonly used for text generation tasks, `model.chat` is used for more interactive and multimodal tasks, and `model(prompt)` is used for straightforward text generation with certain models.

## Difference Between `max_length` and `max_tokens`
- **`max_length`**:
  - This parameter specifies the maximum length of the generated sequence, including both the input tokens and the generated tokens.
  - It is commonly used in Hugging Face's `transformers` library.
  - Example usage:
    ```python
    outputs = model.generate(input_ids, max_length=50)
    ```

- **`max_tokens`**:
  - This parameter specifies the maximum number of tokens to generate, excluding the input tokens.
  - It is often used in other libraries or specific models.
  - Example usage:
    ```python
    response = model(prompt, max_tokens=512)
    ```

In summary, `max_length` includes both the input and generated tokens, while `max_tokens` refers only to the number of tokens to be generated.

## Usage
To use the scripts in this project, ensure you have the necessary dependencies installed, including PyTorch and Hugging Face's Transformers library. Each script can be run independently to perform specific tasks related to language models.

## Requirements
- Python 3.x
- PyTorch
- Transformers library from Hugging Face
- Rich
- Tabulate

## License
This project is licensed under the MIT License. See the LICENSE file for more details.