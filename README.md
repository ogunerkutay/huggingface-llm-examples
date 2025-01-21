# Contents of /README.md

# LLMS Project

## Overview
The LLMS (Large Language Model Suite) project is designed to facilitate the use of various large language models for natural language processing tasks. This suite includes scripts for loading and running different models, checking hardware compatibility, and measuring performance metrics.

## Project Structure
The project is organized into the following directories and files:

- **CODE/**: Contains Python scripts for running various models and checking system capabilities.
  - `check_cuda_availability.py`: Checks the availability of CUDA (GPU support) using PyTorch.
  - `huggingface_causallm_gpt2.py`: Loads and runs the GPT-2 model using the AutoModelForCausalLM class.
  - `huggingface_embed_bert.py`: Loads and runs a BERT model for embedding extraction using Hugging Face's AutoModel.
  - `huggingface_seq2seq_t5.py`: Loads the Google Flan-T5 model for sequence-to-sequence tasks.
  - `llama_gguf_llm_minicpm_watch_time.py`: Measures the execution time of generating a response from a LLaMA GGUF model.
  - `llama_gguf_llm_minicpm.py`: Loads and runs a LLaMA GGUF model optimized for CPU usage.

## Usage
To use the scripts in this project, ensure you have the necessary dependencies installed, including PyTorch and Hugging Face's Transformers library. Each script can be run independently to perform specific tasks related to language models.

## Requirements
- Python 3.x
- PyTorch
- Transformers library from Hugging Face

## License
This project is licensed under the MIT License. See the LICENSE file for more details.