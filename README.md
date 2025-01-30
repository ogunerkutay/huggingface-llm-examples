# Contents of /README.md

# HuggingFace LLM Benchmark

## Overview
This project is designed to facilitate the use of various large language models for natural language processing tasks. It includes scripts for loading and running different models, and measuring their performances.

## Project Structure
The project is organized into the following directories and files:

- **CODE/**: Contains Python scripts for running various models and checking system capabilities.
  - `check_cuda_availability.py`: Checks the availability of CUDA (GPU support) using PyTorch.
  - `hf_transformers_automodel_minicpmo.py`: Loads and runs the MiniCPM-o-2_6 model for multimodal tasks.
  - `hf_transformers_causallm_gpt2.py`: Loads and runs the GPT-2 model using the AutoModelForCausalLM class.
  - `hf_transformers_embed_bert.py`: Loads and runs a BERT model for embedding extraction using Hugging Face's AutoModel.
  - `hf_transformers_image_captioning_blip2.py`: Loads and runs the BLIP-2 model for image captioning tasks.
  - `hf_transformers_multimodalcausallm_januspro1b.py`: Loads and runs the Janus-Pro-1B model for multimodal causal language tasks.
  - `hf_transformers_seq2seq_flant5large.py`: Loads the Google Flan-T5 model for sequence-to-sequence tasks.
  - `llama_gguf_minicpmo.py`: Loads and runs a LLaMA GGUF model optimized for CPU usage.
  - `lmdeploy_image_description.py`: Loads and runs the InternVL2-1B model for image description tasks.
  - `run_all_models.py`: Automates the execution of multiple model scripts, captures their performance metrics, and writes these metrics to an Excel file.


## Usage
To use the scripts in this project, ensure you have the necessary dependencies installed, including PyTorch and Hugging Face's Transformers library. Each script can be run independently to perform specific tasks related to language models.

## Requirements
- Python 3.x
- PyTorch
- Transformers library from Hugging Face

## License
This project is licensed under the MIT License. See the LICENSE file for more details.