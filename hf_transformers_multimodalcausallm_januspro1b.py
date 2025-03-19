# FILE: hf_transformers_multimodalcausallm_januspro1b.py
'''
This script uses the Janus library along with Hugging Face's AutoModelForCausalLM to load and run a multimodal causal language model.
The model used here is "deepseek-ai/Janus-Pro-1B", which is designed for tasks involving both text and images.
In this example, we will load a pretrained multimodal causal language model, process an input image, generate a response to a question about the image, and print the generated response.
'''

# Import necessary libraries
import time  # Library for time-related functions
import torch # Library for tensor computations and GPU support
from transformers import AutoModelForCausalLM # Import AutoModelForCausalLM from Hugging Face
from janus.models import MultiModalityCausalLM, VLChatProcessor # Import MultiModalityCausalLM and VLChatProcessor from Janus
from janus.utils.io import load_pil_images # Import function to load PIL images
from huggingface_hub import scan_cache_dir # Import function to scan the cache directory

# Start the stopwatch
start_time = time.time()

# Set a manual seed for reproducibility
torch.manual_seed(100)

# Define the model name
model_name = "deepseek-ai/Janus-Pro-1B"
print(f"Model name: {model_name}")

# Check if CUDA is available and choose the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_name) # Load the processor for the multimodal causal language model
tokenizer = vl_chat_processor.tokenizer # Get the tokenizer from the processor

model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute model layers across devices CPU and CUDA
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
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

image = r"C:\sise.jpeg"
question = "What is in the image?"

conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(model.device)

with torch.inference_mode(): # Set the model to inference mode, better than torch.no_grad() for inference
    # Use autocast for mixed precision during the model generation
    with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
        # Measure response time
        response_start_time = time.time()
        # # run image encoder to get the image embeddings
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        # # run the model to get the response
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

response_time = time.time() - response_start_time

print(f"{prepare_inputs['sft_format'][0]}", response)

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")