# FILE: hf_transformers_multimodalcausallm_janus.py
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

# Start the stopwatch
start_time = time.time()

# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# specify the path to the model
model_name = "deepseek-ai/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_name)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).eval().to(device)  # Set model to evaluation mode and move it to the appropriate device GPU or CPU

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
).to(vl_gpt.device)

# Measure response time
response_start_time = time.time()
# # run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# # run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

response_time = time.time() - response_start_time

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)

# Stop the stopwatch
elapsed_time = time.time() - start_time

print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Response generation time: {response_time:.2f} seconds")