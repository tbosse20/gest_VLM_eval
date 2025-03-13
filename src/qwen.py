from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import os

# Function to print memory usage
def print_memory_usage(msg):
    allocated_memory = torch.cuda.memory_allocated() / 1024**2  # In MB
    cached_memory = torch.cuda.memory_reserved() / 1024**2  # In MB
    print(msg)
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
    print(f"Allocated Memory: {allocated_memory:.2f} MB")
    print(f"Cached Memory: {cached_memory:.2f} MB")
    print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"Max Cached Memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    print()
import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.cuda.synchronize()  # Synchronize all the GPU operations
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_peak_memory_stats()
print_memory_usage('Init')

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype="auto",
#     device_map="auto",
#     quantization_config=BitsAndBytesConfig(load_in_8bit=True)
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="cuda",
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True, 
    #     bnb_4bit_compute_dtype=torch.float16,  # Use float16 instead of bfloat16
    #     bnb_4bit_use_double_quant=True  # Enable double quantization to save more memory
    # )
)
# model.eval()
# model.gradient_checkpointing_enable()

# default processer
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda"
)

print_memory_usage('Model')

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                # "image": "data/sanity/input/video_0153.png",
                "image": "data/sanity/input/video_0153_down.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda", dtype=torch.float16)  # Use float16 on GPU

# Step 2: Flush processor after generating inputs (free up memory)
# del processor
# torch.cuda.empty_cache()
# print_memory_usage('Input')

inputs = inputs.to(dtype=torch.float32)
# Move tensors explicitly to GPU
inputs = {key: value.to("cuda") for key, value in inputs.items()}
inputs = {key: torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0) for key, value in inputs.items()}

# Verify tensor locations
for key, value in inputs.items():
    print(f"{key}: {value.device}, {value.dtype}, {value}")

def check_for_invalid_values(inputs):
    for key, value in inputs.items():
        if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"Warning: Invalid values found in {key} tensor")
            return True
    return False
# Check for NaN or Inf before generation
if check_for_invalid_values(inputs):
    print("Invalid values detected in inputs. Stopping generation.")
else:
    print("Good to go..")
inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
# inputs = inputs[:1]
# Clamp all values to a range to prevent overflow or underflow
# Apply clamp to each tensor inside the inputs dictionary
for key in inputs:
    if isinstance(inputs[key], torch.Tensor):
        inputs[key] = inputs[key].clamp(min=-1e6, max=1e6).long() 


torch.cuda.empty_cache()
with torch.no_grad():
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=8)

import torch
print(torch.__version__)  # To check your PyTorch version
print(torch.version.cuda)  # To check the CUDA version
# del model
# torch.cuda.empty_cache()
# generated_ids.to("cpu")
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda"
)
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)


# Step 7: Flush processor after decoding (free up memory)
# del processor
# torch.cuda.empty_cache()

print(output_text)
torch.cuda.synchronize()  # Synchronize all the GPU operations
torch.cuda.empty_cache()  # Empty the cache
print_memory_usage()

# Clean up the saved inputs after inference if no longer needed
# os.remove('inputs.pt')