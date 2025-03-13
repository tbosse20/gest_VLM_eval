# Use a pipeline as a high-level helper
from transformers import pipeline
import torch

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline(
    "image-text-to-text",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.dtype.bfloat16,
    )
pipe(messages)