from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pandas as pd
import os
from tqdm import tqdm

def from_end_frame(video_folder, start_frame, interval, end_frame):
    return [
        f"{video_folder}/frame_{frame_count:04d}.png"
        for frame_count in range(start_frame, end_frame, interval)
    ]

def from_n_frame(video_folder, start_frame, interval, n_frames):
    return [
        f"{video_folder}/frame_{start_frame + frame_count:04d}.png"
        for frame_count in range(0, n_frames, interval)
        if os.path.exists(f"{video_folder}/frame_{start_frame + frame_count:04d}.png")
    ]

def generate_frame_list(video_folder, start_frame, interval=1, end_frame=None, n_frames=None):
    if end_frame is not None: 
        return from_end_frame(video_folder, start_frame, interval, end_frame)
    if n_frames is not None: 
        return from_n_frame(video_folder, start_frame, interval, n_frames)

def load_model():
    
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat32,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_name)
    
    return model, processor

def unload_model(model, processor):
    del model
    del processor
    torch.cuda.empty_cache()

def inference(
    prompt: str,
    model_package = None,
    frames_list: list[str] = None,
    ):
    
    if len(frames_list) == 0:
        return 'empty'
    
    unload_model_after = model_package is None
    model, processor = load_model() if model_package is None else model_package
    
    # Messages containing a images list as a video and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames_list,
                    "fps": 1.0,
                },
                {
                    "type": "text",
                    "text": prompt
                    },
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
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    if unload_model_after:
        unload_model(model, processor)

    return output_text[0]

video_folder = "data/sanity/input/video_0153"
start_frame = 0,
interval = 1,
end_frame = None,
n_frames = None,
frame_list = generate_frame_list(video_folder, start_frame, interval, end_frame, n_frames)