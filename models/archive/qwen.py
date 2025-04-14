from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import List
import os
import sys
sys.path.append(".")
import src.utils as utils
import config.hyperparameters as hyperparameters

def load_model():
    torch.cuda.empty_cache()
    
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_name)
    
    return model, processor

def inference(
    prompt: str,
    frames_list: List[str],
    model_package = None,
    ):
    
    # if len(frames_list) > 8:
    #     raise ValueError
    if len(frames_list) == 0:
        return 'empty'
    
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
                }, {
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
    generated_ids = model.generate(**inputs, **hyperparameters.generation_args)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    if model_package is None:
        utils.unload_model(model, processor)

    return output_text[0]

def video_inference(
    prompt: str,
    video_path: str,
    model_package = None,
    ):
    
    model, processor = load_model() if model_package is None else model_package
    
    # Messages containing a images list as a video and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 1.0,
                }, {
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
    generated_ids = model.generate(**inputs, **hyperparameters.generation_args)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    if model_package is None:
        utils.unload_model(model, processor)

    return output_text[0]

if __name__ == "__main__":

    # Example
    """
        python models/archive/qwen.py \
        --video_folder 'data/video_frames/man_0153' \
        --start_frame 85 \
        --n_frames 4 \
        --prompt "What is the man gesturing?"
    """

    args = utils.argparse()
    
    frame_list = utils.generate_frame_list(args.video_folder, args.start_frame, args.interval, end_frame=args.end_frame, n_frames=args.n_frames)
    caption = inference(prompt=args.prompt, frames_list=frame_list)
    print("Caption:", caption)