import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import sys
sys.path.append(".")
import config.hyperparameters as hyperparameters
import src.utils as utils

def load_model():

    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # default processer
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    return model, processor

def inference(
    prompt: str,
    content_setting: str = "You are a helpful assistant.",
    frames_list: list[str] = None,
    model_package = None,
    ):
    
    if len(frames_list) == 0:
        return 'empty'
    
    unload_model_after = model_package is None
    model, processor = load_model() if model_package is None else model_package
    
    conversation = [
        {
            "role": "system",
            "content": content_setting},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": "./assets/cat_and_chicken.mp4",
                        "fps": 1,
                        "max_frames": 180
                    }
                }, {
                    "type": "text",
                    "text":prompt
                },
            ]
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    
    # inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, **hyperparameters.generation_args)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    if unload_model_after:
        utils.unload_model(*model_package)

    return response

if __name__ == "__main__":
    args = utils.argparse()
    
    frame_list = utils.generate_frame_list(args.video_folder, args.start_frame, args.interval, end_frame=args.end_frame, n_frames=args.n_frames)
    caption = inference(prompt="explain the video", frames_list=frame_list)
    print("Caption:", caption)