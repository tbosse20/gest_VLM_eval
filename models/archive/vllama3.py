import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import sys, os
sys.path.append(".")
import config.hyperparameters as hyperparameters
import models.src.utils as utils

def load_model():
    torch.cuda.empty_cache()

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
    input_path: str | list[str] = None,
    model_package = None,
    content_setting: str = "You are a helpful assistant.",
    ):
    
    # Check if frames_list is empty
    if len(input_path) == 0:
        return 'empty'
    
    # Determine modal
    modal = 'image' if len(input_path) == 1 else 'video'
    
    # Create temporary output file as video or image
    if (isinstance(input_path, list) and len(input_path) > 1):
        input_path, tmp_file = utils.create_video_from_str(input_path)
    
    # Load model
    model, processor = load_model() if model_package is None else model_package
    
    conversation = [{
            "role": "system",
            "content": content_setting
        }, {
            "role": "user",
            "content": [{
                "type": modal,
                "video": {
                    "video_path": input_path,
                    "fps": 1,
                    "max_frames": 180
                }
            }, {
                "type": "text",
                "text":prompt
            }]
    }]

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
    
    # Remove temporary file
    if ('tmp_file' in locals() and tmp_file):
        os.remove(input_path)
    
    if model_package is None:
        utils.unload_model(model, processor)

    return response

if __name__ == "__main__":
    args = utils.argparse()
    
    frame_list = utils.generate_frame_list(args.video_folder, args.start_frame, args.interval, end_frame=args.end_frame, n_frames=args.n_frames)
    caption = inference(prompt="explain the video", input_path=frame_list)
    print("Caption:", caption)