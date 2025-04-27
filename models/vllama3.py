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

def build_conversation(content_setting: str, input_path: str, prompt: str, modal: str):
    
    conversation = [{
            "role": "system",
            "content": content_setting
        }, {
            "role": "user",
            "content": [{
                "type": modal,
                "video": {
                    "video_path": input_path,
                    "fps": 30,
                }
            }, {
                "type": "text",
                "text": prompt
            }]
    }]
    
    return conversation

def inference(
    input_path: str | list[str],
    prompt: str = "",
    model_package = None,
    content_setting: str = "You are a helpful assistant.",
    conversation: dict = None,
    ):
    
    # Check if frames_list is empty
    if len(input_path) == 0:
        return 'empty'
    
    # Determine modal
    modal = 'image' if len(input_path) == 1 else 'video'

    # Create temporary output file as video or image
    if not conversation and (isinstance(input_path, list) and len(input_path) > 0):
        input_path = utils.create_video_from_str(input_path)
        tmp_file = True
        if not input_path:
            return 'empty'
    
    # Load model
    model, processor = load_model() if model_package is None else model_package
    
    conversation = build_conversation(content_setting, input_path, prompt, modal) if conversation is None else conversation

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

    # Example
    """
    python models/archive/vllama3.py \
        --video_folder "../video_frames/Follow" \
        --prompt "Explain what the person is during in details, for an LLM to interpret the gesture." \
        --n_frames 8 \
        --start_frame 36
    """

    prompt, input = utils.argparse()
    caption = inference(prompt=prompt, input_path=input)
    print("Caption:", caption)