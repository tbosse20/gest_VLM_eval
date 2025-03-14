from enum import Enum
import numpy as np
from PIL import Image
import torch
import sys, os
sys.path.append("../VideoLLaMA2")
sys.path.append("../../VideoLLaMA2")
sys.path.append(".")
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import RMPS.prompts as prompts

def load_model():

    disable_torch_init()

    # Load the VideoLLaMA2 model
    # model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B' # 8F
    model, processor, tokenizer = model_init(model_path)
    model.to("cuda")

    return model, processor, tokenizer

def unload_model(model, processor, tokenizer):
    
    del model
    model = None
    del processor
    processor = None
    del tokenizer
    tokenizer = None

class Modal(Enum):
    IMAGE = 'image'
    VIDEO = 'video'

def inference(frame: np.ndarray | Image.Image, prompt: str, modal: Modal, vllama2_package=None):

    model, processor, tokenizer = load_model() if vllama2_package is None else vllama2_package

    # Define generation hyperparameters
    generation_args = {
        "temperature": 0.2,          # Controls randomness (lower = more deterministic)
        "top_k": 50,                 # Limits token selection to top 50 choices
        "top_p": 0.95,               # Nucleus sampling threshold
        "max_new_tokens": 512,       # Max number of tokens in response
        "repetition_penalty": 1.2,   # Penalizes repetition
        "no_repeat_ngram_size": 2,   # Prevents repeating n-grams (3-grams)
        "length_penalty": 1.0,       # Adjusts output length preference
    }
    
    processed = processor[modal](frame).to(device="cuda")

    # Perform inference
    with torch.no_grad():
        output = mm_infer(
            processed,
            prompt,
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            modal=modal,
            **generation_args
        )

    return output

def sanity():

    # Video Inference
    modal = 'video'
    modal_path = 'data/sanity/input/video_0153.mp4' 
    instruct = 'What are the pedestrians gesturing to the ego driver?'
    output = inference(modal_path, instruct, modal)
    print("Video output:\n",output)
    print()
	
    # Image Inference
    # modal = 'image'
    # modal_path = 'data/sanity/input/video_0153.png' 
    # instruct = 'What is happening in this image?'
    # output = inference(modal_path, instruct, modal)
    # print("Image output:\n", output)

if __name__ == "__main__":
    sanity()
    # vllama2_package = load_model()

    # # # Image Inference
    # # modal = 'image'
    # # modal_path = 'saved_image0.jpg' 
    # # instruct = """ You are an autonomous vehicle. Explain the pedestrian, and suggest what to do. """

    # # output = inference(modal_path, instruct, modal, vllama2_package)
    # # print("Image output:\n", output)

    # # modal = 'image'
    # # modal_path = 'saved_image1.jpg' 
    # # # instruct = prompts.pose
    # # instruct = """ You are driving a car. What would you do in this situation? """
    # # output = inference(modal_path, instruct, modal, vllama2_package)
    # # print("Image output:\n", output)

    # modal = 'video'
    # # modal_path = 'data/sanity/input/video_0153.mp4' 
    # modal_path = 'data/sanity/output/short.mp4' 
    # # instruct = prompts.pose
    # instruct = ""
    # output = inference(modal_path, instruct, modal, vllama2_package)
    # print("Image output:\n", output)

    # model, processor, tokenizer = vllama2_package
    # unload_model(model, processor, tokenizer)