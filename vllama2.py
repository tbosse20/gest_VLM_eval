from enum import Enum
import numpy as np
from PIL import Image
import sys, os
sys.path.append("../VideoLLaMA2")
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

def load_model():

    disable_torch_init()

    # Load the VideoLLaMA2 model
    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F' # '...-Base'
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
    
    processed = processor[modal](frame).to("cuda")

    output = mm_infer(processed, prompt, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
    # print("Image output:\n", output)

    return output

def sanity():

    # Video Inference
    modal = 'video'
    modal_path = 'data/sanity/video_0153.mp4' 
    instruct = 'What is happening in this video?'
    output = inference(modal_path, instruct, modal)
    print("Video output:\n",output)
    print()
	
    # Image Inference
    modal = 'image'
    modal_path = 'data/sanity/video_0153.png' 
    instruct = 'What is happening in this image?'
    output = inference(modal_path, instruct, modal)
    print("Image output:\n", output)

if __name__ == "__main__":
    sanity()