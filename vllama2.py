from enum import Enum
import numpy as np
from PIL import Image
import sys, os
sys.path.append("../VideoLLaMA2")
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

disable_torch_init()

# Load the VideoLLaMA2 model
model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F' # '...-Base'
model, processor, tokenizer = model_init(model_path)

class Modal(Enum):
    IMAGE = 'image'
    VIDEO = 'video'

def inference(frame: np.ndarray | Image.Image, prompt: str, modal: Modal):
    
    output = mm_infer(processor[modal](frame), prompt, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
    # print("Image output:\n", output)
    
    return output

def sanity():

    # Video Inference
    modal = 'video'
    modal_path = 'data/sanity/video_0153.mp4' 
    instruct = 'What is happening in this video?'
    output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
    print("Video output:\n",output)
    print()
	
    # Image Inference
    modal = 'image'
    modal_path = 'data/sanity/video_0153.png' 
    instruct = 'What is happening in this image?'
    output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
    print("Image output:\n", output)


if __name__ == "__main__":
    sanity()