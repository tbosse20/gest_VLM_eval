from enum import Enum
import numpy as np
from PIL import Image
import torch
import sys, os
sys.path.append(".")
from transformers import AutoProcessor, AutoModelForImageTextToText
import config.hyperparameters as hyperparameters

def load_model():

    # Load the model
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to("cuda")

    return model, processor

def unload_model(model, processor):
    
    del model
    model = None
    del processor
    processor = None

class Modal(Enum):
    IMAGE = 'image'
    VIDEO = 'video'

def inference(frame: np.ndarray | Image.Image, prompt: str, modal: Modal, model_package=None):

    model, processor = load_model() if model_package is None else model_package

    inputs = processor(frame, prompt, return_tensors="pt").to("cuda")

    # Perform inference
    with torch.no_grad():
        output = model.generate(
            **inputs,
            **hyperparameters.generation_args
        )

    caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)

    return caption

def sanity():

    # Video Inference
    modal = 'video'
    modal_path = 'data/sanity/input/video_0153.mp4' 
    instruct = 'What is happening in this video?'
    output = inference(modal_path, instruct, modal)
    print("Video output:\n",output)
    print()
	
    # Image Inference
    modal = 'image'
    modal_path = 'data/sanity/input/video_0153.png' 
    instruct = 'What is happening in this image?'
    output = inference(modal_path, instruct, modal)
    print("Image output:\n", output)

if __name__ == "__main__":
    # sanity()
    model_package = load_model()

    # Image Inference
    modal = 'image'
    modal_path = 'saved_image0.jpg'
    frame = Image.open(modal_path).convert("RGB")
    instruct = """ You are an autonomous vehicle. Explain the pedestrian, and suggest what to do. """
    output = inference(frame, instruct, modal, model_package)
    print("Image output:\n", output)

    modal = 'image'
    modal_path = 'saved_image1.jpg'
    frame = Image.open(modal_path).convert("RGB")
    instruct = ""
    output = inference(frame, instruct, modal, model_package)
    print("Image output:\n", output)

    model, processor = model_package
    unload_model(model, processor)