import torch
import sys, os
sys.path.append("../VideoLLaMA2")
sys.path.append("../../VideoLLaMA2")
sys.path.append(".")
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import config.hyperparameters as hyperparameters
import src.utils as utils

def load_model():

    disable_torch_init()

    # Load the VideoLLaMA2 model
    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    model, processor, tokenizer = model_init(model_path)
    model.to("cuda")

    return model, processor, tokenizer

def inference(
    prompt: str,
    frames_list: list[str] = None,
    model_package = None
    ):

    if frames_list is None or len(frames_list) == 0:
        return 'empty'
    if len(frames_list) > 8:
        raise ValueError("frames_list contains more than 8 elements")
    
    # Determine modal
    modal = 'image' if len(frames_list) == 1 else 'video'
    
    # Create temporary output file as video or image
    OUTPUT_PATH = f"_tmp_output{'.png' if modal == 'image' else '.mp4'}"
    utils.create_video_from_str(frames_list, OUTPUT_PATH)
    
    # Load model
    model, processor, tokenizer = load_model() if model_package is None else model_package

    # Process input
    processed = processor[modal](OUTPUT_PATH).to(device="cuda")

    # Perform inference
    with torch.no_grad():
        output = mm_infer(
            processed,
            prompt,
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            modal=modal,
            **hyperparameters.generation_args
        )
    
    # Remove temporary file
    os.remove(OUTPUT_PATH)
    
    # Unload model
    if model_package is None:
        utils.unload_model(model, processor, tokenizer)

    return output

def sanity():

    # Video Inference
    modal = 'video'
    modal_path = 'data/sanity/input/video_0153.mp4' 
    instruct = 'What are the pedestrians gesturing to the ego driver?'
    output = inference(instruct, modal_path)
    print("Video output:\n",output)
    print()
	
    # Image Inference
    # modal = 'image'
    # modal_path = 'data/sanity/input/video_0153.png' 
    # instruct = 'What is happening in this image?'
    # output = inference(modal_path, instruct, modal)
    # print("Image output:\n", output)

if __name__ == "__main__":
    # sanity()
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
    args = utils.argparse()
    
    frame_list = utils.generate_frame_list(args.video_folder, args.start_frame, args.interval, end_frame=args.end_frame, n_frames=args.n_frames)
    caption = inference(prompt="explain the video", frames_list=frame_list)
    print("Caption:", caption)