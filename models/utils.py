import cv2
import torch

# Function to create a video from images
def create_video(frames, output_video_path):

    # Get frame size
    height, width, layers = frames[0].shape
    size = (width, height)

    # Create video writer
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps=1, size=size)

    for img in frames:
        out.write(img)

    out.release()

def unload_model(*args):
    for obj in args:
        del obj
        obj = None
    
    torch.cuda.empty_cache()
    
def argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Generate video captions using Qwen2VL model.")
    parser.add_argument("--video_folder", type=str, help="Path to the video folder containing frames.", required=True)
    parser.add_argument("--prompt",       type=str, help="The prompt to generate captions.",            required=True)
    parser.add_argument("--start_frame",  type=int, help="The starting frame number.",                  default=0)
    parser.add_argument("--interval",     type=int, help="The interval between frames.",                default=1)
    parser.add_argument("--end_frame",    type=int, help="The ending frame number.",                    default=None)
    parser.add_argument("--n_frames",     type=int, help="The number of frames to process.",            default=None)
    
    return parser.parse_args()