import sys
sys.path.append(".")
import models.src.utils as utils
import enhance.augment.augment as augment
import models.archive.vllama3 as vllama3

def load_model():
    return vllama3.load_model()

def inference(
    input_path: str | list[str],
    prompt: str = "",
    model_package = None,
    content_setting: str = "You are a helpful assistant.",
    ):
    
    conversation = augment.build_conversation(
        input_path  = input_path,
        prompt      = prompt,
    )
    
    response = vllama3.inference(
        input_path      = input_path,
        prompt          = prompt,
        model_package   = model_package,
        content_setting = content_setting,
        conversation    = conversation,
    )
    
    return response

if __name__ == "__main__":
    args = utils.argparse()
    frame_list = utils.generate_frame_list(args.video_folder, args.start_frame, args.interval, end_frame=args.end_frame, n_frames=args.n_frames)
    caption = inference(prompt="explain the video", input_path=frame_list)
    print("Caption:", caption)