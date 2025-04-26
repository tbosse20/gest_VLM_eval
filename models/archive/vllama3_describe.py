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
        frame_list  = input_path,
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

    # Example
    """
    python models/archive/vllama3_describe.py \
        --video_folder "../video_frames/Follow" \
        --prompt "Explain what the person is during in details, for an LLM to interpret the gesture." \
        --n_frames 8 \
        --start_frame 36
    """

    prompt, input = utils.argparse()
    print(f"---> Input: Video of {len(input)} frames")
    caption = inference(prompt=prompt, input_path=input)
    print("Caption:", caption)