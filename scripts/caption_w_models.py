import importlib
import sys
import pandas as pd
from tqdm import tqdm
import re
import os

sys.path.append(".")
import models.src.utils as utils
from config.prompts import prompts
import config.directories as directories
import config.flags as flags


def caption_models(data_folder: str, window: int, interval: int):

    # Validate folder path
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder {data_folder} not found")
    if not os.path.isdir(data_folder):
        raise NotADirectoryError(f"Data folder {data_folder} is not a folder")

    # Load all models modules
    models_folder = directories.MODELS_FOLDER
    model_modules = [
        f"{models_folder}.{module[:-3]}"
        for module in os.listdir(models_folder)
        if module.endswith(".py") and module != "__init__.py"
    ]

    # Iterate over models
    for name in model_modules:
        model_module = importlib.import_module(name)
        print("Processing using:", model_module.__name__.split(".")[-1])

        # Load model
        model_package = model_module.load_model()

        # Caption folder
        caption_folder(
            data_folder, window, interval, model_package, model_module
        )

        # Unload model
        utils.unload_model(*model_package)

        # Delete model from system
        del model_module
        if name in sys.modules:
            del sys.modules[name]


def caption_folder(
    frame_folder: str, window: int, interval: int, model_package, model_module
):

    # Validate folder path
    if not os.path.exists(frame_folder):
        raise FileNotFoundError(f"Data folder {frame_folder} not found")
    if not os.path.isdir(frame_folder):
        raise NotADirectoryError(f"Data folder {frame_folder} is not a folder")

    # Sub-folders
    sub_folders = [
        f.path
        for f in os.scandir(frame_folder)
        if f.is_dir()
    ]

    # Sort subfolders numerically by the number after 'video_'
    sub_folders.sort(key=lambda x: int(os.path.basename(x).split('_')[-1]))

    # Caption each sub-folder
    for sub_path in sub_folders:
        caption_input(sub_path, window, interval, model_package, model_module)

def prep_csv_output(model_module):

    output_folder_path = directories.OUTPUT_FOLDER_PATH
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Get model name
    module_name = model_module.__name__.split(".")[-1]
    csv_path = f"{output_folder_path}/{module_name}.csv"
    csv_path = csv_path.replace(".csv", f"_projection.csv") if flags.projection_enhancement else csv_path
    csv_path = csv_path.replace(".csv", f"_hands.csv") if flags.describe_hands else csv_path

    # Generate csv file if not exists
    columns = [
        "video_name",
        "prompt_type",
        "caption",
        "frame_idx",
        "end_frame",
        "interval",
        "window_size",
    ]

    # Generate csv file if not exists
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_path, mode="w", index=False, header=True)

    return csv_path


def caption_input(
    input_path: str, window: int, interval: int, model_package=None, model_module=None
):
    """Prep either video or folder path for captioning. Auto detects if video or folder.

    Args:
        path (str): Path to video or folder containing frames
        window (int): Number of frames to caption at once
        interval (int): Interval between frames to caption
        model_package: Model package for inference
        model_module: Model module for inference
    """

    # Validate variables
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video path {input_path} not found")
    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"Video path {input_path} is not a folder")

    # Output csv path
    csv_path = prep_csv_output(model_module)

    # Skip computed videos
    computed_video_names = pd.read_csv(csv_path, index_col=False)["video_name"].values
    video_name = os.path.basename(input_path)
    if video_name in computed_video_names:
        print(f"'{video_name}' captioned, skip..")
        return

    # Caption frames
    if os.path.isdir(input_path):
        caption_frames(
            input_path, csv_path, window, interval, model_package, model_module
        )

    # Caption video
    elif os.path.isfile(input_path) and input_path.endswith(
        (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv")
    ):
        caption_videos(
            input_path, csv_path, window, interval, model_package, model_module
        )


def caption_frames(
    video_path: str,
    csv_path: str,
    window: int,
    interval: int,
    model_package = None,
    model_module = None,
):
    video_name = os.path.basename(video_path)
    start_frame, end_frame = utils.get_start_n_end_frames(video_path)
    window = end_frame - start_frame if window == 0 else window

    # Iterate over video frames
    for current_start_frame in tqdm(
        range(start_frame, end_frame, window),
        desc=f"{video_name}",
    ):
        n_frames = window // interval

        # Generate frames list
        frames_list = utils.generate_frame_list(
            video_path, current_start_frame, interval, n_frames = n_frames
        )
        if len(frames_list) == 0:
            continue  # Skip if no frames found

        caption_prompts(
            frames_list, video_path, csv_path, current_start_frame, window, interval, model_package, model_module
        )


def caption_videos(data_folder, csv_path, model_package, model_module):

    # Validate folder path
    if not os.path.isdir(data_folder):
        raise NotADirectoryError(f"Data folder {data_folder} is not a folder")

    # Videos
    video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv")
    video_paths = [
        f.path
        for f in os.scandir(data_folder)
        if f.is_file() and f.path.lower().endswith(video_extensions)
    ]

    # Captions videos across prompts
    for video_path in video_paths:
        caption_prompts(video_path, csv_path, 0, 0, 1, model_package, model_module)


def caption_prompts(
    frames_list: list,
    video_path: str,
    csv_path: str,
    start_frame: int,
    window: int,
    interval: int,
    model_package = None,
    model_module = None,
):

    # Iterate over prompts
    for prompt_type, prompt in prompts.items():
        video_name = os.path.basename(video_path)

        try:
            # Get model response and append to dictionary
            respond = model_module.inference(
                prompt        = prompt,
                input_path    = frames_list,
                model_package = model_package
            )
            respond = re.sub(r" {2,}", "\\\\s", respond.replace("\n", "\\\\n").strip())

            # Save to csv
            save_to_csv(video_name, prompt_type, respond, csv_path, start_frame, window, interval)

        except Exception as e:            
            # Unload model
            utils.unload_model(*model_package)
            # Delete model from system
            del model_module
            
            print(f"Error during inference: {e}")
            print(f'Skip, {video_name}')



def save_to_csv(
    video_name: str,
    prompt_type: str,
    respond: str,
    csv_path: str,
    start_frame: int,
    window: int,
    interval: int,
):

    # Save to csv
    df = pd.DataFrame(
        {
            "video_name":  [video_name],
            "prompt_type": [prompt_type],
            "caption":     [respond],
            "start_frame": [start_frame],
            "end_frame":   [start_frame + window],
            "interval":    [interval],
            "window_size": [window],
        }
    )
    df.to_csv(csv_path, mode="a", index=False, header=False)


if __name__ == "__main__":

    import argparse
    import config.hyperparameters as hyperparameters

    parser = argparse.ArgumentParser(description="Caption frames")
    parser.add_argument(
        "--data_folder",
        type=str,
        help="Data folder path",
        default=directories.VIDEO_FRAMES_FOLDER,
    )
    parser.add_argument(
        "--window",
        type=int,
        help="Window size",
        default=hyperparameters.window,
    )
    parser.add_argument(
        "--interval",
        type=int,
        help="Interval between frames",
        default=hyperparameters.interval,
    )
    args = parser.parse_args()

    # Example
    """ 
    python caption_w_models.py \
        --data_folder "../video_frames" \
        --interval 1 \
        --window 8
    """

    caption_models(args.data_folder, args.window, args.interval)
