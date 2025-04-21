import os
import sys
sys.path.append(".")
from enhance.augment import pose_from_video

def video_process(video, method):
    pass


def dir_process(dir, method, extension_name: str):
    """Process all videos in a directory with the given method.

    Args:
        parent_dir (str):     The parent dir containing sub dirs with videos.
        include_word  (str):  Only include video files containing this word.
        extension_name (str): The name of the output dir.

    """
    pass


def cluster_process(
    parent_dir: str, extension_name: str, include_word: str = None
) -> None:
    """Process subdirectories containing videos.

    Args:
        parent_dir      (str): The parent dir containing sub dirs with videos.
        include_word    (str): Only include video files containing this word.
        extension_name  (str): The name of the output dir.

    """

    # Check if the parent dir exists and is a dir
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"Directory '{parent_dir}' not found.")
    if not os.path.isdir(parent_dir):
        raise NotADirectoryError(f"'{parent_dir}' is not a directory.")

    # Create output dir as sibling of parent dir
    output_dir = os.path.join(f"{parent_dir}_{extension_name}", "videos")
    os.makedirs(output_dir, exist_ok=True)

    # Get sub dirs
    input_dirs = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
    if len(input_dirs) == 0:
        raise FileNotFoundError(f"No sub dirs found in '{parent_dir}'.")
    
    

def arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process videos, directories, or clusters"
    )
    parser.add_argument(
        "input_path", type=str, help="Path to the input video, directory, or cluster"
    )
    return parser.parse_args()


if __name__ == "__main__":
    pass
