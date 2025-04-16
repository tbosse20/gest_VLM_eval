# %%
import subprocess
import os


def concat_videos(input_folder: str, video_list: str) -> None:
    """ Concatenate videos using ffmpeg.
    
    Args:
        input_folder (str): Path to the folder containing input videos.
        video_list   (list): List of video files to concatenate.
        
    Returns:
        None: A new video file is created in the input folder.
    """
    
    # Create a file list for ffmpeg
    list_file = "file_list.txt"
    with open(list_file, "w") as f:
        for video in video_list:
            f.write(f"file '{input_folder}/{video}'/n")

    # Make output file
    output_file = os.path.basename(video_list[0]) + "_concat.mp4"
    if os.path.exists(output_file):
        raise FileExistsError(
            f"Output file '{output_file}' already exists. Please remove it or choose a different name."
        )

    # Run ffmpeg command
    command = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c",
        "copy",
        output_file,
    ]
    subprocess.run(command, check=True)

    # Remove the file list
    os.remove(list_file)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Concatenate videos using ffmpeg.")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing input videos.",
    )
    parser.add_argument(
        "--video_list",
        nargs="+",
        required=True,
        help="List of video files to concatenate.",
    )
    args = parser.parse_args()

    # Example usage
    """
    python concat_videos.py \
        --input_folder "D:/realworldgestures_concat_cut" \
        --video_list "2025-03-15_13-15-15 (6).mp4" "2025-03-15_13-23-36 (6).mp4"
    """

    concat_videos(
        input_folder=args.input_folder,
        video_list=args.video_list,
    )
