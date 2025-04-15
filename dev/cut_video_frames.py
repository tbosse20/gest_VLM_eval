import os
import sys

sys.path.append(".")
from video_edit import cut_video_time, get_fps


def cut_video_frames(input_file: str, start_frame: int, end_frame: int) -> None:

    # File raises
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    if not os.path.isfile(input_file):
        raise NotADirectoryError(f"'{input_file}' is not a file.")

    # Frame raises
    if start_frame < 0:
        raise ValueError(f"Start frame '{start_frame}' cannot be negative.")
    if end_frame < 0:
        raise ValueError(f"End frame '{end_frame}' cannot be negative.")
    if end_frame <= start_frame:
        raise ValueError(
            f"End frame '{end_frame}' must be greater than start frame '{start_frame}'."
        )

    fps = get_fps(input_file)
    print(f"FPS: {fps}")

    start_time = start_frame / fps
    end_time = end_frame / fps
    cut_video_time(input_file, start_time, end_time)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Cut video frames.")
    parser.add_argument("--input_file", type=str, help="Path to the input video file.")
    parser.add_argument("--start_frame", type=int, help="Start frame number.")
    parser.add_argument("--end_frame", type=int, help="End frame number.")
    args = parser.parse_args()

    # Example usage
    """ 
    python cut_video_frames.py \
        --input_file  = "../data/actedgestures_original/video_01.MP4" \
        --start_frame = 36 \
        --end_frame   = 68 + 8
    """

    cut_video_frames(args.input_file, args.start_frame, args.end_frame)
