# %%
import os
import subprocess
import argparse


def get_duration(video_file: str) -> float:
    """Retrieve the duration of a video file in seconds using ffprobe.

    Args:
        video_file (str): Path to the video file.

    Returns:
        float: Duration of the video in seconds.
    """

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_file,
    ]

    # Run the command and capture the output.
    output = subprocess.check_output(cmd).strip()

    try:
        return float(output)
    except ValueError:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {output}")


def build_atempo_chain(speed_factor: float) -> str:
    """
    Build a chain of atempo filters such that each factor is within [0.5, 2.0].
    For example, a speed_factor of 0.4065 is broken into atempo=0.5,atempo=0.813 (0.5 * 0.813 ≈ 0.4065).

    Args:
        speed_factor (float): The speed factor to be applied.

    Returns:
        str: A string representing the filter chain for ffmpeg.
    """

    factors = []
    # If the speed factor is greater than 2.0, repeatedly apply atempo=2.0.
    while speed_factor > 2.0:
        factors.append(2.0)
        speed_factor /= 2.0

    # If the speed factor is less than 0.5, repeatedly apply atempo=0.5.
    while speed_factor < 0.5:
        factors.append(0.5)
        speed_factor /= 0.5

    # Append the remaining factor (if it's 1.0, it doesn't change the audio).
    factors.append(speed_factor)

    # Build the filter chain string.
    atempo_chain = ",".join([f"atempo={f:.6f}" for f in factors])
    return atempo_chain


def match_video_speed(edit_video: str, fixed_video: str) -> None:
    """
    Adjust the speed of the 'edit_video' to match the duration of the 'fixed_video'.

    Args:
        edit_video (str): Path to the edit video file.
        fixed_video (str): Path to the fixed video file.

    Returns:
        None: The function modifies the edit video in place.
    """

    # Raises
    if not os.path.exists(edit_video):
        raise FileNotFoundError(f"Edit video file not found: {edit_video}")
    if not os.path.exists(fixed_video):
        raise FileNotFoundError(f"Fixed video file not found: {fixed_video}")

    # Get durations of both videos.
    edit_duration = get_duration(edit_video)
    fixed_duration = get_duration(fixed_video)

    # Calculate the speed factor.
    speed_factor = edit_duration / fixed_duration
    # For video, 'setpts' uses the reciprocal factor.
    pts_factor = 1.0 / speed_factor

    # Build the appropriate chain for atempo so that each factor is within the allowed range.
    atempo_chain = build_atempo_chain(speed_factor)
    print(f"Using atempo chain for audio: {atempo_chain}")

    # Construct the filter_complex string.
    filter_complex = f"[0:v]setpts={pts_factor:.6f}*PTS[v];[0:a]{atempo_chain}[a]"

    # Create an output folder for the sped-up video.
    output_folder = os.path.dirname(edit_video) + "_sped"
    output_file = os.path.join(output_folder, os.path.basename(edit_video))
    os.makedirs(output_folder, exist_ok=True)

    # Remove the output file if it already exists.
    if os.path.exists(output_file):
        os.remove(output_file)

    # Build and run the ffmpeg command.
    cmd = [
        "ffmpeg",
        "-i",
        edit_video,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        output_file,
    ]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    output_duration = get_duration(output_file)
    if abs(output_duration - fixed_duration) > 1.0:
        os.remove(output_file)
        raise RuntimeError(
            f"Output video duration {output_duration}s differs from target duration {fixed_duration}s"
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Adjust video speed to match another video's duration."
    )
    parser.add_argument("--edit_video", help="Path to the edit video file.")
    parser.add_argument("--fixed_video", help="Path to the fixed video file.")
    args = parser.parse_args()

    # Example usage:
    """
    python match_video_length.py \
        --edit_video  = "../data/actedgestures_original/video_01.MP4" \
        --fixed_video = "../data/actedgestures_original/video_02.MP4"
    """

    # Adjust the speed of the edit video to match the fixed video.
    match_video_speed(args.edit_video, args.fixed_video)
