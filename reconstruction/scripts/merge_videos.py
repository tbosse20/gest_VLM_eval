import os
import cv2
from ultralytics import YOLO
import numpy as np


def merge_videos_with_poses(ground_truth_video, reconstructed_video, output_video):
    """
    Merge two videos by blending their frames (0.5 each) while drawing their pose keypoints.

    For each pair of frames:
      - Runs YOLOv8 pose inference on both frames.
      - Draws the ground truth keypoints in green and reconstructed keypoints in red.
      - Blends the two frames (each with weight 0.5) to produce a merged frame.
      - Writes the merged frame to an output video.

    Args:
        ground_truth_video (str): Path to the ground truth video.
        reconstructed_video (str): Path to the reconstructed video.
        output_video (str): Path for the output merged video.

    """

    # Video files raises
    if not os.path.exists(ground_truth_video):
        raise FileNotFoundError(f"Ground truth video not found: {ground_truth_video}")
    if not os.path.exists(reconstructed_video):
        raise FileNotFoundError(f"Reconstructed video not found: {reconstructed_video}")
    if not os.path.isfile(ground_truth_video):
        raise ValueError(f"Ground truth video is not a file: {ground_truth_video}")
    if not os.path.isfile(reconstructed_video):
        raise ValueError(f"Reconstructed video is not a file: {reconstructed_video}")

    # Create output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_video)):
        os.makedirs(os.path.dirname(output_video))

    # Load the YOLOv8 pose model (only once)
    model = YOLO("yolov8n-pose.pt")

    # Open both videos
    cap_gt = cv2.VideoCapture(ground_truth_video)
    cap_rec = cv2.VideoCapture(reconstructed_video)
    # Ensure both videos are opened successfully
    if not cap_gt.isOpened() or not cap_rec.isOpened():
        raise ValueError("Could not open one of the video files.")

    # Ensure both videos have the same number of frames
    frame_count_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_rec = int(cap_rec.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count_gt != frame_count_rec:
        raise ValueError(
            f"Video lengths do not match: {frame_count_gt} frames (GT) vs {frame_count_rec} frames (Reconstructed)"
        )

    # Get video properties (assuming both videos have the same FPS and frame size)
    fps = cap_gt.get(cv2.CAP_PROP_FPS)
    width = int(cap_gt.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_gt.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while True:
        ret_gt, frame_gt = cap_gt.read()
        ret_rec, frame_rec = cap_rec.read()
        if not ret_gt or not ret_rec:
            break

        # Run pose estimation on both frames
        result_gt = model(frame_gt, verbose=False)
        result_rec = model(frame_rec, verbose=False)
        # keypoints_gt = extract_keypoints(result_gt)
        # keypoints_rec = extract_keypoints(result_rec)

        # Create copies for drawing (so original frames remain unchanged)
        frame_gt_draw = frame_gt.copy()
        frame_rec_draw = frame_rec.copy()
        # draw_pose(frame_gt_draw, keypoints_gt, color=(0, 255, 0))  # Green for ground truth
        # draw_pose(frame_rec_draw, keypoints_rec, color=(0, 0, 255))  # Red for reconstructed

        # Merge the two frames with equal weighting (0.5 each)
        merged_frame = cv2.addWeighted(frame_gt_draw, 0.5, frame_rec_draw, 0.5, 0)
        out.write(merged_frame)

        # Optionally display the merged frame live
        cv2.imshow("Merged Video with Poses", merged_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_gt.release()
    cap_rec.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    import sys
    import argparse

    argparse.ArgumentParser(description="Motion Reconstruction Evaluation")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth_folder",
        type=str,
        default="../data/actedgestures_original",
        help="Path to ground truth folder",
    )
    parser.add_argument(
        "--reconstruct_folder",
        type=str,
        default="../data/reconstruction_cut_sped",
        help="Path to reconstructed video folder",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="results/videos/merged_output.mp4",
        help="Path to output video file",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="video_06.MP4",
        help="Name of the video to merge",
    )
    args = parser.parse_args()

    # Example usage:
    """
    python merge_videos.py \
        --ground_truth_folder   ../data/actedgestures_original \
        --reconstruct_folder    ../data/reconstruction_cut_sped \
        --output_video          results/videos/merged_output.mp4 \
        --video_name            video_06.MP4
    """

    gt_video = os.path.join(args.ground_truth_folder, args.video_name)
    rec_video = os.path.join(args.reconstruct_folder, args.video_name)

    merge_videos_with_poses(gt_video, rec_video, args.output_video)
