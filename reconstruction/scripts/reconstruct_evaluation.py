import os
from ultralytics import YOLO
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess  # if needed for ffmpeg integration


def get_pose_estimation(video_path):
    """Get pose estimation from video using YOLOv8 pose model

    Args:
        video_path (str): Path to video

    Returns:
        list: Pose estimation results, each item corresponds to a frame and includes keypoints.
              (Each frame is a list of numpy arrays; each array contains keypoints for one detected person.)
    """
    # Load YOLOv8 pose model
    model = YOLO("yolov8n-pose.pt")  # or yolov8s-pose.pt, etc.

    # Open the video
    cap = cv2.VideoCapture(video_path)
    results = []

    video_name = os.path.basename(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)

            # Run inference
            pose_result = model(frame, verbose=False)

            # Extract keypoints for each detected person
            frame_poses = []
            # Using the same approach as below: iterate over the keypoints in the first prediction.
            if (
                pose_result
                and hasattr(pose_result[0], "keypoints")
                and pose_result[0].keypoints is not None
            ):
                for pred in pose_result[0].keypoints.xy:

                    points = pred.cpu().numpy()
                    if points.size == 0:
                        continue

                    frame_poses.append(points)

            if frame_poses:
                results.append(frame_poses)

    cap.release()
    return results


def compare_poses(reconstruct_poses, ground_truth_poses):
    """
    Compare poses between reconstructed and ground truth frames.

    Args:
        reconstruct_poses (list): List of pose keypoints from reconstructed video.
        ground_truth_poses (list): List of pose keypoints from ground truth video.

    Returns:
        list: List of frame-by-frame comparison results (e.g., average error per frame).
    """
    if len(reconstruct_poses) == 0 or len(ground_truth_poses) == 0:
        return []

    comparison_results = []
    for recon_frame, gt_frame in zip(reconstruct_poses, ground_truth_poses):
        frame_errors = []

        # Naively assume same number and order of detections per frame
        for recon_person, gt_person in zip(recon_frame, gt_frame):
            if recon_person.shape != gt_person.shape:
                continue  # skip if keypoint shape doesn't match

            # Calculate Euclidean distance for each keypoint
            distances = np.linalg.norm(recon_person - gt_person, axis=1)
            mean_error = np.mean(distances)
            frame_errors.append(mean_error)

        # Sum or average the errors per frame as needed
        comparison_results.append(np.sum(frame_errors) if frame_errors else None)

    return comparison_results


def plot_comparison_results(
    comparison_results: np.ndarray, comparison_names: list
) -> None:
    """Plot comparison results using seaborn violin plot

    Args:
        comparison_results (list):  List of lists of errors.
        comparison_names (list):    Corresponding labels.

    Returns:
        None: Saves the plot in the results/figures folder.

    """
    flat_data = []
    labels = []
    for result, name in zip(comparison_results, comparison_names):
        for val in result:
            flat_data.append(val)
            labels.append(name)
    df = pd.DataFrame({"Error": flat_data, "Version": labels})

    # Define colors for each case
    colors = [
        "#f28e2b",  # Warm Orange
        "#e15759",  # Reddish-Pink
        "#76b7b2",  # Teal
        "#59a14f",  # Green
    ]

    # Rename name
    df["Version"] = df["Version"].replace(
        {
            "qwen": "Qwen",
            "vllama2": "VLLaMA2",
            "vllama3": "VLLaMA3",
            "ground_truth": "Ground Truth",
        }
    )

    plt.figure(figsize=(7.16, 2.5))

    sns.violinplot(
        x="Version",
        y="Error",
        data=df,
        width=0.7,
        palette=colors,
    )

    # Add grid lines
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.gca().set_axisbelow(True)

    # Adjust axes labels
    plt.xlabel("Model", fontstyle="italic")
    plt.ylabel("Average Pose Error", fontstyle="italic")

    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(
        "results/figures/reconstruction_plot.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def reconstruct_evaluation(ground_truth_folder, reconstruct_folder):
    """Compares pose estimation results from reconstructed videos with ground truth data

    Args:
        ground_truth_folder (str): Path to ground truth folder.
        reconstruct_folder (str): Path to reconstructed video folder.
    """

    if not os.path.exists(ground_truth_folder):
        raise FileNotFoundError(f"Ground truth folder not found: {ground_truth_folder}")
    if not os.path.isdir(ground_truth_folder):
        raise NotADirectoryError(
            f"Ground truth folder is not a directory: {ground_truth_folder}"
        )
    if not os.path.exists(reconstruct_folder):
        raise FileNotFoundError(f"Reconstructed folder not found: {reconstruct_folder}")
    if not os.path.isdir(reconstruct_folder):
        raise NotADirectoryError(
            f"Reconstructed folder is not a directory: {reconstruct_folder}"
        )

    # Load info CSV file that contains video mapping information
    info_csv = os.path.join(reconstruct_folder, "info.csv")
    if not os.path.exists(info_csv):
        raise FileNotFoundError(f"Info CSV not found in: {reconstruct_folder}")
    df = pd.read_csv(info_csv)

    ground_truth_videos = df["ground_truth_video"].unique()
    for video in ground_truth_videos:
        comparison_results = []
        comparison_names = []
        ground_truth_video_path = os.path.join(ground_truth_folder, video)
        if not os.path.exists(ground_truth_video_path):
            print(f"Ground truth video not found: {ground_truth_video_path}")
            continue
        ground_truth_poses = get_pose_estimation(ground_truth_video_path)

        for index, row in df[df["ground_truth_video"] == video].iterrows():
            reconstruct_video_path = os.path.join(
                reconstruct_folder, row["reconstruct_video"]
            )

            if not os.path.exists(reconstruct_video_path):
                print(f"Reconstructed video not found: {reconstruct_video_path}")
                continue

            reconstruct_poses = get_pose_estimation(reconstruct_video_path)
            comparison_result = compare_poses(reconstruct_poses, ground_truth_poses)
            comparison_results.append(comparison_result)
            comparison_names.append(row["version"])

        video_name = os.path.splitext(os.path.basename(video))[0]
        np.save(
            f"results/data/reconstruct/comparison_results_{video_name}.npy",
            np.array(comparison_results, dtype=object),
        )

        plot_comparison_results(comparison_results, comparison_names)


def draw_pose(frame, keypoints_list, color=(0, 255, 0), radius=10):
    """
    Draw circles at each keypoint location on the frame.

    Args:
        frame (np.array): The image/frame on which to draw.
        keypoints_list (list): List of numpy arrays, each representing keypoints for one person.
        color (tuple): BGR color tuple for the keypoints.
    """
    for keypoints in keypoints_list:
        for point in keypoints:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), radius, color, -1)


def extract_keypoints(result):
    """
    Helper function to extract keypoints from a YOLO result.

    Args:
        result: YOLO inference result for one frame.

    Returns:
        list: A list of keypoints arrays.
    """
    poses = []
    if result and hasattr(result[0], "keypoints") and result[0].keypoints is not None:
        for pred in result[0].keypoints.xy:
            poses.append(pred.cpu().numpy())
    return poses




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
        "--run_evaluation", action="store_true", help="Run pose comparison evaluation"
    )
    parser.add_argument(
        "--plot_results",
        action="store_true",
        help="Plot comparison results",
    )
    args = parser.parse_args()

    # Run pose comparison evaluation and plot results
    if args.run_evaluation:
        reconstruct_evaluation(args.ground_truth_folder, args.reconstruct_folder)

    elif args.plot_results:
        
        # Load the comparison results from the saved numpy file
        comparison_results = np.load(
            "results/data/reconstruct/comparison_results_video_06.npy",
            allow_pickle=True,
        )
        
        # Code names
        comparison_names = [
            "qwen",
            "vllama2",
            "vllama3",
            "ground_truth",
        ]
        
        # Plot the comparison results
        plot_comparison_results(comparison_results, comparison_names, method="save")

    else:
        print("Please specify an action: --run_evaluation, --plot_results.")
        sys.exit(1)
