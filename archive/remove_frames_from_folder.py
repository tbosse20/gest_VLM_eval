from tqdm import tqdm
import os
import shutil

def remove_frames_from_folder(frames_main_folder:str, frame_rate:int):
    
    # Validate input folder
    if not os.path.exists(frames_main_folder):
        raise FileNotFoundError(f"Input folder '{frames_main_folder}' not found.")
    if not os.path.isdir(frames_main_folder):
        raise NotADirectoryError(f"Input folder '{frames_main_folder}' is not a directory")

    # Make sibling main folder
    main_folder_name = os.path.basename(frames_main_folder)
    main_path = os.path.dirname(frames_main_folder)
    sibling_folder = os.path.join(main_path, f"{main_folder_name}_rate_{frame_rate}")
    os.makedirs(sibling_folder, exist_ok=True)
    
    # Get sub folders
    subfolders = [f.path for f in os.scandir(frames_main_folder) if f.is_dir()]
    
    # Iterate over subfolders
    for subfolder in subfolders:
        
        # Get video name
        video_name = os.path.basename(subfolder)
        # Make folder in sibling folder
        sibling_subfolder = os.path.join(sibling_folder, video_name)
        os.makedirs(sibling_subfolder, exist_ok=True)
    
        # Get frame indices safely
        frame_files = [f for f in os.listdir(subfolder) if f.startswith("frame_") and f.endswith(".png")]
        frame_idx = sorted([int(f.split("_")[-1].split(".")[0]) for f in frame_files])

        if not frame_idx:  # Skip empty folders
            continue

        # Select frames using slicing
        selected_frames = frame_idx[::frame_rate]

        # Copy only selected frames
        for i in tqdm(selected_frames, desc=f"Processing {video_name}"):
            # Get frame filenames
            frame_filename = os.path.join(subfolder, f"frame_{i:04d}.png")
            sibling_frame_filename = os.path.join(sibling_subfolder, f"frame_{i:04d}.png")

            # Copy frame if it exists
            if os.path.exists(frame_filename):
                shutil.copy(frame_filename, sibling_frame_filename)
            
if __name__ == "__main__":
    frames_main_folder = "../video_frames"
    frame_rate = 3 # 3/30fps = 10fps
    remove_frames_from_folder(frames_main_folder, frame_rate)
                