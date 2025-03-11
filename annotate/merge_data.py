import pandas as pd

def merge(keypoints_csv: str, classes_csv: str, signal_csv: str, exp: bool = False):
    """ Merge pose keypoints and annotations classes. Only works for individual poses. """
    
    keypoints_df = pd.read_csv(keypoints_csv)
    classes_df = pd.read_csv(classes_csv)
    
    # Merge on video name and frame
    signal_df = classes_df.merge(keypoints_df, on=["video_name", "frame_idx"])
    
    # Save to csv
    signal_df.to_csv(signal_csv, mode="w", index=False, header=True)
    
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    # import main.signal.config as config
    
    # merge(config.keypoints_csv, config.classes_csv, config.signal_csv, config.sanity)