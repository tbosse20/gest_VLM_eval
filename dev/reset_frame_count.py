import pandas as pd

def reset_frame_count(csv, column):
    """ Zero set frames in a given column of a CSV file relative to the lowest frame in that set. """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv)
        
    # Get set from video name
    video_names = df['video_name'].unique()
    
    # Iterate through each video name
    for video_name in video_names:
        # Get the rows corresponding to the current video name
        video_rows = df[df['video_name'] == video_name]
        
        # Find the minimum frame number in the current set of rows
        min_frame = video_rows[column].min()
        
        # Reset the frame numbers relative to the minimum frame number
        df.loc[video_rows.index, column] -= min_frame
        
    # Save the modified DataFrame back to a CSV file
    df.to_csv(csv, index=False)
    
if __name__ == "__main__":
    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Reset frame count in a CSV file.")
    
    # Add arguments for the CSV file and column name
    parser.add_argument("--csv", type=str, help="Path to the input CSV file.")
    parser.add_argument("--column", type=str, help="Column name to reset frame count.")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the function with the provided arguments
    reset_frame_count(args.csv, args.column)
    
    