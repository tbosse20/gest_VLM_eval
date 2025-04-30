def describe_dataset(csv_path: str) -> None:
    """
    Load a CSV file and print its description.

    Args:
        csv_path (str): Path to the CSV file.
    """
    import pandas as pd

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Print the description of the dataset
    print(df.columns)
    print(df.shape)
    print(df.dtypes)
    
    # Count number of each class
    class_counts = df['gt_id'].value_counts()
    print(class_counts)

import sys
sys.path.append(".")
from config.directories import LABELS_CSV
describe_dataset(LABELS_CSV)