import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append(".")
from results.scripts.compare_captions import compute_similarity_metrics
from tqdm import tqdm

def generate_metrics():
    ground_truth = "A person signals the ego driver to stop by putting their hand towards the ego driver."

    # Define test cases and their expected similarity levels
    test_cases = {
        "Identical": [
            (ground_truth, ground_truth),
        ],
        "High": [
            ("A pedestrian raises their towards the ego driver hand to stop traffic.", ground_truth),
            ("A person puts their hand towards the ego driver to signal 'stop'.", ground_truth),
        ],
        "Moderate": [
            ("A person raises their towards the ego driver.", ground_truth),
            ("A pedestrian signals the ego driver to stop.", ground_truth),
        ],
        "Low": [
            ("A human gestures the ego driver.", ground_truth),
            ("A person puts their hand out to the side.", ground_truth),
            ("A pedestrian puts their hand up.", ground_truth),
        ],
        "Very low": [
            ("The sky is blue and the sun is shining.", ground_truth),
            ("A pedestrian is walking on the sidewalk.", ground_truth),
        ]
    }

    # Define similarity metrics labels
    metric_labels = ["Consine", "Jaccard", "Bleu", "Meteor", "Rouge_L", "Bert"]
    case_labels = list(test_cases.keys())

    # Initialize dictionary to store similarity scores
    data = {case: [] for case in case_labels}

    # Compute similarity scores for each case category
    for case, pairs in tqdm(test_cases.items(), desc="Proc. cases"):
        scores = np.zeros(len(metric_labels))  # Initialize with zeros
        for pred, gt in pairs:
            if pred and gt:
                similarity_scores = compute_similarity_metrics(pred, gt)
                scores += similarity_scores.values[0]  # Aggregate scores
        data[case] = scores / max(1, len(pairs))  # Average over test cases

    return data

def plot_data(data):
    
    metric_labels = ["Consine", "Jaccard", "Bleu", "Meteor", "Rouge_L", "Bert"]

    # Convert to DataFrame
    df = pd.DataFrame(data, index=metric_labels)

    # Plot setup
    fig, ax = plt.subplots(figsize=(7, 3))
    bar_width = 0.15  # Width of each bar
    x = np.arange(len(metric_labels))  # X-axis positions for metrics

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Plot bars for each case
    for i, (case, color) in enumerate(zip(df.columns, colors)):
        ax.bar(x + i * bar_width, df[case], bar_width, label=case, color=color)

    # Labels and formatting
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("Similarity Metrics Analysis")
    ax.set_xticks(x + bar_width * (len(df.columns) / 2 - 0.5))
    ax.set_xticklabels(metric_labels)
    ax.legend(title="Cases", bbox_to_anchor=(1.05, 0.5), loc='center left')

    plt.tight_layout()
    # plt.show()

    # Save plot to file
    plt.savefig("results/figures/similarity_metrics.png")


if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot similarity metrics")
    args = parser.parse_args()
    parser.add_argument("--compute", action="store_true", help="Compute similarity metrics")
    parser.add_argument("--plot",    action="store_true", help="Plot similarity metrics")
    
    args = parser.parse_args()
    
    # Generate similarity metrics
    if args.compute:
        data = generate_metrics()
        # Print the dictionary
        for key, value in data.items():
            print(f"'{key}': {list(value)},")
    
    elif args.plot:
        # Hardcoded true values (to avoid recomputing)
        data = {
            'Identical': [0.9999998807907104, 1.0, 1.0, 0.9998779296875, 1.0, 1.0],
            'High': [0.8912810385227203, 0.5294117647058824, 0.24841392812956903, 0.7420243490624723, 0.5357142857142857, 0.9206320345401764],  
            'Moderate': [0.8548151850700378, 0.42083333333333334, 0.21442117372545666, 0.6724773242630386, 0.5833333333333334, 0.9169631004333496],
            'Low': [0.6131544411182404, 0.25980392156862747, 0.04497292511802876, 0.5752507188442387, 0.4024242424242425, 0.9014791448911031],  
            'Very low': [0.17858769744634628, 0.07763157894736841, 0.013464374971060651, 0.11483753099308364, 0.16695652173913045, 0.861847311258316],
        }
        plot_data(data)