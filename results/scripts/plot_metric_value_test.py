import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append(".")
from results.scripts.compare_captions import compute_similarity_metrics

def generate_metrics():
    ground_truth = "A person signals the ego driver to stop by putting their hand towards the ego driver."

    # Define test cases and their expected similarity levels
    test_cases = {
        "High": [
            ("A pedestrian raises their hand to stop traffic.", ground_truth),
            ("A person puts their hand towards the ego driver.", ground_truth),
        ],
        "Moderate": [
            ("A person makes a signal.", ground_truth),
            ("A pedestrian makes a gesture to communicate with the approaching vehicle.", ground_truth),
        ],
        "Low": [
            ("A cyclist gestures to indicate a turn.", ground_truth),
            ("A cyclist puts their hand out to the side.", ground_truth),
        ],
        "Very low": [
            ("The sky is blue and the sun is shining.", ground_truth),
            ("A pedestrian is walking on the sidewalk.", ground_truth),
        ],
        "empty pred": [
            ("", ground_truth),
        ],
    }

    # Define similarity metrics labels
    metric_labels = ["Consine", "Jaccard", "Bleu", "Meteor", "Rouge_L", "Bert"]
    case_labels = list(test_cases.keys())

    # Initialize dictionary to store similarity scores
    data = {case: [] for case in case_labels}

    # Compute similarity scores for each case category
    for case, pairs in test_cases.items():
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

# Generate similarity metrics
# data = generate_metrics()
# print(data)

# Hardcoded true values  
data = {
    'High': [0.73547599, 0.41372549, 0.18888481, 0.71478112, 0.485, 0.92691967],
    'Moderate': [0.52625793, 0.13392857, 0.0211007 , 0.1471846 , 0.25396825, 0.90059134],
    'Low': [0.42758176, 0.19444444, 0.0236687 , 0.32675047, 0.28695652, 0.89174458],
    'Very low': [0.1785877 , 0.07763158, 0.01346437, 0.11483753, 0.16695652, 0.86184731],
    'empty pred': [0., 0., 0., 0., 0., 0.]
    }
plot_data(data)