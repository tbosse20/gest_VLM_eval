import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(".")
from results.src.compute_similarity_metrics import compute_similarity_metrics

def validate_metrics():
    ground_truths = [
        "A pedestrian signals the ego driver to stop by putting their hand towards the ego driver.",
        "A person signals the driver to stop by raising their hand towards the camera.",
    ]

    # Define test cases and their expected similarity levels
    valid_levels = {
        "Extended": [
            "A pedestrian raises their hand towards the ego driver hand to stop traffic. They are looking scared and in need of help.",
            "A person puts their hand towards the ego driver to signal 'stop'. They are wearing a red t-shirt and blue pants.",
        ],
        "Equivalent": [
            "A pedestrian raises their hand towards the ego driver hand to stop traffic.",
            "A person puts their hand towards the ego driver to signal 'stop'.",
        ],
        "Partial": [
            "A person raises their towards the ego driver.",
            "A pedestrian signals the ego driver to stop.",
        ],
        "Slight": [
            "A human gestures the ego driver.",
            "A person puts their hand out to the side.",
            "A pedestrian puts their hand up.",
        ],
        "Unrelated": [
            "The sky is blue and the sun is shining.",
            "A pedestrian is walking on the sidewalk.",
        ]
    }

    # Define similarity metrics labels
    metric_labels = ["Consine", "Jaccard", "Bleu", "Meteor", "Rouge_L", "Bert", "Cross"]
    metric_labels.sort()
    level_labels = list(valid_levels.keys())

    # Initialize dictionary to store similarity scores
    data = {case: [] for case in level_labels}

    # Compute similarity scores for each level category
    for case, valid_captions in tqdm(valid_levels.items(), desc="Valid. levels"):
        total_scores = np.zeros(len(metric_labels))
        count = 0
        
        # Compute similarity scores for each valid caption
        for valid_caption in valid_captions:
            for ground_truth in ground_truths:
                similarity_scores = compute_similarity_metrics(valid_caption, ground_truth)
                total_scores += similarity_scores.values[0]
                count += 1

        # Average the scores
        data[case] = total_scores / max(1, count)

    return data

def plot_data(data):
    
    # Define metric labels
    metric_labels = ["Consine", "Jaccard", "Bleu", "Meteor", "Rouge_L", "Bert", "Cross"]
    # Sort metrics by alphabetical order
    metric_labels.sort()
    
    # Convert to DataFrame
    df = pd.DataFrame(data, index=metric_labels)

    # Plot setup
    fig, ax = plt.subplots(figsize=(7, 3))
    bar_width = 0.15  # Width of each bar
    x = np.arange(len(metric_labels))  # X-axis positions for metrics

    # Define colors for each case
    colors = [
        "#4e79a7",  # Deep Blue
        "#f28e2b",  # Warm Orange
        "#e15759",  # Reddish-Pink
        "#76b7b2",  # Teal
        "#59a14f",  # Green
    ]

    # Plot bars for each case
    for i, (case, color) in enumerate(zip(df.columns, colors)):
        ax.bar(x + i * bar_width, df[case], bar_width, label=case, color=color)
    
    # Labels and formatting
    ax.set_xlabel("Metrics", fontstyle="italic")
    ax.set_ylabel("Score", fontstyle="italic")
    # ax.set_title("Similarity Metrics Analysis")
    ax.set_xticks(x + bar_width * (len(df.columns) / 2 - 0.5))
    ax.set_xticklabels(metric_labels)
    ax.legend(
        title="Similarity", 
        bbox_to_anchor=(1.05, 0.5), 
        loc='center left', 
        title_fontproperties={'weight': 'bold'})
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.show()

    # Save plot to file
    plt.savefig("results/figures/valid_metrics.png")


if __name__ == "__main__":
    
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot similarity metrics")
    parser.add_argument("--compute", action="store_true", help="Compute similarity metrics")
    parser.add_argument("--plot",    action="store_true", help="Plot similarity metrics")
    args = parser.parse_args()
    
    if not args.compute and not args.plot:
        print("Please specify either --compute or --plot")
        sys.exit(1)
    
    # Generate similarity metrics
    if args.compute:
        data = validate_metrics()
        # Print the dictionary
        for key, value in data.items():
            print(f"'{key}': {list(value)},")
        plot_data(data)
    
    if args.plot:
        # Hardcoded true values (to avoid recomputing)
        data = {
            'Extended': [0.9086198061704636, 0.19185341732200978, 0.7435819804668427, 0.33384615384615385, 0.4139981075491923, 0.3771929824561403],
            'Equivalent': [0.9291751235723495, 0.29055359236162226, 0.8042532205581665, 0.5026552287581699, 0.6854715981118725, 0.507283903835628],
            'Partial': [0.9202924966812134, 0.15804108301809655, 0.7440546303987503, 0.3666666666666667, 0.6224947550373215, 0.5643939393939393],
            'Slight': [0.901681125164032, 0.0387676318991606, 0.5667633761962255, 0.2429874727668845, 0.4691581735485537, 0.37549407114624506],
            'Unrelated': [0.8713605552911758, 0.018089598634690156, 0.20903201028704643, 0.09510233918128655, 0.16542852906926342, 0.19631469979296065],
        }
        plot_data(data)
    
    