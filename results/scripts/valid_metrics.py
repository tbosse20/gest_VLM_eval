import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(".")
from results.src.compute_similarity_metrics import compute_similarity_metrics

def validate_metrics():
    ground_truths = [
        "A pedestrian signals the ego driver to stop, by putting their hand towards the ego driver.",
        "A person signals the driver to stop, by raising their hand towards the camera.",
    ]

    # Define test cases and their expected similarity levels
    valid_levels = {
        "Extended": [
            "A pedestrian raises their hand towards the ego driver to stop traffic. They are looking scared and in need of help.",
            "A person puts their hand towards the ego driver to signal 'stop'. They are wearing a red t-shirt and blue pants.",
        ],
        "Equivalent": [
            "A pedestrian raises their hand towards the ego driver to stop traffic.",
            "A person puts their hand towards the ego driver to signal 'stop'.",
        ],
        "Partial": [
            "A person raises their hand towards the ego driver.",
            "A pedestrian signals the ego driver to stop.",
        ],
        "Slight": [
            "A human gestures to the ego driver.",
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
    metric_labels = ["Cosine", "Jaccard", "BLEU", "METEOR", "ROUGE", "BERT\nScore", "STS"]
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
        'Extended': [0.9148297756910324, 0.2054273528575179, 0.75039142370224, 0.5629934519529343, 0.3082763532763533, 0.4230316873015132, 0.38241888505046395],
'Equivalent': [0.9383604675531387, 0.30044538214535127, 0.8063625693321228, 0.747789740562439, 0.45833333333333337, 0.7091953269344393, 0.5164835164835165],
'Partial': [0.9306317269802094, 0.23553086674620854, 0.7659641951322556, 0.6532697975635529, 0.3979166666666667, 0.6595469121936446, 0.5952042160737813],
'Slight': [0.9089650213718414, 0.03969517740269712, 0.5654146174589793, 0.47863704959551495, 0.2632080610021787, 0.4819230148130227, 0.38497082627517404],
'Unrelated': [0.8736235350370407, 0.018089598634690156, 0.2107335738837719, 0.07154581230133772, 0.09510233918128655, 0.16542852906926342, 0.19631469979296065],}
        plot_data(data)
    
    