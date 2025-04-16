import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.patches as mpatches
sys.path.append(".")
from scripts.src.validation_sentences import VALID_LEVELS, TARGETS, IDEAL

def validate_metrics() -> dict:
    """ Validate similarity metrics for single scenario with multiple targets and validations. 
    
    Find the sentences and idea in 'validation_sentences.py'.
    
    Returns:
        data (dict): Dictionary containing similarity scores for each level.
    """
    
    from scripts.src.compute_similarity_metrics import compute_similarity_metrics

    # Define similarity metrics labels
    metric_labels = ["Cosine", "Jaccard", "Bleu", "Meteor", "Rouge_L", "Bert", "Cross"]
    metric_labels.sort()
    level_labels = list(VALID_LEVELS.keys())

    # Initialize dictionary to store similarity scores
    data = {case: [] for case in level_labels}

    # Compute similarity scores for each level category
    for case, valid_captions in tqdm(VALID_LEVELS.items(), desc="Valid. levels"):
        total_scores = np.zeros(len(metric_labels))
        count = 0
        
        # Compute similarity scores for each valid caption
        for valid_caption in valid_captions:
            for target in TARGETS:
                similarity_scores = compute_similarity_metrics(valid_caption, target, metric_labels)
                total_scores += similarity_scores.values[0]
                count += 1

        # Average the scores
        data[case] = total_scores / max(1, count)

    return data

def plot_metrics(data):
    """ Plot similarity metrics validation. """
    
    if len(data) == 0:
        print("No data to plot")
        return
    
    # Define metric labels
    METRIC_LABELS = ["Cosine", "Jaccard", "BLEU", "METEOR", "ROUGE", "BERT\nScore", "STS"]
    # Define colors for each case
    COLORS = [
        "#4e79a7",  # Deep Blue
        "#f28e2b",  # Warm Orange
        "#e15759",  # Reddish-Pink
        "#76b7b2",  # Teal
        "#59a14f",  # Green
    ]
    BAR_WIDTH = 0.15  # Width of each bar
    
    # Sort metrics by alphabetical order
    METRIC_LABELS.sort()
    
    # Add ideal values to the data if not already present
    if len(data["Extended"]) != len(METRIC_LABELS):
        METRIC_LABELS = ["Ideal"] + METRIC_LABELS
    
    # Check if the data length matches the metric labels
    if len(data["Extended"]) != len(METRIC_LABELS):
        print("Data length does not match metric labels")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data, index=METRIC_LABELS)

    # Plot setup
    fig, ax = plt.subplots(figsize=(7.16, 2.5))
    x = np.arange(len(METRIC_LABELS))  # X-axis positions for metrics
    
    # Store handles for the custom legend
    legend_handles = []
    # Plot bars for each case
    for i, (case, color) in enumerate(zip(df.columns, COLORS)):
        for j, metric in enumerate(df.index):
            tmp_case = case if j == 0 else ""
            bar_alpha = 0.6 if metric == "Ideal" else 1.0
            ax.bar(
                x[j] + i * BAR_WIDTH,
                df[case].iloc[j],
                BAR_WIDTH,
                label=tmp_case,
                color=color,
                alpha=bar_alpha
            )

        # Add proxy legend entry (alpha=1.0)
        legend_handles.append(mpatches.Patch(color=color, label=case, alpha=1.0))
    
    # Add ideal line
    ax.axvline(x=0.8, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel("Metrics", fontstyle="italic")
    ax.set_ylabel("Score", fontstyle="italic")
    
    # Set y-axis limits
    ax.legend(
        title="Similarity", 
        bbox_to_anchor=(1.01, 0.5), 
        loc='center left', 
        title_fontproperties={'weight': 'bold'},
        handles=legend_handles
        )
    
    # Set x-ticks and labels
    ax.set_xticks(x + BAR_WIDTH * (len(df.columns) / 2 - 0.5))
    ax.set_xticklabels(METRIC_LABELS)
    for label in ax.get_xticklabels():
        if label.get_text() != "Ideal": continue
        label.set_fontstyle("italic")
    ax.set_xlim(-0.3, len(METRIC_LABELS) - 1 + BAR_WIDTH * (len(df.columns) + 1.0))
    
    # Set grid
    ax.minorticks_on()
    ax.grid(which='minor', axis='y', linestyle=':', alpha=0.3)
    ax.grid(which='major', axis='y', linestyle='--', alpha=0.7)    
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig("results/figures/valid_metrics.pdf", format="pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    
    VALID_CSV_PATH = "results/data/valid_metrics.csv"
    
    import argparse
    import sys
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot similarity metrics")
    parser.add_argument("--compute", action="store_true", help="Compute similarity metrics")
    parser.add_argument("--plot",    action="store_true", help="Plot similarity metrics")
    args = parser.parse_args()
    
    # Example usage:
    """ 
    python scripts/valid_metrics.py --compute
    """
    
    if not args.compute and not args.plot:
        print("Please specify either --compute or --plot")
        sys.exit(1)
        
    # Generate similarity metrics
    if args.compute:
        data = validate_metrics()
        
        # Save to CSV
        df = pd.DataFrame(data, index=VALID_LEVELS.keys())
        df.to_csv(VALID_CSV_PATH, index=True, header=True)
        
    if args.plot:
        
        # Load data
        df = pd.read_csv(VALID_CSV_PATH, index_col=0)
        data = df.to_dict(orient="list")
        
    # Add the ideal for each case
    for case in data.keys():
        data[case] = [IDEAL[case]] + data[case]
        
    plot_metrics(data)