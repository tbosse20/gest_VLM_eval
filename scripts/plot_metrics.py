import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(".")
from config.prompts import prompts

def plot_metrics(metrics_folder, prompt_types=False, gestures=False, baseline=False) -> pd.DataFrame:
    """ 
    Plot similarity metrics from CSV files in the specified folder.
    
    Args:
        metrics_folder (str): Path to the folder containing metrics CSV files.
        prompt_types (bool): If True, group by prompt types.
        gestures (bool): If True, group by gestures.
        baseline (bool): If True, include baseline models.
        
    Returns:
        pd.DataFrame: Merged DataFrame containing all metrics data.
    """
    
    # Check if the folder exists
    if not os.path.exists(metrics_folder):
        raise FileNotFoundError(f"Folder '{metrics_folder}' not found.")
    if not os.path.isdir(metrics_folder):
        raise NotADirectoryError(f"'{metrics_folder}' is not a folder.")
    
    # Get list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(metrics_folder, "*.csv"))

    # Check if any CSV files were found
    if not csv_files:
        print("No CSV files found in the folder.")
        exit()

    # Create an empty list to store data
    data_list = []
    
    # Loop through each CSV file
    for file in csv_files:
        
        # Load the CSV file
        df = pd.read_csv(file)
        
        # Drop unnecessary columns
        df = df.drop(columns=["frame_idx", "prompt"], errors="ignore")
        
        # Store the model name
        model_name = os.path.basename(file).split(".")[0]
        
        # Reshape DataFrame for boxplot (long format) # Keep prompt_type
        df_melted = df.melt(id_vars=["prompt_type", "video_name"], var_name="Metric", value_name="Score")
        df_melted["Model"] = model_name  # Add model column

        # Append to list
        data_list.append(df_melted)

    # Combine all data into a single DataFrame
    merged_df = pd.concat(data_list, ignore_index=True)
    merged_df["Metric"] = merged_df["Metric"].str.title()
    
    # Print number of samples for each model pr metric
    count = merged_df.groupby(["Model", "Metric"]).size()
    # Format count to a table
    count = count.unstack().fillna(0).astype(int)
    print(count)
    
    # Rename specific video names
    merged_df["video_name"] = merged_df["video_name"].replace({
        "Go left": "Left",
        "Stop + pass": "Stop, pass",
        "Stop + drive": "Stop, go",
        "Go forward": "Forward",
        "Getting a cap": "Hail",
    })
    
    # --- PLOTTING GROUPED BOX PLOT ---

    # Plot
    if not gestures and not prompt_types:
        plt.figure(figsize=(3, 2.5))
    else:
        plt.figure(figsize=(7.16, 2.5))
    
    # Define colors for each case
    colors = [
        "#f28e2b",  # Warm Orange
        "#e15759",  # Reddish-Pink
        "#76b7b2",  # Teal
        "#59a14f",  # Green
    ]
    # Add deep blue for non-human models
    if "human" not in metrics_folder:
        colors.insert(0, "#4e79a7")  # Deep Blue

    # Only keep cos, Jaccard, Bleu, and meteor
    merged_df = merged_df[merged_df["Metric"].isin([
        "Cosine",
        # "Jaccard",
        # "Bleu",
    ])]
    # Sort the metrics
    merged_df["Metric"] = pd.Categorical(merged_df["Metric"], sorted(merged_df["Metric"].unique()))
    
    # Make all values positive
    merged_df["Score"] = merged_df["Score"].abs()
    
    # Print min and max scores
    min_score = merged_df["Score"].min()
    max_score = merged_df["Score"].max()
    print(f"Min score: {min_score:.2f}")
    print(f"Max score: {max_score:.2f}")
    
    # Rename vllama
    merged_df["Model"] = merged_df["Model"].replace({
        "vllama2": "VLLaMA2",
        "vllama3": "VLLaMA3",
        "human": "Expert",
        "qwen": "Qwen",
    })
    
    # Boxplot: Group by prompt type, color by model
    if prompt_types or gestures:
        
        if prompt_types:
            # Sort prompt types
            tmp_prompt_types = prompts.keys()
            merged_df["prompt_type"] = pd.Categorical(merged_df["prompt_type"], tmp_prompt_types)
            merged_df = merged_df.sort_values("prompt_type")
        
        x = "video_name" if gestures else "prompt_type"
        y = "Score"
        hue = "Model"
        xlabel = "Prompt Type" if prompt_types else "Gesture"
        
        # Get cosine scores only
        cosine_df = merged_df[merged_df["Metric"] == "Cosine"]
        
        # Rename "," to "\n"
        cosine_df["video_name"] = cosine_df["video_name"].str.replace(",", ",\n")
          
        sns.boxplot(
            x=x, y=y, hue=hue, data=cosine_df,
            width=0.6, legend=True, palette=colors,
        )
        
        plt.xlabel(xlabel, fontstyle="italic")
        unique_labels = [label for label in cosine_df[x].unique() if pd.notna(label)]
        plt.xticks(
            range(len(unique_labels)),
            [str(label).capitalize() for label in unique_labels]
        )
        if gestures:
            plt.xticks(rotation=45//2)
            
        from matplotlib.lines import Line2D

        # Get and deduplicate handles/labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # Actual handles/labels
        model_handles = list(by_label.values())
        model_labels = list(by_label.keys())

        # Inject bold title at the front (no handle)
        all_handles = model_handles
        all_labels = [r"$\bf{Models:}$"] + model_labels

        # Draw the legend
        plt.legend(
            [Line2D([], [], linestyle='none')] + all_handles,  # insert blank handle for title
            all_labels,
            loc='upper center',
            # bbox_to_anchor=(0.5, 1.15),
            ncol=len(all_labels),
            handlelength=1.0,
            handletextpad=0.4,
            borderpad=0.5,
            frameon=True
        )
        
    # Boxplot: Group by metric, color by model
    else:
        sns.violinplot(
            x="Model", y="Score", data=merged_df,
            width=0.7,
            palette=colors,
        )
        plt.xticks(rotation=45//2)
        plt.xlabel("Model", fontstyle="italic")
        # plt.title("Similarity Metrics Across Models")

    # Configure plot
    plt.ylabel("Score", fontstyle="italic")
    # plt.title(f'Boxplot of Similarity Metrics Across Models (prompt: {include_prompt if include_prompt else "All"})')
    
    plt.ylim(-0.1, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    # Save the plot
    # date_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    boxplot_path = f"results/figures/metrics_boxplot.pdf"
    if prompt_types: name_type = "prompt_types"
    elif gestures:   name_type = "gestures"
    elif baseline:   name_type = "expert"
    else:            name_type = "gt"
    boxplot_path = boxplot_path.replace(".pdf", f"_{name_type}.pdf")
    plt.savefig(boxplot_path, format="pdf", dpi=300, bbox_inches='tight')

    print(f"Boxplot saved to: {boxplot_path}")
    print()
    
    return merged_df

def print_latex_table(merged_df, prompt_types=False, gesture=False) -> None:
    """ 
    Print a LaTeX-style formatted table of the average scores including the percentage difference to the highest score.
    
    Args:
        merged_df (pd.DataFrame):   Merged DataFrame containing all metrics data.
        prompt_types (bool):        If True, group by prompt types.
        gesture (bool):             If True, group by gestures.

    Returns:
        None (prints the table to terminal).
    """

    # Determine the focus for grouping
    if prompt_types:    focus = "prompt_type"
    elif gesture:       focus = "video_name"
    else:               focus = "Metric"
    
    # Compute the average scores
    group_cols = ["Model", focus]
    numeric_cols = merged_df.select_dtypes(include=["number"]).columns.tolist()
    avg_df = merged_df.groupby(group_cols)[numeric_cols].mean().reset_index()
    print(avg_df)
    print()
    
    # Compute the highest score for each Metric
    max_scores = avg_df.groupby(focus)["Score"].transform("max")

    # Compute the percentage difference to the highest score
    avg_df["Difference [%]"] = (avg_df["Score"] / max_scores) * 100

    # Compute the overall average score for each model across all metrics
    model_averages = avg_df.groupby("Model")["Score"].mean()
    # Find the highest model average
    max_avg_score = model_averages.max()
    # Compute percentage difference for model averages
    model_avg_diff = (model_averages / max_avg_score) * 100

    # Convert DataFrame to LaTeX-style formatted output
    latex_table = ""
    models = avg_df["Model"].unique()
    metrics = avg_df[focus].unique()
    
    # Sort in the order of config.prompts
    if prompt_types: metrics = sorted(metrics, key=lambda x: list(prompts.keys()).index(x))
    # Sort in alphabetical order
    else: metrics = sorted(metrics, key=lambda x: x.lower()) 

    # Convert to pivot format for easier LaTeX conversion
    pivot_df = avg_df.pivot(index="Model", columns=focus, values=["Score", "Difference [%]"])

    print("##########################")
    
    # Print unique focus
    print("\\begin{tabular}{|l", end="")
    for metric in metrics:
        print(f"|c", end="")
    print("|c|} \\hline")
    print()
    
    print("\\rowcolor{gray!30}") # Header color
    
    # Print uniqe focus
    for metric in metrics:
        print(f"& \\textbf{{{ metric.capitalize() }}} \\(\\uparrow\\)")
    print("& \\textbf{Average} \\(\\uparrow\\)")
    print('\\\\ \\hline')
    print()
    
    # Build LaTeX-style table
    for model in models:
        row_str = f"\\textbf{{{model.capitalize()}}}"
        
        for metric in metrics:
            score = pivot_df.loc[model, ("Score", metric)]
            diff = pivot_df.loc[model, ("Difference [%]", metric)]
            max_score = max_scores[avg_df[focus] == metric].max()
            row_str += (
                f" & \\textbf{{{score:.2f}}}"
                if score == max_score
                else f" & {score:.2f} ({diff:.0f}\\%)"
            )
        
        # Add model average to the row
        avg_score = model_averages[model]
        avg_diff = model_avg_diff[model]
        
        row_str += (
            f" & \\textbf{{{avg_score:.2f}}}"
            if avg_score == max_avg_score
            else f" & {avg_score:.2f} ({avg_diff:.0f}\\%)"
        )

        latex_table += row_str + " \\\\ \\hline\n"

    # Print the formatted LaTeX-style table
    print(latex_table)
    print()

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Plot similarity metrics from CSV files.")
    parser.add_argument("--prompt_types",   action="store_true", help="Group by prompt types.")
    parser.add_argument("--gestures",       action="store_true", help="Group by gesture.")
                        
    args = parser.parse_args()

    for metrics_folder in ["results/data/metrics/to_gt", "results/data/metrics/to_gt_and_human"]:
        baseline = False if "human" in metrics_folder else True
        merged_df = plot_metrics(metrics_folder, args.prompt_types, args.gestures, baseline)
        print_latex_table(merged_df, args.prompt_types, args.gestures)