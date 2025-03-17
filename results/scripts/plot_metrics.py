import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

def plot_metrics(metrics_folder, include_prompt=None):
    
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
        df = pd.read_csv(file)
        
        if include_prompt:
            df = df[df["prompt"] == include_prompt]
        
        # Drop unnecessary columns
        df = df.drop(columns=["video_name", "frame_idx", "prompt"], errors="ignore")
        
        # Store the model name
        model_name = os.path.basename(file).split(".")[0]
        
        # Reshape DataFrame for boxplot (long format)
        df_melted = df.melt(var_name="Metric", value_name="Score")
        df_melted["Model"] = model_name  # Add model column

        # Append to list
        data_list.append(df_melted)

    # Combine all data into a single DataFrame
    merged_df = pd.concat(data_list, ignore_index=True)
    merged_df["Metric"] = merged_df["Metric"].str.title()
    
    # --- PLOTTING GROUPED BOX PLOT ---

    # Plot
    plt.figure(figsize=(7, 3))
    
    # Only keep cos, jaccard and meteor
    # merged_df = merged_df[merged_df["Metric"].isin(["Cosine", "Jaccard", "Meteor"])]

    # Boxplot: Group by metric, color by model
    import seaborn as sns
    sns.boxplot(x="Metric", y="Score", hue="Model", data=merged_df, showfliers=False, width=0.5)

    # Configure plot
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    # plt.title(f'Boxplot of Similarity Metrics Across Models (prompt: {include_prompt if include_prompt else "All"})')
    plt.legend(title="Models")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    date_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    boxplot_path = f"results/figures/metrics_boxplot2.png"
    boxplot_path = boxplot_path.replace(".png", f"_{include_prompt}.png") if include_prompt else boxplot_path
    # boxplot_path = boxplot_path.replace(".png", f"_{date_time}.png")
    plt.savefig(boxplot_path)
    print(f"Boxplot saved to: {boxplot_path}")
    
    # Get the average of merged_df
    avg_df = merged_df.groupby(["Model", "Metric"]).mean().reset_index()
    print(avg_df)

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    
    # Get list of prompt types
    from config.prompts import prompts
    prompt_types = prompts.keys()
    
    import argparse
    parser = argparse.ArgumentParser(description="Plot similarity metrics from CSV files.")
    parser.add_argument("--metrics_folder", type=str, help="Path to the folder containing metrics CSV files.")
    parser.add_argument("--include_prompt", type=str, help="Include only the specified prompt type.", choices=prompt_types)
    args = parser.parse_args()

    # Define folder containing CSVs
    metrics_folder = args.metrics_folder or "results/data/metrics"

    # Plot metrics
    plot_metrics(metrics_folder, include_prompt=args.include_prompt)