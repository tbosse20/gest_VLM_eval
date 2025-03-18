import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

def plot_metrics(metrics_folder, include_prompt=None, include_gesture=None):
    
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
        
        # Filter by prompt and gesture
        df = df[df["prompt"]     == include_prompt]  if include_prompt  else df
        df = df[df["video_name"] == include_gesture] if include_gesture else df
        
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
    
    # Print number of samples for each model pr metric
    count = merged_df.groupby(["Model", "Metric"]).size()
    # Format count to a table
    count = count.unstack().fillna(0).astype(int)
    print(count)
    
    # --- PLOTTING GROUPED BOX PLOT ---

    # Plot
    plt.figure(figsize=(7, 3))
    
    # Only keep cos, Jaccard, Bleu, and meteor
    merged_df = merged_df[merged_df["Metric"].isin(["Cosine", "Jaccard", "Bleu", "Meteor"])]
    
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
    boxplot_path = f"results/figures/metrics_boxplot.png"
    boxplot_path = boxplot_path.replace(".png", f"_{include_prompt}.png") if include_prompt else boxplot_path
    boxplot_path = boxplot_path.replace(".png", f"_{include_gesture}.png") if include_gesture else boxplot_path
    # boxplot_path = boxplot_path.replace(".png", f"_{date_time}.png")
    plt.savefig(boxplot_path)
    print(f"Boxplot saved to: {boxplot_path}")
    print()
    
    return merged_df

def prin_latex_table(merged_df):

    # Assuming merged_df already exists
    # Compute the average scores
    avg_df = merged_df.groupby(["Model", "Metric"]).mean().reset_index()

    # Compute the highest score for each Metric
    max_scores = avg_df.groupby("Metric")["Score"].transform("max")

    # Compute the percentage difference to the highest score
    avg_df["Difference [%]"] = -((max_scores - avg_df["Score"]) / max_scores) * 100

    # Compute the overall average score for each model across all metrics
    model_averages = avg_df.groupby("Model")["Score"].mean()

    # Find the highest model average
    max_avg_score = model_averages.max()

    # Compute percentage difference for model averages
    model_avg_diff = -((max_avg_score - model_averages) / max_avg_score) * 100

    # Convert DataFrame to LaTeX-style formatted output
    latex_table = ""
    models = avg_df["Model"].unique()
    metrics = avg_df["Metric"].unique()

    # Convert to pivot format for easier LaTeX conversion
    pivot_df = avg_df.pivot(index="Model", columns="Metric", values=["Score", "Difference [%]"])

    # Build LaTeX-style table
    for model in models:
        row_str = f"\\textbf{{{model.capitalize()}}}"
        
        for metric in metrics:
            score = pivot_df.loc[model, ("Score", metric)]
            diff = pivot_df.loc[model, ("Difference [%]", metric)]
            max_score = max_scores[avg_df["Metric"] == metric].max()  # Find max for this metric
            
            if score == max_score:
                row_str += f" & \\textbf{{{score:.2f}}}"
            else:
                row_str += f" & {score:.2f} ({diff:.0f}\\%)"
        
        # Add model average to the row
        avg_score = model_averages[model]
        avg_diff = model_avg_diff[model]
        
        if avg_score == max_avg_score:
            row_str += f" & \\textbf{{{avg_score:.2f}}}"
        else:
            row_str += f" & {avg_score:.2f} ({avg_diff:.0f}\\%)"

        latex_table += row_str + " \\\\ \\hline\n"

    # Print the formatted LaTeX-style table
    print(latex_table)


    print()

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
    parser.add_argument("--include_gesture", type=str, help="Include only the specified gesture type.")
    args = parser.parse_args()

    # Define folder containing CSVs
    metrics_folder = args.metrics_folder or "results/data/metrics"

    # Plot metrics
    merged_df = plot_metrics(metrics_folder, args.include_prompt, args.include_gesture)
    
    # Print LaTeX table
    prin_latex_table(merged_df)