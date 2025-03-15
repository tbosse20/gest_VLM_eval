import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

# Define folder containing CSVs
csv_folder = "results/data/metrics"

# Get list of all CSV files in the folder
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

# Check if any CSV files were found
if not csv_files:
    print("No CSV files found in the folder.")
    exit()

# Create an empty list to store data
data_list = []

# Loop through each CSV file
for file in csv_files:
    df = pd.read_csv(file)
    
    # Drop unnecessary columns
    df = df.drop(columns=["video_name", "frame_idx"], errors="ignore")
    
    # Store the model name
    model_name = os.path.basename(file).split(".")[0]
    
    # Reshape DataFrame for boxplot (long format)
    df_melted = df.melt(var_name="Metric", value_name="Score")
    df_melted["Model"] = model_name  # Add model column

    # Append to list
    data_list.append(df_melted)

# Combine all data into a single DataFrame
merged_df = pd.concat(data_list, ignore_index=True)

# --- PLOTTING GROUPED BOX PLOT ---

# Clean column names
merged_df["Metric"] = merged_df["Metric"].str.replace("_score", "", regex=True)
merged_df["Metric"] = merged_df["Metric"].str.replace("_similarity", "", regex=True)
merged_df["Metric"] = merged_df["Metric"].str.title()

# Plot
plt.figure(figsize=(10, 7))

# Boxplot: Group by metric, color by model
import seaborn as sns
sns.boxplot(x="Metric", y="Score", hue="Model", data=merged_df, showfliers=False)

# Configure plot
plt.xticks(rotation=45, ha="right")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("Boxplot of Similarity Metrics Across Models")
plt.legend(title="Models")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
boxplot_path = "results/figures/similarity_metrics_boxplot.png"
plt.savefig(boxplot_path)

print(f"Boxplot saved to: {boxplot_path}")
