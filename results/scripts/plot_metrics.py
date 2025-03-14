import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load csv
df = pd.read_csv("data/sanity/output/similarity_metrics.csv")

# Post processing
# Remove "_score" and "_similarity" suffix from column names
df.columns = df.columns.str.replace("_score", "")
df.columns = df.columns.str.replace("_similarity", "")
# Make capitalization consistent
df.columns = df.columns.str.title()

# Plot
plt.figure(figsize=(7, 7))

# Define bar width and positions
num_metrics = len(df.columns) - 1  # Excluding 'case' column
num_cases = len(df["Case"])
bar_width = 0.15
x = np.arange(num_metrics)

# Transpose the DataFrame for better grouping by metric
metrics = df.columns[1:]  # Exclude 'case'
cases = df["Case"]

# Plot each case as a separate color group
for i, case in enumerate(cases):
    plt.bar(x + i * bar_width, df.iloc[i, 1:], width=bar_width, label=case)

# Configure x-axis
plt.xticks(x + (num_cases - 1) * bar_width / 2, metrics)
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("Clustered Bar Plot: Metrics as Groups, Cases as Colors")
plt.legend(title="Cases", loc="upper left")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig("data/sanity/output/similarity_metrics.png")