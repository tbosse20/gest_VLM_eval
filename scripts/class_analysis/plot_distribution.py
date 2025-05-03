import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root or config path
sys.path.append(".")
import config.directories as directories
import analysis.plot_matrix as plot_matrix

# === Config ===
model_names = {
    'vllama3.csv': 'Plain',
    'vllama3_projection.csv': 'Project',
    'vllama3_describe.csv': 'Describe',
    'vllama3_describe_projection.csv': 'Proj. + Desc.',
    os.path.basename(directories.LABELS_CSV): 'Ground Truth',
}
model_order = ['Plain', 'Project', 'Describe', 'Proj. + Desc.', 'Ground Truth']
color_map = {
    'Ground Truth':  'green',
    'Plain':         'blue',
    'Project':       'orange',
    'Describe':      'purple',
    'Proj. + Desc.': 'red',
}
caption_map = {
    0: 'Idle', 2: 'Stop', 3: 'Advance', 4: 'Return', 5: 'Accelerate',
    6: 'Decelerate', 7: 'Left', 8: 'Right', 9: 'Hail', 10: 'Attention', 12: 'Other'
}
caption_order = list(caption_map.values())

# === Load model predictions ===
csv_folder = directories.OUTPUT_FOLDER_PATH
dataframes = []
for file in os.listdir(csv_folder):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(csv_folder, file))
        df['source_file'] = file
        df[['pred_class', 'caption_word']] = df['caption'].apply(
            lambda x: pd.Series(plot_matrix.extract_number_and_word(x))
        )
        df["pred_class"] = pd.to_numeric(df["pred_class"], errors="coerce").fillna(-1)
        df["pred_class"] = df["pred_class"].astype(int)
        dataframes.append(df)

# === Load ground truth and merge ===
gt_df = pd.read_csv(directories.LABELS_CSV)
gt_df['source_file'] = os.path.basename(directories.LABELS_CSV)
gt_df['pred_class'] = gt_df['gt_id'].astype(int)
dataframes.append(gt_df)

# # Print samples of each class in gt_df
# print("Ground Truth Samples:")
# class_counts = gt_df['gt_id'].value_counts()
# print(class_counts)

# === Combine and preprocess ===
all_data = pd.concat(dataframes, ignore_index=True)
all_data['source_file'] = all_data['source_file'].replace(model_names)
all_data['source_file'] = pd.Categorical(all_data['source_file'], categories=model_order, ordered=True)

# Map and categorize class labels
all_data['pred_class'] = all_data['pred_class'].map(caption_map)
all_data['pred_class'] = pd.Categorical(all_data['pred_class'], categories=caption_order, ordered=True)

# === Group and Plot ===
caption_distribution = all_data.groupby(['pred_class', 'source_file'], observed=True).size().unstack(fill_value=0)
bar_colors = [color_map.get(col, 'gray') for col in caption_distribution.columns]
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Plot the bar chart
ax = caption_distribution.plot(kind='bar', figsize=(7, 2.5), color=bar_colors, width=0.8, edgecolor='black')

# Get and deduplicate handles/labels
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# Prepare legend components
model_handles = list(by_label.values())
model_labels = list(by_label.keys())

# Add bold "Models:" label (no handle)
all_handles = [Line2D([], [], linestyle="none")] + model_handles
all_labels = [r"$\bf{Method:}$"] + model_labels

# Draw horizontal legend below the plot
legend = ax.legend(
    all_handles,
    all_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.22),
    ncol=len(all_labels),
    handlelength=0.7,
    handletextpad=0.5,
    borderpad=0.3,
    fontsize=8,
)

# Highlight 'Ground Truth' label
for label in legend.get_texts():
    if label.get_text() == 'Ground Truth':
        label.set_fontstyle('italic')
        label.set_color('gray')

ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)
ax.set_xlabel('Gesture Class')
ax.set_ylabel('Frequency')

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)  # Ensure grid is below bars

plt.tight_layout()
plt.savefig("results/figures/distribution.pdf", format="pdf", dpi=300, bbox_inches="tight")