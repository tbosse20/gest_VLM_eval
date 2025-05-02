import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root or config path
sys.path.append(".")
import config.directories as directories
import scripts.plot_matrix as plot_matrix

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
    6: 'Decelerate', 7: 'Left', 8: 'Right', 9: 'Hail', 10: 'Point',
    11: 'Attention', 12: 'Other'
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

ax = caption_distribution.plot(kind='bar', figsize=(12, 7), color=bar_colors)
plt.xlabel('Caption')
plt.ylabel('Frequency')
plt.title('Caption Distribution by Model')
legend = plt.legend(title='Source File')

# Highlight Ground Truth in legend
for label in legend.get_texts():
    if label.get_text() == 'Ground Truth':
        label.set_fontstyle('italic')
        label.set_color('gray')

plt.tight_layout()
plt.show()
