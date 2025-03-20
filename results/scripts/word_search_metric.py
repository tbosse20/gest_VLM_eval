
import pandas as pd

# Load csv
gt = pd.read_csv("data/labels/firsthand.csv")
human = pd.read_csv("results/data/captions/human.csv")
qwen = pd.read_csv("results/data/captions/qwen.csv")

video_name = 'Go forward'
gt = gt[gt["video_name"] == video_name]
human = human[human["video_name"] == video_name]
qwen = qwen[qwen["video_name"] == video_name]

# Define positive and negative words
positive_words = ['forward', 'go', 'drive', 'straight', 'ahead', 'proceed']
negative_words = ['stop', 'back', 'reverse', 'turn', 'drive', 'left', 'right']

# Count how many positive and negative words are in the ground truth caption
count_positives, count_negatives = 0, 0
total_positives, total_negatives = 0, 0

positive_scores, negative_scores = [], []

# Process each row
for index, row in qwen.iterrows():
    frame_idx = row["frame_idx"]
    
    # Retrieve the corresponding ground truth caption
    pred_caption = row["caption"]
    if pred_caption in [None, "empty"]:
        continue
    pred_caption = pred_caption.lower()
    
    # Count how many positive and negative words are in the predicted caption
    contains_positive_pred = sum([1 for word in positive_words if word in pred_caption])
    contains_negative_pred = sum([1 for word in negative_words if word in pred_caption])
    
    count_positives += contains_positive_pred
    count_negatives += contains_negative_pred
    total_positives += len(positive_words)
    total_negatives += len(negative_words)
    
    total_words = len(pred_caption.split())
    
    positive_score = contains_positive_pred / (len(positive_words) * total_words)
    negative_score = contains_negative_pred / (len(negative_words) * total_words)
    positive_scores.append(positive_score)
    negative_scores.append(negative_score)

positive_words = f"{count_positives}/{total_positives}"
negative_words = f"{count_negatives}/{total_negatives}" 

average_positive = count_positives/total_positives
average_negative = count_negatives/total_negatives

average_positive_score = sum(positive_scores) / len(positive_scores)
average_negative_score = sum(negative_scores) / len(negative_scores)

print(f"Positive Words: {positive_words} ({average_positive:.2f}), {average_positive_score:.2f}")
print(f"Negative Words: {negative_words} ({average_negative:.2f}), {average_negative_score:.2f}")