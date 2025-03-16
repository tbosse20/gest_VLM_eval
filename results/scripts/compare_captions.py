import os
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import nltk
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import bert_score
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ensure necessary NLTK resources are available
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Load pre-trained embedding model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ROUGE scorer
rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm
smoothing = SmoothingFunction()

def jaccard_similarity(sent1, sent2):
    """Compute Jaccard Similarity between two sentences."""
    set1, set2 = set(sent1.lower().split()), set(sent2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0.0


def generate_caption(image_path):
    """Placeholder function for image captioning. Replace with an actual model."""
    return "A pedestrian raises their hand to stop traffic."  # Dummy caption


def compute_similarity_metrics(ground_truth, predicted):
    """Computes multiple similarity metrics between the ground truth and predicted captions."""
    
    # Ensure inputs are not empty
    if not ground_truth or not predicted:
        return pd.DataFrame({
            "cosine_similarity": 0.0,
            "jaccard_similarity": 0.0,
            "bleu_score": 0.0,
            "meteor_score": 0.0,
            "rouge_l_score": 0.0,
            "bert_score": 0.0
        })

    # Compute embeddings
    gt_embedding = sbert_model.encode(ground_truth, convert_to_tensor=True)
    pred_embedding = sbert_model.encode(predicted, convert_to_tensor=True)

    # Compute similarity metrics
    metrics = {
        "cosine_similarity":  util.pytorch_cos_sim(gt_embedding, pred_embedding).item(),
        "jaccard_similarity": jaccard_similarity(ground_truth, predicted),
        "bleu_score":         sentence_bleu([ground_truth.split()], predicted.split(), smoothing_function=smoothing.method1),
        "meteor_score":       meteor_score([ground_truth.split()], predicted.split()),
        "rouge_l_score":      rouge_scorer_obj.score(ground_truth, predicted)["rougeL"].fmeasure
    }

    # Compute BERTScore
    _, _, bert_f1 = bert_score.score([predicted], [ground_truth], lang="en")
    metrics["bert_score"] = bert_f1.item()

    return pd.DataFrame([metrics])

def make_sibling_folder(folder_path: str, sibling_name: str):
    """Create a sibling folder to the input folder."""
    
    # Get the parent directory (where sibling folders exist)
    parent_dir = Path(folder_path).parent

    # Define the path for the new sibling folder
    sibling_folder = parent_dir / sibling_name

    # Create the sibling folder if it doesn't exist
    sibling_folder.mkdir(parents=True, exist_ok=True)
    
    return sibling_folder

def process_csv(label_caption_csv, gen_caption_folder):
    """Processes a CSV file, generates captions, computes similarity metrics, and saves results."""
    
    if not os.path.exists(label_caption_csv):
        raise FileNotFoundError(f"Input CSV file '{label_caption_csv}' not found.")

    COLUMNS = ["video_name", "frame_idx", "prompt"]
    METRICS = ["cosine", "jaccard", "bleu", "meteor", "rouge_l", "bert"]
    
    # Load CSV
    label_df = pd.read_csv(label_caption_csv)

    # Ensure necessary columns exist
    required_columns = {"video_name", "frame_idx", "label"}
    if not required_columns.issubset(label_df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")
    
    # Get list of generated caption CSVs (ex.: results/data/CAPTIONS/proxy.csv)
    gen_caption_csvs = [
        os.path.join(gen_caption_folder, f)
        for f in os.listdir(gen_caption_folder)
        if f.endswith(".csv")
    ]
    
    # Make sibling folder to gen_caption_folder
    metrics_folder = make_sibling_folder(gen_caption_folder, "metrics")
    
    # Loop through generated caption CSVs
    for gen_caption_csv in gen_caption_csvs:
        
        # if "proxy" not in gen_caption_csv:
        #     continue

        # Get metric csv file (ex.: results/data/METRICS/proxy.csv)
        metric_path = os.path.join(metrics_folder, os.path.basename(gen_caption_csv))
        
        # Make or skip metric csv file
        if os.path.exists(metric_path):
            continue
        else:
            pd.DataFrame(columns=COLUMNS + METRICS).to_csv(
                metric_path, index=False, header=True)
        
        # Load generated captions
        gen_df = pd.read_csv(gen_caption_csv)
                
        # Process each row
        gen_caption_csv_name = os.path.basename(gen_caption_csv).split(".")[0]
        for index, row in tqdm(gen_df.iterrows(), total=gen_df.shape[0], desc=f"Proc. {gen_caption_csv_name}"):
            # Get image name and frame index
            video_name, frame_idx = row["video_name"], row["frame_idx"]

            # Retrieve the corresponding ground truth caption
            label_caption = label_df.loc[(
                (label_df["video_name"] == video_name) &
                (label_df["frame_idx"] == frame_idx)
                ), "label"
            ]
            if label_caption.empty or label_caption.values[0] in [None, "empty"]:
                continue
            label_caption = label_caption.values[0]
                
            # Get the predicted caption
            pred_caption, prompt_type = row["caption"], row["prompt_type"]
            if pred_caption in [None, "empty"]: continue
            
            # Compute similarity metrics
            metrics_df = compute_similarity_metrics(label_caption, pred_caption)
            
            frame_sample_df = pd.DataFrame({
                "video_name":   [video_name],
                "frame_idx":    [frame_idx],
                "prompt_type":  [prompt_type],
            })
            frame_sample_df = pd.concat([frame_sample_df, metrics_df], axis=1)
            
            frame_sample_df.to_csv(metric_path, mode="a", index=False, header=False)
            
# Example Usage
if __name__ == "__main__":
    label_caption_csv = "data/labels/firsthand.csv"  # Input CSV file
    gen_caption_folder = "results/data/captions/"  # Generated captions folder

    process_csv(label_caption_csv, gen_caption_folder)