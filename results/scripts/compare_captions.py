import os
import pandas as pd
import numpy as np
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
    if not ground_truth or not predicted:
        return {
            "cosine_similarity": 0.0,
            "jaccard_similarity": 0.0,
            "bleu_score": 0.0,
            "meteor_score": 0.0,
            "rouge_l_score": 0.0,
            "bert_score": 0.0
        }

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

    return metrics


def process_csv(input_csv, output_csv, image_folder):
    """Processes a CSV file, generates captions, computes similarity metrics, and saves results."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file '{input_csv}' not found.")

    # Load CSV
    df = pd.read_csv(input_csv)

    # Ensure necessary columns exist
    required_columns = {"image_name", "frame_idx", "ground_truth_caption"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")

    # Add new columns for predicted captions and metrics
    df["predicted_caption"] = ""
    df["cosine_similarity"] = 0.0
    df["jaccard_similarity"] = 0.0
    df["bleu_score"] = 0.0
    df["meteor_score"] = 0.0
    df["rouge_l_score"] = 0.0
    df["bert_score"] = 0.0

    # Process each row
    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, row["image_name"])
        ground_truth_caption = row["ground_truth_caption"]

        # Generate a predicted caption (Replace with actual model)
        predicted_caption = generate_caption(image_path)

        # Compute similarity metrics
        metrics = compute_similarity_metrics(ground_truth_caption, predicted_caption)

        # Store results
        df.at[index, "predicted_caption"] = predicted_caption
        df.at[index, "cosine_similarity"] = metrics["cosine_similarity"]
        df.at[index, "jaccard_similarity"] = metrics["jaccard_similarity"]
        df.at[index, "bleu_score"] = metrics["bleu_score"]
        df.at[index, "meteor_score"] = metrics["meteor_score"]
        df.at[index, "rouge_l_score"] = metrics["rouge_l_score"]
        df.at[index, "bert_score"] = metrics["bert_score"]

    # Save results to CSV
    df.to_csv(output_csv, index=False)
    print(f"Processed data saved to '{output_csv}'")


# Example Usage
if __name__ == "__main__":
    input_csv = "captions_input.csv"  # Input CSV file
    output_csv = "captions_output.csv"  # Output CSV file
    image_folder = "images/"  # Folder where images are stored

    # process_csv(input_csv, output_csv, image_folder)
    
    # # Example test cases with different levels of similarity
    # test_cases = [
    #     ("A pedestrian raises their hand to stop traffic.",
    #         "A person signals a driver to stop with their hand.",
    #         "High"),  # Expect high similarity

    #     ("A pedestrian waves to say thanks.",
    #         "A pedestrian raises their hand to stop traffic.",
    #         "Moderate"),  # Expect moderate similarity

    #     ("A cyclist gestures to indicate a turn.",
    #         "A pedestrian waves to get a taxi.",
    #         "Low"),  # Expect low similarity

    #     ("The sky is blue and the sun is shining.",
    #         "A pedestrian signals a driver to yield.",
    #         "Very low"),  # Expect very low similarity

    #     ("",
    #         "A pedestrian waves to stop a car.",
    #         "empty GT"),  # Edge case: empty string

    #     ("A pedestrian waves at the driver.",
    #         "",
    #         "empty pred"),  # Edge case: empty string
    # ]

    # # Store metrics in a DataFrame
    # df = pd.DataFrame(columns=["case", "consine_similarity", "jaccard_similarity", "bleu_score", "meteor_score", "rouge_l_score", "bert_score"])
    # for ground_truth, predicted, description in test_cases:
    #     metrics = compute_similarity_metrics(ground_truth, predicted)
        
    #     # Convert the dictionary into a DataFrame
    #     new_row = pd.DataFrame([{
    #         "case": description,
    #         "consine_similarity": metrics["cosine_similarity"],
    #         "jaccard_similarity": metrics["jaccard_similarity"],
    #         "bleu_score": metrics["bleu_score"],
    #         "meteor_score": metrics["meteor_score"],
    #         "rouge_l_score": metrics["rouge_l_score"],
    #         "bert_score": metrics["bert_score"]
    #     }])
        
    #     # Append new row using pd.concat()
    #     df = pd.concat([df, new_row], ignore_index=True)
    
    # # Save metrics to CSV
    # df.to_csv("data/sanity/output/similarity_metrics.csv", index=False)