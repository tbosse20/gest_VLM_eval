import numpy as np
import itertools
import nltk
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import bert_score
import pandas as pd
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ensure necessary NLTK resources are available
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Load pre-trained embedding model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# from transformers import BertTokenizer, BertModel
# # Load BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# Initialize ROUGE scorer
rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

from nltk.translate.bleu_score import SmoothingFunction
smoothing = SmoothingFunction()

# Load Cross-Encoder model
cross_model = CrossEncoder('cross-encoder/stsb-roberta-large')

def jaccard_similarity(sent1, sent2):
    """Compute Jaccard Similarity between two sentences."""
    set1, set2 = set(sent1.lower().split()), set(sent2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0.0

def compute_similarity_metrics(ground_truth, predicted):
    """Computes multiple similarity metrics between the ground truth and predicted captions."""
    
    # Ensure inputs are not empty
    if not ground_truth or not predicted:
        return pd.DataFrame({
            "cosine_similarity": 0.0,
            # "jaccard_similarity": 0.0,
            # "bleu_score": 0.0,
            # "meteor_score": 0.0,
            # "rouge_l_score": 0.0,
            # "bert_score": 0.0,
            # "cross_encoder_score": 0.0
        })

    # Compute embeddings 
    gt_embedding = sbert_model.encode(ground_truth, convert_to_tensor=True)
    pred_embedding = sbert_model.encode(predicted, convert_to_tensor=True)
    
    # # Tokenize input
    # ground_truth_inputs = tokenizer(ground_truth, return_tensors="pt", padding=True, truncation=True)
    # predicted_inputs = tokenizer(predicted, return_tensors="pt", padding=True, truncation=True)
    # # Get embeddings
    # with torch.no_grad():
    #     gt_embedding = model(**ground_truth_inputs).last_hidden_state[:, 0, :]
    #     pred_embedding = model(**predicted_inputs).last_hidden_state[:, 0, :]

    # Compute similarity metrics
    metrics = {
        "cosine_similarity":   util.pytorch_cos_sim(gt_embedding, pred_embedding).item(),
        "jaccard_similarity":  jaccard_similarity(ground_truth, predicted),
        "bleu_score":          sentence_bleu([ground_truth.split()], predicted.split(), smoothing_function=smoothing.method1),
        "meteor_score":        meteor_score([ground_truth.split()], predicted.split()),
        "rouge_l_score":       rouge_scorer_obj.score(ground_truth, predicted)["rougeL"].fmeasure,
        "cross_encoder_score": cross_model.predict([(ground_truth, predicted)])[0],
    }

    # Compute BERTScore
    _, _, bert_f1 = bert_score.score([predicted], [ground_truth], lang="en")
    metrics["bert_score"] = bert_f1.item()
    
    # Sort metrics by alphabetical order
    metrics = {k: metrics[k] for k in sorted(metrics)}

    return pd.DataFrame([metrics])