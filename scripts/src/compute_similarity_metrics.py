import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import bert_score
import pandas as pd
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

ENCODER = "sbert"  # Default encoder
# Set the encoder to "bert" if you want to use BERT instead of SBERT

# Ensure necessary NLTK resources are available
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

if ENCODER == "bert":
    # Load pre-trained embedding model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

elif ENCODER == "sbert":
    from transformers import BertTokenizer, BertModel
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

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

def embed_sentence(sentence: str, encoder: str):
    """Embed a sentence using the specified encoder."""
    
    if encoder == "sbert":
        return sbert_model.encode(sentence, convert_to_tensor=True)
    elif encoder == "bert":
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            return model(**inputs).last_hidden_state[:, 0, :]
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")

def compute_similarity_metrics(ground_truth: str, predicted: str, metrics_list: list[str] = None) -> pd.DataFrame:
    """
    Compute multiple similarity metrics between ground truth and predicted captions.

    If 'metrics_list' is provided, only the specified metrics are computed. When 'metrics_list' is None,
    all available metrics are calculated.

    Args:
        ground_truth (str): The reference caption.
        predicted (str): The generated caption to evaluate.
        metrics_list (list[str], optional): List of metric names to compute. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with a single row where each column represents a computed metric,
                      sorted in alphabetical order.
    """
    
    # Ensure inputs are not empty
    if not ground_truth or not predicted:
        return pd.DataFrame({
            "cosine_similarity": 0.0,
            "jaccard_similarity": 0.0,
            "bleu_score": 0.0,
            "meteor_score": 0.0,
            "rouge_l_score": 0.0,
            "bert_score": 0.0,
            "cross_encoder_score": 0.0
        })

    # Flag to check if we compute all metrics
    compute_all = metrics_list is None or len(metrics_list) == 0
    
    # Compute similarity metrics
    metrics = {}
    
    if compute_all or "cosine_similarity" in metrics_list:
        gt_embedding = embed_sentence(ground_truth, encoder=ENCODER)
        pred_embedding = embed_sentence(predicted, encoder=ENCODER)
        metrics["cosine_similarity"] = util.pytorch_cos_sim(gt_embedding, pred_embedding).item()
        
    if compute_all or "jaccard_similarity" in metrics_list:
        metrics["jaccard_similarity"] = jaccard_similarity(ground_truth, predicted)
    
    if compute_all or "bleu_score" in metrics_list:
        metrics["bleu_score"] = sentence_bleu([ground_truth.split()], predicted.split(), smoothing_function=smoothing.method1)
    
    if compute_all or "meteor_score" in metrics_list:
        metrics["meteor_score"] = meteor_score([ground_truth.split()], predicted.split())
        
    if compute_all or "rouge_l_score" in metrics_list:
        metrics["rouge_l_score"] = rouge_scorer_obj.score(ground_truth, predicted)["rougeL"].fmeasure
    
    if compute_all or "cross_encoder_score" in metrics_list:
        metrics["cross_encoder_score"] = cross_model.predict([(ground_truth, predicted)])[0]

    # Compute BERTScore
    if compute_all or "bert_score" in metrics_list:
        _, _, bert_f1 = bert_score.score([predicted], [ground_truth], lang="en")
        metrics["bert_score"] = bert_f1.item()
    
    # Sort metrics by alphabetical order
    metrics = {k: metrics[k] for k in sorted(metrics)}

    return pd.DataFrame([metrics])