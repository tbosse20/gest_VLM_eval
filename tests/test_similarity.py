import unittest
import sys
sys.path.append(".")
from scripts.src.compute_similarity_metrics import compute_similarity_metrics

class TestSimilarityMetrics(unittest.TestCase):

    def test_similarity_metrics(self):
        # Example test cases with different levels of similarity
        GT = "A person signals a driver to stop with their hand."
        test_cases = [
            ("A pedestrian raises their hand to stop traffic.",
             GT, "High"),  # Expect high similarity

            ("A pedestrian waves to say thanks.",
             GT, "Moderate"),  # Expect moderate similarity

            ("A cyclist gestures to indicate a turn.",
             GT, "Low"),  # Expect low similarity

            ("The sky is blue and the sun is shining.",
             GT, "Very low"),  # Expect very low similarity

            ("",
             GT, "Edge case: empty ground truth"),  # Edge case: empty string

            ("A pedestrian waves at the driver.",
             "", "Edge case: empty predicted caption"),  # Edge case: empty string
        ]
        
        for ground_truth, predicted, description in test_cases:
            with self.subTest(msg=description):
                metrics = compute_similarity_metrics(ground_truth, predicted)
                
                # Ensure metrics return valid numeric values
                self.assertTrue(0.0 <= metrics["cosine_similarity"]  <= 1.0)
                self.assertTrue(0.0 <= metrics["jaccard_similarity"] <= 1.0)
                self.assertTrue(0.0 <= metrics["bleu_score"]         <= 1.0)
                self.assertTrue(0.0 <= metrics["meteor_score"]       <= 1.0)
                self.assertTrue(0.0 <= metrics["rouge_l_score"]      <= 1.0)
                self.assertTrue(0.0 <= metrics["bert_score"]         <= 1.0)

                # Edge case handling: If one caption is empty, expect zero similarity
                if not ground_truth or not predicted:
                    for metric in metrics.values():
                        self.assertEqual(metric, 0.0, f"Expected zero for {description}, but got {metric}")

if __name__ == "__main__":
    unittest.main()