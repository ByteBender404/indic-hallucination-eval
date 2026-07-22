import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from evaluator.embeddings import get_embedding

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """Calculates cosine similarity between two texts."""
    emb1 = np.array(get_embedding(text1)).reshape(1, -1)
    emb2 = np.array(get_embedding(text2)).reshape(1, -1)
    return float(cosine_similarity(emb1, emb2)[0][0])

def evaluate_answer(question: str, answer: str, source: str, ground_truth: str) -> dict:
    """Evaluates the answer based on faithfulness, relevance, and consistency."""
    faithfulness = calculate_cosine_similarity(answer, source)
    relevance = calculate_cosine_similarity(question, answer)
    consistency = calculate_cosine_similarity(answer, ground_truth)
    
    overall = (faithfulness + relevance + consistency) / 3.0
    verdict = "PASS" if overall >= 0.7 else "FLAG"
    
    return {
        "faithfulness_score": round(faithfulness, 4),
        "relevance_score": round(relevance, 4),
        "consistency_score": round(consistency, 4),
        "overall_score": round(overall, 4),
        "verdict": verdict
    }
