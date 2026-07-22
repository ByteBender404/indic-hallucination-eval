import numpy as np
from .embeddings import embed

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def faithfulness_score(answer: str, source: str) -> float:
    return round(cosine_similarity(embed(answer), embed(source)), 4)

def relevance_score(question: str, answer: str) -> float:
    return round(cosine_similarity(embed(question), embed(answer)), 4)

def consistency_score(answer: str, ground_truth: str) -> float:
    return round(cosine_similarity(embed(answer), embed(ground_truth)), 4)

def evaluate(question: str, answer: str, source: str, ground_truth: str) -> dict:
    faith = faithfulness_score(answer, source)
    relevance = relevance_score(question, answer)
    consistency = consistency_score(answer, ground_truth)
    overall = round((faith + relevance + consistency) / 3, 4)
    return {
        "faithfulness_score": faith,
        "relevance_score": relevance,
        "consistency_score": consistency,
        "overall_score": overall,
        "verdict": "PASS" if overall >= 0.7 else "FLAG"
    }