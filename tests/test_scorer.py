import json
import os
import pytest
from evaluator.scorer import evaluate_answer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HINDI_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "hindi_samples.json")
TAMIL_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "tamil_samples.json")

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

hindi_samples = load_dataset(HINDI_DATASET_PATH)
tamil_samples = load_dataset(TAMIL_DATASET_PATH)

@pytest.mark.parametrize("sample", hindi_samples + tamil_samples)
def test_evaluation(sample):
    result = evaluate_answer(
        question=sample["question"],
        answer=sample["answer"],
        source=sample["source"],
        ground_truth=sample["ground_truth"]
    )
    assert result["verdict"] in ["PASS", "FLAG"]
    assert 0 <= result["overall_score"] <= 1
