from pydantic import BaseModel

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    source: str
    ground_truth: str

class EvaluationResponse(BaseModel):
    faithfulness_score: float
    relevance_score: float
    consistency_score: float
    overall_score: float
    verdict: str
