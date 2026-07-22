from fastapi import FastAPI
from api.models import EvaluationRequest, EvaluationResponse
from evaluator.scorer import evaluate
from evaluator.embeddings import get_vectorizer

app = FastAPI(
    title="Indic Hallucination Eval API",
    description="API to evaluate LLM answers in Hindi and Tamil for faithfulness, relevance, and consistency."
)

# warm up on startup
get_vectorizer()

@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_endpoint(request: EvaluationRequest):
    result = evaluate(
        question=request.question,
        answer=request.answer,
        source=request.source,
        ground_truth=request.ground_truth
    )
    return EvaluationResponse(**result)