from fastapi import FastAPI
from api.models import EvaluationRequest, EvaluationResponse
from evaluator.scorer import evaluate_answer

app = FastAPI(
    title="Indic Hallucination Eval API",
    description="API to evaluate LLM answers in Hindi and Tamil for faithfulness, relevance, and consistency."
)

@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_endpoint(request: EvaluationRequest):
    result = evaluate_answer(
        question=request.question,
        answer=request.answer,
        source=request.source,
        ground_truth=request.ground_truth
    )
    return EvaluationResponse(**result)
