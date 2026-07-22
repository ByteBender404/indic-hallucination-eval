# Indic Hallucination Eval

A Python library and REST API to evaluate whether an LLM's answer in Hindi or Tamil is faithful to a source document. It uses SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`) to calculate embedding similarities.

## The Problem It Solves
Evaluating LLM responses in Indic languages like Hindi and Tamil is challenging because many mainstream tools lack robust multilingual support natively, or they rely on expensive, API-key-gated language models like GPT-4 for evaluation. `indic-hallucination-eval` solves this by providing a completely offline, lightweight, embeddings-based approach that calculates faithfulness, relevance, and consistency specifically tailored for Indic contexts.

## Comparison Table

| Feature | `indic-hallucination-eval` | RAGAS | DeepEval |
| :--- | :--- | :--- | :--- |
| Offline / Local Model | Yes (SentenceTransformers) | Mostly No (Relies on LLMs) | Mostly No (Relies on LLMs) |
| Cost | Free | Paid (OpenAI API usage) | Paid (OpenAI API usage) |
| Native Indic Focus | Yes | Needs custom setup | Needs custom setup |
| Execution Speed | Fast | Slower (API calls) | Slower (API calls) |
| Requires API Keys | No | Yes | Yes |

## Installation Steps
Ensure you have Python 3.10+ installed.

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Starting the API Server
Start the FastAPI application with Uvicorn:
```bash
uvicorn api.main:app --reload
```

## Example `curl` Request
```bash
curl -X POST http://127.0.0.1:8000/evaluate \
-H "Content-Type: application/json" \
-d '{
  "question": "भारत में खरीफ की फसलें कब बोई जाती हैं?",
  "answer": "भारत में खरीफ की फसलें जून-जुलाई में बोई जाती हैं।",
  "source": "खरीफ की फसलें जून-जुलाई में बोई जाती हैं।",
  "ground_truth": "खरीफ फसलें जून-जुलाई में बोई जाती हैं।"
}'
```

## Example Response
```json
{
  "faithfulness_score": 1.0,
  "relevance_score": 0.8523,
  "consistency_score": 1.0,
  "overall_score": 0.9508,
  "verdict": "PASS"
}
```
