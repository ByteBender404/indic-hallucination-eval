# Indic Hallucination Eval

🚀 **Live Demo:** https://indic-hallucination-eval.onrender.com/docs
💻 **GitHub:** https://github.com/ByteBender404/indic-hallucination-eval

An open-source REST API to evaluate whether an LLM's answer in Hindi 
or Tamil is faithful to a source document. Uses character n-gram TF-IDF 
embeddings — no API keys, no model downloads, runs free on any server.

## The Problem It Solves

Evaluation frameworks like RAGAS and DeepEval are English-only and 
require paid OpenAI API keys. Indian companies deploying AI chatbots 
for agriculture, healthcare, and legal aid in Hindi or Tamil have no 
reliable way to catch hallucinated responses before they reach users.

This tool sits between your LLM and your users — it scores every 
response and flags suspicious ones automatically.

## Comparison Table

| Feature | indic-hallucination-eval | RAGAS | DeepEval |
|---|---|---|---|
| Offline / No model download | ✅ Yes | ❌ No | ❌ No |
| Cost | ✅ Free | ❌ Paid (OpenAI API) | ❌ Paid (OpenAI API) |
| Native Indic Language Focus | ✅ Yes | ⚠️ Needs custom setup | ⚠️ Needs custom setup |
| Requires API Keys | ✅ No | ❌ Yes | ❌ Yes |
| Self-hostable | ✅ Yes | ⚠️ Partial | ⚠️ Partial |

## How It Works

1. You send a POST request with: question, LLM answer, source document, ground truth
2. The API computes three scores using character n-gram similarity
3. Returns faithfulness, relevance, consistency scores + PASS/FLAG verdict

## Installation

```bash
git clone https://github.com/ByteBender404/indic-hallucination-eval
cd indic-hallucination-eval
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Example Request

```bash
curl -X POST https://indic-hallucination-eval.onrender.com/evaluate \
-H "Content-Type: application/json" \
-d '{
  "question": "गेहूं में कौन सा उर्वरक डालना चाहिए?",
  "answer": "गेहूं में यूरिया और DAP उर्वरक डालना चाहिए।",
  "source": "गेहूं की फसल के लिए यूरिया और DAP सबसे उपयुक्त उर्वरक हैं।",
  "ground_truth": "गेहूं में यूरिया और DAP डालें।"
}'
```

## Example Response

```json
{
  "faithfulness_score": 0.87,
  "relevance_score": 0.76,
  "consistency_score": 0.91,
  "overall_score": 0.85,
  "verdict": "PASS"
}
```

## Verdict Logic

- **PASS** — overall score >= 0.7 → safe to show to user
- **FLAG** — overall score < 0.7 → review before showing

## Tech Stack

- Python 3.10+
- FastAPI + Uvicorn
- scikit-learn (TF-IDF embeddings)
- NumPy
- Deployed on Render.com

## Use Cases

- AgriTech chatbots advising farmers in Hindi
- Legal aid tools in Tamil
- Healthcare Q&A bots in regional languages
- Any LLM pipeline serving Indic language users

## Contributing

PRs welcome. Especially interested in:
- Adding more Indian languages (Telugu, Kannada, Bengali)
- Improving scoring with better multilingual models
- Adding a benchmark dataset on Hugging Face
