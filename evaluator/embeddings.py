from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort

_session = None
_tokenizer = None

def get_model():
    global _session, _tokenizer
    if _session is None:
        model_path = hf_hub_download(
            repo_id="optimum/paraphrase-multilingual-MiniLM-L12-v2",
            filename="model.onnx"
        )
        tokenizer_path = hf_hub_download(
            repo_id="optimum/paraphrase-multilingual-MiniLM-L12-v2",
            filename="tokenizer.json"
        )
        _session = ort.InferenceSession(model_path)
        _tokenizer = Tokenizer.from_file(tokenizer_path)
    return _session, _tokenizer

def embed(text: str):
    session, tokenizer = get_model()
    tokenizer.enable_padding(length=128)
    tokenizer.enable_truncation(max_length=128)
    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)
    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    })
    # mean pooling
    embedding = outputs[0][0].mean(axis=0)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding