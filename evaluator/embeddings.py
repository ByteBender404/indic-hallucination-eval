from sentence_transformers import SentenceTransformer

# Load the multilingual model that supports Hindi and Tamil
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_embedding(text: str) -> list[float]:
    """Returns the embedding for a given text as a list of floats."""
    return model.encode(text).tolist()
