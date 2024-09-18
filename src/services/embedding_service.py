from sentence_transformers import SentenceTransformer
from src.utils.config import Config

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL, device="cpu")

    def get_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
