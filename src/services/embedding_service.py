import numpy as np
from typing import Union, List
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None
    _vectorizer = None
    _cache = {}
    _is_fitted = False

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing EmbeddingService singleton")
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._vectorizer is None:
            logger.info("Initializing TF-IDF vectorizer...")
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=1000
            )

    def _preprocess_text(self, text: str) -> str:
        """Simple text preprocessing"""
        return text.lower().strip()

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Create cache keys
        cache_keys = [hash(text) for text in processed_texts]
        
        # Check cache
        if all(key in self._cache for key in cache_keys):
            logger.debug("Using cached embeddings")
            return np.array([self._cache[key] for key in cache_keys])

        # Fit and transform if first time
        if not self._is_fitted:
            logger.debug("First-time fit_transform for vectorizer")
            embeddings = self._vectorizer.fit_transform(processed_texts).toarray()
            self._is_fitted = True
        else:
            logger.debug("Transform only for vectorizer")
            embeddings = self._vectorizer.transform(processed_texts).toarray()

        # Cache the results
        for key, embedding in zip(cache_keys, embeddings):
            self._cache[key] = embedding

        return embeddings
