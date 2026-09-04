import os
import hashlib
import math
import re
from typing import List

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_TOKEN"] = ""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

_FALLBACK_DIMENSION = 384
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _hashed_embedding(text: str) -> List[float]:
    """Create a deterministic, normalized vector without downloading a model."""
    vector = [0.0] * _FALLBACK_DIMENSION
    tokens = _TOKEN_PATTERN.findall(text.lower())
    for token_index, token in enumerate(tokens):
        features = (token, f"{token}__{tokens[token_index + 1]}" if token_index + 1 < len(tokens) else "")
        for feature in features:
            if not feature:
                continue
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % _FALLBACK_DIMENSION
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class _LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        try:
            self._model = SentenceTransformer(model_name, token=False)
        except (OSError, RuntimeError) as error:
            self._model = None
            print(f"  Model unavailable ({error}). Using local hashing embeddings.\n")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            return [_hashed_embedding(text) for text in texts]
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        if self._model is None:
            return _hashed_embedding(text)
        return self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()


def get_embeddings() -> _LocalEmbeddings:
    return _LocalEmbeddings(EMBED_MODEL)
