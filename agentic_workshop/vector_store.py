"""A minimal in-memory vector store: the one built step by step in M03_01, used for the rest of the day.

Change it here if you want to experiment: every notebook imports it from this file.
"""

from collections.abc import Callable

import numpy as np


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class MiniVectorStore:
    """Keeps (embedding, text, metadata) lists and returns the ones closest to a query's embedding."""

    def __init__(self):
        self.embeddings: list[np.ndarray] = []
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def add(
        self, embeddings, texts: list[str], metadata: list[dict] | None = None
    ) -> None:
        if metadata is None:
            metadata = [{} for _ in texts]
        assert len(embeddings) == len(texts) == len(metadata)
        self.embeddings += [np.asarray(e) for e in embeddings]
        self.texts += list(texts)
        self.metadata += list(metadata)

    def query(
        self, embedding, k: int = 3, keep: Callable[[dict], bool] | None = None
    ) -> list[dict]:
        """The k entries most similar to embedding, best first.

        keep: optional filter on the metadata; only entries where keep(metadata) is True are ranked.
        Each result is {"index", "score", "text", "metadata"}.
        """
        candidates = [
            i for i in range(len(self.texts)) if keep is None or keep(self.metadata[i])
        ]
        scored = sorted(
            ((cosine_similarity(embedding, self.embeddings[i]), i) for i in candidates),
            reverse=True,
        )
        return [
            {
                "index": i,
                "score": score,
                "text": self.texts[i],
                "metadata": self.metadata[i],
            }
            for score, i in scored[:k]
        ]

    def __len__(self) -> int:
        return len(self.texts)
