"""TF-IDF cosine retrieval over only the seeded Uber Support sample."""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RetrievedThread:
    thread_id: str
    text: str
    similarity: float


def _thread_text(turns: list[dict[str, str]]) -> str:
    ordered = sorted(turns, key=lambda turn: int(turn["turn_order"]))
    return "\n".join(f"{turn['role'].upper()}: {turn['text']}" for turn in ordered)


class ThreadRetriever:
    def __init__(self, thread_ids: list[str], thread_texts: list[str]) -> None:
        if not thread_ids:
            raise ValueError("No resolved threads are available for retrieval.")
        self.thread_ids = thread_ids
        self.thread_texts = thread_texts
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(thread_texts)

    @classmethod
    def from_csv(cls, path: str | Path) -> "ThreadRetriever":
        """Use threads ending in a brand message as the operational resolved proxy."""
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                grouped[row["thread_id"]].append(row)
        resolved = []
        for thread_id, turns in grouped.items():
            ordered = sorted(turns, key=lambda turn: int(turn["turn_order"]))
            if ordered and ordered[-1]["role"] == "brand":
                resolved.append((thread_id, _thread_text(ordered)))
        if not resolved:
            raise ValueError("No threads ending in a brand turn were found; cannot form a resolved-thread corpus.")
        resolved.sort(key=lambda item: item[0])
        return cls([item[0] for item in resolved], [item[1] for item in resolved])

    def retrieve(self, message: str, top_k: int = 3, exclude_thread_id: str | None = None) -> list[RetrievedThread]:
        if not message or not message.strip():
            raise ValueError("message must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = self.vectorizer.transform([message])
        scores = cosine_similarity(query, self.matrix).ravel()
        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores) if self.thread_ids[index] != exclude_thread_id),
            key=lambda item: (-item[1], self.thread_ids[item[0]]),
        )[:top_k]
        return [RetrievedThread(self.thread_ids[index], self.thread_texts[index], score) for index, score in ranked]


def retrieve_similar_threads(
    message: str, sample_path: str | Path, top_k: int = 3, exclude_thread_id: str | None = None
) -> list[RetrievedThread]:
    """Convenience function for one-off use; batch callers should reuse ThreadRetriever."""
    return ThreadRetriever.from_csv(sample_path).retrieve(message, top_k, exclude_thread_id)
