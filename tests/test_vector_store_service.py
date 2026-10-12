from __future__ import annotations

import unittest
from types import SimpleNamespace

from paper_classifier.config.settings import AppConfig
from paper_classifier.infrastructure.rag.vector_store_service import VectorStoreService


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def similarity_search_with_relevance_scores(
        self, query: str, k: int
    ) -> list[tuple[SimpleNamespace, float]]:
        self.calls.append((query, k))
        return [
            (
                SimpleNamespace(metadata={
                    "discipline": "public administration",
                    "parent": "management",
                    "level": "child",
                }),
                0.96,
            ),
            (
                SimpleNamespace(metadata={
                    "discipline": "administrative management",
                    "parent": "management",
                    "level": "child",
                }),
                0.90,
            ),
        ]


class VectorStoreServiceTests(unittest.TestCase):
    def test_search_uses_one_direct_top_five_query(self) -> None:
        service = object.__new__(VectorStoreService)
        service.config = AppConfig(rag_top_k=5)
        service.vector_store = FakeVectorStore()

        results = service.similarity_search("public administration")

        self.assertEqual(
            [("public administration", 5)], service.vector_store.calls
        )
        self.assertEqual("public administration", results[0]["discipline"])
        self.assertEqual("management", results[0]["parent"])
        self.assertEqual("child", results[0]["level"])


if __name__ == "__main__":
    unittest.main()
