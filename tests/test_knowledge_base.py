from __future__ import annotations

import logging
import unittest
from pathlib import Path

from paper_classifier.config.settings import AppConfig
from paper_classifier.infrastructure.rag.knowledge_base import KnowledgeBaseLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class KnowledgeBaseLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger(self.id())
        self.logger.addHandler(logging.NullHandler())
        self.config = AppConfig(
            knowledge_base_path=PROJECT_ROOT / "学科.csv",
        )

    def test_every_parent_and_child_gets_an_independent_document(self) -> None:
        loader = KnowledgeBaseLoader(self.config, self.logger)

        knowledge_base = loader.load()
        documents = loader.get_all_documents_for_vector_store()

        self.assertEqual(62, len(knowledge_base))
        self.assertEqual(3454, sum(len(children) for children in knowledge_base.values()))
        self.assertEqual(62 + 3454, len(documents))
        self.assertEqual(
            62,
            sum(document["metadata"]["level"] == "parent" for document in documents),
        )
        self.assertEqual(
            3454,
            sum(document["metadata"]["level"] == "child" for document in documents),
        )

    def test_cross_disciplinary_terms_are_not_forced_to_one_parent(self) -> None:
        loader = KnowledgeBaseLoader(self.config, self.logger)
        loader.load()

        self.assertIsNone(loader.get_parent("心理人类学"))
        self.assertEqual(
            ["生物学", "心理学"],
            loader.ambiguous_child_to_parents["心理人类学"],
        )

    def test_math_history_maps_to_math_parent(self) -> None:
        loader = KnowledgeBaseLoader(self.config, self.logger)
        loader.load()

        self.assertEqual("数学", loader.get_parent("数学史"))
        self.assertEqual("数学", loader.get_ground_truth("数学史"))

        math_history_document = next(
            document
            for document in loader.get_all_documents_for_vector_store()
            if document["metadata"]["discipline"] == "数学史"
        )
        self.assertEqual("数学", math_history_document["metadata"]["parent"])
        self.assertEqual("child", math_history_document["metadata"]["level"])
        self.assertEqual(
            "子学科：数学史\n所属一级学科：数学",
            math_history_document["page_content"],
        )


if __name__ == "__main__":
    unittest.main()
