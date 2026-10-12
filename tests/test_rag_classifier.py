from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from paper_classifier.common.api_errors import ApiBalanceExhaustedError
from paper_classifier.infrastructure.rag.rag_classifier import RAGClassifier


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_prompt = ""

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.last_prompt = prompt
        return SimpleNamespace(content=self.content)


class FailingLLM:
    def __init__(self, message: str) -> None:
        self.message = message

    def invoke(self, prompt: str) -> SimpleNamespace:
        raise RuntimeError(self.message)


class FakeVectorStore:
    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def similarity_search(self, query: str, top_k: int | None = None) -> list[dict]:
        return self.results


class RAGClassifierFinalDecisionTests(unittest.TestCase):
    def _make_classifier(self, llm_content: str) -> RAGClassifier:
        classifier = object.__new__(RAGClassifier)
        classifier.logger = logging.getLogger(self.id())
        classifier.logger.addHandler(logging.NullHandler())
        classifier.valid_parents = ["数学", "经济学"]
        classifier.llm = FakeLLM(llm_content)
        return classifier

    def test_valid_parent_is_returned_for_math_history(self) -> None:
        classifier = self._make_classifier('{"0": "数学"}')

        result = classifier._llm_final_classify(
            paper_info="标题：数学史",
            llm_initial="数学史",
            rag_results=[
                {
                    "parent": "数学",
                    "discipline": "数学史",
                    "level": "child",
                    "score": 0.99,
                }
            ],
        )

        self.assertEqual("数学", result)

    def test_child_output_falls_back_to_retrieved_parent(self) -> None:
        classifier = self._make_classifier('{"0": "数学史"}')

        result = classifier._llm_final_classify(
            paper_info="标题：数学史",
            llm_initial="数学史",
            rag_results=[
                {
                    "parent": "数学",
                    "discipline": "数学史",
                    "level": "child",
                    "score": 0.99,
                }
            ],
        )

        self.assertEqual("数学", result)

    def test_parent_outside_top_k_is_rejected(self) -> None:
        classifier = self._make_classifier('{"0": "经济学"}')

        result = classifier._llm_final_classify(
            paper_info="标题：数学史",
            llm_initial="数学史",
            rag_results=[
                {
                    "parent": "数学",
                    "discipline": "数学史",
                    "level": "child",
                    "score": 0.99,
                }
            ],
        )

        self.assertEqual("数学", result)
        self.assertNotIn("经济学", classifier.llm.last_prompt)

    def test_balance_error_is_not_replaced_by_retrieval_fallback(self) -> None:
        classifier = self._make_classifier("")
        classifier.llm = FailingLLM(
            "AccountOverdue: account is overdue, please recharge"
        )

        with self.assertRaises(ApiBalanceExhaustedError):
            classifier._llm_final_classify(
                paper_info="标题：数学史",
                llm_initial="数学史",
                rag_results=[
                    {
                        "parent": "数学",
                        "discipline": "数学史",
                        "level": "child",
                        "score": 0.99,
                    }
                ],
            )

    def test_balance_error_is_propagated_from_initial_classification(self) -> None:
        classifier = self._make_classifier("")
        classifier.llm = FailingLLM("账户余额不足，请充值后重试")

        with self.assertRaises(ApiBalanceExhaustedError):
            classifier._llm_initial_classify("标题：数学史")


class PaperInfoBuildingTests(unittest.TestCase):
    """_build_paper_info 优先级：abstract → filename → title → 兜底。"""

    def _make_classifier(self) -> RAGClassifier:
        classifier = object.__new__(RAGClassifier)
        classifier.logger = logging.getLogger(self.id())
        classifier.logger.addHandler(logging.NullHandler())
        return classifier

    def test_abstract_takes_priority(self) -> None:
        info = self._make_classifier()._build_paper_info(
            title="T", abstract="真实摘要", filename="file.pdf"
        )
        self.assertEqual("摘要：真实摘要", info)

    def test_blank_abstract_falls_back_to_filename(self) -> None:
        classifier = self._make_classifier()
        for blank in ("", "   ", "\n\t ", None):
            info = classifier._build_paper_info(
                title="T", abstract=blank, filename="paper1.pdf"
            )
            self.assertEqual("文件名：paper1.pdf", info)

    def test_blank_abstract_and_filename_fall_back_to_title(self) -> None:
        info = self._make_classifier()._build_paper_info(
            title="标题名", abstract="  ", filename=""
        )
        self.assertEqual("标题：标题名", info)

    def test_all_blank_returns_placeholder(self) -> None:
        info = self._make_classifier()._build_paper_info(
            title="", abstract=" ", filename=None
        )
        self.assertEqual("（无摘要和文件名）", info)


class ClassifyFallbackTests(unittest.TestCase):
    """管线模式下 abstract 为空时用 filename 分类。"""

    def _make_classifier(self, llm_content: str) -> RAGClassifier:
        classifier = object.__new__(RAGClassifier)
        classifier.logger = logging.getLogger(self.id())
        classifier.logger.addHandler(logging.NullHandler())
        classifier.valid_parents = ["数学"]
        classifier.config = SimpleNamespace(rag_top_k=5)
        classifier.vector_store = FakeVectorStore([
            {
                "parent": "数学",
                "discipline": "数学史",
                "level": "child",
                "score": 0.99,
            }
        ])
        classifier.memory_cache = None
        classifier.llm = FakeLLM(llm_content)
        return classifier

    def test_blank_abstract_uses_filename_for_classification(self) -> None:
        classifier = self._make_classifier('{"0": "数学"}')

        result = classifier.classify_single_paper(
            title="某标题", abstract="   ", filename="paper1.pdf"
        )

        self.assertEqual("数学", result["category"])
        self.assertIn("文件名：paper1.pdf", classifier.llm.last_prompt)
        self.assertNotIn("某标题", classifier.llm.last_prompt)

    def test_present_abstract_ignores_filename(self) -> None:
        classifier = self._make_classifier('{"0": "数学"}')

        result = classifier.classify_single_paper(
            title="某标题", abstract="真实摘要内容", filename="paper1.pdf"
        )

        self.assertEqual("数学", result["category"])
        self.assertIn("摘要：真实摘要内容", classifier.llm.last_prompt)
        self.assertNotIn("paper1.pdf", classifier.llm.last_prompt)


class ClassifyBatchCallbackTests(unittest.TestCase):
    """classify_batch 逐篇成功即触发 on_paper_classified 回调。"""

    def _make_classifier(self, llm_content: str = '{"0": "数学"}') -> RAGClassifier:
        classifier = object.__new__(RAGClassifier)
        classifier.logger = logging.getLogger(self.id())
        classifier.logger.addHandler(logging.NullHandler())
        classifier.valid_parents = ["数学"]
        classifier.config = SimpleNamespace(
            rag_top_k=5, title_column="title", abstract_column="abstract",
            filename_column="filename",
        )
        classifier.vector_store = FakeVectorStore([
            {
                "parent": "数学",
                "discipline": "数学史",
                "level": "child",
                "score": 0.99,
            }
        ])
        classifier.memory_cache = None
        classifier.llm = FakeLLM(llm_content)
        return classifier

    def test_callback_receives_index_and_category_per_success(self) -> None:
        classifier = self._make_classifier()
        calls: list[tuple[int, str]] = []
        batch = [
            {"title": "T1", "abstract": "A1", "filename": "f1.pdf"},
            {"title": "T2", "abstract": "A2", "filename": "f2.pdf"},
        ]

        result = classifier.classify_batch(
            batch, lambda idx, cat: calls.append((idx, cat))
        )

        self.assertIsNotNone(result)
        self.assertEqual([(0, "数学"), (1, "数学")], calls)

    def test_failed_paper_skips_callback_but_others_still_trigger(self) -> None:
        classifier = self._make_classifier('{"0": "数学"}')
        classifier.llm = FailingLLM("临时网络错误")
        calls: list[tuple[int, str]] = []
        batch = [
            {"title": "T1", "abstract": "A1", "filename": "f1.pdf"},
            {"title": "T2", "abstract": "A2", "filename": "f2.pdf"},
        ]

        result = classifier.classify_batch(batch, calls.append)

        # 两篇都失败（LLM 不可用），返回 None，不触发任何回调
        self.assertIsNone(result)
        self.assertEqual([], calls)

    def test_balance_error_propagates_without_callback(self) -> None:
        classifier = self._make_classifier()
        classifier.llm = FailingLLM("AccountOverdue: account is overdue")
        calls: list[tuple[int, str]] = []
        batch = [
            {"title": "T1", "abstract": "A1", "filename": "f1.pdf"},
        ]

        with self.assertRaises(ApiBalanceExhaustedError):
            classifier.classify_batch(batch, calls.append)

        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
