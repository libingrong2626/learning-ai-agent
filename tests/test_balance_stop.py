from __future__ import annotations

import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from paper_classifier.common.api_errors import ApiBalanceExhaustedError
from paper_classifier.config.settings import AppConfig
from paper_classifier.services.rag_classifier_app import RAGClassifierApp


class BalanceFailingClassifier:
    def __init__(self) -> None:
        self.call_count = 0

    def classify_batch(
        self, papers_batch: list[dict], on_paper_classified=None
    ) -> None:
        self.call_count += 1
        raise ApiBalanceExhaustedError("API 账户余额不足")


class RecordingOutputStore:
    def __init__(self) -> None:
        self.failed_save_count = 0

    def recover_processed_ids(self, paper_df, existing_ids):
        return existing_ids

    def collect_success_ids(self, paper_df):
        return set()

    def save_failed(self, failed_df):
        self.failed_save_count += 1


class BalanceStopPipelineTests(unittest.TestCase):
    def test_balance_error_stops_without_saving_current_batch_as_failed(self) -> None:
        app = object.__new__(RAGClassifierApp)
        app.config = AppConfig(batch_size=5, retry_times=3)
        app.logger = logging.getLogger(self.id())
        app.logger.addHandler(logging.NullHandler())
        app.log_file = Path("test.log")
        app.kb_loader = SimpleNamespace(load=lambda: None, all_parents=["数学"])
        app.vector_store_service = object()
        app.memory_cache = object()
        app.paper_repo = SimpleNamespace(
            load=lambda: pd.DataFrame(
                {
                    "title": ["论文标题"],
                    "abstract": ["论文摘要"],
                    "论文唯一标识": ["paper-id"],
                }
            )
        )
        app.checkpoint_store = SimpleNamespace(
            read=lambda: set(),
            write=lambda processed_ids: None,
        )
        output_store = RecordingOutputStore()
        app.output_store = output_store
        app._log_start_banner = lambda: None
        app._log_batch_banner = lambda *args: None
        app._init_vector_store = lambda: None

        failing_classifier = BalanceFailingClassifier()
        with patch(
            "paper_classifier.services.rag_classifier_app.RAGClassifier",
            return_value=failing_classifier,
        ):
            app.run_pipeline()

        self.assertEqual(1, failing_classifier.call_count)
        self.assertEqual(0, output_store.failed_save_count)


if __name__ == "__main__":
    unittest.main()
