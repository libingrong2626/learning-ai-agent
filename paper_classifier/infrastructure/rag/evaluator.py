#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from typing import Any


class Evaluator:
    """RAG 检索与分类评估器。

    评估指标：
    - 检索召回率 (Recall@K)：正确的一级学科是否出现在 RAG 检索的 top-k 中
    - 检索平均相关度：正确方向在检索结果中的平均相似度得分
    - 分类准确率 (Accuracy)：最终 LLM 分类结果是否与真实标签一致
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.total_queries = 0
        self.retrieval_hits = 0
        self.classification_correct = 0
        self.total_classified = 0
        self.retrieval_scores: list[float] = []
        self._details: list[dict[str, Any]] = []

    def record(
        self,
        query: str,
        ground_truth_parent: str,
        rag_results: list[dict],
        predicted_category: str | None = None,
    ) -> dict[str, bool]:
        """记录单次查询的检索和分类结果。"""
        self.total_queries += 1

        retrieved_parents = [r["parent"] for r in rag_results]
        retrieval_hit = ground_truth_parent in retrieved_parents

        if retrieval_hit:
            self.retrieval_hits += 1
            for r in rag_results:
                if r["parent"] == ground_truth_parent:
                    self.retrieval_scores.append(r["score"])
                    break

        classification_correct = None
        if predicted_category is not None:
            self.total_classified += 1
            classification_correct = (predicted_category == ground_truth_parent)
            if classification_correct:
                self.classification_correct += 1

        self._details.append({
            "query": query,
            "ground_truth": ground_truth_parent,
            "retrieval_hit": retrieval_hit,
            "retrieved_parents": retrieved_parents,
            "predicted": predicted_category,
            "classification_correct": classification_correct,
        })

        return {
            "retrieval_hit": retrieval_hit,
            "classification_correct": classification_correct or False,
        }

    def get_metrics(self) -> dict[str, float | int]:
        """计算并返回所有评估指标。"""
        recall = (
            self.retrieval_hits / self.total_queries
            if self.total_queries > 0 else 0.0
        )
        accuracy = (
            self.classification_correct / self.total_classified
            if self.total_classified > 0 else 0.0
        )
        avg_score = (
            sum(self.retrieval_scores) / len(self.retrieval_scores)
            if self.retrieval_scores else 0.0
        )

        return {
            "total_queries": self.total_queries,
            "retrieval_recall": round(recall, 4),
            "retrieval_hits": self.retrieval_hits,
            "avg_retrieval_score": round(avg_score, 4),
            "classification_accuracy": round(accuracy, 4),
            "classification_correct": self.classification_correct,
            "total_classified": self.total_classified,
        }

    def print_report(self) -> str:
        """生成格式化的评估报告文本。"""
        metrics = self.get_metrics()

        report_lines = [
            "",
            "=" * 60,
            "RAG 评估报告",
            "=" * 60,
            f"检索测试总数:       {metrics['total_queries']}",
            f"检索召回率(Recall@K): {metrics['retrieval_recall']:.2%}"
            f"  ({metrics['retrieval_hits']}/{metrics['total_queries']})",
            f"平均检索相关度:       {metrics['avg_retrieval_score']:.4f}",
            "-" * 60,
            f"分类测试总数:         {metrics['total_classified']}",
            f"分类准确率(Accuracy): {metrics['classification_accuracy']:.2%}"
            f"  ({metrics['classification_correct']}/{metrics['total_classified']})",
            "=" * 60,
            "",
        ]

        report = "\n".join(report_lines)
        self.logger.info(report)
        return report

    def get_details(self) -> list[dict[str, Any]]:
        """返回所有详细评估记录。"""
        return self._details
