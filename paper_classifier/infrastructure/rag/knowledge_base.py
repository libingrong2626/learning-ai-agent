#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import difflib
import logging
import re
from pathlib import Path

from paper_classifier.config.settings import AppConfig


class KnowledgeBaseLoader:
    """学科知识库加载器。

    CSV 每行表示一个一级学科，第一列为“一级学科”，第二列为用
    中文分号分隔的“子学科”。一级学科和每个子学科分别生成向量文档。
    """

    EXPECTED_COLUMNS = ("一级学科", "子学科")
    SUPPORTED_ENCODINGS = ("utf-8-sig", "gb18030")

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.parent_to_children: dict[str, list[str]] = {}
        self.child_to_parent: dict[str, str] = {}
        self.ambiguous_child_to_parents: dict[str, list[str]] = {}
        self.all_parents: list[str] = []

    def load(self) -> dict[str, list[str]]:
        """加载知识库 CSV 并构建一级学科与子学科映射。"""
        kb_path = self.config.knowledge_base_path

        if not kb_path.exists():
            raise FileNotFoundError(f"知识库文件不存在: {kb_path}")

        rows, encoding = self._read_csv_rows(kb_path)
        parent_to_children: dict[str, list[str]] = {}
        child_to_parent: dict[str, str] = {}
        ambiguous_child_to_parents: dict[str, list[str]] = {}

        for row_number, row in enumerate(rows, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 2:
                raise ValueError(
                    f"知识库第 {row_number} 行应有 2 列，实际为 {len(row)} 列"
                )

            parent, children_text = (value.strip() for value in row)
            if not parent or not children_text:
                raise ValueError(
                    f"知识库第 {row_number} 行的一级学科或子学科为空"
                )
            if parent in parent_to_children:
                raise ValueError(f"知识库一级学科重复: {parent}")

            children = [
                child.strip()
                for child in re.split(r"[；;]", children_text)
                if child.strip()
            ]
            if not children:
                raise ValueError(f"知识库一级学科“{parent}”没有有效子学科")
            if len(children) != len(set(children)):
                raise ValueError(f"知识库一级学科“{parent}”存在重复子学科")

            for child in children:
                if child in ambiguous_child_to_parents:
                    ambiguous_child_to_parents[child].append(parent)
                    continue
                existing_parent = child_to_parent.get(child)
                if existing_parent:
                    ambiguous_child_to_parents[child] = [existing_parent, parent]
                    del child_to_parent[child]
                else:
                    child_to_parent[child] = parent

            parent_to_children[parent] = children

        if not parent_to_children:
            raise ValueError("知识库 CSV 没有有效数据行")

        self.parent_to_children = parent_to_children
        self.child_to_parent = child_to_parent
        self.ambiguous_child_to_parents = ambiguous_child_to_parents
        self.all_parents = list(parent_to_children)

        total_children = sum(len(children) for children in parent_to_children.values())
        self.logger.info(
            f"知识库加载完成：{len(parent_to_children)} 个一级学科，"
            f"{total_children} 个子学科（编码: {encoding}）"
        )
        if ambiguous_child_to_parents:
            self.logger.warning(
                f"知识库包含 {len(ambiguous_child_to_parents)} 个跨学科同名词，"
                "不对它们建立唯一一级学科映射"
            )
        return self.parent_to_children

    def _read_csv_rows(self, kb_path: Path) -> tuple[list[list[str]], str]:
        """读取 UTF-8 或 GBK/GB18030 编码的两列 CSV。"""
        last_decode_error: UnicodeDecodeError | None = None

        for encoding in self.SUPPORTED_ENCODINGS:
            try:
                with kb_path.open("r", encoding=encoding, newline="") as file:
                    reader = csv.reader(file)
                    header = next(reader, None)
                    if header is None:
                        raise ValueError("知识库 CSV 为空")

                    normalized_header = tuple(value.strip() for value in header)
                    if normalized_header != self.EXPECTED_COLUMNS:
                        raise ValueError(
                            "知识库 CSV 列名必须依次为“一级学科”和“子学科”，"
                            f"实际为: {normalized_header}"
                        )
                    return list(reader), encoding
            except UnicodeDecodeError as exc:
                last_decode_error = exc

        raise ValueError(
            f"知识库 CSV 编码无法识别，请使用 UTF-8 或 GBK: {last_decode_error}"
        )

    def get_parent(self, sub_direction: str) -> str | None:
        """根据子方向查找对应的一级学科。"""
        return self.child_to_parent.get(sub_direction)

    def get_ground_truth(self, query: str) -> str | None:
        """获取查询的 ground truth 一级学科。

        三种输入均能识别：
        - 输入子学科名称（如"数学史"）-> 返回其一级学科（"数学"）
        - 输入一级学科名称（如"经济学"）-> 直接返回"经济学"
        - 输入模糊学科名（如"环境科学"）-> 模糊匹配到"环境科学技术及资源科学技术"
        """
        # 优先检查是否为子方向
        parent = self.child_to_parent.get(query)
        if parent:
            return parent
        # 再检查是否为一级学科名称本身
        if query in self.parent_to_children:
            return query
        # 模糊匹配兜底：复用 normalize_parent_name 的逻辑
        normalized = self.normalize_parent_name(query)
        if normalized != query and normalized in self.parent_to_children:
            return normalized
        return None

    def normalize_parent_name(self, name: str) -> str:
        """将 LLM 输出的学科名归一化到知识库中的合法一级学科名。

        匹配策略（优先级从高到低）：
        1. 精确匹配：输出名与某个一级学科名完全一致
        2. 包含匹配：输出名包含某个一级学科名，或某个一级学科名包含输出名
           例：LLM 输出 "土木工程" -> 匹配到 "土木建筑工程"
        3. 字符相似度：SequenceMatcher 计算相似度，阈值 0.85
           例：LLM 输出 "环境科学技术及资源科学技"（截断）-> "环境科学技术及资源科学技术"
        4. 无匹配时返回原始输出
        """
        if not name:
            return name

        name = name.strip()

        # 1. 精确匹配
        if name in self.parent_to_children:
            return name

        # 2. 子串包含匹配（优先匹配更长的学科名，避免短名误匹配）
        best_match = None
        best_len = 0

        for parent in self.parent_to_children:
            if name in parent or parent in name:
                match_len = min(len(name), len(parent))
                if match_len > best_len:
                    best_len = match_len
                    best_match = parent

        if best_match:
            return best_match

        # 3. 字符相似度匹配（处理截断、笔误等场景）
        best_similarity = 0.0
        best_sim_match = None

        for parent in self.parent_to_children:
            ratio = difflib.SequenceMatcher(None, name, parent).ratio()
            if ratio > best_similarity:
                best_similarity = ratio
                best_sim_match = parent

        # 阈值 0.85：允许少量字符差异（如截断1字、漏字等）
        if best_similarity >= 0.85 and best_sim_match:
            if self.logger:
                self.logger.info(
                    f"模糊匹配: '{name}' -> '{best_sim_match}' "
                    f"(相似度: {best_similarity:.3f})"
                )
            return best_sim_match

        # 4. 无匹配，返回原始名称
        return name

    def get_all_documents_for_vector_store(self) -> list[dict[str, str]]:
        """为向量数据库生成文档列表。

        文本分割策略：每个一级学科和每个子学科都单独生成向量文档。
        所有文档都在元数据中保存所属一级学科，便于终判只输出一级学科。

        示例：
        - 一级文档: "一级学科：数学"
        - 子学科文档: "子学科：数学史\n所属一级学科：数学"
        """
        documents = []
        for parent, children in self.parent_to_children.items():
            documents.append({
                "page_content": f"一级学科：{parent}",
                "metadata": {
                    "parent": parent,
                    "discipline": parent,
                    "level": "parent",
                }
            })

            for child in children:
                documents.append({
                    "page_content": (
                        f"子学科：{child}\n"
                        f"所属一级学科：{parent}"
                    ),
                    "metadata": {
                        "parent": parent,
                        "discipline": child,
                        "level": "child",
                    }
                })
        return documents
