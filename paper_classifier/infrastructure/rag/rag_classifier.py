#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, TYPE_CHECKING

from langchain_openai import ChatOpenAI

from paper_classifier.common.api_errors import (
    ApiBalanceExhaustedError,
    raise_for_api_balance_error,
)
from paper_classifier.config.prompts import RAG_SYSTEM_PROMPT
from paper_classifier.config.settings import AppConfig

if TYPE_CHECKING:
    from paper_classifier.infrastructure.rag.memory_cache import MemoryCache
    from paper_classifier.infrastructure.rag.vector_store_service import VectorStoreService


class RAGClassifier:
    """RAG 增强分类器（基于 LangChain）。

    分类流程：
    1. 使用 LLM 对论文进行初步方向判断（自主分类）
    2. 将 LLM 的判断结果作为查询，在 Chroma 向量数据库中进行相似度检索
    3. 获取最相关的一级/子学科及其对应的一级学科
    4. 结合 RAG 检索结果，由 LLM 给出最终的一级学科分类
    """

    def __init__(
        self,
        config: AppConfig,
        logger: logging.Logger,
        vector_store: "VectorStoreService",
        valid_parents: list[str] | None = None,
        memory_cache: "MemoryCache | None" = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.vector_store = vector_store
        self.valid_parents = valid_parents  # 知识库中所有合法一级学科名
        self.memory_cache = memory_cache    # 记忆缓存（LLM初判→终判映射）

        self.llm = ChatOpenAI(
            model=config.llm_model_name,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            timeout=config.llm_timeout,
        )

    def classify_single_paper(
        self,
        title: str,
        abstract: str,
        filename: str = "",
    ) -> dict[str, Any]:
        """对单篇论文进行 RAG 增强分类（带记忆加速）。

        分类依据：abstract 优先；abstract 为空（含空白字符）时用
        filename 兜底；两者都空时退回 title。

        流程：
        1. LLM 初步方向判断
        2. 查记忆缓存 → 命中则直接返回
        3. 未命中 → 向量检索 + LLM 终判 → 写入记忆
        """
        paper_info = self._build_paper_info(title, abstract, filename)

        # Step 1: LLM 初次方向判断
        llm_initial = self._llm_initial_classify(paper_info)
        self.logger.debug(f"LLM 初始判断: {llm_initial}")

        # Step 2: 查记忆缓存
        if self.memory_cache:
            cached = self.memory_cache.lookup(llm_initial)
            if cached and (
                not self.valid_parents or cached in self.valid_parents
            ):
                return {
                    "category": cached,
                    "llm_initial": llm_initial,
                    "rag_results": [],
                    "from_memory": True,
                }
            if cached:
                self.logger.warning(
                    f"忽略不在当前学科库中的旧缓存结果: {cached}"
                )

        # Step 3: 向量数据库检索
        rag_results = self.vector_store.similarity_search(
            llm_initial, top_k=self.config.rag_top_k
        )
        self.logger.debug(f"RAG 检索结果: {rag_results[:3]}")

        # Step 4: LLM 最终分类
        final_category = self._llm_final_classify(
            paper_info, llm_initial, rag_results
        )

        # Step 5: 写入记忆缓存
        if self.memory_cache:
            self.memory_cache.record(llm_initial, final_category)

        return {
            "category": final_category,
            "llm_initial": llm_initial,
            "rag_results": rag_results,
            "from_memory": False,
        }

    def classify_batch(
        self,
        papers_batch: list[dict[str, Any]],
        on_paper_classified: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any] | None:
        """批量分类论文，返回与原始分类器兼容的 JSON 字典。

        on_paper_classified(idx, category)：可选回调，每篇论文分类
        成功后立即调用，用于逐篇落盘（中断不丢已分类结果）。
        """
        results = {}

        for idx, paper in enumerate(papers_batch):
            title = paper.get(self.config.title_column, "")
            abstract = paper.get(self.config.abstract_column, "")
            filename = paper.get(self.config.filename_column, "")

            try:
                result = self.classify_single_paper(title, abstract, filename)
                results[str(idx)] = result["category"]
                if on_paper_classified:
                    on_paper_classified(idx, result["category"])
                self.logger.info(
                    f"论文 [{idx}] {title[:30]}... -> "
                    f"LLM初判: {result['llm_initial']} -> "
                    f"RAG终判: {result['category']}"
                )
            except ApiBalanceExhaustedError:
                raise
            except Exception as exc:
                raise_for_api_balance_error(exc)
                self.logger.error(f"论文 [{idx}] 分类失败: {exc}")
                results[str(idx)] = None

        return {k: v for k, v in results.items() if v is not None} or None

    def _llm_initial_classify(self, paper_info: str) -> str:
        """Step 1: LLM 自主判断论文可能的研究方向（不用 RAG）。"""
        prompt = (
            "请识别以下印尼语学术论文最核心的研究方向，"
            "输出一个尽可能具体、可用于学科知识库检索的中文子学科"
            "或研究方向名称。不要输出解释、句子或多个备选：\n\n"
            f"{paper_info}"
        )

        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            result = result.strip('"\'').strip()
            return result if result else "未知方向"
        except Exception as exc:
            raise_for_api_balance_error(exc)
            self.logger.error(f"LLM 初始分类失败: {exc}")
            return "未知方向"

    def _llm_final_classify(
        self,
        paper_info: str,
        llm_initial: str,
        rag_results: list[dict],
    ) -> str:
        """Step 3: 结合 RAG 检索结果，由 LLM 给出最终一级学科分类。"""

        fallback_category = next(
            (
                str(result.get("parent", "")).strip()
                for result in rag_results
                if str(result.get("parent", "")).strip()
            ),
            llm_initial,
        )

        candidate_parents = list(dict.fromkeys(
            str(result.get("parent", "")).strip()
            for result in rag_results
            if str(result.get("parent", "")).strip()
        ))
        allowed_parents = candidate_parents or list(self.valid_parents or [])

        rag_context_lines = []
        for i, r in enumerate(rag_results):
            discipline = str(
                r.get("discipline", r.get("sub_direction", r["parent"]))
            ).strip()
            level_name = (
                "一级学科" if r.get("level") == "parent" else "子学科"
            )
            rag_context_lines.append(
                f"  {i + 1}. 匹配{level_name}：{discipline}\n"
                f"     所属一级学科：{r['parent']}\n"
                f"     检索相关度：{r['score']:.4f}"
            )
        rag_context = "\n".join(rag_context_lines)

        # 只允许从实际 Top-K 结果对应的一级学科中选择。
        valid_parents_hint = ""
        if allowed_parents:
            parents_str = "、".join(allowed_parents)
            valid_parents_hint = (
                f"\n\n【重要】你必须从以下一级学科名称中选择一个，"
                f"输出名称必须与列表中的完全一致，不可自行简化或改写：\n"
                f"{parents_str}"
            )

        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\n"
            f"论文信息：\n{paper_info}\n\n"
            f"LLM 初步判断方向：{llm_initial}\n\n"
            f"通过向量数据库检索到的相关学科参考方向：\n{rag_context}"
            f"{valid_parents_hint}\n\n"
            "请综合以上信息，输出该论文的一级学科分类。"
            '严格按 JSON 格式输出：{"0": "一级学科名称"}'
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            parsed = self._parse_json_response(content)
            if parsed and "0" in parsed:
                category = str(parsed["0"]).strip()
                if not allowed_parents or category in allowed_parents:
                    return category
                self.logger.warning(
                    f"LLM 终判输出了 Top-K 之外的一级学科“{category}”，"
                    f"回退到检索首位“{fallback_category}”"
                )
            else:
                self.logger.warning(
                    f"LLM 终判未返回有效 JSON，"
                    f"回退到检索首位“{fallback_category}”"
                )
            return fallback_category
        except Exception as exc:
            raise_for_api_balance_error(exc)
            self.logger.error(f"LLM 最终分类失败: {exc}")
            return fallback_category

    def _build_paper_info(
        self, title: str, abstract: str, filename: str = ""
    ) -> str:
        """构建传给 LLM 的论文信息文本。

        优先级：abstract（非空白）→ filename → title → 兜底提示。
        只取优先级最高的一项，避免与分类依据无关的信息干扰 LLM。
        """
        abstract_clean = str(abstract or "").strip()
        if abstract_clean:
            return f"摘要：{abstract_clean}"

        filename_clean = str(filename or "").strip()
        if filename_clean:
            return f"文件名：{filename_clean}"

        title_clean = str(title or "").strip()
        if title_clean:
            return f"标题：{title_clean}"

        return "（无摘要和文件名）"

    def _parse_json_response(self, content: str) -> dict | None:
        """健壮的 JSON 解析。"""
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        fixed = content.replace("'", '"')
        try:
            parsed = json.loads(fixed)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return None
