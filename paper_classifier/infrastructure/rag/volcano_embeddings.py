#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
火山引擎多模态嵌入模型适配器。

doubao-embedding-vision 系列模型仅支持 /api/v3/embeddings/multimodal REST 接口，
不兼容标准 OpenAI /embeddings 接口。本类实现 LangChain Embeddings 协议，
将调用转发到火山引擎的多模态嵌入端点。
"""

from __future__ import annotations

import logging
import time
from typing import List

import requests
from langchain_core.embeddings import Embeddings

from paper_classifier.common.api_errors import (
    ApiBalanceExhaustedError,
    raise_for_api_balance_error,
)


class VolcanoMultimodalEmbeddings(Embeddings):
    """火山引擎多模态嵌入模型适配器（LangChain Embeddings 协议）。

    通过 /api/v3/embeddings/multimodal 端点调用文本嵌入，
    兼容 doubao-embedding-vision 系列模型。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        logger: logging.Logger | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        batch_size: int = 5,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.logger = logger
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size

        # 智能 URL 构建：兼容用户传入 base_url 或完整端点 URL
        clean_url = base_url.rstrip("/")
        if clean_url.endswith("/embeddings/multimodal"):
            # 用户已提供完整端点路径，直接使用
            self.endpoint = clean_url
        else:
            # 用户只提供 base_url，追加端点路径
            self.endpoint = f"{clean_url}/embeddings/multimodal"

        if self.logger:
            self.logger.info(f"嵌入模型端点: {self.endpoint}")

        # 请求头
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档文本。

        将文本列表按小批次发送到多模态嵌入接口，返回对应的向量列表。
        多模态接口对批量大小有限制，使用较小的 batch_size 避免 500 错误。
        """
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._call_api_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            if self.logger and len(all_embeddings) % 10 == 0 and i > 0:
                self.logger.info(
                    f"嵌入进度: {min(len(all_embeddings), len(texts))}/{len(texts)}"
                )

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询文本。"""
        result = self._call_api_with_retry([text])
        return result[0]

    def _call_api_with_retry(self, texts: List[str]) -> List[List[float]]:
        """带重试的 API 调用，处理瞬时 500 错误。"""
        for attempt in range(self.max_retries):
            try:
                return self._call_api(texts)
            except ApiBalanceExhaustedError:
                raise
            except RuntimeError as exc:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    if self.logger:
                        self.logger.warning(
                            f"嵌入 API 第 {attempt + 1} 次失败，"
                            f"{wait_time} 秒后重试: {exc}"
                        )
                    time.sleep(wait_time)
                else:
                    raise

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用火山引擎多模态嵌入 API。

        请求格式：
        {
            "model": "doubao-embedding-vision-251215",
            "input": [
                {"type": "text", "text": "文本内容1"},
                {"type": "text", "text": "文本内容2"},
                ...
            ]
        }
        """
        # 构建多模态格式的 input
        input_items = [{"type": "text", "text": text} for text in texts]

        payload = {
            "model": self.model,
            "input": input_items,
        }

        try:
            response = requests.post(
                self.endpoint,
                headers=self._headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()

            # 从响应中提取嵌入向量
            if "data" not in data:
                raise ValueError(f"API 响应缺少 data 字段: {list(data.keys())}")

            raw_data = data["data"]

            # 调试日志：打印 data 字段的实际类型和结构
            if self.logger:
                self.logger.debug(
                    f"嵌入 API 响应顶层 keys: {list(data.keys())}"
                )
                self.logger.debug(
                    f"嵌入 API data 字段类型: {type(raw_data).__name__}"
                )
                if isinstance(raw_data, dict):
                    self.logger.debug(
                        f"嵌入 API data 字段 keys: {list(raw_data.keys())}"
                    )
                    # 打印每个 key 对应的值类型和前 80 字符
                    for k, v in raw_data.items():
                        val_preview = str(v)[:80]
                        self.logger.debug(
                            f"  data['{k}']: type={type(v).__name__}, "
                            f"preview={val_preview}"
                        )
                elif isinstance(raw_data, list) and len(raw_data) > 0:
                    self.logger.debug(
                        f"嵌入 API data[0] 类型: {type(raw_data[0]).__name__}, "
                        f"内容: {str(raw_data[0])[:200]}"
                    )

            # 防御性解析：兼容多种响应格式
            embeddings: List[List[float]] = []

            if isinstance(raw_data, dict):
                # ── data 是字典（火山引擎多模态接口可能的格式）──
                # 格式 C: {"embedding": [0.1, 0.2, ...]}  单条结果
                # 格式 D: {"embeddings": [[...], [...], ...]}  多条结果
                if "embeddings" in raw_data:
                    embeddings = raw_data["embeddings"]
                elif "embedding" in raw_data:
                    # 单条嵌入，包装成列表
                    embeddings = [raw_data["embedding"]]
                else:
                    raise ValueError(
                        f"data 字典中找不到 embedding/embeddings 字段，"
                        f"可用 keys: {list(raw_data.keys())}"
                    )

            elif isinstance(raw_data, list):
                if len(raw_data) == 0:
                    raise ValueError("API 返回 data 数组为空")

                first = raw_data[0]
                if isinstance(first, dict):
                    # 格式 A（标准 OpenAI）: [{"embedding": [...], "index": 0}, ...]
                    sorted_items = sorted(
                        raw_data, key=lambda x: x.get("index", 0)
                    )
                    embeddings = [item["embedding"] for item in sorted_items]
                elif isinstance(first, list):
                    # 格式 B: [[0.1, 0.2, ...], ...]  直接是向量数组
                    embeddings = raw_data
                else:
                    raise ValueError(
                        f"无法解析 data 数组元素类型: "
                        f"{type(first).__name__}: {str(first)[:200]}"
                    )
            else:
                raise ValueError(
                    f"data 字段类型异常: {type(raw_data).__name__}"
                )

            if self.logger:
                self.logger.debug(
                    f"嵌入 API 调用成功: {len(texts)} 条文本 -> "
                    f"{len(embeddings)} 个向量 (维度: {len(embeddings[0])})"
                )

            return embeddings

        except requests.exceptions.HTTPError as exc:
            error_detail = ""
            try:
                error_detail = exc.response.json()
            except Exception:
                error_detail = exc.response.text[:500]

            error_msg = (
                f"嵌入 API 调用失败 (HTTP {exc.response.status_code}): "
                f"{error_detail}"
            )
            if self.logger:
                self.logger.error(error_msg)
            raise_for_api_balance_error(RuntimeError(error_msg))
            raise RuntimeError(error_msg) from exc

        except requests.exceptions.RequestException as exc:
            error_msg = f"嵌入 API 网络请求失败: {exc}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from exc
