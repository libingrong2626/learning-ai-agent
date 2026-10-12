#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from paper_classifier.config.prompts import SYSTEM_PROMPT
from paper_classifier.config.settings import AppConfig


class AgnesClassifier:
    """Agnes/OpenAI 兼容接口客户端。"""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.api_timeout_seconds,
        )

    def classify_batch(self, papers_batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        prompt_text = self._build_user_prompt(papers_batch)

        try:
            self.logger.debug(f"发送API请求，包含 {len(papers_batch)} 篇论文")
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.api_timeout_seconds,
            )

            content = response.choices[0].message.content.strip()
            self.logger.debug(f"API响应: {content[:200]}...")
            return self._parse_json_response(content)

        except Exception as exc:
            self.logger.error(f"API调用异常: {exc}")
            return None

    def _build_user_prompt(self, papers_batch: list[dict[str, Any]]) -> str:
        paper_summary_list = []
        for idx, paper in enumerate(papers_batch):
            abstract = paper.get(self.config.abstract_column, "")
            title = paper.get(self.config.title_column, "")

            if title:
                summary_text = f"标题：{title}。摘要：{abstract}"
            else:
                summary_text = abstract

            paper_summary_list.append(f"[{idx}] {summary_text}")

        return f"""
请对以下 {len(papers_batch)} 篇印尼语学术论文进行分类。

每篇论文的摘要如下：
{chr(10).join(paper_summary_list)}

请为每篇论文分配一个核心研究方向（如：经济学、计算机科学、医学、环境科学、管理学等）。
请严格按照 JSON 格式输出，键为论文编号，值为类别名称。
"""

    def _parse_json_response(self, content: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        fixed_content = content.replace("'", '"')
        try:
            parsed = json.loads(fixed_content)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        self.logger.warning(f"无法解析 API 响应: {content[:200]}...")
        return None
