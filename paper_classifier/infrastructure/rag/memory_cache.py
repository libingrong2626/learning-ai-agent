#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
记忆缓存系统。

数据结构：终判结果 → [LLM初判词列表]
- 外层 key：RAG/向量匹配的最终一级学科名（确定且唯一）
- 内层 value：历史上命中该终判的所有 LLM 初判词

示例：
{
  "计算机科学与技术": ["人工智能", "深度学习", "机器学习"],
  "环境科学": ["环境学", "环境变化"],
  "生命科学": ["生物学"]
}

查找方式：给定 LLM初判"人工智能"，扫描所有 value 列表，找到它属于"计算机科学与技术"，直接返回。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path


class MemoryCache:
    """终判结果 → LLM初判词 的反向映射缓存。

    初判词与终判结果严格一一对应：每当你发现某个 LLM初判词
    的终判结果变了，旧映射会自动移除，新映射写入。
    """

    def __init__(self, cache_path: Path, logger: logging.Logger) -> None:
        self.cache_path = cache_path
        self.logger = logger
        # {终判学科名: {LLM初判词集合}}
        self._store: dict[str, set[str]] = {}
        # {LLM初判词: 终判学科名}  快速反向索引
        self._reverse: dict[str, str] = {}
        self._loaded = False

    # ── 文件读写 ──────────────────────────────────────────

    def load(self) -> None:
        if not self.cache_path.exists():
            self.logger.info("记忆缓存文件不存在，将创建新的")
            self._loaded = True
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                self.logger.warning("记忆缓存格式异常，将重置")
                self._loaded = True
                return

            self._store.clear()
            self._reverse.clear()
            for final, initials in data.items():
                if isinstance(initials, list):
                    self._store[final] = set(initials)
                    for init in initials:
                        self._reverse[str(init)] = final

            total = sum(len(v) for v in self._store.values())
            self.logger.info(
                f"记忆缓存加载完成: {len(self._store)} 个终判学科, "
                f"{total} 条初判映射"
            )
        except Exception as exc:
            self.logger.warning(f"读取记忆缓存失败: {exc}，将创建新的")
        self._loaded = True

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        output = {key: sorted(val) for key, val in self._store.items()}
        tmp_path = self.cache_path.with_name(f"{self.cache_path.name}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"记忆缓存已保存: {len(self._store)} 个终判, "
                              f"{len(self._reverse)} 条映射")
        except Exception as exc:
            self.logger.error(f"保存记忆缓存失败: {exc}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    # ── 核心操作 ──────────────────────────────────────────

    def lookup(self, llm_initial: str) -> str | None:
        """根据 LLM 初判词查找记忆中的终判结果。

        Returns:
            终判学科名，未找到返回 None
        """
        if not self._loaded:
            self.load()

        result = self._reverse.get(llm_initial)
        if result:
            self.logger.info(f"记忆命中: LLM初判 '{llm_initial}' → 终判 '{result}'")
        return result

    def record(self, llm_initial: str, final_result: str) -> bool:
        """记录一条映射。如果该初判词之前已映射到其他终判，会先移除旧映射。

        Returns:
            True 如果是新增，False 如果已存在完全相同映射
        """
        if not self._loaded:
            self.load()

        # 检查是否已有完全相同的映射
        existing = self._reverse.get(llm_initial)
        if existing == final_result:
            return False

        # 如果该初判词之前映射到别的终判，先移除
        if existing and existing != final_result:
            self._store[existing].discard(llm_initial)
            if not self._store[existing]:
                del self._store[existing]
            self.logger.info(
                f"记忆更新: LLM初判 '{llm_initial}' "
                f"从 '{existing}' 迁移到 '{final_result}'"
            )

        # 写入新映射
        if final_result not in self._store:
            self._store[final_result] = {llm_initial}
        else:
            self._store[final_result].add(llm_initial)
        self._reverse[llm_initial] = final_result
        self.save()
        self.logger.info(f"记忆记录: LLM初判 '{llm_initial}' → 终判 '{final_result}'")
        return True

    # ── 统计 ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        total_finals = len(self._store)
        total_mappings = len(self._reverse)
        max_initials = max((len(v) for v in self._store.values()), default=0)
        return {
            "finals": total_finals,
            "mappings": total_mappings,
            "max_initials": max_initials,
        }

    def __len__(self) -> int:
        return len(self._reverse)
