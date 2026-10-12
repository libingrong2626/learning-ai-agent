#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from paper_classifier.config.settings import AppConfig


class CheckpointStore:
    """读取、备份、原子写入断点。"""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def read(self) -> set[str]:
        checkpoint_path = self.config.checkpoint_file
        if not checkpoint_path.exists():
            self.logger.info("无断点文件，将尝试从已有分类结果恢复断点")
            return set()

        try:
            with checkpoint_path.open("r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
        except Exception as exc:
            self.logger.warning(f"读取断点文件失败: {exc}，将尝试恢复")
            self.backup("broken")
            return set()

        if isinstance(data, dict):
            return self._read_current_checkpoint(data)

        if isinstance(data, list):
            return self._read_legacy_checkpoint(data)

        self.logger.warning("断点文件格式异常，将不直接使用")
        self.backup("unknown")
        return set()

    def write(self, processed_ids: set[str]) -> None:
        checkpoint_path = self.config.checkpoint_file
        tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")

        payload = {
            "id_version": self.config.checkpoint_id_version,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "processed_count": len(processed_ids),
            "processed_ids": sorted(processed_ids),
        }

        try:
            with tmp_path.open("w", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            os.replace(tmp_path, checkpoint_path)
            self.logger.debug(f"断点记录已更新，当前已处理 {len(processed_ids)} 篇")
        except Exception as exc:
            self.logger.error(f"写入断点文件失败: {exc}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def backup(self, reason: str) -> Path | None:
        checkpoint_path = self.config.checkpoint_file
        if not checkpoint_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}.{reason}_backup_{timestamp}{checkpoint_path.suffix}"
        )
        shutil.copy2(checkpoint_path, backup_path)
        self.logger.warning(f"旧断点文件已备份到: {backup_path}")
        return backup_path

    def _read_current_checkpoint(self, data: dict[str, Any]) -> set[str]:
        version = data.get("id_version")
        processed_ids = data.get("processed_ids", [])

        if version == self.config.checkpoint_id_version and isinstance(processed_ids, list):
            result = set(str(item) for item in processed_ids)
            self.logger.info(f"断点记录：已处理 {len(result)} 篇论文")
            return result

        self.logger.warning("检测到非当前版本断点文件，将不直接使用")
        self.backup("incompatible")
        return set()

    def _read_legacy_checkpoint(self, data: list[Any]) -> set[str]:
        stable_like = all(
            isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
            for item in data
        )
        if stable_like:
            result = set(data)
            self.logger.info(f"读取到旧格式稳定断点：已处理 {len(result)} 篇论文")
            return result

        self.logger.warning("检测到旧版 hash() 断点文件，旧 ID 跨重启不可靠，将不直接使用")
        self.backup("legacy_hash")
        return set()

