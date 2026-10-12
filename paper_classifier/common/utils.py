#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def make_paper_id(title: Any, abstract: Any) -> str:
    text = f"{normalize_cell(title)}||{normalize_cell(abstract)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_legacy_output_key(title: Any, abstract: Any) -> str:
    title_text = normalize_cell(title).replace(",", "，")
    abstract_text = normalize_cell(abstract).replace(",", "，")
    return f"{title_text}||{abstract_text}"


def sanitize_filename(name: Any) -> str:
    safe_name = str(name).strip() or "未分类"
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", safe_name)
    safe_name = safe_name.strip().strip(".")
    return safe_name[:120] or "未分类"


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(csv_path, encoding=encoding, dtype=str).fillna("")
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法读取CSV文件: {csv_path}，最后错误: {last_error}")

