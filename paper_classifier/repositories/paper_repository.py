#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from paper_classifier.common.utils import make_paper_id
from paper_classifier.config.settings import AppConfig


class PaperRepository:
    """根据配置读取并标准化 XLSX 或 CSV 论文数据。"""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def load(self) -> pd.DataFrame:
        input_type = self.config.input_file_type.strip().lower()
        input_path = self.config.input_path
        self.logger.info(
            f"开始读取 {input_type.upper()} 文件: {input_path}"
        )

        try:
            if input_type == "xlsx":
                df = pd.read_excel(input_path, engine="openpyxl")
            else:
                df = self._read_csv(input_path)
        except Exception as exc:
            raise ValueError(
                f"读取 {input_type.upper()} 文件失败: {exc}"
            ) from exc

        self.logger.info(
            f"成功读取 {input_type.upper()} 文件，共 {len(df)} 行"
        )
        df = self._normalize_required_columns(df, input_type.upper())

        original_count = len(df)
        title_column = self.config.title_column
        abstract_column = self.config.abstract_column

        df[title_column] = df[title_column].fillna("").astype(str)
        df[abstract_column] = df[abstract_column].fillna("").astype(str)
        df = df.drop_duplicates(
            subset=[title_column, abstract_column], keep="first"
        ).reset_index(drop=True)

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("").astype(str)

        df["论文唯一标识"] = df.apply(
            lambda row: make_paper_id(
                row[title_column], row[abstract_column]
            ),
            axis=1,
        )

        empty_abstract_count = len(
            df[df[abstract_column].str.strip() == ""]
        )
        if empty_abstract_count > 0:
            self.logger.info(
                f"发现 {empty_abstract_count} 篇论文摘要为空，将仅依据标题进行分类"
            )

        self.logger.info(
            f"处理后共 {len(df)} 篇有效论文"
            f"（原始 {original_count} 篇，过滤 {original_count - len(df)} 篇）"
        )
        return df

    def _read_csv(self, input_path: Path) -> pd.DataFrame:
        """读取常见 CSV 编码：UTF-16（BOM/启发式）→ UTF-8 → GB18030。"""
        last_decode_error: UnicodeDecodeError | None = None
        for encoding in self._candidate_encodings(input_path):
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                self.logger.info(f"CSV 文件编码: {encoding}")
                return df
            except UnicodeDecodeError as exc:
                last_decode_error = exc

        raise ValueError(
            "CSV 文件编码无法识别，请使用 UTF-8、GB18030 或 UTF-16: "
            f"{last_decode_error}"
        )

    def _candidate_encodings(self, input_path: Path) -> list[str]:
        """根据 BOM 与字节特征推断候选编码顺序。

        Excel/WPS 另存为"Unicode 文本"时产出 UTF-16 CSV：每个字符后
        跟一个 0x00 字节。这类文件开头若全是 ASCII 字符，UTF-8 检测
        会误判通过，读到一半遇到非 ASCII 字符才失败——因此必须提前
        通过 BOM 或 NUL 字节占比识别，否则会静默读出带 NUL 的乱码。
        """
        with input_path.open("rb") as file_obj:
            head = file_obj.read(8192)

        if head.startswith(b"\xef\xbb\xbf"):
            return ["utf-8-sig", "gb18030", "utf-16"]
        if head.startswith(b"\xfe\xff"):
            return ["utf-16-be", "utf-8-sig", "gb18030"]
        if head.startswith(b"\xff\xfe"):
            return ["utf-16", "utf-8-sig", "gb18030"]
        if head and head.count(b"\x00") / len(head) >= 0.2:
            # 无 BOM 的 UTF-16：Python 的 'utf-16' 要求 BOM，
            # 必须显式字节序；Windows/Excel 默认小端，LE 优先
            return ["utf-16-le", "utf-16-be", "gb18030", "utf-8-sig"]
        return ["utf-8-sig", "gb18030", "utf-16"]

    def _normalize_required_columns(
        self, df: pd.DataFrame, source_type: str
    ) -> pd.DataFrame:
        """忽略大小写和首尾空格识别必需表头，并统一为配置名称。

        重复检查只针对必需列（title/abstract 及其别名），避免无关列
        （如 doi/DOI、md5/MD5 同时存在）误触发校验导致无法读取。
        """
        required_normalized = {
            str(column).strip().casefold()
            for column in self.config.required_columns
        }
        normalized_to_original: dict[str, object] = {}
        for column in df.columns:
            normalized = str(column).strip().casefold()
            if normalized not in required_normalized:
                continue
            if normalized in normalized_to_original:
                raise ValueError(
                    f"{source_type} 存在规范化后重复的必需表头: {column!r}"
                )
            normalized_to_original[normalized] = column

        missing_cols = [
            column
            for column in self.config.required_columns
            if column.casefold() not in normalized_to_original
        ]
        if missing_cols:
            raise ValueError(f"{source_type} 缺少列: {missing_cols}")

        rename_map = {
            normalized_to_original[column.casefold()]: column
            for column in self.config.required_columns
            if normalized_to_original[column.casefold()] != column
        }
        return df.rename(columns=rename_map)
