#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

from paper_classifier.common.utils import (
    make_legacy_output_key,
    make_paper_id,
    normalize_cell,
    read_csv_with_fallback,
    sanitize_filename,
)
from paper_classifier.config.settings import AppConfig


class OutputStore:
    """恢复历史输出、保存分类结果和失败记录。"""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def recover_processed_ids(self, paper_df: pd.DataFrame, existing_ids: set[str]) -> set[str]:
        recovered_ids = self.collect_success_ids(paper_df)
        if self.config.include_failed_output_as_processed:
            recovered_ids |= self.collect_failed_ids(paper_df)

        if not recovered_ids:
            self.logger.info("未从已有输出中恢复到可用断点记录")
            return existing_ids

        merged_ids = set(existing_ids) | recovered_ids
        added_count = len(merged_ids) - len(existing_ids)

        if added_count > 0:
            self.logger.info(
                f"已从历史输出恢复 {added_count} 条断点记录，"
                f"当前断点共 {len(merged_ids)} 篇"
            )
        else:
            self.logger.info("历史输出与当前断点一致，无需补充")

        return merged_ids

    def collect_success_ids(self, paper_df: pd.DataFrame) -> set[str]:
        if not self.config.save_dir.exists():
            return set()

        valid_ids, exact_key_to_id, legacy_key_to_id = self._build_source_lookup(paper_df)
        recovered_ids: set[str] = set()

        for csv_path in sorted(self.config.save_dir.glob("*.csv")):
            try:
                output_df = read_csv_with_fallback(csv_path)
            except Exception as exc:
                self.logger.warning(f"读取历史分类结果失败，已跳过 {csv_path}: {exc}")
                continue

            if not self._has_title_abstract_columns(output_df):
                self.logger.warning(
                    f"历史分类结果缺少 title 或 abstract 列，已跳过: {csv_path}"
                )
                continue

            before_count = len(recovered_ids)
            for _, row in output_df.iterrows():
                paper_id = self._match_output_row(
                    row,
                    valid_ids,
                    exact_key_to_id,
                    legacy_key_to_id,
                )
                if paper_id:
                    recovered_ids.add(paper_id)

            self.logger.info(
                f"扫描历史分类文件 {csv_path.name}，恢复 "
                f"{len(recovered_ids) - before_count} 条已分类记录"
            )

        return recovered_ids

    def collect_failed_ids(self, paper_df: pd.DataFrame) -> set[str]:
        fail_path = self.config.fail_dir / "分类失败论文.xlsx"
        if not fail_path.exists():
            return set()

        valid_ids, exact_key_to_id, legacy_key_to_id = self._build_source_lookup(paper_df)
        recovered_ids: set[str] = set()

        try:
            failed_df = pd.read_excel(fail_path, engine="openpyxl").fillna("")
        except Exception as exc:
            self.logger.warning(f"读取历史失败文件失败，已跳过: {exc}")
            return set()

        for _, row in failed_df.iterrows():
            paper_id = self._match_output_row(
                row,
                valid_ids,
                exact_key_to_id,
                legacy_key_to_id,
            )
            if paper_id:
                recovered_ids.add(paper_id)

        self.logger.info(f"从历史失败文件恢复 {len(recovered_ids)} 条记录为已处理")
        return recovered_ids

    def save_classified(
        self,
        batch_df: pd.DataFrame,
        category_by_local_idx: dict[int, str],
        already_saved_ids: set[str],
    ) -> set[str]:
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        success_ids: set[str] = set()
        saved_count = 0
        skipped_duplicate_count = 0

        for local_idx, category in category_by_local_idx.items():
            try:
                paper_row = batch_df.iloc[local_idx]
                paper_id = str(paper_row["论文唯一标识"])
                success_ids.add(paper_id)

                if paper_id in already_saved_ids:
                    skipped_duplicate_count += 1
                    continue

                category_name = sanitize_filename(category)
                save_path = self.config.save_dir / f"{category_name}.csv"
                header = self._get_output_header(batch_df, save_path)
                write_header = not save_path.exists() or save_path.stat().st_size == 0

                row_dict = {
                    col: self._get_output_value(paper_row, col)
                    for col in header
                }

                with save_path.open("a", encoding="utf-8-sig", newline="") as file_obj:
                    writer = csv.DictWriter(
                        file_obj,
                        fieldnames=header,
                        extrasaction="ignore",
                    )
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row_dict)

                already_saved_ids.add(paper_id)
                saved_count += 1

            except Exception as exc:
                self.logger.error(f"保存论文 {local_idx} 失败: {exc}")

        self.logger.info(
            f"本批次结果已保存至 {self.config.save_dir}，新增保存 {saved_count} 篇，"
            f"跳过重复 {skipped_duplicate_count} 篇"
        )
        return success_ids

    def save_one(
        self,
        paper_row: pd.Series,
        category: str,
        already_saved_ids: set[str],
    ) -> bool:
        """单篇论文立即追加写入对应学科 CSV（中断不丢已分类结果）。

        与 save_classified（整批落盘）的区别：本篇独立落盘，不等待
        整批处理完。返回 True 表示本次新增保存，False 表示重复跳过。
        """
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        paper_id = str(paper_row["论文唯一标识"])
        if paper_id in already_saved_ids:
            return False

        category_name = sanitize_filename(category)
        save_path = self.config.save_dir / f"{category_name}.csv"
        header = self._get_output_header(pd.DataFrame([paper_row]), save_path)
        write_header = not save_path.exists() or save_path.stat().st_size == 0

        row_dict = {
            col: self._get_output_value(paper_row, col)
            for col in header
        }

        with save_path.open("a", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=header,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row_dict)

        already_saved_ids.add(paper_id)
        return True

    def save_failed(self, failed_papers_df: pd.DataFrame) -> None:
        if failed_papers_df.empty:
            return

        self.config.fail_dir.mkdir(parents=True, exist_ok=True)
        fail_path = self.config.fail_dir / "分类失败论文.xlsx"
        save_df = failed_papers_df.drop(columns=["论文唯一标识"], errors="ignore")

        try:
            if fail_path.exists():
                existing_df = pd.read_excel(fail_path, engine="openpyxl").fillna("")
                combined_df = pd.concat([existing_df, save_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(
                    subset=[
                        self.config.title_column,
                        self.config.abstract_column,
                    ],
                    keep="first",
                )
                combined_df.to_excel(fail_path, index=False, engine="openpyxl")
            else:
                save_df.to_excel(fail_path, index=False, engine="openpyxl")

            self.logger.info(f"失败论文已保存至：{fail_path}，本次 {len(save_df)} 篇")
        except Exception as exc:
            self.logger.error(f"保存失败论文时出错: {exc}")

    def _build_source_lookup(
        self,
        paper_df: pd.DataFrame,
    ) -> tuple[set[str], dict[str, str], dict[str, str]]:
        valid_ids = set(str(item) for item in paper_df["论文唯一标识"].tolist())
        exact_key_to_id: dict[str, str] = {}
        legacy_key_to_id: dict[str, str] = {}

        for _, row in paper_df.iterrows():
            paper_id = str(row["论文唯一标识"])
            title = row[self.config.title_column]
            abstract = row[self.config.abstract_column]
            exact_key = f"{normalize_cell(title)}||{normalize_cell(abstract)}"
            legacy_key = make_legacy_output_key(title, abstract)
            exact_key_to_id[exact_key] = paper_id
            legacy_key_to_id[legacy_key] = paper_id

        return valid_ids, exact_key_to_id, legacy_key_to_id

    def _match_output_row(
        self,
        row: pd.Series,
        valid_ids: set[str],
        exact_key_to_id: dict[str, str],
        legacy_key_to_id: dict[str, str],
    ) -> str | None:
        title = normalize_cell(
            row.get(self.config.title_column, row.get("标题", ""))
        )
        abstract = normalize_cell(
            row.get(self.config.abstract_column, row.get("摘要", ""))
        )

        direct_id = make_paper_id(title, abstract)
        if direct_id in valid_ids:
            return direct_id

        exact_key = f"{title}||{abstract}"
        if exact_key in exact_key_to_id:
            return exact_key_to_id[exact_key]

        legacy_key = f"{title}||{abstract}"
        if legacy_key in legacy_key_to_id:
            return legacy_key_to_id[legacy_key]

        return None

    def _has_title_abstract_columns(self, df: pd.DataFrame) -> bool:
        """兼容当前英文输出和旧版中文输出。"""
        has_title = (
            self.config.title_column in df.columns or "标题" in df.columns
        )
        has_abstract = (
            self.config.abstract_column in df.columns or "摘要" in df.columns
        )
        return has_title and has_abstract

    def _get_output_value(self, row: pd.Series, column: str) -> str:
        """写出整行数据，并兼容已有结果文件的旧中文表头。"""
        if column in row.index:
            return normalize_cell(row[column])
        if column == "标题" and self.config.title_column in row.index:
            return normalize_cell(row[self.config.title_column])
        if column == "摘要" and self.config.abstract_column in row.index:
            return normalize_cell(row[self.config.abstract_column])
        return ""

    def _get_output_header(self, paper_df: pd.DataFrame, save_path: Path) -> list[str]:
        default_header = [col for col in paper_df.columns if col != "论文唯一标识"]
        if save_path.exists() and save_path.stat().st_size > 0:
            try:
                with save_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
                    reader = csv.reader(file_obj)
                    existing_header = next(reader, None)
                if existing_header:
                    return existing_header
            except Exception:
                pass
        return default_header
