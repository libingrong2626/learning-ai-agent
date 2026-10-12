#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""切分未分类论文为 10 个 xlsx 文件。

流程：
1. 读取断点记录（分类断点记录.json）并从已有分类结果 CSV 恢复，
   确定已分类到哪个位置；
2. 从总输入文件（settings.py 中的 input_path，当前为印尼语0.xlsx）
   中筛出未分类部分，保持原始行顺序；
3. 均分为 10 份（余数分给前几份），复制原表头，写出
   印尼a.xlsx、印尼b.xlsx …… 印尼j.xlsx，保存在输入文件同目录；
4. 原始总文件只读，不做任何修改（留作备份）。

用法：python 切分未分类为10份.py
"""

import logging
import string
from pathlib import Path

import pandas as pd

from paper_classifier.config.settings import AppConfig
from paper_classifier.infrastructure.checkpoint_store import CheckpointStore
from paper_classifier.infrastructure.output_store import OutputStore
from paper_classifier.repositories.paper_repository import PaperRepository

SPLIT_COUNT = 10
NAME_LETTERS = string.ascii_lowercase[:SPLIT_COUNT]  # a~j


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("split")

    config = AppConfig()
    input_path = config.input_path
    print(f"输入文件: {input_path}")
    if not input_path.exists():
        raise SystemExit(f"输入文件不存在: {input_path}")

    # 1. 读取并标准化（复用分类器同一套去重和唯一标识逻辑）
    paper_repo = PaperRepository(config, logger)
    paper_df = paper_repo.load()
    print(f"去重后论文总数: {len(paper_df)}")

    # 2. 断点 + 已有输出 CSV 双重恢复，确定已分类位置
    checkpoint_store = CheckpointStore(config, logger)
    output_store = OutputStore(config, logger)
    processed_ids = checkpoint_store.read()
    print(f"断点记录已处理: {len(processed_ids)} 篇")

    processed_ids = output_store.recover_processed_ids(
        paper_df, processed_ids
    )
    print(f"恢复后已处理总数: {len(processed_ids)} 篇")

    # 3. 筛出未分类部分，保持原始顺序
    unprocessed_df = paper_df[
        ~paper_df["论文唯一标识"].isin(processed_ids)
    ].reset_index(drop=True)
    print(f"未分类论文: {len(unprocessed_df)} 篇")

    if unprocessed_df.empty:
        print("没有需要切分的未分类论文，退出。")
        return

    # 去掉内部唯一标识列；同时排除 pandas 从原文件读出的
    # Unnamed 溢出脏列（原文件少数行数据串列所致），
    # 只保留原文件真实表头，保证"复制表头"
    output_df = unprocessed_df.drop(columns=["论文唯一标识"])
    real_header = [
        col for col in pd.read_excel(input_path, nrows=0).columns
        if not str(col).startswith("Unnamed")
    ]
    output_df = output_df[[c for c in real_header if c in output_df.columns]]
    print(f"输出表头列数: {len(output_df.columns)}（与原文件有效表头一致）")

    # 4. 均分 10 份：前 (N % 10) 份多 1 行
    total = len(output_df)
    base = total // SPLIT_COUNT
    extra = total % SPLIT_COUNT
    part_sizes = [base + (1 if i < extra else 0) for i in range(SPLIT_COUNT)]

    out_dir = input_path.parent
    start = 0
    print(f"切分为 {SPLIT_COUNT} 份，输出目录: {out_dir}")
    for letter, size in zip(NAME_LETTERS, part_sizes):
        part_df = output_df.iloc[start:start + size]
        start += size
        out_path = out_dir / f"印尼{letter}.xlsx"
        part_df.to_excel(out_path, index=False, engine="openpyxl")
        print(
            f"  印尼{letter}.xlsx: {size} 行"
            f"（原文件第 {start - size + 2}~{start + 1} 行未分类部分）"
        )

    print(f"完成，共写出 {total} 行；原文件 {input_path.name} 未做任何修改。")


if __name__ == "__main__":
    main()
