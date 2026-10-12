#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ╔══════════════════════════════════════════════════════════════════╗
# ║  方案一：原始 LLM 直接分类流程（已注释，保留供参考）            ║
# ║                                                                  ║
# ║  此方案使用 LLM 直接对论文进行一级学科分类，不使用向量数据库。  ║
# ║  如需恢复此方案，请在 run.py 中将 use_rag_mode 设为 False，      ║
# ║  并取消下方代码的注释。                                          ║
# ║                                                                  ║
# ║  当前激活的是方案二（RAG 增强分类），见 rag_classifier_app.py   ║
# ╚══════════════════════════════════════════════════════════════════╝

#
# ---------- 方案一代码开始（已注释）----------
#
# from __future__ import annotations
#
# import logging
# import time
# from pathlib import Path
# from typing import Any
#
# from paper_classifier.config.settings import AppConfig
# from paper_classifier.infrastructure.checkpoint_store import CheckpointStore
# from paper_classifier.infrastructure.output_store import OutputStore
# from paper_classifier.repositories.paper_repository import PaperRepository
# from paper_classifier.services.agnes_classifier import AgnesClassifier
#
#
# class PaperClassifierApp:
#     """应用编排层（方案一：LLM 直接分类，无 RAG 增强）。"""
#
#     def __init__(self, config: AppConfig, logger: logging.Logger, log_file: Path) -> None:
#         self.config = config
#         self.logger = logger
#         self.log_file = log_file
#         self.paper_repo = PaperRepository(config, logger)
#         self.checkpoint_store = CheckpointStore(config, logger)
#         self.output_store = OutputStore(config, logger)
#         self.classifier = AgnesClassifier(config, logger)
#
#     def run(self) -> None:
#         start_time = time.time()
#         total_attempted = 0
#         total_success = 0
#         total_failed = 0
#
#         self._log_start_banner()
#
#         try:
#             paper_df = self.paper_repo.load()
#             if paper_df.empty:
#                 self.logger.error("未读取到有效论文，程序退出")
#                 return
#
#             processed_ids = self.checkpoint_store.read()
#             recovered_ids = self.output_store.recover_processed_ids(paper_df, processed_ids)
#             if recovered_ids != processed_ids:
#                 processed_ids = recovered_ids
#                 self.checkpoint_store.write(processed_ids)
#
#             unprocessed_df = paper_df[
#                 ~paper_df["论文唯一标识"].isin(processed_ids)
#             ].reset_index(drop=True)
#
#             if unprocessed_df.empty:
#                 self.logger.info("所有论文均已处理完成！")
#                 return
#
#             self.logger.info(f"待处理论文数量：{len(unprocessed_df)} 篇")
#             self.logger.info(
#                 f"预计需要 {(len(unprocessed_df) + self.config.batch_size - 1) // self.config.batch_size} 个批次"
#             )
#
#             already_saved_ids = self.output_store.collect_success_ids(paper_df)
#
#             for batch_num, current_start in enumerate(
#                 range(0, len(unprocessed_df), self.config.batch_size),
#                 start=1,
#             ):
#                 current_end = min(current_start + self.config.batch_size, len(unprocessed_df))
#                 batch_df = unprocessed_df.iloc[current_start:current_end]
#                 batch_papers = batch_df.to_dict("records")
#                 remaining = len(unprocessed_df) - current_end
#
#                 self._log_batch_banner(batch_num, current_start, current_end, remaining)
#                 category_result = self._classify_with_retry(batch_papers)
#
#                 if category_result:
#                     category_by_idx, missing_indexes = self._normalize_category_result(
#                         category_result,
#                         len(batch_papers),
#                     )
#
#                     success_ids = self.output_store.save_classified(
#                         batch_df,
#                         category_by_idx,
#                         already_saved_ids,
#                     )
#
#                     if success_ids:
#                         processed_ids.update(success_ids)
#                         self.checkpoint_store.write(processed_ids)
#
#                     if missing_indexes:
#                         failed_df = batch_df.iloc[missing_indexes]
#                         self.output_store.save_failed(failed_df)
#                         total_failed += len(failed_df)
#                         if self.config.mark_failed_as_processed:
#                             processed_ids.update(failed_df["论文唯一标识"].tolist())
#                             self.checkpoint_store.write(processed_ids)
#
#                     batch_success = len(success_ids)
#                     total_success += batch_success
#                     self.logger.info(f"本批次分类完成，成功 {batch_success} 篇")
#                 else:
#                     self.logger.error(
#                         f"本批次 {self.config.retry_times} 次重试均失败，已保存失败记录"
#                     )
#                     self.output_store.save_failed(batch_df)
#                     total_failed += len(batch_papers)
#
#                     if self.config.mark_failed_as_processed:
#                         processed_ids.update(batch_df["论文唯一标识"].tolist())
#                         self.checkpoint_store.write(processed_ids)
#
#                 total_attempted += len(batch_papers)
#
#             self._log_finish_summary(
#                 start_time=start_time,
#                 total_attempted=total_attempted,
#                 total_success=total_success,
#                 total_failed=total_failed,
#                 processed_count=len(processed_ids),
#             )
#
#         except Exception as exc:
#             self.logger.error(f"程序异常: {exc}")
#             import traceback
#             self.logger.error(traceback.format_exc())
#
#     def _classify_with_retry(self, batch_papers: list[dict[str, Any]]) -> dict[str, Any] | None:
#         for retry_count in range(self.config.retry_times):
#             self.logger.info(f"第 {retry_count + 1} 次尝试分类...")
#             result = self.classifier.classify_batch(batch_papers)
#             if result:
#                 return result
#
#             if retry_count < self.config.retry_times - 1:
#                 wait_time = 2 ** retry_count
#                 self.logger.warning(f"分类失败，{wait_time} 秒后重试...")
#                 time.sleep(wait_time)
#
#         return None
#
#     def _normalize_category_result(
#         self,
#         category_result: dict[str, Any],
#         batch_size: int,
#     ) -> tuple[dict[int, str], list[int]]:
#         category_by_idx: dict[int, str] = {}
#         invalid_keys = []
#
#         for key, value in category_result.items():
#             try:
#                 idx = int(key)
#             except Exception:
#                 invalid_keys.append(key)
#                 continue
#
#             if idx < 0 or idx >= batch_size:
#                 invalid_keys.append(key)
#                 continue
#
#             category = str(value).strip()
#             if category:
#                 category_by_idx[idx] = category
#
#         if invalid_keys:
#             self.logger.warning(f"API 返回了无效编号，已忽略: {invalid_keys}")
#
#         missing_indexes = [idx for idx in range(batch_size) if idx not in category_by_idx]
#         if missing_indexes:
#             self.logger.warning(f"API 结果缺失编号: {missing_indexes}，这些论文不会写入断点")
#
#         return category_by_idx, missing_indexes
#
#     def _log_start_banner(self) -> None:
#         self.logger.info("=" * 70)
#         self.logger.info("印尼语学术论文分类器 (Agnes API) 启动 - 企业级项目版")
#         self.logger.info("=" * 70)
#         self.logger.info(f"日志文件: {self.log_file}")
#         self.logger.info("当前配置:")
#         self.logger.info(f"  - API Base URL: {self.config.base_url}")
#         self.logger.info(f"  - 模型: {self.config.model_name}")
#         self.logger.info(f"  - 批次大小: {self.config.batch_size}")
#         self.logger.info(f"  - 重试次数: {self.config.retry_times}")
#         self.logger.info(f"  - 输入文件: {self.config.xlsx_input_path}")
#         self.logger.info(f"  - 失败是否写入已完成断点: {self.config.mark_failed_as_processed}")
#         self.logger.info(f"  - 断点 ID 版本: {self.config.checkpoint_id_version}")
#
#     def _log_batch_banner(
#         self,
#         batch_num: int,
#         current_start: int,
#         current_end: int,
#         remaining: int,
#     ) -> None:
#         self.logger.info("")
#         self.logger.info("-" * 60)
#         self.logger.info(f"第 {batch_num} 批次：处理第 {current_start + 1}-{current_end} 篇")
#         self.logger.info(f"剩余待处理：{remaining} 篇")
#         self.logger.info("-" * 60)
#
#     def _log_finish_summary(
#         self,
#         start_time: float,
#         total_attempted: int,
#         total_success: int,
#         total_failed: int,
#         processed_count: int,
#     ) -> None:
#         elapsed_time = time.time() - start_time
#         self.logger.info("")
#         self.logger.info("=" * 70)
#         self.logger.info("分类处理完成")
#         self.logger.info("处理统计:")
#         self.logger.info(f"  - 本次尝试处理论文: {total_attempted} 篇")
#         self.logger.info(f"  - 本次成功分类: {total_success} 篇")
#         self.logger.info(f"  - 本次失败或缺失: {total_failed} 篇")
#         self.logger.info(f"  - 当前断点已完成: {processed_count} 篇")
#         self.logger.info(f"  - 耗时: {elapsed_time:.2f} 秒")
#         if total_attempted > 0:
#             self.logger.info(f"  - 平均每篇: {elapsed_time / total_attempted:.2f} 秒")
#         self.logger.info(f"结果保存位置: {self.config.save_dir}")
#         self.logger.info(f"失败记录保存位置: {self.config.fail_dir}")
#         self.logger.info(f"日志文件保存位置: {self.log_file}")
#         self.logger.info("=" * 70)
#
# ---------- 方案一代码结束 ----------
