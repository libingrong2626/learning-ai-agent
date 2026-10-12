#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from paper_classifier.common.api_errors import ApiBalanceExhaustedError
from paper_classifier.config.settings import AppConfig
from paper_classifier.infrastructure.checkpoint_store import CheckpointStore
from paper_classifier.infrastructure.output_store import OutputStore
from paper_classifier.repositories.paper_repository import PaperRepository
from paper_classifier.infrastructure.rag.knowledge_base import KnowledgeBaseLoader
from paper_classifier.infrastructure.rag.vector_store_service import VectorStoreService
from paper_classifier.infrastructure.rag.rag_classifier import RAGClassifier
from paper_classifier.infrastructure.rag.evaluator import Evaluator
from paper_classifier.infrastructure.rag.memory_cache import MemoryCache


class RAGClassifierApp:
    """RAG 增强分类应用编排层。

    提供两种运行模式：
    - 测试模式 (run_test): 交互式测试 RAG 检索效果，或使用测试数据集评估
    - 管线模式 (run_pipeline): 对 XLSX 论文数据进行批量 RAG 增强分类
    """

    def __init__(
        self, config: AppConfig, logger: logging.Logger, log_file: Path
    ) -> None:
        self.config = config
        self.logger = logger
        self.log_file = log_file

        # 复用已有的基础设施组件
        self.paper_repo = PaperRepository(config, logger)
        self.checkpoint_store = CheckpointStore(config, logger)
        self.output_store = OutputStore(config, logger)

        # RAG 组件
        self.kb_loader = KnowledgeBaseLoader(config, logger)
        self.vector_store_service = VectorStoreService(config, logger)
        self.evaluator = Evaluator(logger)

        # 记忆缓存（LLM初判→最终分类映射，加速重复方向）
        self.memory_cache = MemoryCache(config.memory_cache_path, logger)

    # ══════════════════════════════════════════════════════════
    # 方案二：测试模式（当前激活）
    # 用于测试 RAG 检索效果，支持交互式输入和批量测试数据集
    # ══════════════════════════════════════════════════════════

    def run_test(self) -> None:
        """测试模式：输入方向 -> RAG 检索 -> LLM 终判 -> 评估指标。

        支持两种测试方式：
        1. 交互式输入：直接在终端输入任意方向名称，走完整 RAG 流程
        2. 测试数据集：从 test_dataset.json 批量评估召回率和准确率
        """
        self._log_start_banner()

        # 加载知识库
        self.kb_loader.load()

        # 构建或加载向量数据库
        self._init_vector_store()

        # 初始化 RAG 分类器（用于完整流程测试）
        rag_classifier = RAGClassifier(
            self.config, self.logger, self.vector_store_service,
            valid_parents=self.kb_loader.all_parents,
            memory_cache=self.memory_cache,
        )

        # 检查是否有测试数据集
        test_path = self.config.test_dataset_path
        if test_path.exists():
            self.logger.info(f"检测到测试数据集: {test_path}")
            self._run_test_dataset(test_path, rag_classifier)
            self.logger.info("")
            self.logger.info("数据集测试完成。下面进入交互式测试：")
            self.logger.info("")

        # 交互式测试循环
        self._run_interactive_test(rag_classifier)

    def _run_interactive_test(self, rag_classifier: RAGClassifier) -> None:
        """交互式测试：输入方向名称，走完整 RAG 流程（检索 + LLM 终判）。"""
        print("\n" + "=" * 60)
        print("RAG 完整流程测试模式")
        print("=" * 60)
        print("输入任意学科方向名称，系统将执行：")
        print("  1. 向量数据库检索 Top-K 相似一级/子学科")
        print("  2. LLM 结合检索结果进行最终分类")
        print("  3. 返回最终一级学科 + 累计召回率/准确率")
        print("")
        print("例如：计量经济学、深度学习、环境化学、教育心理学")
        print("输入 quit 或 exit 退出")
        print("=" * 60)

        while True:
            try:
                query = input("\n请输入方向名称 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出测试模式")
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("退出测试模式")
                break

            self._test_single_query(query, rag_classifier)

        # 打印累计评估报告
        if self.evaluator.total_queries > 0:
            self.evaluator.print_report()

    def _test_single_query(self, query: str, rag_classifier: RAGClassifier) -> None:
        """执行单次完整 RAG 流程测试：检索 + LLM 终判 + 指标记录。"""
        # 查找 ground truth：支持子方向名和一级学科名
        ground_truth = self.kb_loader.get_ground_truth(query)
        is_parent_input = (
            ground_truth is not None
            and self.kb_loader.get_parent(query) is None
        )

        # ── Step 1: 向量数据库检索 ──
        rag_results = self.vector_store_service.similarity_search(
            query, top_k=self.config.rag_top_k
        )

        # 显示检索结果
        print(f"\n查询：{query}")
        if ground_truth and not is_parent_input:
            print(f"知识库匹配：{query} -> {ground_truth}（子方向，将作为评测基准）")
        elif ground_truth and is_parent_input:
            print(f"知识库匹配：{query} 是一级学科名（其下所有子方向均为正确结果）")
        else:
            print("知识库匹配：该方向不在知识库中（将仅展示检索和分类结果，不计入指标）")

        print(f"\n[检索阶段] Top-{self.config.rag_top_k} 结果：")
        print("-" * 55)
        for i, r in enumerate(rag_results, 1):
            match_mark = " *命中*" if ground_truth and r["parent"] == ground_truth else ""
            level_name = "一级学科" if r["level"] == "parent" else "子学科"
            print(
                f"  {i}. [{r['parent']}] 匹配{level_name}: {r['discipline']} "
                f"(相关度: {r['score']:.4f}){match_mark}"
            )

        # ── Step 2: LLM 终判 ──
        print("\n[分类阶段] LLM 终判中...")
        predicted_category = None
        try:
            result = rag_classifier._llm_final_classify(
                paper_info=rag_classifier._build_paper_info(
                    title=query, abstract="", filename=""
                ),
                llm_initial=query,
                rag_results=rag_results,
            )
            # 模糊匹配兜底：将 LLM 输出归一化到知识库中的合法学科名
            normalized = self.kb_loader.normalize_parent_name(result)
            if normalized != result:
                print(f"  LLM 原始输出：{result}")
                print(f"  归一化后：{normalized}")
            predicted_category = normalized
            print(f"  LLM 最终分类结果：{predicted_category}")
        except Exception as exc:
            print(f"  LLM 分类失败: {exc}")

        # ── Step 3: 记录评估指标 ──
        if ground_truth:
            eval_result = self.evaluator.record(
                query=query,
                ground_truth_parent=ground_truth,
                rag_results=rag_results,
                predicted_category=predicted_category,
            )

            retrieval_hit = eval_result["retrieval_hit"]
            classification_hit = eval_result["classification_correct"]

            print(f"\n[评估]")
            print(f"  检索召回：{'是' if retrieval_hit else '否'}"
                  f"（正确学科 {ground_truth} 是否在 Top-K 中）")
            print(f"  分类正确：{'是' if classification_hit else '否'}"
                  f"（期望: {ground_truth}, 实际: {predicted_category}）")

            # 实时显示累计指标
            metrics = self.evaluator.get_metrics()
            print(f"\n[累计指标]")
            print(f"  RAG 召回率(Recall@{self.config.rag_top_k}): "
                  f"{metrics['retrieval_recall']:.2%} "
                  f"({metrics['retrieval_hits']}/{metrics['total_queries']})")
            print(f"  分类准确率(Accuracy): "
                  f"{metrics['classification_accuracy']:.2%} "
                  f"({metrics['classification_correct']}/{metrics['total_classified']})")
        else:
            print(f"\n[评估] 输入不在知识库中，本次不计入指标")
            print(f"  LLM 最终分类：{predicted_category}")

            # 即使本次不计入，仍显示历史累计指标
            metrics = self.evaluator.get_metrics()
            if metrics["total_queries"] > 0:
                print(f"\n[累计指标]（基于已标注数据）")
                print(f"  RAG 召回率(Recall@{self.config.rag_top_k}): "
                      f"{metrics['retrieval_recall']:.2%} "
                      f"({metrics['retrieval_hits']}/{metrics['total_queries']})")
                print(f"  分类准确率(Accuracy): "
                      f"{metrics['classification_accuracy']:.2%} "
                      f"({metrics['classification_correct']}/{metrics['total_classified']})")

    def _run_test_dataset(self, test_path: Path, rag_classifier: RAGClassifier) -> None:
        """从测试数据集批量评估 RAG 完整流程（检索 + LLM 终判）。

        测试数据集格式 (test_dataset.json)：
        [
            {
                "query": "计量经济学",
                "ground_truth_parent": "经济学"
            },
            ...
        ]
        """
        with test_path.open("r", encoding="utf-8") as f:
            test_data = json.load(f)

        if not isinstance(test_data, list):
            self.logger.error("测试数据集格式错误，应为 JSON 数组")
            return

        self.logger.info(f"开始批量评估，共 {len(test_data)} 条测试数据...")
        start_time = time.time()

        for i, item in enumerate(test_data, 1):
            query = item.get("query", "")
            ground_truth = item.get("ground_truth_parent", "")

            if not query or not ground_truth:
                self.logger.warning(f"[{i}] 跳过无效测试项: {item}")
                continue

            # 向量检索
            rag_results = self.vector_store_service.similarity_search(
                query, top_k=self.config.rag_top_k
            )

            # LLM 终判（每次都执行）
            predicted = None
            try:
                predicted = rag_classifier._llm_final_classify(
                    paper_info=rag_classifier._build_paper_info(
                        title=query,
                        abstract=item.get("abstract", ""),
                        filename="",
                    ),
                    llm_initial=query,
                    rag_results=rag_results,
                )
                # 模糊匹配兜底
                predicted = self.kb_loader.normalize_parent_name(predicted)
            except Exception as exc:
                self.logger.error(f"[{i}] LLM 分类失败: {exc}")

            hit_info = self.evaluator.record(
                query=query,
                ground_truth_parent=ground_truth,
                rag_results=rag_results,
                predicted_category=predicted,
            )

            retrieval_status = "OK" if hit_info["retrieval_hit"] else "MISS"
            cls_status = "OK" if hit_info["classification_correct"] else "WRONG"
            self.logger.info(
                f"[{i}/{len(test_data)}] {query} -> "
                f"期望: {ground_truth}, "
                f"LLM终判: {predicted} "
                f"[检索:{retrieval_status} 分类:{cls_status}]"
            )

        elapsed = time.time() - start_time
        self.logger.info(f"批量评估完成，耗时: {elapsed:.2f} 秒")

        # 打印评估报告
        self.evaluator.print_report()

        # 保存详细结果（随其他输出一起放在输入文件所在目录）
        details_path = self.config.save_dir / "rag_evaluation_details.json"
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": self.evaluator.get_metrics(),
                    "details": self.evaluator.get_details(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        self.logger.info(f"详细评估结果已保存至: {details_path}")

    # ══════════════════════════════════════════════════════════
    # 管线模式：批量论文分类（RAG 增强）
    # ══════════════════════════════════════════════════════════

    def run_pipeline(self) -> None:
        """管线模式：对论文数据进行 RAG 增强批量分类。"""
        start_time = time.time()
        total_attempted = 0
        total_success = 0
        total_failed = 0

        self._log_start_banner()

        try:
            # 加载知识库和向量数据库
            self.kb_loader.load()
            self._init_vector_store()

            # 初始化 RAG 分类器
            rag_classifier = RAGClassifier(
                self.config, self.logger, self.vector_store_service,
                valid_parents=self.kb_loader.all_parents,
                memory_cache=self.memory_cache,
            )

            # 加载论文数据
            paper_df = self.paper_repo.load()
            if paper_df.empty:
                self.logger.error("未读取到有效论文，程序退出")
                return

            # 恢复断点
            processed_ids = self.checkpoint_store.read()
            recovered_ids = self.output_store.recover_processed_ids(
                paper_df, processed_ids
            )
            if recovered_ids != processed_ids:
                processed_ids = recovered_ids
                self.checkpoint_store.write(processed_ids)

            unprocessed_df = paper_df[
                ~paper_df["论文唯一标识"].isin(processed_ids)
            ].reset_index(drop=True)

            if unprocessed_df.empty:
                self.logger.info("所有论文均已处理完成！")
                return

            self.logger.info(f"待处理论文数量：{len(unprocessed_df)} 篇")
            batch_count = (
                (len(unprocessed_df) + self.config.batch_size - 1)
                // self.config.batch_size
            )
            self.logger.info(f"预计需要 {batch_count} 个批次")

            already_saved_ids = self.output_store.collect_success_ids(paper_df)

            for batch_num, current_start in enumerate(
                range(0, len(unprocessed_df), self.config.batch_size),
                start=1,
            ):
                current_end = min(
                    current_start + self.config.batch_size, len(unprocessed_df)
                )
                batch_df = unprocessed_df.iloc[current_start:current_end]
                batch_papers = batch_df.to_dict("records")
                remaining = len(unprocessed_df) - current_end

                self._log_batch_banner(
                    batch_num, current_start, current_end, remaining
                )

                # RAG 增强分类（带重试）；每篇分类成功后立即落盘，
                # 中途中断只丢正在分类的 1 篇，已完成的论文下次从 CSV 恢复
                success_ids: set[str] = set()

                def on_paper_classified(idx: int, category: str) -> None:
                    paper_row = batch_df.iloc[idx]
                    paper_id = str(paper_row["论文唯一标识"])
                    if self.output_store.save_one(
                        paper_row, category, already_saved_ids
                    ):
                        success_ids.add(paper_id)

                category_result = self._classify_with_retry(
                    rag_classifier, batch_papers, on_paper_classified
                )

                if category_result:
                    _, missing_indexes = (
                        self._normalize_category_result(
                            category_result, len(batch_papers)
                        )
                    )

                    if success_ids:
                        processed_ids.update(success_ids)
                        self.checkpoint_store.write(processed_ids)

                    if missing_indexes:
                        failed_df = batch_df.iloc[missing_indexes]
                        self.output_store.save_failed(failed_df)
                        total_failed += len(failed_df)
                        if self.config.mark_failed_as_processed:
                            processed_ids.update(
                                failed_df["论文唯一标识"].tolist()
                            )
                            self.checkpoint_store.write(processed_ids)

                    batch_success = len(success_ids)
                    total_success += batch_success
                    self.logger.info(f"本批次分类完成，成功 {batch_success} 篇")
                else:
                    self.logger.error(
                        f"本批次 {self.config.retry_times} 次重试均失败"
                    )
                    self.output_store.save_failed(batch_df)
                    total_failed += len(batch_papers)

                    if self.config.mark_failed_as_processed:
                        processed_ids.update(
                            batch_df["论文唯一标识"].tolist()
                        )
                        self.checkpoint_store.write(processed_ids)

                total_attempted += len(batch_papers)

            self._log_finish_summary(
                start_time=start_time,
                total_attempted=total_attempted,
                total_success=total_success,
                total_failed=total_failed,
                processed_count=len(processed_ids),
            )

        except ApiBalanceExhaustedError as exc:
            self.logger.error("")
            self.logger.error("=" * 70)
            self.logger.error("检测到 API 余额不足或账户欠费，程序已立即停止")
            self.logger.error(f"原因: {exc}")
            self.logger.error(
                "当前批次未写入分类失败文件，也未标记为已处理；"
                "此前成功结果和断点均已保留"
            )
            self.logger.error("充值后重新运行程序，将从当前未完成批次继续")
            self.logger.error("=" * 70)
        except Exception as exc:
            self.logger.error(f"程序异常: {exc}")
            import traceback
            self.logger.error(traceback.format_exc())

    # ══════════════════════════════════════════════════════════
    # 公共辅助方法
    # ══════════════════════════════════════════════════════════

    def _init_vector_store(self) -> None:
        """初始化向量数据库（构建或加载）。"""
        if self.vector_store_service.should_build():
            self.logger.info("准备构建向量数据库...")
            documents = self.kb_loader.get_all_documents_for_vector_store()
            self.vector_store_service.build_from_documents(documents)
        else:
            self.logger.info("加载已有向量数据库...")
            vs = self.vector_store_service.load_existing()
            if vs is None:
                self.logger.warning(
                    "向量数据库加载失败，将尝试重新构建..."
                )
                documents = self.kb_loader.get_all_documents_for_vector_store()
                self.vector_store_service.build_from_documents(documents)

    def _classify_with_retry(
        self,
        rag_classifier: RAGClassifier,
        batch_papers: list[dict[str, Any]],
        on_paper_classified=None,
    ) -> dict[str, Any] | None:
        """带重试的批量分类（逐篇成功回调由 classify_batch 触发）。"""
        for retry_count in range(self.config.retry_times):
            self.logger.info(f"第 {retry_count + 1} 次尝试分类...")
            result = rag_classifier.classify_batch(
                batch_papers, on_paper_classified
            )
            if result:
                return result

            if retry_count < self.config.retry_times - 1:
                wait_time = 2 ** retry_count
                self.logger.warning(f"分类失败，{wait_time} 秒后重试...")
                time.sleep(wait_time)

        return None

    def _normalize_category_result(
        self,
        category_result: dict[str, Any],
        batch_size: int,
    ) -> tuple[dict[int, str], list[int]]:
        """标准化分类结果。"""
        category_by_idx: dict[int, str] = {}
        invalid_keys = []

        for key, value in category_result.items():
            try:
                idx = int(key)
            except Exception:
                invalid_keys.append(key)
                continue

            if idx < 0 or idx >= batch_size:
                invalid_keys.append(key)
                continue

            category = str(value).strip()
            if category:
                category_by_idx[idx] = category

        if invalid_keys:
            self.logger.warning(f"API 返回了无效编号: {invalid_keys}")

        missing_indexes = [
            idx for idx in range(batch_size) if idx not in category_by_idx
        ]
        if missing_indexes:
            self.logger.warning(f"API 结果缺失编号: {missing_indexes}")

        return category_by_idx, missing_indexes

    def _log_start_banner(self) -> None:
        """打印启动信息。"""
        actual_embed_url = self.config.embed_base_url or self.config.llm_base_url

        self.logger.info("=" * 70)
        self.logger.info("印尼语学术论文分类器 (RAG 增强版) 启动")
        self.logger.info("=" * 70)
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info("")
        self.logger.info("[LLM 分类模型]")
        self.logger.info(f"  - API 地址:  {self.config.llm_base_url}")
        self.logger.info(f"  - 模型名称:  {self.config.llm_model_name}")
        self.logger.info(f"  - 温度:      {self.config.llm_temperature}")
        self.logger.info(f"  - 最大Token: {self.config.llm_max_tokens}")
        self.logger.info(f"  - 超时(秒):  {self.config.llm_timeout}")
        self.logger.info("")
        self.logger.info("[嵌入向量模型]")
        self.logger.info(f"  - API 地址:  {actual_embed_url}")
        self.logger.info(f"  - 模型名称:  {self.config.embed_model_name}")
        self.logger.info(f"  - 向量维度:  {self.config.embed_dimensions}")
        self.logger.info(f"  - 批量大小:  {self.config.embed_batch_size}")
        self.logger.info(f"  - 超时(秒):  {self.config.embed_timeout}")
        self.logger.info("")
        self.logger.info("[RAG / 管线配置]")
        self.logger.info(f"  - 向量数据库:     {self.config.chroma_persist_dir}")
        self.logger.info(f"  - 向量数据库构建: {self.config.vector_db_build}")
        self.logger.info(f"  - RAG Top-K:      {self.config.rag_top_k}")
        self.logger.info(f"  - 论文批次大小:   {self.config.batch_size}")
        self.logger.info(
            f"  - 输入模式:       {self.config.input_file_type.upper()}"
        )
        self.logger.info(f"  - 输入文件:       {self.config.input_path}")
        self.logger.info("")
        stats = self.memory_cache.get_stats()
        self.logger.info("[记忆缓存]")
        self.logger.info(f"  - 缓存文件:     {self.config.memory_cache_path}")
        self.logger.info(f"  - 终判学科数:   {stats['finals']}")
        self.logger.info(f"  - 初判映射数:   {stats['mappings']}")
        if stats['max_initials'] > 1:
            self.logger.info(f"  - 最多初判词数: {stats['max_initials']}（同一终判包含多个LLM初判方向）")

    def _log_batch_banner(
        self, batch_num: int, start: int, end: int, remaining: int
    ) -> None:
        """打印批次信息。"""
        self.logger.info("")
        self.logger.info("-" * 60)
        self.logger.info(f"第 {batch_num} 批次：处理第 {start + 1}-{end} 篇")
        self.logger.info(f"剩余待处理：{remaining} 篇")
        self.logger.info("-" * 60)

    def _log_finish_summary(
        self,
        start_time: float,
        total_attempted: int,
        total_success: int,
        total_failed: int,
        processed_count: int,
    ) -> None:
        """打印完成统计。"""
        elapsed = time.time() - start_time
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("RAG 增强分类处理完成")
        self.logger.info("处理统计:")
        self.logger.info(f"  - 本次尝试处理论文: {total_attempted} 篇")
        self.logger.info(f"  - 本次成功分类: {total_success} 篇")
        self.logger.info(f"  - 本次失败或缺失: {total_failed} 篇")
        self.logger.info(f"  - 当前断点已完成: {processed_count} 篇")
        self.logger.info(f"  - 耗时: {elapsed:.2f} 秒")
        if total_attempted > 0:
            self.logger.info(f"  - 平均每篇: {elapsed / total_attempted:.2f} 秒")
        self.logger.info(f"结果保存位置: {self.config.save_dir}")
        self.logger.info(f"失败记录保存位置: {self.config.fail_dir}")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info("=" * 70)
