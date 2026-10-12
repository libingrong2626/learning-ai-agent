#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
印尼语学术论文分类器 - 主入口

双方案切换：
  - use_rag_mode = True  -> 方案二：RAG 增强分类（LangChain + Chroma 向量数据库）
  - use_rag_mode = False -> 方案一：原始 LLM 直接分类（已注释，需取消注释后使用）

切换方法：修改 paper_classifier/config/settings.py 中的 use_rag_mode 配置项
"""

from paper_classifier.config.settings import CONFIG
from paper_classifier.infrastructure.logging_setup import setup_logger


def main() -> None:
    logger, log_file = setup_logger(CONFIG)

    if CONFIG.use_rag_mode:
        # ════════════════════════════════════════════════════════
        # 方案二：RAG 增强分类（当前激活）
        # ════════════════════════════════════════════════════════
        #
        # 提供两种运行模式：
        #    run_test()     - 测试模式：交互式测试向量检索效果，评估召回率/准确率
        #   run_pipeline() - 管线模式：对论文数据进行批量 RAG 增强分类
        #
        from paper_classifier.services.rag_classifier_app import RAGClassifierApp

        app = RAGClassifierApp(CONFIG, logger, log_file)

        # 选择运行模式（取消注释你需要的那个）：
        
        #app.run_test()        # <- 测试模式（当前激活，用于调试和评估）
        app.run_pipeline()  # <- 管线模式（正式批量分类时取消注释此行）

    else:
        # ════════════════════════════════════════════════════════
        # 方案一：原始 LLM 直接分类（已注释）
        # ════════════════════════════════════════════════════════
        #
        # 如需恢复原始分类流程：
        # 1. 将 use_rag_mode 设为 False
        # 2. 取消下方代码的注释
        # 3. 同时取消 classifier_app.py 中代码的注释
        #
        # from paper_classifier.services.classifier_app import PaperClassifierApp
        # app = PaperClassifierApp(CONFIG, logger, log_file)
        # app.run()
        #
        logger.error(
            "方案一（原始分类）当前已注释。"
            "如需使用，请取消 run.py 和 classifier_app.py 中的注释。"
        )


if __name__ == "__main__":
    main()
