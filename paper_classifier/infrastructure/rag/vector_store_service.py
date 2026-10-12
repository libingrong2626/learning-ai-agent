#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from paper_classifier.config.settings import AppConfig
from paper_classifier.infrastructure.rag.volcano_embeddings import (
    VolcanoMultimodalEmbeddings,
)


class VectorStoreService:
    """Chroma 向量数据库管理服务。

    负责知识库文档的向量化、存储、加载和相似度检索。
    使用余弦相似度（cosine similarity）作为检索度量。
    """

    MANIFEST_FILENAME = "index_manifest.json"

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

        embed_base_url = config.embed_base_url or config.llm_base_url
        self.embeddings = VolcanoMultimodalEmbeddings(
            api_key=config.embed_api_key,
            base_url=embed_base_url,
            model=config.embed_model_name,
            logger=logger,
            timeout=config.embed_timeout,
            batch_size=config.embed_batch_size,
        )

        self.persist_dir = Path(config.chroma_persist_dir)
        self.vector_store: Chroma | None = None

    def should_build(self) -> bool:
        """判断是否需要构建向量数据库。"""
        if self.config.vector_db_build or not self.persist_dir.exists():
            return True

        manifest_path = self.persist_dir / self.MANIFEST_FILENAME
        if not manifest_path.exists():
            self.logger.info("向量索引缺少数据签名，将根据当前知识库重建")
            return True

        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                stored_manifest = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning(f"无法读取向量索引签名，将重建: {exc}")
            return True

        current_manifest = self._build_manifest()
        if stored_manifest != current_manifest:
            self.logger.info("知识库或嵌入配置已变更，将自动重建向量索引")
            return True
        return False

    def build_from_documents(self, documents: list[dict[str, str]]) -> Chroma:
        """从文档列表构建 Chroma 向量数据库。"""
        if not documents:
            raise ValueError("文档列表为空，无法构建向量数据库")

        # 重建前先清空旧数据，避免 Chroma 追加导致重复
        if self.persist_dir.exists():
            self.logger.info(f"清空旧的向量数据库: {self.persist_dir}")
            shutil.rmtree(str(self.persist_dir))

        self.logger.info(f"开始构建向量数据库，共 {len(documents)} 条文档...")

        langchain_docs = [
            Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"],
            )
            for doc in documents
        ]

        self.vector_store = Chroma.from_documents(
            documents=langchain_docs,
            embedding=self.embeddings,
            collection_name=self.config.chroma_collection_name,
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": self.config.similarity_metric},
        )

        manifest_path = self.persist_dir / self.MANIFEST_FILENAME
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(self._build_manifest(), file, ensure_ascii=False, indent=2)

        self.logger.info(f"向量数据库构建完成，已持久化到: {self.persist_dir}")
        return self.vector_store

    def _build_manifest(self) -> dict[str, str | int]:
        """生成用于判断索引是否过期的稳定签名。"""
        knowledge_base_path = Path(self.config.knowledge_base_path)
        if not knowledge_base_path.exists():
            raise FileNotFoundError(f"知识库文件不存在: {knowledge_base_path}")

        digest = hashlib.sha256(knowledge_base_path.read_bytes()).hexdigest()
        return {
            "knowledge_base_sha256": digest,
            "embedding_model": self.config.embed_model_name,
            "embedding_dimensions": self.config.embed_dimensions,
            "collection_name": self.config.chroma_collection_name,
            "similarity_metric": self.config.similarity_metric,
            "document_strategy": "one_discipline_per_document_v2",
        }

    def load_existing(self) -> Chroma | None:
        """加载已有的 Chroma 向量数据库。"""
        if not self.persist_dir.exists():
            self.logger.warning(f"向量数据库目录不存在: {self.persist_dir}")
            return None

        try:
            self.vector_store = Chroma(
                collection_name=self.config.chroma_collection_name,
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings,
            )
            count = self.vector_store._collection.count()
            if count <= 0:
                self.logger.error("已有向量数据库为空，需要重建")
                self.vector_store = None
                return None
            self.logger.info(f"已加载向量数据库，包含 {count} 条记录")
            return self.vector_store
        except Exception as exc:
            self.logger.error(f"加载向量数据库失败: {exc}")
            return None

    def similarity_search(
        self, query: str, top_k: int | None = None
    ) -> list[dict]:
        """执行余弦相似度检索。

        Returns:
            检索结果列表，每项包含 discipline、level、parent、score
        """
        if self.vector_store is None:
            raise RuntimeError("向量数据库未加载，请先调用 build 或 load")

        k = top_k or self.config.rag_top_k
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=k
        )

        return [
            {
                "discipline": doc.metadata.get(
                    "discipline", doc.metadata.get("direction", "")
                ),
                "level": doc.metadata.get("level", "child"),
                "parent": doc.metadata.get("parent", ""),
                "score": float(score),
            }
            for doc, score in results
        ]

    def similarity_search_parents(self, query: str, top_k: int | None = None) -> list[str]:
        """检索并返回匹配的一级学科列表（去重，保持顺序）。"""
        results = self.similarity_search(query, top_k)
        seen = set()
        parents = []
        for r in results:
            if r["parent"] not in seen:
                seen.add(r["parent"])
                parents.append(r["parent"])
        return parents
