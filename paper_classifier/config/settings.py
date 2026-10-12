#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """程序集中配置。API Key 按要求使用明文。"""

    # ══════════════════════════════════════════════════════════
    # LLM 大模型配置（用于论文分类：初判 + 终判）
    # ══════════════════════════════════════════════════════════
    llm_api_key: str = ""
    llm_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    llm_model_name: str = "doubao-seed-2-0-mini-260428"
    llm_temperature: float = 0.1       # 生成温度，越低越确定性
    llm_max_tokens: int = 1000         # 单次最大输出 token 数
    llm_timeout: int = 60              # API 超时秒数
    
    # ══════════════════════════════════════════════════════════
    # 嵌入模型配置（用于向量数据库索引与检索）
    # ══════════════════════════════════════════════════════════
    embed_api_key: str = ""
    embed_base_url: str = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
    embed_model_name: str = "doubao-embedding-vision-251215"
    embed_dimensions: int = 2048       # 输出向量维度（取决于模型）
    embed_timeout: int = 60            # 嵌入 API 超时秒数
    embed_batch_size: int = 1          # 多模态接口每次仅返回1条嵌入，必须设为1

    # ── 向量数据库配置（Chroma）──
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "subject_directions"
    similarity_metric: str = "cosine"

    # ── RAG 检索配置 ──
    rag_top_k: int = 5
    # True=每次启动强制重建；False=数据或嵌入配置变化时自动重建
    vector_db_build: bool = False

    # ── 输入/输出路径 ──
    # 可选 "xlsx" 或 "csv"
    input_file_type: str = "xlsx"
    xlsx_input_path: Path = Path(r"C:\Users\asus\Desktop\论文分类器\印尼语论文信息\印尼语0.xlsx")
    csv_input_path: Path = Path(r"C:\Users\asus\Desktop\论文分类器\印尼语论文信息\印尼语0.csv")
    knowledge_base_path: Path = Path("学科.csv")
    test_dataset_path: Path = Path("test_dataset.json")

    @property
    def input_path(self) -> Path:
        """返回当前读取模式对应的论文文件路径。"""
        input_type = self.input_file_type.strip().lower()
        if input_type == "xlsx":
            return self.xlsx_input_path
        if input_type == "csv":
            return self.csv_input_path
        raise ValueError(
            "input_file_type 只能设置为 'xlsx' 或 'csv'，"
            f"当前值为: {self.input_file_type!r}"
        )

    # ── 批量处理配置（分类管线级别）──
    batch_size: int = 15                # 每批论文数量
    retry_times: int = 3               # 分类失败重试次数

    # ── 断点与存储 ──
    # 以下输出路径默认以相对路径声明，实例化时（__post_init__）
    # 会自动挂到输入文件（csv_input_path/xlsx_input_path）所在目录下，
    # 保证所有产物与输入 CSV 同目录。显式传入绝对路径时保持原样。
    checkpoint_file: Path = Path("分类断点记录.json")
    memory_cache_path: Path = Path("memory_cache.json")  # LLM初判→终判 记忆缓存
    save_dir: Path = Path("印尼语论文分类结果")
    fail_dir: Path = Path("分类失败")
    log_dir: Path = Path("logs")

    def __post_init__(self) -> None:
        """将输出类相对路径重定向到输入文件所在目录。

        输入 CSV 在 A 目录，则断点、记忆缓存、分类结果、失败记录、
        日志全部生成在 A 目录下；换输入文件时输出自动跟随。
        """
        try:
            output_root = self.input_path.parent
        except ValueError:
            # input_file_type 非法时保持原行为，由 load() 阶段报错
            return

        for field_name in (
            "checkpoint_file",
            "memory_cache_path",
            "save_dir",
            "fail_dir",
            "log_dir",
        ):
            value = getattr(self, field_name)
            if not value.is_absolute():
                object.__setattr__(self, field_name, output_root / value)

    checkpoint_id_version: str = "sha256_title_abstract_v1"
    mark_failed_as_processed: bool = False
    include_failed_output_as_processed: bool = False

    # ── 运行模式 ──
    use_rag_mode: bool = True

    title_column: str = "title"
    abstract_column: str = "abstract"
    # abstract 为空时的兜底列（论文文件名）
    filename_column: str = "filename"
    required_columns: tuple[str, ...] = ("title", "abstract")

    # ── 向后兼容（旧方案 agnes_classifier.py 引用的字段名）──
    @property
    def api_key(self) -> str:
        return self.llm_api_key

    @property
    def base_url(self) -> str:
        return self.llm_base_url

    @property
    def model_name(self) -> str:
        return self.llm_model_name

    @property
    def temperature(self) -> float:
        return self.llm_temperature

    @property
    def max_tokens(self) -> int:
        return self.llm_max_tokens

    @property
    def api_timeout_seconds(self) -> int:
        return self.llm_timeout


CONFIG = AppConfig()
