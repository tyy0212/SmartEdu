import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置"""

    # 基础配置
    app_name: str = "SmartEdu-MCP"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")

    # 服务器配置
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    mcp_server_port: int = Field(default=8001, env="MCP_SERVER_PORT")

    # 数据库配置
    database_url: str = Field(default="sqlite:///./smartedu.db", env="DATABASE_URL")

    # AI服务配置
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    default_llm_model: str = Field(default="gpt-3.5-turbo", env="DEFAULT_LLM_MODEL")
    fallback_llm_model: str = Field(default="claude-3-haiku-20240307", env="FALLBACK_LLM_MODEL")
    llm_max_tokens: int = Field(default=1000, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    enable_llm_streaming: bool = Field(default=False, env="ENABLE_LLM_STREAMING")

    # ContextGraph 配置
    max_active_context_graphs: int = Field(default=10, env="MAX_ACTIVE_CONTEXT_GRAPHS")
    context_graph_ttl_seconds: int = Field(default=3600, env="CONTEXT_GRAPH_TTL_SECONDS")
    semantic_similarity_model: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2", env="SEMANTIC_SIMILARITY_MODEL")
    initial_similarity_threshold: float = Field(default=0.7, env="INITIAL_SIMILARITY_THRESHOLD")
    adaptive_threshold_enabled: bool = Field(default=True, env="ADAPTIVE_THRESHOLD_ENABLED")
    adaptive_learning_rate: float = Field(default=0.1, env="ADAPTIVE_LEARNING_RATE")
    adaptive_window_size: int = Field(default=20, env="ADAPTIVE_WINDOW_SIZE")

    # 日志配置
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    enable_structlog: bool = Field(default=True, env="ENABLE_STRUCTLOG")

    # 图像分析配置
    image_analysis_enabled: bool = Field(default=True, env="IMAGE_ANALYSIS_ENABLED")
    llava_model_path: Optional[str] = Field(default=None, env="LLAVA_MODEL_PATH")

    # 视频分析配置
    video_analysis_enabled: bool = Field(default=False, env="VIDEO_ANALYSIS_ENABLED")

    # 路径配置
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    images_dir: Path = base_dir / "images"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 创建必要的目录
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)


# 全局配置实例
settings = Settings()