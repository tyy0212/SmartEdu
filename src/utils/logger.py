# -*- coding: utf-8 -*-
"""
结构化日志配置模块
"""
import sys
import logging
from typing import Optional

import structlog
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config.settings import settings


def setup_logging():
    """配置结构化日志系统"""

    # 基础 logging 配置
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # 如果启用 structlog
    if settings.enable_structlog:
        # structlog 配置
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # 根据格式选择处理器
        if settings.log_format == "json":
            formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer()
            )
        else:  # console
            formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer()
            )

        # 配置 root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

        # 设置日志级别
        root_logger.setLevel(getattr(logging, settings.log_level.upper()))

        # 文件日志（如果配置了 log_file）
        if settings.log_file:
            file_handler = logging.FileHandler(settings.log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        logger = structlog.get_logger(__name__)
        logger.info("结构化日志系统已初始化", log_format=settings.log_format, log_level=settings.log_level)
    else:
        # 使用标准 logging
        logger = logging.getLogger(__name__)
        logger.info("标准日志系统已初始化")

    return logger


def get_logger(name: str):
    """获取 logger 实例"""
    if settings.enable_structlog:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


# 全局 logger 实例
logger = setup_logging()