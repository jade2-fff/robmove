"""
日志模块
使用 loguru 提供专业的日志功能

使用方式:
    from logger import logger
    logger.info("信息")
    logger.debug("调试信息")
    logger.error("错误信息")
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger as _logger

from config import cfg


def setup_logger():
    """
    配置日志系统
    - 控制台输出：彩色格式
    - 文件输出：JSON格式或普通文本格式
    """
    # 清除默认处理器
    _logger.remove()

    # 控制台输出（彩色）
    if cfg.log_to_console:
        _logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=cfg.log_level,
            colorize=True,
        )

    # 文件输出
    if cfg.log_to_file:
        log_dir = Path(cfg.log_dir)
        if not log_dir.is_absolute():
            log_dir = Path(__file__).resolve().parent / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件名：log_YYYYMMDD_HHMMSS.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"log_{timestamp}.log"

        _logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level=cfg.log_level,
            encoding="utf-8",
            rotation="10 MB",
            retention="7 days",
        )

    return _logger


# 初始化日志
setup_logger()


# 方便使用的全局 logger
logger = _logger
