"""
日志配置模块
提供统一的日志记录功能
"""
import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "pose_estimation",
                log_dir: str = "logs",
                log_level: int = logging.INFO,
                log_to_file: bool = True,
                log_to_console: bool = True) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
        log_level: 日志级别
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_to_file:
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件名（按日期）
        log_filename = os.path.join(
            log_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 默认日志记录器
default_logger = setup_logger()

