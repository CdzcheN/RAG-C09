#!/usr/bin/env python3
"""统一日志：同时输出到 stdout 与 results/logs/<exp_id>.log。"""
from __future__ import annotations

import logging
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / "results" / "logs"
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str, exp_id: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """取得带统一格式的 logger；给出 exp_id 时额外落盘。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(_FORMAT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if exp_id:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_DIR / f"{exp_id}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
