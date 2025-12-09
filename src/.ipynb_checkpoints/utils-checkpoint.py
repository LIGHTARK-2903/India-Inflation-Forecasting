"""
utils.py
--------
Utility helpers for the India-Inflation-Forecasting project.
Includes logging configuration, safe IO, argument parsing helpers, and formatting utilities.

Author: LIGHTARK (Naman)
"""

import logging
from pathlib import Path
from typing import Any, Iterable
import json


# -----------------------------
# Logging configuration helper
# -----------------------------

def configure_logging(level: str = 'INFO') -> None:
    """Configure root logger for consistent formatting across modules."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


# -----------------------------
# Safe IO helpers
# -----------------------------

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent)


# -----------------------------
# Formatting helpers
# -----------------------------

def pct(x: float, digits: int = 2) -> str:
    """Format a float as percent string."""
    return f"{x:.{digits}f}%"


def safe_list(obj: Any) -> list:
    """Ensure an object is returned as a list."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, (tuple, set)):
        return list(obj)
    return [obj]


# -----------------------------
# Misc
# -----------------------------

def chunk_iterable(iterable: Iterable, chunk_size: int):
    """Yield successive chunk_size-sized chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk