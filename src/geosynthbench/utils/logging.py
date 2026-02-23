from __future__ import annotations

import sys

from loguru import logger
from loguru._logger import Logger


def _ensure_utf8_stdout() -> None:
    """
    Best-effort: ensure stdout uses UTF-8 when possible.

    On some environments (VS Code/Jupyter), sys.stdout may be a wrapped stream
    that does not implement `.reconfigure()`.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if not callable(reconfigure):
        return

    encoding = getattr(sys.stdout, "encoding", None)
    if isinstance(encoding, str) and encoding.lower() != "utf-8":
        # reconfigure exists here by callable() check above
        reconfigure(encoding="utf-8")


def setup_logging() -> None:
    """
    Configure global logger.
    Call this once at program entrypoints.
    """
    _ensure_utf8_stdout()

    logger.remove()

    logger.add(
        sys.stdout,
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- {message}"
        ),
        colorize=True,
    )


def get_logger() -> Logger:
    return logger  # type: ignore
