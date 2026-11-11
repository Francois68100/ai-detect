"""ai-detect — per-sentence AI text detection with pattern diagnostics.

Public API:
    from ai_detect import classify_text, get_detector
"""

from .detector import (
    DEFAULT_MODEL,
    MODELS,
    classify_text,
    classify_docx,
    get_detector,
    save_report_json,
)
from .patterns import diagnose_sentence

__all__ = [
    "classify_text",
    "classify_docx",
    "get_detector",
    "save_report_json",
    "diagnose_sentence",
    "MODELS",
    "DEFAULT_MODEL",
]

__version__ = "1.1.0"
