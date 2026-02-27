"""
qwen_asr - Python API for Qwen3-ASR speech-to-text inference.

This module provides a clean Python interface to the qwen_asr C library,
supporting both Qwen3-ASR-0.6B and Qwen3-ASR-1.7B models.

Example usage:
    from qwen_asr import QwenASR, TranscriptionConfig

    with QwenASR("/path/to/model") as model:
        result = model.transcribe("audio.wav")
        print(result.text)
"""

from qwen_asr.api import (
    QwenASR,
    TranscriptionConfig,
    TranscriptionResult,
    StreamingSession,
    QwenASRError,
    ModelLoadError,
    TranscriptionError,
)

__version__ = "0.1.0"
__all__ = [
    "QwenASR",
    "TranscriptionConfig",
    "TranscriptionResult",
    "StreamingSession",
    "QwenASRError",
    "ModelLoadError",
    "TranscriptionError",
]
