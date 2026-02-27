"""Tests for module constants and exported symbols."""

import pytest


class TestPublicAPI:
    """Tests for public API exports."""

    def test_package_exports(self):
        """Test that __all__ contains expected exports."""
        import qwen_asr

        expected = [
            "QwenASR",
            "TranscriptionConfig",
            "TranscriptionResult",
            "StreamingSession",
            "QwenASRError",
            "ModelLoadError",
            "TranscriptionError",
        ]

        for name in expected:
            assert hasattr(qwen_asr, name), f"Missing export: {name}"
            assert name in qwen_asr.__all__

    def test_version_defined(self):
        """Test that __version__ is defined."""
        import qwen_asr

        assert hasattr(qwen_asr, "__version__")
        assert isinstance(qwen_asr.__version__, str)
        assert len(qwen_asr.__version__) > 0


class TestAudioConstants:
    """Tests for audio processing constants."""

    def test_sample_rate(self):
        """Test SAMPLE_RATE constant."""
        from qwen_asr.api import SAMPLE_RATE

        assert SAMPLE_RATE == 16000

    def test_mel_bins(self):
        """Test MEL_BINS constant."""
        from qwen_asr.api import MEL_BINS

        assert MEL_BINS == 128

    def test_hop_length(self):
        """Test HOP_LENGTH constant."""
        from qwen_asr.api import HOP_LENGTH

        assert HOP_LENGTH == 160

    def test_window_size(self):
        """Test WINDOW_SIZE constant."""
        from qwen_asr.api import WINDOW_SIZE

        assert WINDOW_SIZE == 400


class TestImports:
    """Tests for module import structure."""

    def test_import_from_package(self):
        """Test importing classes from package root."""
        from qwen_asr import (
            QwenASR,
            TranscriptionConfig,
            TranscriptionResult,
            StreamingSession,
            QwenASRError,
            ModelLoadError,
            TranscriptionError,
        )

        # Just verify they're importable and are classes/types
        assert callable(QwenASR)
        assert callable(TranscriptionConfig)
        assert callable(TranscriptionResult)
        assert callable(StreamingSession)
        assert issubclass(QwenASRError, Exception)
        assert issubclass(ModelLoadError, QwenASRError)
        assert issubclass(TranscriptionError, QwenASRError)

    def test_import_from_api_module(self):
        """Test importing from api submodule."""
        from qwen_asr.api import (
            QwenASR,
            TranscriptionConfig,
            TranscriptionResult,
            StreamingSession,
            QwenASRError,
            ModelLoadError,
            TranscriptionError,
            ConfigurationError,
            PastTextMode,
            SAMPLE_RATE,
            SUPPORTED_LANGUAGES,
        )

        # Verify imports succeeded
        assert QwenASR is not None
        assert TranscriptionConfig is not None
        assert SAMPLE_RATE == 16000
        assert len(SUPPORTED_LANGUAGES) > 0
