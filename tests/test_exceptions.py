"""Tests for exception classes and error handling."""

import pytest

from qwen_asr import QwenASRError, ModelLoadError, TranscriptionError
from qwen_asr.api import ConfigurationError


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_qwen_asr_error_is_base(self):
        """Test QwenASRError is the base exception."""
        assert issubclass(ModelLoadError, QwenASRError)
        assert issubclass(TranscriptionError, QwenASRError)
        assert issubclass(ConfigurationError, QwenASRError)

    def test_qwen_asr_error_inherits_exception(self):
        """Test QwenASRError inherits from Exception."""
        assert issubclass(QwenASRError, Exception)

    def test_can_catch_all_with_base(self):
        """Test catching all errors with base class."""
        errors = [
            QwenASRError("base error"),
            ModelLoadError("model error"),
            TranscriptionError("transcription error"),
            ConfigurationError("config error"),
        ]

        for error in errors:
            with pytest.raises(QwenASRError):
                raise error


class TestExceptionMessages:
    """Tests for exception message handling."""

    def test_qwen_asr_error_message(self):
        """Test QwenASRError preserves message."""
        error = QwenASRError("test message")
        assert str(error) == "test message"

    def test_model_load_error_message(self):
        """Test ModelLoadError preserves message."""
        error = ModelLoadError("failed to load model")
        assert str(error) == "failed to load model"

    def test_transcription_error_message(self):
        """Test TranscriptionError preserves message."""
        error = TranscriptionError("transcription failed")
        assert str(error) == "transcription failed"

    def test_configuration_error_message(self):
        """Test ConfigurationError preserves message."""
        error = ConfigurationError("invalid config")
        assert str(error) == "invalid config"


class TestExceptionChaining:
    """Tests for exception chaining."""

    def test_model_load_error_with_cause(self):
        """Test ModelLoadError can chain cause."""
        cause = OSError("file not found")
        error = ModelLoadError("failed to load")
        error.__cause__ = cause

        assert error.__cause__ is cause

    def test_exception_from_clause(self):
        """Test 'from' clause for chaining."""
        try:
            try:
                raise OSError("original error")
            except OSError as e:
                raise ModelLoadError("wrapped error") from e
        except ModelLoadError as wrapped:
            assert isinstance(wrapped.__cause__, OSError)
            assert str(wrapped.__cause__) == "original error"
