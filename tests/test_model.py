"""Tests for QwenASR model loading and basic operations."""

from pathlib import Path
import sys

import pytest

from qwen_asr import QwenASR, TranscriptionConfig, QwenASRError
from qwen_asr.api import ModelLoadError

# Project root for finding models
PROJECT_ROOT = Path(__file__).parent.parent

# Check availability at import time
_LIB_PATHS = [
    PROJECT_ROOT / "libqwen_asr.so",
    PROJECT_ROOT / "libqwen_asr.dylib",
    PROJECT_ROOT / "qwen_asr.so",
]
_LIBRARY_AVAILABLE = any(p.exists() for p in _LIB_PATHS)


def _model_available(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    required = ["vocab.json", "config.json"]
    if not all((model_dir / f).exists() for f in required):
        return False
    return (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "model.safetensors.index.json").exists()
    )


_MODEL_06B = PROJECT_ROOT / "qwen3-asr-0.6b"
_MODEL_17B = PROJECT_ROOT / "qwen3-asr-1.7b"
_ANY_MODEL_AVAILABLE = _model_available(_MODEL_06B) or _model_available(_MODEL_17B)

requires_library = pytest.mark.skipif(
    not _LIBRARY_AVAILABLE,
    reason="C library not built. Run 'make shared' first.",
)

requires_model = pytest.mark.skipif(
    not _ANY_MODEL_AVAILABLE,
    reason="No model available. Run 'bash download_model.sh' first.",
)


class TestModelLoadErrors:
    """Tests for model loading error conditions."""

    def test_nonexistent_model_dir(self):
        """Test error when model directory doesn't exist."""
        with pytest.raises(ModelLoadError) as exc_info:
            QwenASR("/nonexistent/path/to/model")

        assert "not found" in str(exc_info.value).lower()

    def test_empty_model_dir(self, tmp_path: Path):
        """Test error when model directory is empty."""
        with pytest.raises(ModelLoadError) as exc_info:
            QwenASR(tmp_path)

        assert "Missing required file" in str(exc_info.value)

    def test_missing_vocab_json(self, tmp_path: Path):
        """Test error when vocab.json is missing."""
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"")

        with pytest.raises(ModelLoadError) as exc_info:
            QwenASR(tmp_path)

        assert "vocab.json" in str(exc_info.value)

    def test_missing_config_json(self, tmp_path: Path):
        """Test error when config.json is missing."""
        (tmp_path / "vocab.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"")

        with pytest.raises(ModelLoadError) as exc_info:
            QwenASR(tmp_path)

        assert "config.json" in str(exc_info.value)

    def test_missing_weights(self, tmp_path: Path):
        """Test error when model weights are missing."""
        (tmp_path / "vocab.json").write_text("{}")
        (tmp_path / "config.json").write_text("{}")

        with pytest.raises(ModelLoadError) as exc_info:
            QwenASR(tmp_path)

        assert "weights" in str(exc_info.value).lower()

    def test_path_as_string(self, tmp_path: Path):
        """Test that string paths are accepted."""
        with pytest.raises(ModelLoadError):
            # Should fail but accept string path
            QwenASR(str(tmp_path))


class TestModelProperties:
    """Tests for model properties and attributes."""

    @requires_library
    @requires_model
    def test_model_dir_property(self, model_dir: Path):
        """Test model_dir property returns correct path."""
        with QwenASR(model_dir) as model:
            assert model.model_dir == model_dir

    @requires_library
    @requires_model
    def test_config_property(self, model_dir: Path):
        """Test config property returns configuration."""
        config = TranscriptionConfig(segment_sec=30.0)

        with QwenASR(model_dir, config=config) as model:
            assert model.config == config
            assert model.config.segment_sec == 30.0

    @requires_library
    @requires_model
    def test_default_config_when_none_provided(self, model_dir: Path):
        """Test default config is used when none provided."""
        with QwenASR(model_dir) as model:
            assert model.config is not None
            assert isinstance(model.config, TranscriptionConfig)


class TestModelContextManager:
    """Tests for context manager functionality."""

    @requires_library
    @requires_model
    def test_context_manager_closes_model(self, model_dir: Path):
        """Test that context manager closes model on exit."""
        with QwenASR(model_dir) as model:
            assert model._ctx is not None

        # After context exit, model should be closed
        assert model._closed is True

    @requires_library
    @requires_model
    def test_context_manager_on_exception(self, model_dir: Path):
        """Test that context manager closes model even on exception."""
        try:
            with QwenASR(model_dir) as model:
                ctx_before = model._ctx
                raise ValueError("test exception")
        except ValueError:
            pass

        assert model._closed is True

    @requires_library
    @requires_model
    def test_explicit_close(self, model_dir: Path):
        """Test explicit close() method."""
        model = QwenASR(model_dir)
        assert model._closed is False

        model.close()
        assert model._closed is True

    @requires_library
    @requires_model
    def test_double_close_is_safe(self, model_dir: Path):
        """Test that closing twice doesn't raise error."""
        model = QwenASR(model_dir)
        model.close()
        model.close()  # Should not raise

        assert model._closed is True


class TestModelOperationsAfterClose:
    """Tests for operations on closed model."""

    @requires_library
    @requires_model
    def test_transcribe_after_close_raises(self, model_dir: Path, sample_wav: Path):
        """Test that transcribe raises error after close."""
        model = QwenASR(model_dir)
        model.close()

        with pytest.raises(QwenASRError) as exc_info:
            model.transcribe(sample_wav)

        assert "closed" in str(exc_info.value).lower()

    @requires_library
    @requires_model
    def test_stream_transcribe_after_close_raises(self, model_dir: Path):
        """Test that stream_transcribe raises error after close."""
        import numpy as np

        model = QwenASR(model_dir)
        model.close()

        samples = np.zeros(16000, dtype=np.float32)

        with pytest.raises(QwenASRError) as exc_info:
            model.stream_transcribe(samples)

        assert "closed" in str(exc_info.value).lower()
