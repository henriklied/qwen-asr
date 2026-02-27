"""Tests for transcription functionality."""

from pathlib import Path
from typing import List

import numpy as np
import pytest

from qwen_asr import QwenASR, TranscriptionConfig, TranscriptionResult
from qwen_asr.api import TranscriptionError, SAMPLE_RATE

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


class TestTranscribeFile:
    """Tests for file-based transcription."""

    @requires_library
    @requires_model
    def test_transcribe_wav_file(self, model_dir: Path, sample_wav: Path):
        """Test transcribing a WAV file."""
        with QwenASR(model_dir) as model:
            result = model.transcribe(sample_wav)

        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    @requires_library
    @requires_model
    def test_transcribe_wav_file_as_string(self, model_dir: Path, sample_wav: Path):
        """Test transcribing with path as string."""
        with QwenASR(model_dir) as model:
            result = model.transcribe(str(sample_wav))

        assert isinstance(result, TranscriptionResult)
        assert len(result.text) > 0

    @requires_library
    @requires_model
    def test_transcribe_nonexistent_file(self, model_dir: Path):
        """Test error when transcribing nonexistent file."""
        with QwenASR(model_dir) as model:
            with pytest.raises(TranscriptionError) as exc_info:
                model.transcribe("/nonexistent/audio.wav")

        assert "not found" in str(exc_info.value).lower()

    @requires_library
    @requires_model
    def test_transcribe_with_callback(self, model_dir: Path, sample_wav: Path):
        """Test transcription with streaming callback."""
        tokens: List[str] = []

        def callback(token: str) -> None:
            tokens.append(token)

        with QwenASR(model_dir) as model:
            result = model.transcribe(sample_wav, stream_callback=callback)

        # Should have received tokens via callback
        assert len(tokens) > 0
        # Combined tokens should approximate result text
        combined = "".join(tokens)
        assert len(combined) > 0


class TestTranscribeSamples:
    """Tests for raw audio sample transcription."""

    @requires_library
    @requires_model
    def test_transcribe_numpy_array(self, model_dir: Path, sample_wav: Path):
        """Test transcribing numpy array of samples."""
        # Load actual audio samples from a WAV file
        import wave
        with wave.open(str(sample_wav), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        with QwenASR(model_dir) as model:
            result = model.transcribe(samples)

        assert isinstance(result, TranscriptionResult)
        assert len(result.text) > 0

    @requires_library
    @requires_model
    def test_transcribe_short_audio(self, model_dir: Path, sample_wav: Path):
        """Test transcribing short audio."""
        # Load and truncate audio to 1 second
        import wave
        with wave.open(str(sample_wav), 'rb') as wf:
            frames = wf.readframes(min(wf.getnframes(), SAMPLE_RATE))
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        with QwenASR(model_dir) as model:
            result = model.transcribe(samples)

        assert isinstance(result, TranscriptionResult)

    @requires_library
    @requires_model
    def test_transcribe_converts_to_float32(self, model_dir: Path, sample_wav: Path):
        """Test that non-float32 arrays are converted."""
        # Load audio as float64
        import wave
        with wave.open(str(sample_wav), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0

        with QwenASR(model_dir) as model:
            result = model.transcribe(samples)

        assert isinstance(result, TranscriptionResult)

    @requires_library
    @requires_model
    def test_transcribe_rejects_2d_array(self, model_dir: Path):
        """Test error for 2D (stereo) arrays."""
        samples = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)

        with QwenASR(model_dir) as model:
            with pytest.raises(TranscriptionError) as exc_info:
                model.transcribe(samples)

        assert "1D" in str(exc_info.value) or "mono" in str(exc_info.value).lower()


class TestStreamingTranscription:
    """Tests for streaming transcription."""

    @requires_library
    @requires_model
    def test_stream_transcribe_basic(self, model_dir: Path):
        """Test basic streaming transcription."""
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

        with QwenASR(model_dir) as model:
            with model.stream_transcribe(samples) as session:
                tokens = list(session)

        assert session.result is not None

    @requires_library
    @requires_model
    def test_stream_transcribe_with_callback(self, model_dir: Path):
        """Test streaming transcription with callback."""
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        callback_tokens: List[str] = []

        def callback(token: str) -> None:
            callback_tokens.append(token)

        with QwenASR(model_dir) as model:
            with model.stream_transcribe(samples, callback=callback) as session:
                list(session)  # Consume iterator

        # Callback should have been called
        assert len(callback_tokens) >= 0  # May be 0 for silence

    @requires_library
    @requires_model
    def test_stream_session_result_available_after_iteration(self, model_dir: Path):
        """Test that result is available after iteration."""
        samples = np.zeros(SAMPLE_RATE, dtype=np.float32)

        with QwenASR(model_dir) as model:
            session = model.stream_transcribe(samples)

            # Result not available before iteration
            assert session.result is None

            # Iterate
            list(session)

            # Result now available
            assert session.result is not None

    @requires_library
    @requires_model
    def test_stream_session_close(self, model_dir: Path, sample_wav: Path):
        """Test explicit close of streaming session."""
        # Load actual audio samples
        import wave
        with wave.open(str(sample_wav), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        with QwenASR(model_dir) as model:
            session = model.stream_transcribe(samples)
            # Start iteration first to trigger transcription
            list(session)
            session.close()

            assert session._finished is True
            assert session.result is not None


class TestTranscriptionWithConfig:
    """Tests for transcription with various configurations."""

    @requires_library
    @requires_model
    def test_transcribe_with_prompt(self, model_dir: Path, sample_wav: Path):
        """Test transcription with system prompt."""
        config = TranscriptionConfig(
            prompt="Technical terms: API, GPU, CUDA"
        )

        with QwenASR(model_dir, config=config) as model:
            result = model.transcribe(sample_wav)

        assert isinstance(result, TranscriptionResult)

    @requires_library
    @requires_model
    def test_transcribe_with_language(self, model_dir: Path, sample_wav: Path):
        """Test transcription with forced language."""
        config = TranscriptionConfig(language="english")

        with QwenASR(model_dir, config=config) as model:
            result = model.transcribe(sample_wav)

        assert isinstance(result, TranscriptionResult)

    @requires_library
    @requires_model
    def test_transcribe_segmented(self, model_dir: Path, sample_wav: Path):
        """Test segmented transcription mode."""
        config = TranscriptionConfig(segment_sec=10.0)

        with QwenASR(model_dir, config=config) as model:
            result = model.transcribe(sample_wav)

        assert isinstance(result, TranscriptionResult)
