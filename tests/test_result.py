"""Tests for TranscriptionResult dataclass."""

import pytest

from qwen_asr import TranscriptionResult


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_default_values(self):
        """Test default values for optional fields."""
        result = TranscriptionResult(text="Hello world")

        assert result.text == "Hello world"
        assert result.audio_duration_ms == 0.0
        assert result.inference_time_ms == 0.0
        assert result.encoding_time_ms == 0.0
        assert result.decoding_time_ms == 0.0
        assert result.tokens_generated == 0

    def test_full_result(self):
        """Test result with all fields populated."""
        result = TranscriptionResult(
            text="This is a transcription.",
            audio_duration_ms=5000.0,
            inference_time_ms=1000.0,
            encoding_time_ms=200.0,
            decoding_time_ms=800.0,
            tokens_generated=25,
        )

        assert result.text == "This is a transcription."
        assert result.audio_duration_ms == 5000.0
        assert result.inference_time_ms == 1000.0
        assert result.encoding_time_ms == 200.0
        assert result.decoding_time_ms == 800.0
        assert result.tokens_generated == 25

    def test_tokens_per_second_calculation(self):
        """Test tokens_per_second property calculation."""
        result = TranscriptionResult(
            text="test",
            inference_time_ms=1000.0,
            tokens_generated=50,
        )

        assert result.tokens_per_second == 50.0

    def test_tokens_per_second_with_fast_inference(self):
        """Test tokens_per_second with sub-second inference."""
        result = TranscriptionResult(
            text="test",
            inference_time_ms=500.0,
            tokens_generated=100,
        )

        assert result.tokens_per_second == 200.0

    def test_tokens_per_second_zero_time(self):
        """Test tokens_per_second returns 0 when inference_time is 0."""
        result = TranscriptionResult(
            text="test",
            inference_time_ms=0.0,
            tokens_generated=50,
        )

        assert result.tokens_per_second == 0.0

    def test_realtime_factor_calculation(self):
        """Test realtime_factor property calculation."""
        result = TranscriptionResult(
            text="test",
            audio_duration_ms=10000.0,  # 10 seconds of audio
            inference_time_ms=2000.0,  # processed in 2 seconds
        )

        assert result.realtime_factor == 5.0  # 5x realtime

    def test_realtime_factor_slower_than_realtime(self):
        """Test realtime_factor when slower than realtime."""
        result = TranscriptionResult(
            text="test",
            audio_duration_ms=5000.0,
            inference_time_ms=10000.0,
        )

        assert result.realtime_factor == 0.5  # 0.5x realtime (slower)

    def test_realtime_factor_zero_inference_time(self):
        """Test realtime_factor returns 0 when inference_time is 0."""
        result = TranscriptionResult(
            text="test",
            audio_duration_ms=5000.0,
            inference_time_ms=0.0,
        )

        assert result.realtime_factor == 0.0

    def test_realtime_factor_zero_audio_duration(self):
        """Test realtime_factor returns 0 when audio_duration is 0."""
        result = TranscriptionResult(
            text="test",
            audio_duration_ms=0.0,
            inference_time_ms=1000.0,
        )

        assert result.realtime_factor == 0.0

    def test_empty_text(self):
        """Test result with empty text."""
        result = TranscriptionResult(text="")

        assert result.text == ""

    def test_unicode_text(self):
        """Test result with unicode text."""
        result = TranscriptionResult(text="こんにちは世界 🌍")

        assert result.text == "こんにちは世界 🌍"

    def test_multiline_text(self):
        """Test result with multiline text."""
        text = "Line one.\nLine two.\nLine three."
        result = TranscriptionResult(text=text)

        assert result.text == text
        assert result.text.count("\n") == 2

    def test_result_is_mutable(self):
        """Test that result fields can be modified."""
        result = TranscriptionResult(text="original")
        result.text = "modified"

        assert result.text == "modified"
