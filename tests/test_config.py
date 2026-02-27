"""Tests for TranscriptionConfig and configuration validation."""

import pytest

from qwen_asr import TranscriptionConfig, QwenASRError
from qwen_asr.api import ConfigurationError, PastTextMode, SUPPORTED_LANGUAGES


class TestTranscriptionConfig:
    """Tests for TranscriptionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TranscriptionConfig()

        assert config.segment_sec == 0.0
        assert config.search_sec == 3.0
        assert config.stream_mode is False
        assert config.stream_max_new_tokens == 32
        assert config.encoder_window_sec == 8.0
        assert config.past_text_mode == PastTextMode.AUTO
        assert config.skip_silence is False
        assert config.prompt is None
        assert config.language is None
        assert config.num_threads == 0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TranscriptionConfig(
            segment_sec=30.0,
            search_sec=5.0,
            stream_mode=True,
            stream_max_new_tokens=64,
            encoder_window_sec=4.0,
            past_text_mode=PastTextMode.YES,
            skip_silence=True,
            prompt="Technical terms: API, GPU",
            language="english",
            num_threads=4,
        )

        assert config.segment_sec == 30.0
        assert config.search_sec == 5.0
        assert config.stream_mode is True
        assert config.stream_max_new_tokens == 64
        assert config.encoder_window_sec == 4.0
        assert config.past_text_mode == PastTextMode.YES
        assert config.skip_silence is True
        assert config.prompt == "Technical terms: API, GPU"
        assert config.language == "english"
        assert config.num_threads == 4

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        config = TranscriptionConfig()

        with pytest.raises(AttributeError):
            config.segment_sec = 10.0  # type: ignore

    def test_invalid_segment_sec_negative(self):
        """Test validation rejects negative segment_sec."""
        with pytest.raises(ConfigurationError) as exc_info:
            TranscriptionConfig(segment_sec=-1.0)

        assert "segment_sec" in str(exc_info.value)

    def test_invalid_segment_sec_too_large(self):
        """Test validation rejects segment_sec > 3600."""
        with pytest.raises(ConfigurationError) as exc_info:
            TranscriptionConfig(segment_sec=4000.0)

        assert "segment_sec" in str(exc_info.value)

    def test_invalid_search_sec_zero(self):
        """Test validation rejects search_sec <= 0."""
        with pytest.raises(ConfigurationError) as exc_info:
            TranscriptionConfig(search_sec=0.0)

        assert "search_sec" in str(exc_info.value)

    def test_invalid_search_sec_too_large(self):
        """Test validation rejects search_sec > 60."""
        with pytest.raises(ConfigurationError) as exc_info:
            TranscriptionConfig(search_sec=100.0)

        assert "search_sec" in str(exc_info.value)

    def test_invalid_encoder_window_too_small(self):
        """Test validation rejects encoder_window_sec < 1."""
        with pytest.raises(ConfigurationError) as exc_info:
            TranscriptionConfig(encoder_window_sec=0.5)

        assert "encoder_window_sec" in str(exc_info.value)

    def test_invalid_encoder_window_too_large(self):
        """Test validation rejects encoder_window_sec > 8."""
        with pytest.raises(ConfigurationError) as exc_info:
            TranscriptionConfig(encoder_window_sec=10.0)

        assert "encoder_window_sec" in str(exc_info.value)

    def test_invalid_stream_max_new_tokens(self):
        """Test validation rejects stream_max_new_tokens < 1."""
        with pytest.raises(ConfigurationError) as exc_info:
            TranscriptionConfig(stream_max_new_tokens=0)

        assert "stream_max_new_tokens" in str(exc_info.value)

    def test_invalid_language(self):
        """Test validation rejects unsupported language."""
        with pytest.raises(ConfigurationError) as exc_info:
            TranscriptionConfig(language="klingon")

        assert "Unsupported language" in str(exc_info.value)
        assert "klingon" in str(exc_info.value)

    def test_valid_languages(self):
        """Test all supported languages are accepted."""
        for lang in SUPPORTED_LANGUAGES:
            config = TranscriptionConfig(language=lang)
            assert config.language == lang

    def test_language_case_insensitive(self):
        """Test language validation is case-insensitive."""
        config = TranscriptionConfig(language="ENGLISH")
        assert config.language == "ENGLISH"

        config = TranscriptionConfig(language="English")
        assert config.language == "English"

    def test_boundary_values_valid(self):
        """Test boundary values are accepted."""
        # Minimum valid values
        config = TranscriptionConfig(
            segment_sec=0.0,
            search_sec=0.001,
            encoder_window_sec=1.0,
            stream_max_new_tokens=1,
        )
        assert config.segment_sec == 0.0
        assert config.search_sec == 0.001
        assert config.encoder_window_sec == 1.0
        assert config.stream_max_new_tokens == 1

        # Maximum valid values
        config = TranscriptionConfig(
            segment_sec=3600.0,
            search_sec=60.0,
            encoder_window_sec=8.0,
        )
        assert config.segment_sec == 3600.0
        assert config.search_sec == 60.0
        assert config.encoder_window_sec == 8.0


class TestPastTextMode:
    """Tests for PastTextMode enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert PastTextMode.AUTO.value == "auto"
        assert PastTextMode.YES.value == "yes"
        assert PastTextMode.NO.value == "no"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert PastTextMode("auto") == PastTextMode.AUTO
        assert PastTextMode("yes") == PastTextMode.YES
        assert PastTextMode("no") == PastTextMode.NO


class TestSupportedLanguages:
    """Tests for supported languages constant."""

    def test_supported_languages_is_frozenset(self):
        """Test SUPPORTED_LANGUAGES is immutable."""
        assert isinstance(SUPPORTED_LANGUAGES, frozenset)

    def test_common_languages_present(self):
        """Test common languages are in the set."""
        common = ["english", "chinese", "japanese", "french", "german", "spanish"]
        for lang in common:
            assert lang in SUPPORTED_LANGUAGES

    def test_languages_are_lowercase(self):
        """Test all language names are lowercase."""
        for lang in SUPPORTED_LANGUAGES:
            assert lang == lang.lower()
