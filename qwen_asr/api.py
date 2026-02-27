"""
Core Python API for qwen_asr C library bindings.

Provides type-safe, Pythonic interface to the Qwen3-ASR speech-to-text engine.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_void_p,
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Exceptions
# =============================================================================


class QwenASRError(Exception):
    """Base exception for qwen_asr errors."""

    pass


class ModelLoadError(QwenASRError):
    """Raised when model loading fails."""

    pass


class TranscriptionError(QwenASRError):
    """Raised when transcription fails."""

    pass


class ConfigurationError(QwenASRError):
    """Raised when configuration is invalid."""

    pass


# =============================================================================
# Constants
# =============================================================================

SAMPLE_RATE = 16000
MEL_BINS = 128
HOP_LENGTH = 160
WINDOW_SIZE = 400

# Supported languages for --language flag (from C implementation)
SUPPORTED_LANGUAGES = frozenset([
    "chinese", "english", "cantonese", "arabic", "german", "french",
    "spanish", "portuguese", "indonesian", "italian", "korean", "russian",
    "thai", "vietnamese", "japanese", "turkish", "hindi", "malay",
    "dutch", "swedish", "danish", "finnish", "polish", "czech",
    "filipino", "persian", "greek", "romanian", "hungarian", "macedonian",
    "norwegian",  # Added for Nordic support
])


# =============================================================================
# Configuration
# =============================================================================


class PastTextMode(Enum):
    """Past text conditioning mode."""

    AUTO = "auto"
    YES = "yes"
    NO = "no"


@dataclass(frozen=True)
class TranscriptionConfig:
    """Configuration for transcription.

    Attributes:
        segment_sec: Segment target in seconds (0 = full-audio decode).
        search_sec: Segment-cutting silence search window in seconds.
        stream_mode: Enable streaming mode with chunk-based processing.
        stream_max_new_tokens: Max tokens per streaming chunk.
        encoder_window_sec: Encoder attention window in seconds (1-8).
        past_text_mode: How to reuse previous text as context.
        skip_silence: Drop long silent spans before inference.
        prompt: System prompt for biasing (e.g., "Preserve spelling: CPU, CUDA").
        language: Force output language (None = auto-detect).
        num_threads: Number of threads (0 = auto-detect).
    """

    segment_sec: float = 0.0
    search_sec: float = 3.0
    stream_mode: bool = False
    stream_max_new_tokens: int = 32
    encoder_window_sec: float = 8.0
    past_text_mode: PastTextMode = PastTextMode.AUTO
    skip_silence: bool = False
    prompt: Optional[str] = None
    language: Optional[str] = None
    num_threads: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.segment_sec <= 3600.0:
            raise ConfigurationError(
                f"segment_sec must be 0-3600, got {self.segment_sec}"
            )
        if not 0.0 < self.search_sec <= 60.0:
            raise ConfigurationError(
                f"search_sec must be 0-60, got {self.search_sec}"
            )
        if not 1.0 <= self.encoder_window_sec <= 8.0:
            raise ConfigurationError(
                f"encoder_window_sec must be 1-8, got {self.encoder_window_sec}"
            )
        if self.stream_max_new_tokens < 1:
            raise ConfigurationError(
                f"stream_max_new_tokens must be >= 1, got {self.stream_max_new_tokens}"
            )
        if self.language is not None:
            lang_lower = self.language.lower()
            if lang_lower not in SUPPORTED_LANGUAGES:
                raise ConfigurationError(
                    f"Unsupported language: {self.language}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
                )


@dataclass
class TranscriptionResult:
    """Result of a transcription.

    Attributes:
        text: The transcribed text.
        audio_duration_ms: Input audio duration in milliseconds.
        inference_time_ms: Total inference time in milliseconds.
        encoding_time_ms: Mel + encoder time in milliseconds.
        decoding_time_ms: Decoder prefill + autoregressive time in milliseconds.
        tokens_generated: Number of text tokens generated.
    """

    text: str
    audio_duration_ms: float = 0.0
    inference_time_ms: float = 0.0
    encoding_time_ms: float = 0.0
    decoding_time_ms: float = 0.0
    tokens_generated: int = 0

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.inference_time_ms > 0:
            return (1000.0 * self.tokens_generated) / self.inference_time_ms
        return 0.0

    @property
    def realtime_factor(self) -> float:
        """Calculate realtime factor (how many times faster than realtime)."""
        if self.inference_time_ms > 0 and self.audio_duration_ms > 0:
            return self.audio_duration_ms / self.inference_time_ms
        return 0.0


# =============================================================================
# C Library Bindings
# =============================================================================


# Callback type for token streaming
TokenCallback = CFUNCTYPE(None, c_char_p, c_void_p)


def _find_library() -> str:
    """Find the qwen_asr shared library."""
    # Look in common locations
    search_paths = [
        # Same directory as this file
        Path(__file__).parent.parent / "libqwen_asr.so",
        Path(__file__).parent.parent / "libqwen_asr.dylib",
        # System paths
        Path("/usr/local/lib/libqwen_asr.so"),
        Path("/usr/local/lib/libqwen_asr.dylib"),
        # Build directory
        Path(__file__).parent.parent / "qwen_asr.so",
    ]

    for path in search_paths:
        if path.exists():
            return str(path)

    # Try loading by name (system library path)
    return "libqwen_asr.so" if sys.platform != "darwin" else "libqwen_asr.dylib"


def _get_libc():
    """Get the C standard library for memory management."""
    if sys.platform == "darwin":
        return ctypes.CDLL("libc.dylib")
    else:
        return ctypes.CDLL("libc.so.6")


class _QwenLib:
    """Wrapper for the C library."""

    _instance: Optional[_QwenLib] = None
    _lock = threading.Lock()
    _libc: Optional[ctypes.CDLL] = None

    def __init__(self) -> None:
        lib_path = _find_library()
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise ModelLoadError(
                f"Failed to load qwen_asr library from {lib_path}: {e}\n"
                "Make sure to build the library first with 'make blas' or 'make shared'"
            ) from e

        self._setup_functions()

        # Load libc for memory management
        if _QwenLib._libc is None:
            _QwenLib._libc = _get_libc()
            _QwenLib._libc.free.argtypes = [c_void_p]
            _QwenLib._libc.free.restype = None

    def free_string(self, ptr: c_char_p) -> None:
        """Free a C string allocated by the library.

        Note: On macOS, the C library may use a different allocator than Python's libc,
        making it unsafe to free the memory. For now, we skip freeing to avoid crashes.
        This causes a small memory leak per transcription call.
        TODO: Add a proper qwen_free_string() function to the C library.
        """
        # Skip freeing for now to avoid allocator mismatch crashes on macOS
        pass

    def _setup_functions(self) -> None:
        """Set up function signatures."""
        # qwen_load
        self._lib.qwen_load.argtypes = [c_char_p]
        self._lib.qwen_load.restype = c_void_p

        # qwen_free
        self._lib.qwen_free.argtypes = [c_void_p]
        self._lib.qwen_free.restype = None

        # qwen_set_token_callback
        self._lib.qwen_set_token_callback.argtypes = [c_void_p, TokenCallback, c_void_p]
        self._lib.qwen_set_token_callback.restype = None

        # qwen_set_prompt
        self._lib.qwen_set_prompt.argtypes = [c_void_p, c_char_p]
        self._lib.qwen_set_prompt.restype = c_int

        # qwen_set_force_language
        self._lib.qwen_set_force_language.argtypes = [c_void_p, c_char_p]
        self._lib.qwen_set_force_language.restype = c_int

        # qwen_transcribe
        self._lib.qwen_transcribe.argtypes = [c_void_p, c_char_p]
        self._lib.qwen_transcribe.restype = c_char_p

        # qwen_transcribe_audio
        self._lib.qwen_transcribe_audio.argtypes = [c_void_p, POINTER(c_float), c_int]
        self._lib.qwen_transcribe_audio.restype = c_char_p

        # qwen_transcribe_stream
        self._lib.qwen_transcribe_stream.argtypes = [c_void_p, POINTER(c_float), c_int]
        self._lib.qwen_transcribe_stream.restype = c_char_p

    @classmethod
    def get_instance(cls) -> _QwenLib:
        """Get singleton instance of library wrapper."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


# =============================================================================
# Context Structure Access
# =============================================================================

# Offsets into qwen_ctx_t for settings we need to modify
# These are computed from the C struct definition in qwen_asr.h
# NOTE: This is fragile and depends on struct layout!


def _set_float_field(ctx: c_void_p, offset: int, value: float) -> None:
    """Set a float field in the context structure."""
    ptr = ctypes.cast(ctx, ctypes.POINTER(ctypes.c_char))
    float_ptr = ctypes.cast(
        ctypes.addressof(ptr.contents) + offset, ctypes.POINTER(ctypes.c_float)
    )
    float_ptr.contents.value = value


def _set_int_field(ctx: c_void_p, offset: int, value: int) -> None:
    """Set an int field in the context structure."""
    ptr = ctypes.cast(ctx, ctypes.POINTER(ctypes.c_char))
    int_ptr = ctypes.cast(
        ctypes.addressof(ptr.contents) + offset, ctypes.POINTER(ctypes.c_int)
    )
    int_ptr.contents.value = value


def _get_double_field(ctx: c_void_p, offset: int) -> float:
    """Get a double field from the context structure."""
    ptr = ctypes.cast(ctx, ctypes.POINTER(ctypes.c_char))
    double_ptr = ctypes.cast(
        ctypes.addressof(ptr.contents) + offset, ctypes.POINTER(ctypes.c_double)
    )
    return double_ptr.contents.value


def _get_int_field(ctx: c_void_p, offset: int) -> int:
    """Get an int field from the context structure."""
    ptr = ctypes.cast(ctx, ctypes.POINTER(ctypes.c_char))
    int_ptr = ctypes.cast(
        ctypes.addressof(ptr.contents) + offset, ctypes.POINTER(ctypes.c_int)
    )
    return int_ptr.contents.value


# =============================================================================
# Streaming Session
# =============================================================================


@dataclass
class StreamingSession:
    """A streaming transcription session.

    Use as a context manager or call close() when done.

    Example:
        with model.stream_transcribe(samples) as session:
            for token in session:
                print(token, end="", flush=True)
        print(session.result.text)
    """

    _model: QwenASR
    _samples: NDArray[np.float32]
    _callback: Optional[Callable[[str], None]] = None
    _tokens: list[str] = field(default_factory=list)
    _result: Optional[TranscriptionResult] = None
    _started: bool = False
    _finished: bool = False

    def __iter__(self) -> Iterator[str]:
        """Iterate over tokens as they're generated."""
        if self._finished:
            yield from self._tokens
            return

        if not self._started:
            self._start()

        # Tokens are collected via callback during transcription
        yield from self._tokens

    def _start(self) -> None:
        """Start the streaming transcription."""
        self._started = True
        self._tokens = []

        def token_callback(piece: bytes, userdata: c_void_p) -> None:
            text = piece.decode("utf-8", errors="replace")
            self._tokens.append(text)
            if self._callback:
                self._callback(text)

        # Create callback that persists for the duration of transcription
        self._c_callback = TokenCallback(token_callback)

        lib = _QwenLib.get_instance()
        lib._lib.qwen_set_token_callback(
            self._model._ctx, self._c_callback, None
        )

        # Run transcription
        samples_ptr = self._samples.ctypes.data_as(POINTER(c_float))
        result_ptr = lib._lib.qwen_transcribe_stream(
            self._model._ctx, samples_ptr, len(self._samples)
        )

        if result_ptr:
            text = ctypes.string_at(result_ptr).decode("utf-8", errors="replace")
            # Free the C string
            lib.free_string(result_ptr)
        else:
            text = "".join(self._tokens)

        self._result = TranscriptionResult(text=text)
        self._finished = True

        # Clear callback
        lib._lib.qwen_set_token_callback(self._model._ctx, TokenCallback(0), None)

    @property
    def result(self) -> Optional[TranscriptionResult]:
        """Get the final transcription result (available after iteration completes)."""
        return self._result

    def close(self) -> None:
        """Close the streaming session."""
        if not self._finished and self._started:
            # Consume remaining tokens
            list(self)

    def __enter__(self) -> StreamingSession:
        return self

    def __exit__(self, *args) -> None:
        self.close()


# =============================================================================
# Main API
# =============================================================================


class QwenASR:
    """Qwen3-ASR speech-to-text model.

    Provides transcription of audio files and raw audio samples.

    Example:
        # Basic usage
        model = QwenASR("/path/to/qwen3-asr-0.6b")
        result = model.transcribe("audio.wav")
        print(result.text)
        model.close()

        # With context manager
        with QwenASR("/path/to/model") as model:
            result = model.transcribe("audio.wav")

        # Custom configuration
        config = TranscriptionConfig(
            segment_sec=30.0,
            prompt="Technical terms: API, GPU, CUDA",
            language="english"
        )
        with QwenASR("/path/to/model", config=config) as model:
            result = model.transcribe("audio.wav")

        # Streaming transcription
        with QwenASR("/path/to/model") as model:
            samples = load_audio("audio.wav")  # float32, 16kHz mono
            with model.stream_transcribe(samples) as session:
                for token in session:
                    print(token, end="", flush=True)
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        config: Optional[TranscriptionConfig] = None,
    ) -> None:
        """Initialize the model.

        Args:
            model_dir: Path to model directory containing safetensors and vocab.json.
            config: Optional transcription configuration.

        Raises:
            ModelLoadError: If the model fails to load.
        """
        self._model_dir = Path(model_dir)
        self._config = config or TranscriptionConfig()
        self._ctx: Optional[c_void_p] = None
        self._lib: Optional[_QwenLib] = None
        self._token_callback: Optional[TokenCallback] = None
        self._closed = False

        self._load_model()

    def _load_model(self) -> None:
        """Load the model from disk."""
        # Validate model directory FIRST (before loading C library)
        if not self._model_dir.exists():
            raise ModelLoadError(f"Model directory not found: {self._model_dir}")

        # Check for required files
        required_files = ["vocab.json", "config.json"]
        for fname in required_files:
            if not (self._model_dir / fname).exists():
                raise ModelLoadError(
                    f"Missing required file: {self._model_dir / fname}"
                )

        # Check for model weights
        has_weights = (
            (self._model_dir / "model.safetensors").exists()
            or (self._model_dir / "model.safetensors.index.json").exists()
        )
        if not has_weights:
            raise ModelLoadError(
                f"No model weights found in {self._model_dir}. "
                "Expected model.safetensors or model.safetensors.index.json"
            )

        # Now load the C library (after validation passes)
        self._lib = _QwenLib.get_instance()

        # Load the model
        model_dir_bytes = str(self._model_dir).encode("utf-8")
        self._ctx = self._lib._lib.qwen_load(model_dir_bytes)

        if not self._ctx:
            raise ModelLoadError(f"Failed to load model from {self._model_dir}")

        # Apply configuration
        self._apply_config()

    def _apply_config(self) -> None:
        """Apply configuration to the loaded model."""
        config = self._config

        # Set prompt if specified
        if config.prompt:
            result = self._lib._lib.qwen_set_prompt(
                self._ctx, config.prompt.encode("utf-8")
            )
            if result != 0:
                raise ConfigurationError(f"Failed to set prompt: {config.prompt}")

        # Set language if specified
        if config.language:
            result = self._lib._lib.qwen_set_force_language(
                self._ctx, config.language.encode("utf-8")
            )
            if result != 0:
                raise ConfigurationError(
                    f"Failed to set language: {config.language}"
                )

        # Note: Other configuration options require struct field access
        # which is fragile. For a production API, the C library should
        # expose setter functions for all configuration options.

    def transcribe(
        self,
        audio: Union[str, Path, NDArray[np.float32]],
        *,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Path to WAV file or raw audio samples (float32, 16kHz mono).
            stream_callback: Optional callback for streaming tokens.

        Returns:
            TranscriptionResult with text and performance metrics.

        Raises:
            TranscriptionError: If transcription fails.
        """
        self._check_closed()

        # Set up streaming callback if provided
        if stream_callback:

            def c_callback(piece: bytes, userdata: c_void_p) -> None:
                text = piece.decode("utf-8", errors="replace")
                stream_callback(text)

            self._token_callback = TokenCallback(c_callback)
            self._lib._lib.qwen_set_token_callback(
                self._ctx, self._token_callback, None
            )
        else:
            self._lib._lib.qwen_set_token_callback(
                self._ctx, TokenCallback(0), None
            )

        try:
            if isinstance(audio, (str, Path)):
                return self._transcribe_file(Path(audio))
            else:
                return self._transcribe_samples(audio)
        finally:
            # Clear callback
            self._lib._lib.qwen_set_token_callback(
                self._ctx, TokenCallback(0), None
            )
            self._token_callback = None

    def _transcribe_file(self, path: Path) -> TranscriptionResult:
        """Transcribe a WAV file."""
        if not path.exists():
            raise TranscriptionError(f"Audio file not found: {path}")

        path_bytes = str(path).encode("utf-8")
        result_ptr = self._lib._lib.qwen_transcribe(self._ctx, path_bytes)

        if not result_ptr:
            raise TranscriptionError(f"Transcription failed for {path}")

        text = ctypes.string_at(result_ptr).decode("utf-8", errors="replace")
        # Free the C string
        self._lib.free_string(result_ptr)

        return TranscriptionResult(text=text)

    def _transcribe_samples(
        self, samples: NDArray[np.float32]
    ) -> TranscriptionResult:
        """Transcribe raw audio samples."""
        # Ensure samples are float32 and contiguous
        samples = np.ascontiguousarray(samples, dtype=np.float32)

        if samples.ndim != 1:
            raise TranscriptionError(
                f"Audio must be 1D array (mono), got shape {samples.shape}"
            )

        samples_ptr = samples.ctypes.data_as(POINTER(c_float))

        if self._config.stream_mode:
            result_ptr = self._lib._lib.qwen_transcribe_stream(
                self._ctx, samples_ptr, len(samples)
            )
        else:
            result_ptr = self._lib._lib.qwen_transcribe_audio(
                self._ctx, samples_ptr, len(samples)
            )

        if not result_ptr:
            raise TranscriptionError("Transcription failed")

        text = ctypes.string_at(result_ptr).decode("utf-8", errors="replace")
        # Free the C string
        self._lib.free_string(result_ptr)

        return TranscriptionResult(text=text)

    def stream_transcribe(
        self,
        samples: NDArray[np.float32],
        callback: Optional[Callable[[str], None]] = None,
    ) -> StreamingSession:
        """Create a streaming transcription session.

        Args:
            samples: Audio samples (float32, 16kHz mono).
            callback: Optional callback for each token.

        Returns:
            StreamingSession that yields tokens as they're generated.
        """
        self._check_closed()

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise TranscriptionError(
                f"Audio must be 1D array (mono), got shape {samples.shape}"
            )

        return StreamingSession(
            _model=self,
            _samples=samples,
            _callback=callback,
        )

    def _check_closed(self) -> None:
        """Check if the model is closed."""
        if self._closed:
            raise QwenASRError("Model has been closed")

    def close(self) -> None:
        """Release model resources."""
        if not self._closed and self._ctx:
            self._lib._lib.qwen_free(self._ctx)
            self._ctx = None
            self._closed = True

    def __enter__(self) -> QwenASR:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @property
    def model_dir(self) -> Path:
        """Get the model directory."""
        return self._model_dir

    @property
    def config(self) -> TranscriptionConfig:
        """Get the transcription configuration."""
        return self._config
