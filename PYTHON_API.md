# Qwen3-ASR Python API

Python bindings for the [qwen_asr](README.md) C library. Provides a clean, typed interface for Qwen3-ASR speech-to-text inference with full support for offline, segmented, and streaming transcription modes.

**Important**: This API wraps the C library — you must build the shared library first (`make shared`) and download a model before using it. The bindings are designed for production use with proper error handling, context managers, and type hints.

## Quick Start

```bash
# Build the shared library
make shared

# Download a model
./download_model.sh --model small

# Install Python dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v
```

```python
from qwen_asr import QwenASR, TranscriptionConfig

# Basic transcription
with QwenASR("qwen3-asr-0.6b") as model:
    result = model.transcribe("audio.wav")
    print(result.text)

# With configuration
config = TranscriptionConfig(
    language="english",
    prompt="Technical terms: PostgreSQL, CUDA, FFmpeg"
)
with QwenASR("qwen3-asr-0.6b", config=config) as model:
    result = model.transcribe("lecture.wav")
    print(f"Transcribed in {result.inference_time_ms:.0f}ms")
```

## Features

- **Type-safe API**: Full type hints for IDE autocompletion and static analysis.
- **Context managers**: Automatic resource cleanup with `with` statements.
- **Multiple input formats**: WAV files, file paths, or raw NumPy arrays.
- **Streaming support**: Token-by-token output via callbacks or iterators.
- **Configuration validation**: Catches invalid settings before C library calls.
- **Proper error handling**: Typed exceptions for different failure modes.

## Installation

The Python API requires the C shared library to be built first:

```bash
# macOS (uses Accelerate framework)
make shared

# Linux (requires OpenBLAS)
sudo apt install libopenblas-dev  # Ubuntu/Debian
make shared

# Install Python package
uv pip install -e .

# Or with dev dependencies for testing
uv pip install -e ".[dev]"
```

## API Reference

### QwenASR

The main model class. Use as a context manager for automatic cleanup.

```python
from qwen_asr import QwenASR, TranscriptionConfig

# Basic usage
model = QwenASR("path/to/model")
result = model.transcribe("audio.wav")
model.close()

# With context manager (recommended)
with QwenASR("path/to/model") as model:
    result = model.transcribe("audio.wav")

# With configuration
config = TranscriptionConfig(segment_sec=30.0)
with QwenASR("path/to/model", config=config) as model:
    result = model.transcribe("long_recording.wav")
```

#### Methods

**`transcribe(audio, *, stream_callback=None)`**

Transcribe audio to text.

```python
# From WAV file
result = model.transcribe("audio.wav")
result = model.transcribe(Path("audio.wav"))

# From NumPy array (float32, 16kHz, mono)
import numpy as np
samples = np.zeros(16000, dtype=np.float32)  # 1 second of silence
result = model.transcribe(samples)

# With streaming callback
def on_token(token: str) -> None:
    print(token, end="", flush=True)

result = model.transcribe("audio.wav", stream_callback=on_token)
```

**`stream_transcribe(samples, callback=None)`**

Create a streaming transcription session for iterating over tokens.

```python
import numpy as np

samples = load_audio("audio.wav")  # float32, 16kHz, mono

with model.stream_transcribe(samples) as session:
    for token in session:
        print(token, end="", flush=True)

print(f"\nFinal: {session.result.text}")
```

### TranscriptionConfig

Immutable configuration for transcription settings. All values are validated on creation.

```python
from qwen_asr import TranscriptionConfig
from qwen_asr.api import PastTextMode

# Default configuration
config = TranscriptionConfig()

# Custom configuration
config = TranscriptionConfig(
    segment_sec=30.0,              # Split into 30s segments (0 = full audio)
    search_sec=3.0,                # Silence search window for segment cuts
    stream_mode=False,             # Use offline mode
    stream_max_new_tokens=32,      # Max tokens per streaming chunk
    encoder_window_sec=8.0,        # Encoder attention window (1-8)
    past_text_mode=PastTextMode.AUTO,  # Text conditioning mode
    skip_silence=False,            # Don't skip silent spans
    prompt="Technical: API, GPU",  # System prompt for biasing
    language="english",            # Force output language
    num_threads=0,                 # Auto-detect thread count
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `segment_sec` | `float` | `0.0` | Segment target in seconds. `0` = full-audio decode. |
| `search_sec` | `float` | `3.0` | Silence search window for segment boundaries. |
| `stream_mode` | `bool` | `False` | Enable streaming mode with chunk processing. |
| `stream_max_new_tokens` | `int` | `32` | Max tokens generated per streaming chunk. |
| `encoder_window_sec` | `float` | `8.0` | Encoder attention window in seconds (1-8). |
| `past_text_mode` | `PastTextMode` | `AUTO` | How to reuse previous text as context. |
| `skip_silence` | `bool` | `False` | Drop long silent spans before inference. |
| `prompt` | `str \| None` | `None` | System prompt for biasing model output. |
| `language` | `str \| None` | `None` | Force output language (auto-detect if `None`). |
| `num_threads` | `int` | `0` | Number of threads (`0` = auto-detect). |

#### Supported Languages

```
arabic, chinese, czech, danish, dutch, english, finnish, french, german,
greek, hebrew, hindi, hungarian, indonesian, italian, japanese, korean,
malay, norwegian, polish, portuguese, romanian, russian, spanish, swedish,
tagalog, thai, turkish, ukrainian, vietnamese
```

### TranscriptionResult

Result object returned by `transcribe()`.

```python
result = model.transcribe("audio.wav")

print(result.text)                    # Transcribed text
print(result.audio_duration_ms)       # Input audio duration
print(result.inference_time_ms)       # Total inference time
print(result.encoding_time_ms)        # Mel + encoder time
print(result.decoding_time_ms)        # Decoder time
print(result.tokens_generated)        # Number of tokens

# Computed properties
print(result.tokens_per_second)       # Throughput
print(result.realtime_factor)         # How many times faster than realtime
```

### Exceptions

All exceptions inherit from `QwenASRError`:

```python
from qwen_asr import QwenASRError, ModelLoadError, TranscriptionError
from qwen_asr.api import ConfigurationError

try:
    with QwenASR("invalid/path") as model:
        result = model.transcribe("audio.wav")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except TranscriptionError as e:
    print(f"Transcription failed: {e}")
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
except QwenASRError as e:
    print(f"General error: {e}")
```

| Exception | When Raised |
|-----------|-------------|
| `ModelLoadError` | Model directory not found, missing files, or C library load failure. |
| `TranscriptionError` | Audio file not found, invalid format, or transcription failure. |
| `ConfigurationError` | Invalid configuration values (e.g., unsupported language). |

## Examples

### Basic Transcription

```python
from qwen_asr import QwenASR

with QwenASR("qwen3-asr-0.6b") as model:
    result = model.transcribe("meeting.wav")
    print(result.text)
    print(f"Processed {result.audio_duration_ms/1000:.1f}s in {result.inference_time_ms/1000:.1f}s")
    print(f"({result.realtime_factor:.1f}x realtime)")
```

### Streaming Output

```python
from qwen_asr import QwenASR

def print_token(token: str) -> None:
    print(token, end="", flush=True)

with QwenASR("qwen3-asr-0.6b") as model:
    result = model.transcribe("audio.wav", stream_callback=print_token)
    print()  # Newline after streaming
```

### Processing NumPy Arrays

```python
import numpy as np
import wave
from qwen_asr import QwenASR

# Load audio from WAV file
with wave.open("audio.wav", "rb") as wf:
    frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

with QwenASR("qwen3-asr-0.6b") as model:
    result = model.transcribe(samples)
    print(result.text)
```

### Iterator-based Streaming

```python
import numpy as np
from qwen_asr import QwenASR

samples = np.zeros(16000 * 10, dtype=np.float32)  # 10 seconds

with QwenASR("qwen3-asr-0.6b") as model:
    with model.stream_transcribe(samples) as session:
        for token in session:
            print(token, end="", flush=True)

    print(f"\n\nFinal result: {session.result.text}")
```

### Language and Prompt Control

```python
from qwen_asr import QwenASR, TranscriptionConfig

# Force Italian output (may translate if source is different language)
config = TranscriptionConfig(language="italian")
with QwenASR("qwen3-asr-0.6b", config=config) as model:
    result = model.transcribe("english_speech.wav")
    print(result.text)  # Output in Italian

# Technical term biasing
config = TranscriptionConfig(
    prompt="Preserve spelling: PostgreSQL, Redis, Kubernetes, CUDA"
)
with QwenASR("qwen3-asr-0.6b", config=config) as model:
    result = model.transcribe("tech_talk.wav")
```

### Segmented Processing for Long Files

```python
from qwen_asr import QwenASR, TranscriptionConfig

# Process in 30-second segments (better for long files)
config = TranscriptionConfig(segment_sec=30.0)
with QwenASR("qwen3-asr-0.6b", config=config) as model:
    result = model.transcribe("hour_long_lecture.wav")
    print(result.text)
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=qwen_asr

# Run specific test file
uv run pytest tests/test_transcription.py -v
```

Test requirements:
- Built shared library (`make shared`)
- Downloaded model (`./download_model.sh`)
- Sample WAV files in `samples/` directory

## Thread Safety

The `QwenASR` class is **not thread-safe**. Each thread should create its own model instance. The underlying C library uses a global thread pool, so multiple Python instances will share compute resources efficiently.

```python
from concurrent.futures import ThreadPoolExecutor
from qwen_asr import QwenASR

def transcribe_file(path: str) -> str:
    # Each thread creates its own model instance
    with QwenASR("qwen3-asr-0.6b") as model:
        return model.transcribe(path).text

with ThreadPoolExecutor(max_workers=4) as executor:
    files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    results = list(executor.map(transcribe_file, files))
```

## Memory Management

The Python bindings manage C library memory automatically:
- Model resources are freed when `close()` is called or the context manager exits.
- Transcription result strings are copied to Python strings (small leak per call, see note below).

**Note**: Due to allocator compatibility issues on macOS, the C library's result strings are not freed by Python. This causes a small memory leak (~KB per transcription). For long-running applications, consider periodically restarting or using the C API directly.

## Comparison with C API

| Feature | C API | Python API |
|---------|-------|------------|
| Performance | Native | ~Same (thin wrapper) |
| Memory control | Manual | Automatic |
| Type safety | None | Full type hints |
| Error handling | Return codes | Exceptions |
| Streaming | Callback | Callback or iterator |

The Python API adds minimal overhead — all heavy computation happens in the C library.

## License

MIT (same as the C implementation)
