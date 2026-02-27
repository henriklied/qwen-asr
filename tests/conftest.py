"""Pytest configuration and fixtures for qwen_asr tests."""

from pathlib import Path

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Model directories
MODEL_DIR_06B = PROJECT_ROOT / "qwen3-asr-0.6b"
MODEL_DIR_17B = PROJECT_ROOT / "qwen3-asr-1.7b"

# Sample audio files
SAMPLES_DIR = PROJECT_ROOT / "samples"


def _check_library_available() -> bool:
    """Check if the C library is available."""
    lib_paths = [
        PROJECT_ROOT / "libqwen_asr.so",
        PROJECT_ROOT / "libqwen_asr.dylib",
        PROJECT_ROOT / "qwen_asr.so",
    ]
    return any(p.exists() for p in lib_paths)


def _check_model_available(model_dir: Path) -> bool:
    """Check if a model is available."""
    if not model_dir.exists():
        return False
    required = ["vocab.json", "config.json"]
    if not all((model_dir / f).exists() for f in required):
        return False
    has_weights = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "model.safetensors.index.json").exists()
    )
    return has_weights


# Availability flags
LIBRARY_AVAILABLE = _check_library_available()
MODEL_06B_AVAILABLE = _check_model_available(MODEL_DIR_06B)
MODEL_17B_AVAILABLE = _check_model_available(MODEL_DIR_17B)
ANY_MODEL_AVAILABLE = MODEL_06B_AVAILABLE or MODEL_17B_AVAILABLE


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def samples_dir() -> Path:
    """Get the samples directory."""
    return SAMPLES_DIR


@pytest.fixture
def model_dir() -> Path:
    """Get an available model directory."""
    if MODEL_06B_AVAILABLE:
        return MODEL_DIR_06B
    if MODEL_17B_AVAILABLE:
        return MODEL_DIR_17B
    pytest.skip("No model available")


@pytest.fixture
def model_dir_06b() -> Path:
    """Get the 0.6B model directory."""
    if not MODEL_06B_AVAILABLE:
        pytest.skip("0.6B model not available")
    return MODEL_DIR_06B


@pytest.fixture
def model_dir_17b() -> Path:
    """Get the 1.7B model directory."""
    if not MODEL_17B_AVAILABLE:
        pytest.skip("1.7B model not available")
    return MODEL_DIR_17B


@pytest.fixture
def sample_wav(samples_dir: Path) -> Path:
    """Get a sample WAV file for testing."""
    # Look for any .wav file in samples
    wav_files = list(samples_dir.glob("*.wav"))
    if not wav_files:
        pytest.skip("No sample WAV files found")
    return wav_files[0]
