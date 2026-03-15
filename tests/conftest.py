"""
pytest configuration and shared fixtures for Typhoon ASR tests.

Model-dependent tests are marked with @pytest.mark.integration and are
skipped by default unless --run-integration is passed on the CLI.
"""

import io
import os
import struct
import tempfile
import wave
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests that require the actual NeMo model (slow, ~30 s startup).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires the NeMo Typhoon model (skip with --no-integration)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="pass --run-integration to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    num_channels: int = 1,
    amplitude: float = 0.5,
    frequency: float = 440.0,
    silent: bool = False,
) -> bytes:
    """Return raw WAV file bytes for a sine-wave (or silence) clip."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    if silent:
        pcm = np.zeros(n_samples, dtype=np.float32)
    else:
        pcm = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    pcm_int16 = (pcm * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _write_tmp_wav(
    tmp_path: Path,
    name: str = "audio.wav",
    **kwargs,
) -> Path:
    """Write a WAV file to tmp_path and return its Path."""
    p = tmp_path / name
    p.write_bytes(_make_wav_bytes(**kwargs))
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_wav(tmp_path) -> Path:
    """Standard 1-second 16 kHz mono WAV."""
    return _write_tmp_wav(tmp_path, duration_s=1.0, sample_rate=16000)


@pytest.fixture
def tmp_wav_8khz(tmp_path) -> Path:
    """8 kHz WAV (needs resampling to 16 kHz)."""
    return _write_tmp_wav(tmp_path, name="audio_8k.wav", duration_s=1.0, sample_rate=8000)


@pytest.fixture
def tmp_wav_44khz(tmp_path) -> Path:
    """44.1 kHz WAV (needs resampling down to 16 kHz)."""
    return _write_tmp_wav(tmp_path, name="audio_44k.wav", duration_s=1.0, sample_rate=44100)


@pytest.fixture
def tmp_wav_stereo(tmp_path) -> Path:
    """Stereo WAV (model expects mono; pre-processing should handle this)."""
    return _write_tmp_wav(tmp_path, name="audio_stereo.wav", duration_s=1.0, num_channels=2)


@pytest.fixture
def tmp_wav_silence(tmp_path) -> Path:
    """1-second silent WAV."""
    return _write_tmp_wav(tmp_path, name="silence.wav", duration_s=1.0, silent=True)


@pytest.fixture
def tmp_wav_very_short(tmp_path) -> Path:
    """50 ms WAV — below typical ASR minimum segment length."""
    return _write_tmp_wav(tmp_path, name="very_short.wav", duration_s=0.05, sample_rate=16000)


@pytest.fixture
def tmp_wav_clipped(tmp_path) -> Path:
    """WAV with saturated (clipped) audio — amplitude > 1.0 before normalisation."""
    p = tmp_path / "clipped.wav"
    n = 16000
    # All values at max int16 — simulates a clipped microphone
    pcm = np.full(n, 32767, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    p.write_bytes(buf.getvalue())
    return p


@pytest.fixture
def tmp_wav_noisy(tmp_path) -> Path:
    """WAV with heavy white noise (SNR ≈ 0 dB)."""
    p = tmp_path / "noisy.wav"
    rng = np.random.default_rng(42)
    pcm = rng.uniform(-1.0, 1.0, 16000).astype(np.float32)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_int16.tobytes())
    p.write_bytes(buf.getvalue())
    return p


@pytest.fixture
def mock_nemo_model():
    """A MagicMock that replaces nemo_asr.models.ASRModel.from_pretrained."""
    model = MagicMock()
    model.transcribe.return_value = ["สวัสดีครับ"]
    return model


@pytest.fixture
def mock_nemo_model_with_hypothesis():
    """Mock model that returns a Hypothesis-style object."""
    hypothesis = MagicMock()
    hypothesis.text = "สวัสดีครับ ยินดีต้อนรับ"
    model = MagicMock()
    model.transcribe.return_value = [hypothesis]
    return model
