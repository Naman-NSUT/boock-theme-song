"""
Smoke / integration tests.
These do NOT run actual YuE inference (too slow/heavy for CI).
They test: validation logic, metadata writing, output structure.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.validation.audio_validator import validate_audio_file, validate_output_set, ValidationResult
from src.postprocessing.audio_processor import trim_and_fade
from src.preprocessing.formatter import load_request_from_json, write_yue_inputs


FIRELIGHT_JSON = os.path.join(
    os.path.dirname(__file__), "..",
    "provided_inputs", "task_01_firelight.json"
)


def _make_sine_wav(path: str, duration_s: float = 25.0, sr: int = 44100, freq: float = 440.0):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    data = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, data, sr)


# ── Validation unit tests ────────────────────────────────────────────────────

def test_validate_missing_file():
    result = validate_audio_file("/nonexistent/path.wav")
    assert not result.valid
    assert "not found" in result.warnings[0].lower()


def test_validate_valid_20s_sine():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        _make_sine_wav(path, duration_s=20.0)
        result = validate_audio_file(path)
        assert result.valid
        assert 18_000 <= result.duration_ms <= 22_000
        assert result.warnings == []
    finally:
        os.unlink(path)


def test_validate_too_short_raises_warning():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        _make_sine_wav(path, duration_s=5.0)
        result = validate_audio_file(path)
        assert result.valid  # not invalid, just warned
        assert any("below minimum" in w for w in result.warnings)
    finally:
        os.unlink(path)


def test_validate_silent_file_fails():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        sr = 44100
        silent = np.zeros(sr * 20, dtype=np.float32)
        sf.write(path, silent, sr)
        result = validate_audio_file(path)
        assert not result.valid
        assert any("silent" in w.lower() for w in result.warnings)
    finally:
        os.unlink(path)


# ── Trim-and-fade smoke test ─────────────────────────────────────────────────

def test_trim_and_fade_produces_20s():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "long.wav")
        dst = os.path.join(tmpdir, "trimmed.wav")
        _make_sine_wav(src, duration_s=35.0)
        trim_and_fade(src, dst, target_s=20.0)
        info = sf.info(dst)
        actual_ms = int(info.frames / info.samplerate * 1000)
        assert 19_800 <= actual_ms <= 20_200


# ── Request / formatter tests ────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.isfile(FIRELIGHT_JSON), reason="provided_inputs not available")
def test_load_firelight_request():
    req = load_request_from_json(FIRELIGHT_JSON)
    assert req.theme_title == "Firelight"
    assert req.tempo_bpm_hint == 98
    assert "fire" in req.hook_lyrics.lower()


@pytest.mark.skipif(not os.path.isfile(FIRELIGHT_JSON), reason="provided_inputs not available")
def test_write_yue_inputs_files():
    req = load_request_from_json(FIRELIGHT_JSON)
    with tempfile.TemporaryDirectory() as tmpdir:
        genre_path, lyrics_path = write_yue_inputs(req, tmpdir)
        assert os.path.isfile(genre_path)
        assert os.path.isfile(lyrics_path)
        with open(genre_path) as f:
            genre_content = f.read()
        assert "98 bpm" in genre_content
        with open(lyrics_path) as f:
            lyrics_content = f.read()
        assert "fire" in lyrics_content.lower()


# ── Output set validation smoke test ─────────────────────────────────────────

def test_validate_output_set_missing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        warnings = validate_output_set(tmpdir)
        assert len(warnings) > 0


def test_validate_output_set_all_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in ["full_mix.wav", "vocals.wav", "instrumental.wav"]:
            _make_sine_wav(os.path.join(tmpdir, fname), duration_s=20.0)
        meta = {"task_id": "test", "status": "completed"}
        with open(os.path.join(tmpdir, "metadata.json"), "w") as f:
            json.dump(meta, f)

        warnings = validate_output_set(tmpdir)
        # All duration warnings should be empty for valid 20s files
        duration_warnings = [w for w in warnings if "duration" in w.lower()]
        assert duration_warnings == []
