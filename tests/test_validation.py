"""
Tests for src/validate.py — output validation module.
Also fixes test_to_lyrics_txt_splits_on_slash for the P1 [chorus] change.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.validate import (
    validate_task_output,
    _check_audio_file,
    _check_metadata,
    DURATION_MIN_MS,
    DURATION_MAX_MS,
    SILENCE_RMS_THRESHOLD,
)
from src.service.schemas import ThemeSongRequest


# ── helpers ─────────────────────────────────────────────────────────────────

SR = 44100

def _sine(path: str, duration_s: float = 20.0, amplitude: float = 0.3, freq: float = 440.0):
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    sf.write(path, (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)

def _silent(path: str, duration_s: float = 20.0):
    sf.write(path, np.zeros(int(SR * duration_s), dtype=np.float32), SR)

def _clipped(path: str, duration_s: float = 20.0):
    data = np.ones(int(SR * duration_s), dtype=np.float32)
    sf.write(path, data, SR)

def _make_full_output(tmpdir: str, duration_s: float = 20.0):
    """Write a complete, valid task output set."""
    for fname in ["full_mix.wav", "vocals.wav", "instrumental.wav"]:
        _sine(os.path.join(tmpdir, fname), duration_s=duration_s)
    meta = {
        "task_id": "test_task",
        "actual_duration_ms": int(duration_s * 1000),
        "model_track": "B",
        "warnings": [],
    }
    with open(os.path.join(tmpdir, "metadata.json"), "w") as f:
        json.dump(meta, f)


# ── _check_audio_file ────────────────────────────────────────────────────────

def test_audio_missing_file():
    fr = _check_audio_file("/nonexistent/path.wav")
    assert not fr.ok
    assert any("not found" in e for e in fr.errors)


def test_audio_valid_20s():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        p = f.name
    try:
        _sine(p, duration_s=20.0)
        fr = _check_audio_file(p)
        assert fr.ok
        assert DURATION_MIN_MS <= fr.duration_ms <= DURATION_MAX_MS
        assert fr.errors == []
        assert fr.warnings == []
    finally:
        os.unlink(p)


def test_audio_too_short():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        p = f.name
    try:
        _sine(p, duration_s=5.0)
        fr = _check_audio_file(p)
        assert not fr.ok
        assert any("too short" in e for e in fr.errors)
    finally:
        os.unlink(p)


def test_audio_silent():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        p = f.name
    try:
        _silent(p)
        fr = _check_audio_file(p)
        assert not fr.ok
        assert any("silent" in e for e in fr.errors)
    finally:
        os.unlink(p)


def test_audio_clipped_warns():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        p = f.name
    try:
        _clipped(p)
        fr = _check_audio_file(p)
        assert any("clipping" in w for w in fr.warnings)
    finally:
        os.unlink(p)


# ── _check_metadata ──────────────────────────────────────────────────────────

def test_metadata_missing():
    fr = _check_metadata("/nonexistent/metadata.json")
    assert not fr.ok
    assert any("not found" in e for e in fr.errors)


def test_metadata_invalid_json():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("{broken json")
        p = f.name
    try:
        fr = _check_metadata(p)
        assert not fr.ok
        assert any("invalid JSON" in e for e in fr.errors)
    finally:
        os.unlink(p)


def test_metadata_valid():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"task_id": "t", "actual_duration_ms": 20000, "model_track": "B"}, f)
        p = f.name
    try:
        fr = _check_metadata(p)
        assert fr.ok
        assert fr.errors == []
    finally:
        os.unlink(p)


def test_metadata_missing_fields_warns():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"task_id": "t"}, f)
        p = f.name
    try:
        fr = _check_metadata(p)
        assert fr.ok          # not an error, just a warning
        assert any("actual_duration_ms" in w for w in fr.warnings)
    finally:
        os.unlink(p)


# ── validate_task_output ─────────────────────────────────────────────────────

def test_full_output_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_full_output(tmpdir)
        report = validate_task_output(tmpdir)
        assert report.passed
        assert report.errors == []


def test_missing_directory_fails():
    report = validate_task_output("/nonexistent/task_dir")
    assert not report.passed
    assert any("does not exist" in e for e in report.errors)


def test_missing_stem_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_full_output(tmpdir)
        os.unlink(os.path.join(tmpdir, "vocals.wav"))
        report = validate_task_output(tmpdir)
        assert not report.passed


def test_silent_full_mix_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_full_output(tmpdir)
        _silent(os.path.join(tmpdir, "full_mix.wav"))
        report = validate_task_output(tmpdir)
        assert not report.passed


def test_summary_string():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_full_output(tmpdir)
        report = validate_task_output(tmpdir)
        summary = report.summary()
        assert "[PASS]" in summary
        assert "full_mix.wav" in summary


# ── schema: lyrics now include [chorus] header (P1 fix) ─────────────────────

VALID_PAYLOAD = {
    "task_id": "task_01_firelight",
    "theme_title": "Firelight",
    "hook_lyrics": "Carry the fire, carry the light / Hold to the dream through the longest night",
    "genre": "cinematic pop anthem",
    "emotion": "hopeful",
    "language": "en",
    "tempo_bpm_hint": 98,
    "duration_ms": 20000,
}


def test_to_lyrics_txt_has_chorus_tag():
    req = ThemeSongRequest(**VALID_PAYLOAD)
    lyrics = req.to_lyrics_txt()
    assert lyrics.startswith("[chorus]\n")


def test_to_lyrics_txt_lyric_lines():
    req = ThemeSongRequest(**VALID_PAYLOAD)
    lyrics = req.to_lyrics_txt()
    # Non-empty lines after stripping: [chorus] + 2 lyric lines = 3
    lines = [l for l in lyrics.strip().splitlines() if l.strip()]
    assert len(lines) == 3
    assert lines[0] == "[chorus]"
    assert "fire" in lines[1].lower()
