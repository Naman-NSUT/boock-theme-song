"""Unit tests: request parsing, validation logic, schema contracts."""

import pytest
from pydantic import ValidationError

from src.service.schemas import ThemeSongRequest


VALID_PAYLOAD = {
    "task_id": "task_01_firelight",
    "theme_title": "Firelight",
    "hook_lyrics": "Carry the fire, carry the light / Hold to the dream through the longest night",
    "genre": "cinematic pop anthem with Indian percussion",
    "emotion": "hopeful, rising, heroic",
    "language": "en",
    "vocal_style": "young lead vocal",
    "tempo_bpm_hint": 98,
    "duration_ms": 20000,
}


def test_valid_request_parses():
    req = ThemeSongRequest(**VALID_PAYLOAD)
    assert req.theme_title == "Firelight"
    assert req.tempo_bpm_hint == 98
    assert req.duration_ms == 20000


def test_derived_task_id_uses_provided():
    req = ThemeSongRequest(**VALID_PAYLOAD)
    assert req.derived_task_id() == "task_01_firelight"


def test_derived_task_id_generated_from_title():
    payload = {**VALID_PAYLOAD, "task_id": None}
    req = ThemeSongRequest(**payload)
    assert req.derived_task_id() == "task_firelight"


def test_empty_lyrics_rejected():
    payload = {**VALID_PAYLOAD, "hook_lyrics": "   "}
    with pytest.raises(ValidationError) as exc_info:
        ThemeSongRequest(**payload)
    assert "hook_lyrics" in str(exc_info.value)


def test_empty_genre_rejected():
    payload = {**VALID_PAYLOAD, "genre": ""}
    with pytest.raises(ValidationError):
        ThemeSongRequest(**payload)


def test_tempo_out_of_range_rejected():
    payload = {**VALID_PAYLOAD, "tempo_bpm_hint": 300}
    with pytest.raises(ValidationError):
        ThemeSongRequest(**payload)


def test_to_genre_txt_includes_bpm():
    req = ThemeSongRequest(**VALID_PAYLOAD)
    genre_txt = req.to_genre_txt()
    assert "98 bpm" in genre_txt
    assert "cinematic pop anthem" in genre_txt


def test_to_lyrics_txt_splits_on_slash():
    req = ThemeSongRequest(**VALID_PAYLOAD)
    lyrics = req.to_lyrics_txt()
    lines = [l for l in lyrics.strip().splitlines() if l.strip()]
    # [chorus] header + 2 lyric lines = 3 (P1 fix added section tag for YuE)
    assert lines[0] == "[chorus]"
    assert len(lines) == 3


def test_optional_fields_have_defaults():
    minimal = {
        "theme_title": "Test",
        "hook_lyrics": "Some lyrics here",
        "genre": "pop",
        "emotion": "happy",
    }
    req = ThemeSongRequest(**minimal)
    assert req.language == "en"
    assert req.duration_ms == 20000
    assert req.vocal_style is None


def test_duration_bounds():
    payload_short = {**VALID_PAYLOAD, "duration_ms": 1000}
    with pytest.raises(ValidationError):
        ThemeSongRequest(**payload_short)

    payload_long = {**VALID_PAYLOAD, "duration_ms": 70000}
    with pytest.raises(ValidationError):
        ThemeSongRequest(**payload_long)
