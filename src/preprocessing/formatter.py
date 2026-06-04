import json
import os
import tempfile
from pathlib import Path
from typing import Tuple

from src.service.schemas import ThemeSongRequest


def load_request_from_json(json_path: str) -> ThemeSongRequest:
    with open(json_path) as f:
        data = json.load(f)
    return ThemeSongRequest(**data)


def write_yue_inputs(request: ThemeSongRequest, tmp_dir: str) -> Tuple[str, str]:
    """Write genre.txt and lyrics.txt to tmp_dir for YuE inference. Returns (genre_path, lyrics_path)."""
    genre_path = os.path.join(tmp_dir, "genre.txt")
    lyrics_path = os.path.join(tmp_dir, "lyrics.txt")

    with open(genre_path, "w") as f:
        f.write(request.to_genre_txt())

    with open(lyrics_path, "w") as f:
        f.write(request.to_lyrics_txt())

    return genre_path, lyrics_path


def prepare_yue_inputs(request: ThemeSongRequest) -> Tuple[str, str, str]:
    """
    Creates a temp directory, writes genre.txt and lyrics.txt.
    Returns (tmp_dir, genre_path, lyrics_path). Caller owns the tmp_dir lifetime.
    """
    tmp_dir = tempfile.mkdtemp(prefix="boock_yue_")
    genre_path, lyrics_path = write_yue_inputs(request, tmp_dir)
    return tmp_dir, genre_path, lyrics_path
