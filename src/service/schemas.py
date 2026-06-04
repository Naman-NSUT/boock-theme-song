from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class ThemeSongRequest(BaseModel):
    task_id: Optional[str] = None
    theme_title: str
    hook_lyrics: str
    genre: str
    emotion: str
    language: str = "en"
    vocal_style: Optional[str] = None
    tempo_bpm_hint: Optional[int] = Field(default=None, ge=40, le=240)
    duration_ms: int = Field(default=20000, ge=5000, le=60000)

    @field_validator("hook_lyrics")
    @classmethod
    def lyrics_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("hook_lyrics must not be empty")
        return v

    @field_validator("genre")
    @classmethod
    def genre_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("genre must not be empty")
        return v

    def derived_task_id(self) -> str:
        if self.task_id:
            return self.task_id
        slug = self.theme_title.lower().replace(" ", "_")
        return f"task_{slug}"

    def to_genre_txt(self) -> str:
        parts = [self.genre]
        if self.emotion:
            parts.append(self.emotion)
        if self.tempo_bpm_hint:
            parts.append(f"{self.tempo_bpm_hint} bpm")
        if self.vocal_style:
            parts.append(self.vocal_style)
        return ", ".join(parts)

    def to_lyrics_txt(self) -> str:
        lines = self.hook_lyrics.replace(" / ", "\n").replace("/", "\n").strip()
        return f"[chorus]\n{lines}\n"


class ThemeSongResponse(BaseModel):
    task_id: str
    status: str
    duration_ms: int
    full_mix_path: str
    instrumental_stem_path: str
    vocal_stem_path: str
    metadata_path: str
    warnings: List[str] = []


class ErrorResponse(BaseModel):
    task_id: Optional[str] = None
    status: str = "failed"
    error: str
    warnings: List[str] = []
