import os
from dataclasses import dataclass, field
from typing import List

import soundfile as sf
import numpy as np


TARGET_MIN_MS = 18_000
TARGET_MAX_MS = 22_000
SILENCE_RMS_THRESHOLD = 1e-4
CLIPPING_THRESHOLD = 0.99


@dataclass
class ValidationResult:
    valid: bool
    warnings: List[str] = field(default_factory=list)
    duration_ms: int = 0


def validate_audio_file(path: str) -> ValidationResult:
    warnings: List[str] = []

    if not os.path.isfile(path):
        return ValidationResult(valid=False, warnings=[f"File not found: {path}"])

    if os.path.getsize(path) == 0:
        return ValidationResult(valid=False, warnings=[f"File is empty: {path}"])

    try:
        data, samplerate = sf.read(path)
    except Exception as e:
        return ValidationResult(valid=False, warnings=[f"Cannot read audio: {e}"])

    if data.ndim > 1:
        mono = data.mean(axis=1)
    else:
        mono = data

    duration_ms = int(len(mono) / samplerate * 1000)

    if duration_ms < TARGET_MIN_MS:
        warnings.append(f"Duration {duration_ms}ms is below minimum {TARGET_MIN_MS}ms")
    if duration_ms > TARGET_MAX_MS:
        warnings.append(f"Duration {duration_ms}ms exceeds maximum {TARGET_MAX_MS}ms")

    rms = float(np.sqrt(np.mean(mono ** 2)))
    if rms < SILENCE_RMS_THRESHOLD:
        return ValidationResult(valid=False, warnings=[f"Output appears silent (RMS={rms:.6f})"])

    peak = float(np.max(np.abs(mono)))
    if peak >= CLIPPING_THRESHOLD:
        warnings.append(f"Output may be clipping (peak={peak:.4f})")

    return ValidationResult(valid=True, warnings=warnings, duration_ms=duration_ms)


def validate_output_set(output_dir: str) -> List[str]:
    """Validate all required output files. Returns list of warnings (empty = all good)."""
    warnings: List[str] = []
    required = ["full_mix.wav", "vocals.wav", "instrumental.wav", "metadata.json"]

    for fname in required:
        fpath = os.path.join(output_dir, fname)
        if fname.endswith(".json"):
            if not os.path.isfile(fpath):
                warnings.append(f"Missing metadata file: {fname}")
        else:
            result = validate_audio_file(fpath)
            if not result.valid:
                warnings.extend(result.warnings)
            else:
                warnings.extend(result.warnings)

    return warnings
