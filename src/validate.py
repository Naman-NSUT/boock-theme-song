"""
Output validation for a completed theme-song task directory.

validate_task_output(task_dir) → ValidationReport
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import soundfile as sf


DURATION_MIN_MS = 18_000
DURATION_MAX_MS = 22_000
SILENCE_RMS_THRESHOLD = 1e-4
CLIPPING_THRESHOLD = 0.9999   # samples ≥ this absolute value count as clipped

REQUIRED_AUDIO = ["full_mix.wav", "vocals.wav", "instrumental.wav"]
REQUIRED_META  = ["metadata.json"]


@dataclass
class FileReport:
    path: str
    ok: bool
    duration_ms: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    task_dir: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    files: List[FileReport] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.task_dir}"]
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING {w}")
        for fr in self.files:
            icon = "✓" if fr.ok else "✗"
            lines.append(f"  {icon} {os.path.basename(fr.path)}  ({fr.duration_ms} ms)")
            for e in fr.errors:
                lines.append(f"      ERROR   {e}")
            for w in fr.warnings:
                lines.append(f"      WARNING {w}")
        return "\n".join(lines)


def _check_audio_file(path: str) -> FileReport:
    report = FileReport(path=path, ok=False)

    if not os.path.isfile(path):
        report.errors.append("file not found")
        return report

    if os.path.getsize(path) == 0:
        report.errors.append("file is empty (0 bytes)")
        return report

    try:
        data, sr = sf.read(path)
    except Exception as e:
        report.errors.append(f"unreadable: {e}")
        return report

    mono = data.mean(axis=1) if data.ndim > 1 else data
    report.duration_ms = int(len(mono) / sr * 1000)

    if report.duration_ms < DURATION_MIN_MS:
        report.errors.append(
            f"too short: {report.duration_ms} ms (min {DURATION_MIN_MS} ms)"
        )
    elif report.duration_ms > DURATION_MAX_MS:
        report.warnings.append(
            f"slightly long: {report.duration_ms} ms (max {DURATION_MAX_MS} ms)"
        )

    rms = float(np.sqrt(np.mean(mono ** 2)))
    if rms < SILENCE_RMS_THRESHOLD:
        report.errors.append(f"silent output (RMS {rms:.2e} < {SILENCE_RMS_THRESHOLD:.0e})")

    peak = float(np.max(np.abs(mono)))
    if peak >= CLIPPING_THRESHOLD:
        report.warnings.append(f"possible clipping (peak {peak:.4f})")

    report.ok = len(report.errors) == 0
    return report


def _check_metadata(path: str) -> FileReport:
    report = FileReport(path=path, ok=False)

    if not os.path.isfile(path):
        report.errors.append("metadata.json not found")
        return report

    try:
        with open(path) as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        report.errors.append(f"invalid JSON: {e}")
        return report

    for key in ("task_id", "actual_duration_ms", "model_track"):
        if key not in meta:
            report.warnings.append(f"missing field: {key}")

    report.ok = len(report.errors) == 0
    return report


def validate_task_output(task_dir: str) -> ValidationReport:
    """
    Validate all required files in a task output directory.
    Returns a ValidationReport with per-file detail and an overall passed flag.
    """
    report = ValidationReport(task_dir=task_dir, passed=False)

    if not os.path.isdir(task_dir):
        report.errors.append(f"output directory does not exist: {task_dir}")
        return report

    for fname in REQUIRED_AUDIO:
        fr = _check_audio_file(os.path.join(task_dir, fname))
        report.files.append(fr)
        report.errors.extend(fr.errors)
        report.warnings.extend(fr.warnings)

    for fname in REQUIRED_META:
        fr = _check_metadata(os.path.join(task_dir, fname))
        report.files.append(fr)
        report.errors.extend(fr.errors)
        report.warnings.extend(fr.warnings)

    report.passed = len(report.errors) == 0
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m src.validate <task_output_dir>")
        sys.exit(1)
    r = validate_task_output(sys.argv[1])
    print(r.summary())
    sys.exit(0 if r.passed else 1)
