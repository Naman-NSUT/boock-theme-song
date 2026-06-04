import os
import subprocess
import soundfile as sf
import numpy as np


TARGET_DURATION_S = 20.0
FADE_IN_S = 0.4
FADE_OUT_S = 0.8


def trim_and_fade(input_path: str, output_path: str, target_s: float = TARGET_DURATION_S,
                  normalize: bool = True) -> None:
    """
    Trim or pad audio to target_s and apply fade in/out.
    normalize=True peak-normalizes to -1 dBFS (for the full mix / master).
    normalize=False preserves the source's level — use for stems so vocals and
    instrumental keep their natural relative balance and still sum to the mix.
    """
    data, sr = sf.read(input_path)
    target_samples = int(target_s * sr)

    if len(data) > target_samples:
        data = data[:target_samples]
    elif len(data) < target_samples:
        pad = target_samples - len(data)
        if data.ndim == 1:
            data = np.concatenate([data, np.zeros(pad)])
        else:
            data = np.concatenate([data, np.zeros((pad, data.shape[1]))])

    # Fade in
    fade_in_samples = int(FADE_IN_S * sr)
    fade_in = np.linspace(0, 1, fade_in_samples)
    if data.ndim == 1:
        data[:fade_in_samples] *= fade_in
    else:
        data[:fade_in_samples] *= fade_in[:, np.newaxis]

    # Fade out
    fade_out_samples = int(FADE_OUT_S * sr)
    fade_out = np.linspace(1, 0, fade_out_samples)
    if data.ndim == 1:
        data[-fade_out_samples:] *= fade_out
    else:
        data[-fade_out_samples:] *= fade_out[:, np.newaxis]

    # Peak normalize to -1 dBFS (master only; stems keep relative balance)
    if normalize:
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak * 0.891  # -1 dBFS

    sf.write(output_path, data, sr, subtype="PCM_24")


def apply_joint_peak_ceiling(paths: list, ceiling: float = 0.95) -> None:
    """
    Apply a single shared gain across multiple stems so the loudest peak across all of
    them sits at `ceiling`, only if it currently exceeds it. Using one shared factor (not
    per-file normalization) preserves the relative balance between stems and guarantees a
    little headroom so nothing clips on playback or format conversion.
    """
    arrs, srs = [], []
    for p in paths:
        data, sr = sf.read(p)
        arrs.append(data)
        srs.append(sr)
    if not arrs:
        return
    peak = max(float(np.max(np.abs(a))) for a in arrs)
    if peak > ceiling:
        gain = ceiling / peak
        for p, a, sr in zip(paths, arrs, srs):
            sf.write(p, a * gain, sr, subtype="PCM_24")


def normalize_lufs(input_path: str, output_path: str, target_lufs: float = -14.0) -> None:
    """
    Light mastering chain via ffmpeg:
      highpass=35Hz   — remove inaudible sub-bass rumble that wastes headroom
      loudnorm        — normalize perceived loudness to target LUFS with true-peak limiting
    Output is 44.1kHz, which is the standard delivery sample rate.
    """
    af = f"highpass=f=35,loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", af,
        "-ar", "44100",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg loudnorm failed: {result.stderr[-500:]}")


def mix_stereo(input_path: str, output_path: str) -> None:
    """Ensure output is stereo 44.1kHz."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "2", "-ar", "44100",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
