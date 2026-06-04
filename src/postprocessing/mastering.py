"""
Mastering chain for the final mix.

Turns YuE's raw mono output into a wider, clearer, more "produced" master:
  mono -> stereo  ·  corrective + tonal EQ  ·  glue compression  ·  short reverb
  ·  mid/side widening  ·  peak safety

Includes its own makeup gain + brickwall limiter so the master is loud and
peak-safe (≈ -1 dBFS ceiling); no external loudness stage is required.
"""

import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard,
    HighpassFilter,
    LowShelfFilter,
    PeakFilter,
    HighShelfFilter,
    Compressor,
    Reverb,
    Gain,
    Limiter,
)


def _to_stereo(data: np.ndarray) -> np.ndarray:
    """Return audio as (num_samples, 2)."""
    if data.ndim == 1:
        return np.stack([data, data], axis=1)
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2]


def _ms_widen(stereo: np.ndarray, amount: float = 1.4) -> np.ndarray:
    """Mid/side widening: boost the side signal to open up the stereo image.
    amount=1.0 is unchanged; ~1.4 is a tasteful widen. Input/-output: (n, 2)."""
    left, right = stereo[:, 0], stereo[:, 1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * amount
    return np.stack([mid + side, mid - side], axis=1)


def master(input_path: str, output_path: str, width: float = 1.4,
           makeup_db: float = 2.5, ceiling: float = 0.95) -> None:
    """Apply the full mastering chain and write a loud, stereo WAV.

    Creative stage (in order):
      HighpassFilter 40Hz   — strip sub-bass rumble
      LowShelf -1.5dB @220  — tame low-mid mud
      PeakFilter +2dB @3.2k — vocal/melody presence
      HighShelf +2.5dB @9k  — air / brightness
      Compressor 2:1 @-16dB — glue the mix together, add density
      Reverb (subtle)       — space + natural stereo decorrelation
      mid/side widen        — open the stereo image
    Loudness stage:
      Gain (makeup) + Limiter @ -1 dB — push perceived loudness, brickwall the peaks
    """
    data, sr = sf.read(input_path, dtype="float32", always_2d=False)
    audio = _to_stereo(data).T.copy()          # pedalboard wants (channels, samples)

    creative = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=40.0),
        LowShelfFilter(cutoff_frequency_hz=220.0, gain_db=-1.5, q=0.7),
        PeakFilter(cutoff_frequency_hz=3200.0, gain_db=2.0, q=0.8),
        HighShelfFilter(cutoff_frequency_hz=9000.0, gain_db=2.5, q=0.7),
        Compressor(threshold_db=-16.0, ratio=2.0, attack_ms=8.0, release_ms=120.0),
        Reverb(room_size=0.18, damping=0.5, wet_level=0.06, dry_level=0.94, width=1.0),
    ])
    processed = creative(audio, sr)            # (2, n)

    wide = _ms_widen(processed.T, amount=width)  # (n, 2) — widening can raise peaks

    # Loudness: modest makeup gain + limiter for density, then a hard numpy peak
    # normalize to `ceiling` as the final guarantee against any clipping (pedalboard's
    # Limiter is not a strict brickwall, so we enforce the ceiling ourselves).
    loud = Pedalboard([
        Gain(gain_db=makeup_db),
        Limiter(threshold_db=-1.0, release_ms=100.0),
    ])
    final = loud(wide.T.copy(), sr).T          # (n, 2)

    peak = float(np.max(np.abs(final)))
    if peak > 0:
        final = final / peak * ceiling

    sf.write(output_path, final, sr, subtype="PCM_24")
