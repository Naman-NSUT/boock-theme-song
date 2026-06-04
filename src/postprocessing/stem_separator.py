import os

import numpy as np
import soundfile as sf
import torch


def native_stem_paths(yue_out_dir: str) -> tuple[str, str] | None:
    """
    Locate YuE's native vocoder stems (the model's own vocal/instrumental tracks,
    decoded at 44.1kHz before mixing). These are higher quality than re-separating
    the synthetic full mix with Demucs — Demucs is trained on real music and tends to
    misclassify AI-generated audio, dumping the whole mix into "vocals".

    Returns (vtrack_path, itrack_path) if both exist, else None.
    """
    stems = os.path.join(yue_out_dir, "vocoder", "stems")
    vtrack = os.path.join(stems, "vtrack.wav")
    itrack = os.path.join(stems, "itrack.wav")
    if os.path.isfile(vtrack) and os.path.isfile(itrack):
        return vtrack, itrack
    return None


def separate_stems(full_mix_path: str, output_dir: str) -> tuple[str, str]:
    """
    Separate full_mix into vocal + instrumental stems using Demucs htdemucs.

    Runs Demucs as an in-process library call (not a subprocess) and writes the
    stems with soundfile. This avoids two pitfalls: a bare `python3` subprocess
    resolving to a system interpreter without demucs, and Demucs' own
    `torchaudio.save` hitting the missing torchcodec backend on torchaudio 2.11.

    Stems are post-separated (not natively emitted by YuE).
    Returns (vocals_path, instrumental_path) in output_dir.
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    data, sr = sf.read(full_mix_path, dtype="float32", always_2d=True)  # (frames, channels)
    wav = torch.from_numpy(data.T.copy())                               # (channels, frames)

    model = get_model("htdemucs")
    model.eval()

    # htdemucs expects stereo audio at model.samplerate.
    if sr != model.samplerate:
        import torchaudio  # functional.resample is a pure tensor op (no codec backend)
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate
    if wav.shape[0] == 1:
        wav = wav.repeat(model.audio_channels, 1)        # mono -> stereo
    elif wav.shape[0] > model.audio_channels:
        wav = wav[: model.audio_channels]

    # Normalize before separation, denormalize after (standard demucs preprocessing).
    ref = wav.mean(0)
    wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)

    # Run on CPU: by this point the YuE inference subprocess has exited, but CPU
    # keeps stem separation independent of GPU memory state. A 20s clip is fast.
    with torch.no_grad():
        out = apply_model(model, wav_n[None], device="cpu", split=True, overlap=0.25)[0]
    out = out * ref.std() + ref.mean()                   # (stems, channels, frames)

    vocals_idx = model.sources.index("vocals")
    vocals = out[vocals_idx]
    instrumental = sum(out[i] for i in range(len(model.sources)) if i != vocals_idx)

    vocals_dest = os.path.join(output_dir, "vocals.wav")
    instrumental_dest = os.path.join(output_dir, "instrumental.wav")
    sf.write(vocals_dest, vocals.cpu().numpy().T, sr, subtype="PCM_16")
    sf.write(instrumental_dest, instrumental.cpu().numpy().T, sr, subtype="PCM_16")

    return vocals_dest, instrumental_dest
