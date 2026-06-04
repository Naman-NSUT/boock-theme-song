"""
Main orchestrator: JSON task → full_mix.wav + stems + metadata.json
"""

import json
import os
import shutil
import time
from typing import Optional

from src.preprocessing.formatter import load_request_from_json, prepare_yue_inputs
from src.models.yue_pipeline import run_yue_inference
from src.postprocessing.audio_processor import trim_and_fade, normalize_lufs, apply_joint_peak_ceiling
from src.postprocessing.mastering import master
from src.postprocessing.stem_separator import separate_stems, native_stem_paths
from src.validation.audio_validator import validate_output_set
from src.service.schemas import ThemeSongRequest, ThemeSongResponse


OUTPUTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def run_pipeline(request: ThemeSongRequest, outputs_root: Optional[str] = None, seed: int = 42) -> ThemeSongResponse:
    if outputs_root is None:
        outputs_root = OUTPUTS_ROOT

    task_id = request.derived_task_id()
    task_out = os.path.join(outputs_root, task_id)
    os.makedirs(task_out, exist_ok=True)

    warnings: list[str] = []
    t0 = time.time()

    # 1. Prepare YuE inputs
    tmp_dir, genre_txt, lyrics_txt = prepare_yue_inputs(request)

    # 2. Run YuE inference
    yue_out_dir = os.path.join(task_out, "yue_raw")
    # ~2400 tokens yields >20s of real audio (1500 produced only ~15s, forcing a 5s
    # silence pad). The extra budget also gives YuE room to develop the instrumental
    # track instead of collapsing to vocals-only. We then trim real content to 20s.
    # `seed` controls the sampling trajectory: some prompt+seed combinations collapse the
    # instrumental track to silence, and re-rolling the seed escapes that trajectory.
    # ~2200 tokens -> ~22s of audio. Stage 2 splits this into three 6s segments, which
    # fit in one stage2_batch_size=3 pass on 8GB (one pass instead of two => faster),
    # while still leaving margin to trim a clean 20s. (2400 -> 24s -> 4 segments needs
    # batch 4 for a single pass, which OOMs on 8GB.)
    max_new_tokens = 2200
    raw_wav = run_yue_inference(
        genre_txt=genre_txt,
        lyrics_txt=lyrics_txt,
        output_dir=yue_out_dir,
        run_n_segments=1,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )

    # 3. Post-process: trim to 20s + fade, then master (EQ / glue comp / reverb /
    #    mono->stereo widen / makeup + limiter). Falls back to LUFS normalize, then to
    #    the trimmed mix, so a full_mix always exists.
    target_s = request.duration_ms / 1000.0
    trimmed_wav = os.path.join(task_out, "full_mix_trimmed.wav")
    trim_and_fade(raw_wav, trimmed_wav, target_s=target_s)

    full_mix_path = os.path.join(task_out, "full_mix.wav")
    mastered = True
    try:
        master(trimmed_wav, full_mix_path)
    except Exception as e:
        mastered = False
        warnings.append(f"Mastering failed, falling back to LUFS normalize: {e}")
        try:
            normalize_lufs(trimmed_wav, full_mix_path)
        except Exception as e2:
            warnings.append(f"LUFS normalization also failed, using trimmed mix: {e2}")
            shutil.copy2(trimmed_wav, full_mix_path)

    # 4. Stems. Prefer YuE's native vocoder stems (the model's own vocal/instrumental
    #    tracks) — they're higher quality than re-separating the synthetic mix with
    #    Demucs, which misclassifies AI-generated audio and collapses to vocals-only.
    #    Fall back to Demucs, then to copying the mix, so stems always exist.
    vocals_path = os.path.join(task_out, "vocals.wav")
    instrumental_path = os.path.join(task_out, "instrumental.wav")
    native = native_stem_paths(yue_out_dir)
    if native is not None:
        vtrack_raw, itrack_raw = native
        # normalize=False keeps the stems' relative balance (so they sum to the mix)
        trim_and_fade(vtrack_raw, vocals_path, target_s=target_s, normalize=False)
        trim_and_fade(itrack_raw, instrumental_path, target_s=target_s, normalize=False)
        # Shared peak ceiling: leave headroom so neither stem clips, balance preserved.
        apply_joint_peak_ceiling([vocals_path, instrumental_path], ceiling=0.95)
        stem_source = "yue_native_vocoder_stems"
    else:
        try:
            vocals_path, instrumental_path = separate_stems(full_mix_path, task_out)
            stem_source = "demucs_htdemucs_post_separated"
        except Exception as e:
            warnings.append(f"Stem separation failed: {e}. Stems fall back to full mix.")
            shutil.copy2(full_mix_path, vocals_path)
            shutil.copy2(full_mix_path, instrumental_path)
            stem_source = "full_mix_copy_fallback"

    # 5. Write metadata.json (before validation, so validate_output_set finds it)
    import soundfile as sf
    info = sf.info(full_mix_path)
    actual_duration_ms = int(info.frames / info.samplerate * 1000)

    metadata = {
        "task_id": task_id,
        "theme_title": request.theme_title,
        "genre": request.genre,
        "emotion": request.emotion,
        "language": request.language,
        "vocal_style": request.vocal_style,
        "tempo_bpm_hint": request.tempo_bpm_hint,
        "target_duration_ms": request.duration_ms,
        "actual_duration_ms": actual_duration_ms,
        "model_track": "B",
        "stage1_model": "m-a-p/YuE-s1-7B-anneal-en-cot",
        "stage2_model": "m-a-p/YuE-s2-1B-general",
        "stem_source": stem_source,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "mastered": mastered,
        "mastering": ("pedalboard: HPF + EQ(lowshelf/presence/air) + 2:1 glue comp + "
                      "reverb + M/S widen + makeup + limiter, peak 0.95, stereo"
                      if mastered else None),
        "generation_time_s": round(time.time() - t0, 1),
        "warnings": warnings,
    }
    metadata_path = os.path.join(task_out, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # 6. Validate the complete output set, then re-write metadata with any new warnings
    validation_warnings = validate_output_set(task_out)
    warnings.extend(validation_warnings)
    if validation_warnings:
        metadata["warnings"] = warnings
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return ThemeSongResponse(
        task_id=task_id,
        status="completed",
        duration_ms=actual_duration_ms,
        full_mix_path=full_mix_path,
        instrumental_stem_path=instrumental_path,
        vocal_stem_path=vocals_path,
        metadata_path=metadata_path,
        warnings=warnings,
    )


def run_pipeline_from_json(json_path: str, outputs_root: Optional[str] = None, seed: int = 42) -> ThemeSongResponse:
    request = load_request_from_json(json_path)
    return run_pipeline(request, outputs_root, seed=seed)
