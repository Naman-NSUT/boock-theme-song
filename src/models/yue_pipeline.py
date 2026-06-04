"""
YuE inference wrapper.

Primary model track: Track B — YuE
  Stage 1: m-a-p/YuE-s1-7B-anneal-en-cot  (4-bit quantized for 8GB VRAM)
  Stage 2: m-a-p/YuE-s2-1B-general

Inference is delegated to the official YuE/inference/infer.py script via subprocess,
using a patched copy that replaces flash_attention_2 with sdpa and adds 4-bit
quantization so it fits on an RTX 4060 8GB GPU.
"""

import os
import subprocess
import sys
import shutil
import glob
from pathlib import Path

YUE_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "YuE")
YUE_INFER_DIR = os.path.join(YUE_REPO, "inference")
PATCHED_INFER = os.path.join(os.path.dirname(__file__), "yue_infer_patched.py")

STAGE1_MODEL = "m-a-p/YuE-s1-7B-anneal-en-cot"
STAGE2_MODEL = "m-a-p/YuE-s2-1B-general"


def _build_patched_infer():
    """
    Read the official infer.py and write a patched version that:
    - Replaces flash_attention_2 with sdpa (no flash-attn install needed)
    - Adds load_in_4bit=True to fit 7B on 8GB VRAM
    - Removes torch.compile (saves startup time, avoids triton dependency)
    """
    src = os.path.join(YUE_INFER_DIR, "infer.py")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"YuE infer.py not found at {src}")

    with open(src) as f:
        code = f.read()

    # Replace flash_attention_2 with sdpa
    code = code.replace(
        'attn_implementation="flash_attention_2"',
        'attn_implementation="sdpa"'
    )

    # Add 4-bit quantization to stage1 model load
    old_load = (
        'model = AutoModelForCausalLM.from_pretrained(\n'
        '    stage1_model, \n'
        '    torch_dtype=torch.bfloat16,\n'
        '    attn_implementation="sdpa",\n'
        '    # device_map="auto",\n'
        '    )'
    )
    new_load = (
        'from transformers import BitsAndBytesConfig as _BnBConfig\n'
        '_bnb_cfg = _BnBConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)\n'
        'model = AutoModelForCausalLM.from_pretrained(\n'
        '    stage1_model,\n'
        '    quantization_config=_bnb_cfg,\n'
        '    attn_implementation="sdpa",\n'
        '    device_map="auto",\n'
        '    )'
    )
    if old_load in code:
        code = code.replace(old_load, new_load)
    else:
        # Fallback: inject 4bit config just before the model.to(device) line
        code = code.replace(
            'model = AutoModelForCausalLM.from_pretrained(\n    stage1_model, \n    torch_dtype=torch.bfloat16,',
            'from transformers import BitsAndBytesConfig as _BnBConfig\n'
            '_bnb_cfg = _BnBConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)\n'
            'model = AutoModelForCausalLM.from_pretrained(\n    stage1_model,\n    quantization_config=_bnb_cfg,\n    device_map="auto",',
        )
        # Remove the attn_implementation line that references flash_attention_2
        code = code.replace('    attn_implementation="sdpa",\n', '')

    # Remove torch.compile calls (avoids triton issues, saves startup time)
    import re
    code = re.sub(r'if torch\.__version__ >= "2\.0\.0":\n\s+model = torch\.compile\(model\)\n', '', code)
    code = re.sub(r'if torch\.__version__ >= "2\.0\.0":\n\s+model_stage2 = torch\.compile\(model_stage2\)\n', '', code)
    # Remove model.to(device) for stage-1: 4-bit models are already placed by device_map="auto"
    # Only strip the first occurrence (stage-1); stage-2 uses the standard to(device) call
    code = code.replace('model.to(device)\n', 'pass  # device_map="auto" handles placement\n', 1)
    # Fix off-by-one: len(lyrics) caps the loop before the first real segment runs.
    # prompt_texts = [full_prompt] + lyrics, so len(prompt_texts) = len(lyrics)+1.
    code = code.replace(
        'run_n_segments = min(args.run_n_segments+1, len(lyrics))',
        'run_n_segments = min(args.run_n_segments+1, len(prompt_texts))',
    )
    # Truncate genre slug to 40 chars to avoid OSError: filename too long (>255 chars).
    code = code.replace(
        'f"{genres.replace(\' \', \'-\')}_tp',
        'f"{genres.replace(\' \', \'-\')[:40]}_tp',
    )
    # torchaudio 2.11+ defaults to torchcodec backend which isn't installed.
    # Replace torchaudio.save with soundfile.write which is always available.
    code = code.replace(
        "    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)",
        "    import soundfile as _sf\n"
        "    _w = wav.cpu().numpy()\n"
        "    _sf.write(str(path), _w.T if _w.ndim == 2 else _w, sample_rate, subtype='PCM_16')",
    )
    # soundfile cannot write .mp3 — change all intermediate save paths to .wav.
    code = code.replace('+ ".mp3")', '+ ".wav")')
    code = code.replace("'itrack.mp3'", "'itrack.wav'")
    code = code.replace("'vtrack.mp3'", "'vtrack.wav'")

    # Global compat shim: torchaudio 2.11 defaults to the torchcodec backend (not installed),
    # so every torchaudio.save/.load in YuE's imported helpers (vocoder.py, post_process_audio.py)
    # raises ImportError. Monkey-patch both onto the torchaudio module so all call sites — across
    # every module that did `import torchaudio` — route through soundfile instead. Inserted right
    # after the torchaudio import so it is active before any helper runs.
    _shim = (
        "\n# --- compat shim: route torchaudio I/O through soundfile (torchcodec not installed) ---\n"
        "import soundfile as _sf_compat\n"
        "import torch as _torch_compat\n"
        "def _ta_save(path, wav, sample_rate=None, **kw):\n"
        "    p = str(path)\n"
        "    if p.lower().endswith('.mp3'):\n"
        "        p = p[:-4] + '.wav'\n"
        "    arr = wav.detach().cpu().numpy() if hasattr(wav, 'detach') else wav\n"
        "    if getattr(arr, 'ndim', 1) == 2:\n"
        "        arr = arr.T\n"
        "    sr = sample_rate if sample_rate is not None else kw.get('sample_rate', 44100)\n"
        "    _sf_compat.write(p, arr, sr, subtype='PCM_16')\n"
        "def _ta_load(path, *a, **k):\n"
        "    import os as _os\n"
        "    p = str(path)\n"
        "    if p.lower().endswith('.mp3') and not _os.path.exists(p):\n"
        "        alt = p[:-4] + '.wav'\n"
        "        if _os.path.exists(alt):\n"
        "            p = alt\n"
        "    data, sr = _sf_compat.read(p, dtype='float32', always_2d=True)\n"
        "    return _torch_compat.from_numpy(data.T.copy()), sr\n"
        "torchaudio.save = _ta_save\n"
        "torchaudio.load = _ta_load\n"
        "# --- end compat shim ---\n"
    )
    code = code.replace(
        "import torchaudio\nfrom torchaudio.transforms import Resample\n",
        "import torchaudio\nfrom torchaudio.transforms import Resample\n" + _shim,
        1,
    )

    with open(PATCHED_INFER, "w") as f:
        f.write(code)


def run_yue_inference(
    genre_txt: str,
    lyrics_txt: str,
    output_dir: str,
    stage1_model: str = STAGE1_MODEL,
    stage2_model: str = STAGE2_MODEL,
    run_n_segments: int = 1,
    max_new_tokens: int = 1500,
    seed: int = 42,
    stage2_batch_size: int = 3,
) -> str:
    """
    Run YuE inference and return path to the generated full_mix WAV.
    `output_dir` is where YuE writes its stage1/stage2 subdirectories.
    Returns path to the merged stereo output WAV.
    """
    _build_patched_infer()

    xcodec_dir = os.path.join(YUE_INFER_DIR, "xcodec_mini_infer")
    basic_config = os.path.join(xcodec_dir, "final_ckpt", "config.yaml")
    resume_path = os.path.join(xcodec_dir, "final_ckpt", "ckpt_00360000.pth")
    config_path = os.path.join(xcodec_dir, "decoders", "config.yaml")
    vocal_decoder = os.path.join(xcodec_dir, "decoders", "decoder_131000.pth")
    inst_decoder = os.path.join(xcodec_dir, "decoders", "decoder_151000.pth")

    for required in [basic_config, resume_path, config_path, vocal_decoder, inst_decoder]:
        if not os.path.isfile(required):
            raise FileNotFoundError(
                f"Required codec file missing: {required}\n"
                "Run: cd YuE/inference && git clone https://huggingface.co/m-a-p/xcodec_mini_infer"
            )

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, PATCHED_INFER,
        "--stage1_model", stage1_model,
        "--stage2_model", stage2_model,
        "--genre_txt", genre_txt,
        "--lyrics_txt", lyrics_txt,
        "--run_n_segments", str(run_n_segments),
        "--stage2_batch_size", str(stage2_batch_size),
        "--output_dir", output_dir,
        "--max_new_tokens", str(max_new_tokens),
        "--repetition_penalty", "1.1",
        "--seed", str(seed),
        "--basic_model_config", basic_config,
        "--resume_path", resume_path,
        "--config_path", config_path,
        "--vocal_decoder_path", vocal_decoder,
        "--inst_decoder_path", inst_decoder,
        "--rescale",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        YUE_INFER_DIR + os.pathsep +
        os.path.join(YUE_INFER_DIR, "xcodec_mini_infer") + os.pathsep +
        os.path.join(YUE_INFER_DIR, "xcodec_mini_infer", "descriptaudiocodec") + os.pathsep +
        env.get("PYTHONPATH", "")
    )
    # Reclaim fragmented VRAM so the larger stage-2 batch fits on 8GB. At batch_size=4
    # the run OOM'd by only ~82MB while ~945MB sat reserved-but-unallocated (fragmentation);
    # expandable_segments defragments the allocator and lets the faster batch run.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    result = subprocess.run(cmd, capture_output=False, text=True, env=env, cwd=YUE_INFER_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"YuE inference failed (exit {result.returncode})")

    return _find_generated_wav(output_dir)


def _find_generated_wav(output_dir: str) -> str:
    """Find the final WAV output from YuE inference.

    YuE's actual output layout (stage2/ only has .npy codec files):
      output_dir/*.wav              — final post-processed blend (replace_low_freq…)
      output_dir/vocoder/mix/*.wav  — 44.1kHz vocoder upsampled mix
      output_dir/recons/mix/*.wav   — 16kHz reconstructed mix
    """
    searched = []

    # 1. Final post-processed output at output_dir root (highest quality)
    top_wavs = sorted(glob.glob(os.path.join(output_dir, "*.wav")))
    searched.append(os.path.join(output_dir, "*.wav"))
    mixed = [w for w in top_wavs if "vtrack" not in w and "itrack" not in w]
    if mixed:
        return mixed[-1]

    # 2. Vocoder upsampled mix (44.1kHz)
    vocoder_wavs = sorted(glob.glob(os.path.join(output_dir, "vocoder", "mix", "*.wav")))
    searched.append(os.path.join(output_dir, "vocoder", "mix", "*.wav"))
    if vocoder_wavs:
        return vocoder_wavs[-1]

    # 3. Reconstructed 16kHz mix
    recons_wavs = sorted(glob.glob(os.path.join(output_dir, "recons", "mix", "*.wav")))
    searched.append(os.path.join(output_dir, "recons", "mix", "*.wav"))
    if recons_wavs:
        return recons_wavs[-1]

    # 4. Any remaining WAV (vtrack/itrack stems) — merge them
    all_wavs = sorted(glob.glob(os.path.join(output_dir, "**", "*.wav"), recursive=True))
    vtracks = [w for w in all_wavs if "vtrack" in w]
    itracks = [w for w in all_wavs if "itrack" in w]
    if vtracks and itracks:
        return _merge_tracks(vtracks[-1], itracks[-1], output_dir)

    raise FileNotFoundError(
        f"No WAV output found in {output_dir}. Searched:\n" +
        "\n".join(f"  {p}" for p in searched)
    )


def _merge_tracks(vocal_path: str, inst_path: str, output_dir: str) -> str:
    """Mix vocal + instrumental track into a single stereo WAV."""
    import soundfile as sf
    import numpy as np

    v_data, v_sr = sf.read(vocal_path)
    i_data, i_sr = sf.read(inst_path)

    # Resample instrumental if sample rates differ
    if v_sr != i_sr:
        import torchaudio
        i_tensor = _to_tensor(i_data)
        i_tensor = torchaudio.functional.resample(i_tensor, i_sr, v_sr)
        i_data = i_tensor.numpy().T

    # Align lengths
    min_len = min(len(v_data), len(i_data))
    v_data = v_data[:min_len]
    i_data = i_data[:min_len]

    mixed = (v_data + i_data) * 0.5
    out_path = os.path.join(output_dir, "stage2", "full_mix_raw.wav")
    sf.write(out_path, mixed, v_sr)
    return out_path


def _to_tensor(data):
    import torch
    import numpy as np
    arr = np.array(data)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    else:
        arr = arr.T
    return torch.from_numpy(arr.astype(np.float32))
