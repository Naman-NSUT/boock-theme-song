# Boock Theme Song Generator

Generates 20-second original songs (vocals + accompaniment) from short lyrics, a genre,
and an emotion.

**Model track:** Track B — YuE (`m-a-p/YuE-s1-7B-anneal-en-cot` + `m-a-p/YuE-s2-1B-general`)

Both provided tasks run end-to-end and produce validated 20s output:
`outputs/task_01_firelight/` and `outputs/task_02_neonrain/`.

---

## Requirements

- Python 3.10+ (developed on 3.12)
- NVIDIA GPU with ≥8 GB VRAM (tested: RTX 4060 Laptop 8GB, CUDA 13, PyTorch 2.11)
- ≥16 GB system RAM
- ≥25 GB free disk (model weights + outputs)
- `ffmpeg` installed (`sudo apt install ffmpeg`) — used for LUFS normalization

---

## Setup

### 1. Clone YuE and the codec weights

```bash
git clone https://github.com/multimodal-art-projection/YuE.git
cd YuE/inference
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
cd ../..
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the working set (torch 2.11, transformers 5.9, bitsandbytes 0.49,
demucs 4.0, torchaudio 2.11, soundfile, librosa, fastapi, uvicorn, pytest, …).

> **Note on torchaudio/torchcodec:** torchaudio 2.11 defaults to a `torchcodec` backend that
> is often not installed. This project does **not** require torchcodec — all audio I/O is
> routed through `soundfile`, and Demucs runs in-process rather than as a CLI subprocess.
> No extra action needed.

The YuE Stage-1/Stage-2 and xcodec weights (~14 GB + ~2 GB + ~0.5 GB) auto-download from
HuggingFace on first run and are cached for subsequent runs. Set `HF_TOKEN` for faster,
rate-limit-free downloads.

---

## How to run

### CLI

```bash
python run_task.py --input provided_inputs/task_01_firelight.json
```

Output is written to `outputs/<task_id>/`:

| File | Description |
|---|---|
| `full_mix.wav` | 20-second full mix, trimmed/faded, LUFS-normalized to −14 |
| `vocals.wav` | vocal stem (YuE native vocoder stem) |
| `instrumental.wav` | instrumental stem (YuE native vocoder stem) |
| `metadata.json` | task params, model IDs, seed, tokens, duration, stem source, warnings |

By default stems come from YuE's own internal vocal/instrumental tracks (`stem_source:
yue_native_vocoder_stems`), which are higher quality than re-separating the synthetic mix.
See DECISIONS.md for why. If native stems are unavailable, the pipeline falls back to
in-process Demucs.

#### Choosing a seed

YuE generation is sampled, and some prompt+seed combinations collapse the instrumental track
to silence. Pass `--seed` to re-roll:

```bash
python run_task.py --input provided_inputs/task_01_firelight.json --seed 7
```

`task_01_firelight` uses **seed 7** (seed 42 produced a silent instrumental for that prompt);
`task_02_neonrain` uses the default **seed 42**.

### HTTP service (FastAPI)

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
# health
curl http://localhost:8000/health

# render (synchronous — generation takes ~10 min on RTX 4060)
curl -X POST http://localhost:8000/theme-song/render \
  -H "Content-Type: application/json" \
  -d @provided_inputs/task_01_firelight.json
```

### Validate an output directory

```bash
python -m src.validate outputs/task_01_firelight/
```

Checks each file exists/reads, duration is 18–22 s, audio is non-silent and non-clipping,
and `metadata.json` is valid JSON.

---

## Models used

| Role | Model ID | Size | Notes |
|---|---|---|---|
| Stage 1 (LLM) | `m-a-p/YuE-s1-7B-anneal-en-cot` | 7B | Loaded in 4-bit on 8 GB GPU |
| Stage 2 (codec) | `m-a-p/YuE-s2-1B-general` | 1B | bfloat16 |
| Codec / vocoder | `m-a-p/xcodec_mini_infer` | ~0.5 GB | xcodec + Vocos decoders |
| Stems (primary) | YuE native vocoder stems | — | Model's own vocal/instrumental tracks |
| Stems (fallback) | Demucs `htdemucs` | ~80 MB | In-process, if native stems missing |

---

## Hardware assumptions

- GPU: NVIDIA RTX 4060 8GB (or any GPU ≥8 GB), CUDA 13, PyTorch 2.11
- RAM: ≥16 GB
- Disk: ≥25 GB free
- Generation time: ~10 min per song on RTX 4060 (measured 634 s for task_01)

---

## Run tests

```bash
pytest tests/ -v
```

35 tests covering schema validation, audio validation, trim-and-fade, formatter
round-trips, and output-directory validation. Tests do **not** run model inference, so they
finish in <1 s.

---

## Known limitations

- **Speed:** RTX 4060 8GB is much slower than the reference H800 (~10 min vs ~2.5 min).
- **4-bit quantization:** reduces stage-1 precision vs. bfloat16.
- **Seed sensitivity:** YuE sampling means some prompt+seed combinations collapse the
  instrumental to silence. Use `--seed` to re-roll (task_01 needs seed 7, not the default 42).
- **Tempo:** the BPM hint is a soft prompt, not enforced at the model level.
- **Synchronous API:** `/theme-song/render` blocks for the full generation time; a
  production deployment should use an async job queue (see DECISIONS.md).
