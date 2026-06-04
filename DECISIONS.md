# DECISIONS.md

## Primary model track chosen: Track B — YuE

**Exact checkpoints:**
| Role | Model ID |
|---|---|
| Stage 1 (LLM) | `m-a-p/YuE-s1-7B-anneal-en-cot` (4-bit quantized) |
| Stage 2 (codec upsampler) | `m-a-p/YuE-s2-1B-general` (bfloat16) |
| Codec / vocoder | `m-a-p/xcodec_mini_infer` (xcodec + Vocos decoders) |
| Stems (primary) | YuE native vocoder stems (`vtrack.wav` / `itrack.wav`) |
| Stems (fallback) | Demucs `htdemucs`, in-process |

Both provided tasks generate successfully end-to-end on an RTX 4060 8GB:
`task_01_firelight` and `task_02_neonrain` each produce a 20.0s `full_mix.wav` plus
vocal/instrumental stems and `metadata.json`, with zero validation warnings.

---

## Why YuE (Track B) over SongGeneration (Track A)

The assignment provided two prepared input sets (`provided_inputs/yue/` and
`provided_inputs/songgeneration/`). Both were evaluated against the 8 GB VRAM constraint.

| Factor | SongGeneration (Track A) | YuE (Track B — chosen) |
|---|---|---|
| Preferred checkpoint | `v2-large` — est. >14 GB VRAM in fp16 | `s1-7B` — fits in 8 GB at 4-bit |
| 8 GB fallback | `base-new` — lower quality, no documented quant path | 4-bit s1-7B — well-documented `BitsAndBytesConfig` path |
| Vocal quality | Adequate, structured tags | Excellent — LLM→codec, explicit lyric conditioning |
| Inference path | Single model, single pass | Two-stage: stage-1 LLM tokens → stage-2 codec → Vocos |
| Native stems | Yes (`--separate`) | No — Demucs post-separation |

**The deciding factor:** SongGeneration's preferred `v2-large` checkpoint does not fit in
8 GB VRAM. The `base-new` fallback would yield weaker vocals than YuE's 7B model quantized
to 4-bit, which was confirmed to fit and run on the RTX 4060 8GB.

**Why not `base-new` specifically:** it has no published quantization config, so fitting it
in 8 GB would have meant custom patching with no reference. YuE 4-bit had a documented,
tested HuggingFace path at this VRAM tier.

**Why not Track C (ACE-Step):** the assignment warns it requires "clearly audible sung
vocals" as a proof criterion; ACE-Step skews instrumental and adds vocal-presence
uncertainty within the timebox. YuE is purpose-built for lyric→song with vocals.

---

## Stem source — and why it changed from Demucs to native

The pipeline first followed the assignment's suggested path: generate `full_mix.wav`, then
derive stems with Demucs `htdemucs`. **Testing on real output proved this was the wrong
choice for YuE audio.** Demucs is trained on real recorded music; YuE's output is synthetic
and out-of-distribution, so Demucs misclassified almost the entire mix as "vocals" — the
derived instrumental stem came out near-silent (RMS ~0.0008) even when the song clearly had
instrumentation.

YuE, however, **generates its own vocal and instrumental tracks internally**
(`vtrack.wav` / `itrack.wav` in `yue_raw/vocoder/stems/`) before mixing them. These are the
model's *intended* stems and are dramatically better — the firelight instrumental went from
RMS 0.0008 (Demucs) to 0.072 (native), a ~90× improvement in audible content.

**Decision:** use YuE's native vocoder stems as the primary source
(`src/postprocessing/stem_separator.py::native_stem_paths`), trimmed/faded with
`normalize=False` to preserve the vocal/instrumental balance so they sum to the mix. Demucs
remains as an in-process fallback if native stems are missing; copying the full mix is the
last-resort fallback. The `stem_source` field in `metadata.json` records which path was used.

Demucs, when used, runs **in-process as a library** (`get_model('htdemucs')` +
`apply_model`), not as a CLI subprocess — see the hardware-modifications section for why.

---

## Hardware modifications made to YuE's inference path

The pipeline runs a patched copy of YuE's `inference/infer.py`, generated at runtime by
`src/models/yue_pipeline.py::_build_patched_infer()` via string replacement (the original
repo is left untouched). The static snapshot is `src/models/yue_infer_patched.py`. Each
fix lives in both places so the regenerated script always carries it.

Patches applied, and why:

1. **`flash_attention_2` → `sdpa`.** Flash-attn has no pre-built wheel for CUDA 13 +
   PyTorch 2.11 on this machine; SDPA is PyTorch-native and needs no install.
2. **4-bit quantization on stage-1.** `BitsAndBytesConfig(load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.bfloat16)`. The 7B model in bfloat16 is ~14 GB; 4-bit
   brings it under 8 GB. Stage-2 (1 B) stays in bfloat16 — quantizing it would add codec
   error for no memory benefit.
3. **Removed `model.to(device)` for stage-1.** 4-bit models are placed by
   `device_map="auto"`; calling `.to()` on them raises `ValueError`.
4. **`run_n_segments` off-by-one.** Original caps the loop at `len(lyrics)`; with a single
   `[chorus]` section that capped to 1 and the only real segment (index 1) never ran,
   leaving `raw_output` undefined. Changed cap to `len(prompt_texts)` (= `len(lyrics)+1`).
5. **Lyrics need a section tag.** `to_lyrics_txt()` wraps the hook in `[chorus]` so YuE's
   `split_lyrics()` regex `r"\[(\w+)\]..."` finds a segment (`src/service/schemas.py`).
6. **Stage-1 output filename truncation.** YuE builds the `.npy` name from the full genre
   string (~150 chars), exceeding the 255-char filename limit. Truncated to 40 chars.
7. **torchaudio 2.11 → soundfile.** torchaudio 2.11 defaults to a `torchcodec` backend that
   is not installed, so every `torchaudio.save`/`.load` raises `ImportError`. A global
   monkey-patch routes both through `soundfile`, coercing `.mp3`→`.wav` and handling the
   (channels, frames)↔(frames, channels) layout difference. Injected after `import
   torchaudio` so it covers YuE's imported helpers (`vocoder.py`, `post_process_audio.py`).
8. **Demucs in-process.** The original called Demucs as a `python3 -m demucs` subprocess,
   which (a) resolved to the system interpreter without demucs and (b) hit the same
   torchcodec wall in Demucs' own save path (a separate process the monkey-patch can't
   reach). Rewrote `separate_stems` to apply the model in-process and save via soundfile.

---

## What was weak or failed

- **Generation speed.** ~10 min per 20s song on RTX 4060 8GB (measured: 634s for task_01).
  Reference H800 hardware does this in ~2.5 min. Hardware-bound.
- **Generation is seed-sensitive.** At seed 42, the firelight prompt collapsed YuE's
  instrumental track to silence (itrack RMS 0.0002), while neonrain produced a full
  instrumental at the same seed. Re-rolling firelight to seed 7 gave it a real instrumental
  (itrack RMS 0.072). This is inherent to sampled generation — there is no single seed that
  is best for every prompt. The pipeline exposes `--seed` so a bad trajectory can be escaped;
  a production system would generate a few candidates and auto-select the one with the most
  balanced stems.
- **Token budget vs. duration.** 1500 tokens produced only ~15s of audio (the rest was
  silence padding); raised to 2400 for >20s of real content. Pushing much higher risks OOM
  on 8 GB.
- **4-bit precision loss.** Stage-1 quantization trades some fidelity for fitting in 8 GB.
- **Tempo is a soft prompt.** The BPM hint conditions but does not enforce tempo at the
  token level.

---

## What I would do next to productionize for Boock

1. **Bigger GPU** (A10G/A100 24 GB+) to run stage-1 in bfloat16 — removes quant loss, cuts
   generation to ~2–5 min.
2. **Async job queue** (Celery/Redis or Vercel Queues). 10-min generation is too slow for a
   synchronous HTTP response — return a `job_id` and expose a status/poll endpoint.
3. **Warm model cache** across requests to avoid per-request cold load.
4. **Tempo enforcement** via `librosa.effects.time_stretch` post-generation.
5. **Native stems** — SongGeneration `--separate` on larger hardware, or YuE's planned
   native stemgen, to avoid post-separation artifacts.
6. **Output caching** keyed on a hash of the request params.
7. **API versioning** (`/v1/theme-song/render`) and a locked contract.
