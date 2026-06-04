# AI_USAGE.md

## Tools used

- **Claude Code** (Anthropic) — primary coding and debugging assistant throughout, using
  Claude Sonnet and Claude Opus.

---

## Representative prompts / tasks given to Claude Code

1. **"Read the full project structure and all files, then summarize what's built, what's
   missing, and any bugs you can spot without running the code."**
   Claude read every source file, the provided inputs, and the YuE inference script, then
   produced a prioritized report. It caught three crash-class bugs by static reading alone:
   missing `[chorus]` section tag, `model.to(device)` on a 4-bit model, and a WAV finder
   pointed at the wrong directory.

2. **"Fix the 3 critical bugs. Show me the diff for each before applying, and wait for
   approval after each."**
   Claude made surgical, reviewed edits to `src/service/schemas.py` and
   `src/models/yue_pipeline.py` / `yue_infer_patched.py`, presenting each diff for sign-off
   before moving on.

3. **"Create requirements.txt from the working venv, a FastAPI service, an output
   validator, and a test suite."**
   Claude inspected the actual installed package versions, wrote `requirements.txt`,
   `src/api.py` (with `/health` + `/theme-song/render` and a clean 422 error model),
   `src/validate.py`, and `tests/test_validation.py` — bringing the suite to 35 passing.

4. **"Run a test inference on task_01 and show me the output."**
   This kicked off an iterative debugging loop. Each run surfaced the next failure deeper in
   YuE's pipeline; Claude diagnosed each from the traceback, patched both the static script
   and the runtime patch-generator, and re-ran.

5. **"Make it work this time — don't fix one bug at a time."**
   Instead of patching the current crash, Claude traced the *entire* remaining execution
   path, enumerated all six `torchaudio.save`/`.load` call sites across YuE's imported
   modules, confirmed none used a patch-defeating `from torchaudio import save`, and
   installed a single global compatibility shim — ending the cycle.

6. **"Make the songs sound as good as possible — use your own judgment."**
   Claude diagnosed *why* a stem was empty by measuring YuE's internal tracks (not just the
   final output), discovered the bottleneck was Demucs misclassifying synthetic audio,
   and switched the pipeline to YuE's native stems — a ~90× improvement in the instrumental
   (RMS 0.0008 → 0.072). It also raised the token budget (15s→20s of real audio) and added a
   `--seed` lever after proving the empty instrumental was a seed-trajectory collapse.

7. **"Write the three docs to match the working pipeline."**
   DECISIONS.md, README.md, and AI_USAGE.md, written against the code as it actually runs.

---

## Where AI materially helped

- **Static bug detection.** Three crash-class bugs were identified by reading the code
  before a single run — including the subtle `run_n_segments` interaction that only bites
  with a single lyric section.
- **Deep traceback diagnosis.** Across ten distinct failures in YuE's two-stage pipeline,
  Claude read each traceback, identified root cause (not just symptom), and located the fix
  in both the static snapshot and the runtime patch-generator so it wouldn't regress.
- **Knowing when to stop whacking moles.** The pivotal call was recognizing that
  per-call-site patching of `torchaudio.save` would never converge (the calls live in
  imported third-party modules), and replacing it with one module-level monkey-patch.
- **Environment archaeology.** Claude found that the system `python3` lacked torch/soundfile
  but a separate venv (`/home/nonu/PYTorch/venv`) had the full working stack, and that
  Demucs was failing because the subprocess used the wrong interpreter.

---

## One AI suggestion that was reviewed and verified (not taken on faith)

When the torchcodec backend problem appeared, the tempting one-line fix was
`pip install torchcodec`. Before doing that, we checked compatibility: this machine runs an
unusual `torch 2.11.0+cu130` build, and a prebuilt torchcodec wheel carries real ABI-mismatch
risk against a nightly/custom torch. Rather than add a fragile dependency, we chose the
self-contained path — route all audio I/O through `soundfile` (already installed and proven)
and run Demucs in-process. This was then **verified empirically**: the shim's save/load
round-trip and `.mp3`→`.wav` coercion were unit-tested standalone, and the in-process Demucs
was tested on the existing `full_mix.wav` to confirm it produced real, differing stems before
committing to a full 10-minute run.

---

## How correctness was verified

- **Unit tests:** `pytest tests/` — 35/35 passing (schemas, audio validation, trim-and-fade,
  formatter round-trips, output-set validation). Re-run after every change.
- **Pre-flight checks before long runs:** confirmed the runtime-generated patched script
  compiles (`py_compile`), that the shim is injected before any torchaudio usage, and that
  the shim round-trips audio correctly — so a 10-minute inference run was never spent
  discovering a syntax slip.
- **End-to-end runs on both provided inputs:** `task_01_firelight` and `task_02_neonrain`
  each completed with `"warnings": []` and passed `src/validate.py` (all four files present,
  duration 20000 ms, non-silent, non-clipping, valid metadata).
- **Manual audio-level inspection:** measured per-file RMS/peak to confirm the stems were
  genuinely separated (not silent, not copies of the mix) — which is also how the
  vocal-dominated nature of task_01's output was discovered and documented honestly rather
  than hidden.
