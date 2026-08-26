# Run Log

Chronological record of every `vdl locate` invocation (and the acquisition-
level diagnostics run alongside them) during this session, with what each
one actually returned and what it revealed. Complements `prompt.txt` (which
records prompts/decisions, not execution results) and `DESIGN.md` (which
records the resulting architecture decisions, not the raw run history).

## Environment discoveries (not runs themselves, but shaped everything below)

- **`ffmpeg`/`ffprobe` were not installed** at the start of this session —
  the ASR pipeline cannot run at all without them (Run 1 below is the
  direct proof). Installed via `sudo apt-get install -y ffmpeg` partway
  through the session.
- **`tesseract-ocr` was never installed** this session — the OCR fallback
  path was never exercised against a real downloaded video in this
  environment, only via mocked unit tests and one integration test that
  draws synthetic text (`test_real_ocr_reads_rendered_text`, itself
  skipped without tesseract).
- **No GPU on this machine** — `nvidia-smi` not found, no `torch`/CUDA.
  Confirmed before recommending anything GPU-related; led to building
  `notebooks/gpu_asr_benchmark.ipynb` for GPU testing on Colab instead.
- **The pre-existing `outputs/frame_7786.png`** (present before this
  session started) turned out to be a byproduct of the integration test
  suite's synthetic 192x144 fixture video, **not** evidence of a prior real
  pipeline run. Traced via file resolution (192x144 doesn't match any real
  downloaded format) and confirmed against `tests/integration/test_video_pipeline_integration.py`'s
  `synthetic_video` fixture. Nobody had actually run `vdl locate`
  successfully against a real video before this session.

## Runs

### Run 1 — FAILED: missing `ffprobe`

```
vdl locate --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation" --json
```

Downloaded the full video successfully via yt-dlp (proving acquisition
itself worked even before ffmpeg was installed), then crashed at the next
stage:

```
FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'
```

**Discovery:** confirmed `ffmpeg`/`ffprobe` genuinely absent, not a PATH
issue — nothing in the environment could stand in for it.

### Run 2 — SUCCESS: first successful real run

Same command, re-run after `ffmpeg` was installed.

- **Result:** `status=ok`, `source=asr`, timestamp `00:05:24.860`, frame
  `7788`, text `"My mind rebels at stagnation."`, confidence `1.0`.
- Took roughly 23 minutes end-to-end (fresh download + ASR, no cache yet).

**Discovery:** this ok.ru source is a **full ~54-60 minute video**, not a
short clip — confirmed later via ASR debug logs processing segments out to
`~54:10` even though the target line is at `5:24`. The pipeline transcribes
the whole file before matching, regardless of where the answer falls.

### Diagnostic — yt-dlp connectivity and real format ladder

`yt-dlp -F` against the ok.ru URL: first attempt failed with
`Connection reset by peer`; retried and succeeded, revealing the real
available formats: `mobile` (unverifiable), `hls-193` (192x144, ~75MB),
`hls-372` (320x240, ~145MB — the one selected by `DEFAULT_FORMAT_SELECTOR`),
`hls-749` (480x360, ~291MB), `hls-1222` (640x480, ~475MB), `hls-2565`
(960x720, ~997MB).

**Discovery:** ok.ru intermittently resets connections at the initial
webpage-fetch step, independent of format/size — confirmed this wasn't a
full IP block since plain `curl` reached the same URL fine at the same
time. Purely transient flakiness on their end.

*(Between here and Run 3: implemented URL-keyed download caching and
concurrent-fragment downloads in `acquisition.py`/`config.py`/`pipeline.py`/`cli.py`.)*

### Diagnostic — cache-hit verification, ok.ru (inconclusive, abandoned)

Direct `acquisition.acquire_video(..., cache_dir=Path(".vdl_cache"))` calls
(not the full CLI), attempting a clean cache-miss-then-hit proof against
ok.ru. Hit the same connection-reset flakiness repeatedly (multiple failed
attempts); abandoned in favor of a fresh URL rather than continuing to
retry blind.

### Run 3 — SUCCESS: fresh YouTube video

```
vdl locate --url "https://www.youtube.com/watch?v=MvXVDje91BE" --text "That's the structure of almost any story" --json
```

- **Result:** `status=ok`, `source=asr`, timestamp `00:01:30.000`, frame
  `2250`, text `"That's the structure of almost any story."`, confidence
  `1.0`.
- Completed much faster than the ok.ru runs (~5:10 total video length vs.
  ~54-60 minutes).

### Run 4 — SUCCESS: same YouTube URL, cache-hit proof

Identical command, re-run immediately after Run 3.

```
using cached download for https://www.youtube.com/watch?v=MvXVDje91BE: .vdl_cache/f9f67cc35ebe2f49/source.mp4
acquired video: fps=25.000 duration=310.80s frame_count=7770 vfr=False
```

**Discovery:** download+probe took **0.18 seconds** (vs. multi-minute on a
miss) — the caching feature's first clean, complete proof. The full re-run
still took ~1m39s wall-clock, but that was now 100% ASR transcription time.

### Run 5 — INTERRUPTED: ok.ru, stopped externally mid-run

```
vdl locate --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation" --json -v
```

Stopped externally before completion (`status: killed`) — but had already
finished downloading (populating `.vdl_cache/df82ebd20fb0ee3f/`) before
being interrupted, likely during ASR.

### Run 6 — SUCCESS: ok.ru re-run, cache hit — key evidence for the later architecture audit

Same command, re-run after Run 5's interruption. Cache hit on the download.

- **Result:** `status=ok`, `source=asr`, timestamp `00:05:24.520`, frame
  `7780`, text `"My mind rebels at stagnation"`, confidence `1.0`.
- **`asr_candidates` contained 6 entries** for what is genuinely one
  spoken occurrence:

  | score | start_s | text |
  |---|---|---|
  | 1.0 | 324.52 | My mind rebels at stagnation |
  | 0.943 | 325.38 | mind rebels at stagnation |
  | 0.918 | 320.96 | time My mind rebels at stagnation |
  | 0.918 | 324.52 | My mind rebels at stagnation Give |
  | 0.862 | 325.38 | mind rebels at stagnation Give |
  | 0.820 | 325.38 | mind rebels at stagnation Give me |

**Discovery (the important one):** the top-two scores (1.0 vs. 0.943) were
only **0.057 apart** — just 0.007 above the default 0.05 `ambiguous_margin`.
This one true occurrence was one narrowly-averted false `"ambiguous"` away
from being misreported as two. This exact table became the central,
concrete evidence for the architecture audit's two most important findings:
(1) candidate selection picked the highest-scoring window globally rather
than the chronologically first occurrence — a real violation of "finds
where dialogue **first** appears"; (2) window-size-delta duplicates
(`target_len±1`) weren't deduplicated before ambiguity was judged.

### Architecture audit and fixes (not a run — see `DESIGN.md` "Occurrence deduplication and first-occurrence selection")

Full audit of every module against the stated requirements; findings and
the four approved fixes (candidate deduplication by `word_span` overlap +
earliest-occurrence selection, OCR prefilter reuse in
`visual_refinement.py`, VFR test coverage) are recorded in `DESIGN.md`
§10 and §9. Validated with **95 passed, 2 skipped** across the full test
suite (unit + integration) before being considered done.

### Run 7 — SUCCESS: ok.ru, post-fix — dedup confirmed live in production

Same ok.ru command, re-run after the fixes above landed. Cache hit on the
download.

- **Result:** `status=ok`, `source=asr`, timestamp `00:05:24.800`, frame
  `7787`, text `"My mind rebels at stagnation"`, confidence `1.0`.
- **`asr_candidates` now contains exactly 1 entry** — the same 6
  near-duplicates from Run 6 correctly collapsed into one occurrence by
  `matching.dedupe_by_occurrence()`, with the highest-scoring reading
  (324.8s) selected as the representative onset.

**Discovery:** direct, real-world confirmation the fix works as designed —
not just the unit/integration tests, an actual production run against the
actual problematic input.

### Run 8 — SUCCESS, with a real accuracy regression: `--asr-model tiny`

```
vdl locate --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation" --asr-model tiny --json -v
```

- **Result:** `status=ok`, `source=asr`, timestamp `00:05:25.300`, frame
  `7799`, text `"My mind reveals its stagnation."` (not "rebels at"),
  confidence `0.897` (barely above the 0.80 threshold).
- Wall time: **2m36s**, vs. ~4.5-9 minutes with the default `small` model
  on the same ~54-minute video — roughly 2-3x faster.

**Discovery:** `tiny` genuinely mis-transcribed the target phrase
("rebels at" → "reveals its", phonetically close but wrong), surviving the
fuzzy-match threshold only narrowly. This is a live instance of the same
category of error already documented in this project's history
(`DESIGN.md`'s acquisition-format regression note references an earlier
"rebels at" → "verbels its" mis-transcription from a different cause) —
concrete, repeated evidence that model-size/speed tradeoffs here have a
real accuracy cost, not just a theoretical one.
