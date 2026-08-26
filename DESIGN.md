# Design: Video Dialogue Locator

## Scope

The problem statement itself describes a visual event ("an **on-screen**
dialogue appears," "the exact video frame in which the dialogue first
appears") — so on-screen text (OCR) is the phenomenon the problem is
literally describing, and must remain a real part of the system, not just a
mentioned possibility. Two things are also true at the same time:

- The supplied sample video does **not** carry the target dialogue as
  on-screen text — it's spoken (confirmed by the user having watched it;
  not independently verified by frame inspection in this session, since
  that download was skipped — see `prompt.txt`). ASR is the only pipeline
  that resolves this specific input.
- The evaluator may substitute a different video (stated explicitly in the
  problem statement), which could rely on visible text instead of speech.
  A system that only handles the sample input is solving the wrong,
  narrower problem.
- On-screen text extraction is also just less reliable than a direct
  transcript — rendering lag, low contrast, misreads — which is exactly
  why it stays secondary rather than being run with equal priority.

So: **ASR is the primary pipeline (source of truth when it finds a
confident match); OCR is a secondary/fallback pipeline**, used when ASR
doesn't find the phrase spoken. This matches the original Prompt #2
decision in `prompt.txt`; a later message had dropped OCR entirely, and
this reinstates it. No VLM, database, frontend, queue, or cloud
infrastructure is introduced — none of this requires them.

**Why fallback (sequential), not parallel (always run both):** ASR is one
linear pass over the audio track only — cost scales with audio duration,
not video resolution or frame count. The OCR pipeline's shot-boundary
detection, by contrast, requires decoding the video stream frame-by-frame
to compute scene-change diffs — a full sequential video decode, before any
OCR is even attempted. Under fallback, total cost = ASR cost, plus OCR cost
*only if ASR misses*. Under parallel, total cost = ASR cost + OCR cost on
*every* run, regardless of whether ASR alone would have answered correctly.
Fallback's total cost is a strict subset of parallel's — it is never more
expensive, and whenever ASR alone succeeds (the expected common case for
spoken dialogue, and what happens on the supplied sample video), fallback
pays nothing for the video-decode pass, shot detection, or OCR calls that
parallel would pay unconditionally every time. Parallel's only edge is
wall-clock latency if concurrency hardware is available — but that requires
added complexity (concurrency handling, plus a fusion step to arbitrate
disagreeing results) that isn't justified by any stated latency/throughput
requirement in the problem statement.

**Acquisition format selection:** the video is fetched with yt-dlp's
`wv*[height>=240]+wa/w[height>=240]/bv*+ba/b` selector — a low-but-
verified-quality stream (>=240p) when the source reports resolution
metadata, falling back to **best** quality (not worst) when it doesn't.
ASR, the primary pipeline, needs no video quality at all, and frame
extraction only ever needs one still frame, so downloading less is
attractive — but this default went through a real correctness regression
first, worth recording: an earlier version used a blind `wv*+wa/w` ("worst
video + worst audio, or the single worst combined format"), reasoning that
any low-quality tier would do. Run against the real target video, it
resolved to ok.ru's `mobile` format — a tier with unverifiable
audio/resolution metadata, one step below the tier actually validated —
and it measurably degraded transcription accuracy: "rebels at" was misheard
as "verbels its" (see `prompt.txt`). The fix is to only take the cheap
path when a quality floor can be confirmed (height is broadly-reported,
standard metadata, unlike this source's audio bitrate fields), and to fall
back to the known-safe best-quality selector — never to an unverified
"worst" — when it can't be. Verified against the real target: this
resolves to a real 320×240/373kbps tier rather than gambling on whatever
the site's internal ranking considers "worst". Deliberately not
`bestaudio` alone: a source offering separate audio-only streams would then
yield a file with no video track at all, breaking frame extraction, which
always has to run regardless of which pipeline resolves the match. This is
configurable (`--video-format`) for a caller who wants a higher-quality
output frame image or better OCR-fallback fidelity at the cost of download
time.

## Pipeline

```
Video URL
   │
   ▼
Acquisition  (resolve URL → local file; measure fps, duration, frame_count)
   │
   ├──► Audio extraction ──► Timestamped ASR ──► ASR dialogue matching
   │                                                    │
   │                                          confident match?
   │                                         yes │           │ no
   │                                             ▼           ▼
   │                                    Onset refinement   Shot-boundary keyframe
   │                                    (ASR timestamp,    sampling (ffmpeg scene
   │                                     optional VAD       detection, one frame
   │                                     snap)              per shot, full duration)
   │                                             │                   │
   │                                             │                   ▼
   │                                             │          Cascade prefilter (cheap
   │                                             │          text-region check) → OCR
   │                                             │          + text matching on survivors
   │                                             │                   │
   │                                             │           shot flagged as candidate?
   │                                             │          yes │           │ no
   │                                             │               ▼           ▼
   │                                             │   Linear scan of the      status =
   │                                             │   flagged shot's frames   "not_found"
   │                                             │   (small window — cheap,   (both
   │                                             │    robust to one noisy     pipelines
   │                                             │    OCR read)               attempted)
   │                                             │           │
   └─────────────────────────────────────────────┴───────────┘
                              │
                              ▼
                     Frame extraction (fps-aware)
                              │
                              ▼
              Structured result (source="asr"|"ocr", full candidate trace)
```

---

## 1. Component responsibilities

| Component | Responsibility | Does NOT do |
|---|---|---|
| **Acquisition** | Resolve a URL to a decodable local video file; determine real fps, duration, frame count, VFR flag | Audio/visual analysis |
| **AudioExtraction** | Pull a clean mono audio track for ASR | Interpretation of content |
| **TranscriptionEngine** | Run ASR once, produce a timestamped transcript (segment + word level) | Matching |
| **ASR DialogueMatcher** | Search the transcript for the target phrase using shared text-matching logic | Frame/video I/O |
| **OnsetRefiner** | Tighten an ASR match's approximate onset time | Deciding which candidate is correct |
| **VisualSampler** | Detect shot/scene boundaries (ffmpeg scene-change filter) and extract one keyframe per shot — targets actual visual structure instead of guessing a time interval | Text reading |
| **OCRMatcher** | Cheap text-region prefilter (edge-density/MSER) to reject frames with no plausible text before spending OCR on them; OCR + match survivors using the same shared text-matching logic | Sampling, frame math |
| **VisualRefiner** | Linear scan of the (already small, shot-bounded) candidate window to find the first frame where text is legible/matches | Deciding which candidate is correct |
| **FrameExtractor** | Convert a refined time/frame into a concrete frame index + image, fps-aware | Any text/audio logic |
| **Pipeline** | Orchestrate: try ASR, fall back to OCR, assemble result, own error handling | Its own I/O — delegates everything |
| **CLI** | Parse args, call the pipeline, format output | Any business logic |

## 2. Project structure

```
video-dialogue-locator/
├── prompt.txt
├── DESIGN.md
├── README.md
├── pyproject.toml
├── src/vdl/
│   ├── cli.py               # argument parsing only
│   ├── pipeline.py           # orchestration: ASR first, OCR fallback, error boundary
│   ├── acquisition.py
│   ├── audio.py
│   ├── transcription.py
│   ├── text_match.py         # SHARED normalization + fuzzy scoring (used by ASR and OCR paths)
│   ├── matching.py            # ASR: Transcript + text_match -> MatchCandidate[]
│   ├── refinement.py           # ASR onset refinement (baseline + optional VAD snap)
│   ├── ocr.py                   # coarse-to-fine visual sampling + OCR + text_match -> OCRCandidate[]
│   ├── visual_refinement.py    # frame-boundary bisection on OCR legibility
│   ├── frames.py
│   ├── models.py                # all shared dataclasses
│   ├── config.py                 # thresholds/model names in one place
│   └── errors.py
├── tests/
│   ├── test_matching.py          # synthetic transcripts, no network
│   ├── test_ocr_matching.py       # synthetic frame-text sequences, no network
│   ├── test_refinement.py
│   ├── test_frames.py              # tiny local fixture clip
│   └── fixtures/
└── outputs/                          # frame images + result JSON per run
```

`text_match.py` exists specifically so the normalization/fuzzy-matching rules in §6 are written once and used identically by both pipelines — the ASR and OCR paths differ in *where the text comes from*, not in how it's compared to the target phrase.

## 3. Interfaces between modules

```python
def acquire_video(url: str, workdir: Path) -> AcquiredVideo: ...
def extract_audio(video: AcquiredVideo, workdir: Path) -> AudioAsset: ...
def transcribe(audio: AudioAsset, model_name: str) -> Transcript: ...

def find_dialogue(transcript: Transcript, target_text: str, cfg: MatchConfig) -> list[MatchCandidate]: ...
def refine_onset(candidate: MatchCandidate, audio: AudioAsset, cfg: RefineConfig) -> RefinedTime: ...

def sample_candidate_windows(video: AcquiredVideo, cfg: OCRConfig) -> list[TimeWindow]: ...   # shot-boundary keyframes, not fixed intervals
def has_probable_text_region(frame: np.ndarray) -> bool: ...                                    # cheap prefilter, runs before OCR
def find_onscreen_dialogue(video: AcquiredVideo, windows: list[TimeWindow], target_text: str, cfg: OCRConfig) -> list[OCRCandidate]: ...
def refine_frame_boundary(candidate: OCRCandidate, video: AcquiredVideo, cfg: RefineConfig) -> RefinedTime: ...   # linear scan, not bisection

def extract_frame(video: AcquiredVideo, refined: RefinedTime) -> FrameResult: ...

def locate_dialogue(url: str, target_text: str, cfg: PipelineConfig) -> PipelineResult: ...
```

`locate_dialogue` is still the single library entrypoint; it decides whether the ASR or OCR branch runs, nothing above it needs to know that.

## 4. Data structures

```python
@dataclass
class AcquiredVideo:
    source_url: str
    local_path: Path
    duration_s: float
    fps: float                # measured, never assumed
    frame_count: int | None
    is_vfr: bool

@dataclass
class AudioAsset:
    path: Path
    sample_rate: int
    duration_s: float

@dataclass
class Word:
    text: str
    start_s: float
    end_s: float
    confidence: float | None

@dataclass
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str
    words: list[Word]

@dataclass
class Transcript:
    segments: list[TranscriptSegment]
    language: str
    model_name: str
    def words(self) -> list[Word]: ...   # flattened, segment-boundary-agnostic view

@dataclass
class MatchCandidate:                     # ASR path
    matched_text: str
    score: float
    start_s: float
    end_s: float
    word_span: tuple[int, int]

@dataclass
class TimeWindow:
    start_s: float
    end_s: float

@dataclass
class OCRCandidate:                        # OCR path
    matched_text: str
    score: float
    window: TimeWindow                      # coarse window to refine within

@dataclass
class RefinedTime:
    onset_s: float
    method: Literal["asr_word_timestamp", "vad_snap", "ocr_window_scan"]
    confidence: float

@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    image_path: Path

@dataclass
class PipelineResult:
    status: Literal["ok", "ambiguous", "not_found", "error"]
    source: Literal["asr", "ocr", "none"]
    text: str | None
    timestamp: str | None
    frame_index: int | None
    image_path: Path | None
    confidence: float | None
    asr_candidates: list[MatchCandidate]
    ocr_candidates: list[OCRCandidate]
    warnings: list[str]
    error: str | None
```

## 5. Timestamped transcription representation

Unchanged from the ASR-only draft: segment + word level from a single ASR pass, with `Transcript.words()` giving a flattened, segment-boundary-agnostic view that matching always operates on. Degraded mode (segment-level only) is logged as a warning and lowers confidence rather than being silently treated as equal precision.

## 6. Dialogue matching robustness

The normalization and scoring rules are defined **once**, in `text_match.py`, and applied identically regardless of source:

- **Capitalization** — both sides lowercased before comparison.
- **Punctuation** — stripped from both sides; original text kept for reporting.
- **Whitespace** — collapsed/stripped on both sides.
- **Transcription/OCR errors** — never exact string equality; fuzzy similarity score against a configurable threshold.
- **Split across segments (ASR)** — solved structurally by matching over the flattened word list rather than per-segment text.
- **Multiple occurrences** — the full transcript (ASR) or full set of sampled windows (OCR) is scanned; every candidate above threshold is kept, not just the first. One dominant candidate → `status="ok"`; several close-scoring ones → `status="ambiguous"`, all returned.

OCR-specific additions on top of the shared logic:
- OCR misreads are noisier than ASR misrecognitions (font, contrast, motion blur), so the OCR match threshold is configured more leniently than the ASR one.
- A single keyframe testing positive is enough to flag its shot as a candidate — each keyframe represents an entire shot (typically several seconds), and a real text/dialogue card is very likely to be its own shot, so it's expected to register on that one keyframe rather than needing repetition across samples. The actual noise-filtering happens one stage later: within the flagged shot's window, the linear scan (§9) requires the match to hold on at least two consecutive *real frames* before accepting it, which is what actually filters out a one-off OCR misread.

## 7–8. Timestamp/frame conversion and FPS handling

Unchanged: `frame_index = floor((onset_s - stream_start_offset) * fps)`, clamped to valid range, with `fps`/offset measured per-video via `ffprobe`, never assumed. VFR is detected by comparing `r_frame_rate` vs `avg_frame_rate`; when detected, extraction seeks by timestamp and reads back whichever frame the decoder presents, rather than trusting linear arithmetic. The OCR path already works in frame/sample space, so for it this is used in the reverse direction — frame → timestamp — for reporting.

## 9. Localization and temporal refinement

This section supersedes the original "binary search across all frames" idea from Prompt #2. That idea bundled two different problems — *where in a possibly hour-long video is the text at all* (no prior, unstructured) and *what is the exact first frame within a known few-second window* (structured, monotonic) — and binary search only ever validly applies to the second one. Splitting them out:

**Global localization (replaces fixed-interval coarse sampling):**
- **Shot-boundary / keyframe sampling.** `ffmpeg`'s built-in scene-change filter (`select='gt(scene,X)'`) segments the video into shots; one keyframe is decoded per shot. This is not a search-algorithm optimization, it's a better sampling strategy: a text/dialogue card is almost always its own static shot, so this samples where the actual structure of the video suggests content changes, instead of a blind fixed time interval that might straddle or miss a short card entirely.
- **Cascade prefilter.** Before running full OCR on a keyframe, a cheap "does this frame plausibly contain a text region" check (edge-density / MSER via OpenCV) rejects the vast majority of shots with no chance of containing text. Full OCR — the expensive step — only runs on frames that pass this filter. Same principle as a Viola-Jones cascade: reject cheap negatives fast, spend the expensive step only on plausible positives.
- Together, these attack the actual bottleneck (frame count × OCR cost across a potentially 50,000+ frame video) far more directly than any search-order cleverness over a fixed sampling grid would.

**Local boundary refinement, per pipeline:**
- **ASR path**: baseline is the matched word's `start_s` directly (`method="asr_word_timestamp"`). An optional local audio-energy/VAD onset snap can tighten this further within a short window around the reported onset — still proposed, not required, since ASR word timestamps are usually precise enough on their own; flag if you want it built.
- **OCR path**: once a shot is flagged as a candidate, its window is already small (typically 50–150 frames, since shots are short). Rather than bisecting this window (`method="ocr_window_scan"` — deliberately *not* binary search):
  - **Why not binary search here**: it needs a strictly monotonic "text present" predicate across the window, but OCR is a noisy oracle — a single misread at the midpoint sends the search into the wrong half with no way to recover, unlike the two-consecutive-frame tolerance used elsewhere in this design.
  - **Why linear scan is the better fit**: the window is already so small (post shot-detection) that scanning every frame in it costs a negligible amount more than the ~7 OCR calls binary search would use, and it's completely robust to a one-off misread — it just looks for the first point where the match holds on two consecutive frames.
  - **Noted but not adopted**: *noisy binary search* (Feige–Raghavan–Peleg–Upfal 1994; Karp–Kleinberg) is the algorithm actually designed for threshold-finding under an unreliable oracle — it queries adaptively against a belief distribution instead of trusting one comparison. It's the theoretically "correct" tool for a noisy monotonic search, but it's built for search spaces much larger than a 50–150 frame window; adopting it here would add real complexity to optimize a stage that's already cheap.

## 10. Representing uncertainty and failure

`PipelineResult.status`:
- `"ok"` — one pipeline produced a clearly dominant candidate.
- `"ambiguous"` — multiple candidates within a small score margin, either within one pipeline or (rarer) both pipelines returning conflicting confident hits at different times — all are returned, none silently chosen.
- `"not_found"` — neither ASR nor OCR found a confident match anywhere.
- `"error"` — an upstream stage failed.

`source` records which pipeline actually resolved the answer. `asr_candidates` and `ocr_candidates` are both always populated with whatever was considered (empty list if that pipeline wasn't triggered), so the result is fully auditable regardless of which path won.

## 11. Logging and error handling

Unchanged in structure: module-level loggers per stage, `VDLError` exception hierarchy, pipeline is the only boundary that converts exceptions into `PipelineResult(status="error", ...)`. No silent fallbacks (e.g. no guessed fps, no defaulting to "probably no on-screen text" without the OCR pass actually running when ASR misses).

## 12. CLI interface

```
vdl locate --url <VIDEO_URL> --text "My mind rebels at stagnation" \
           [--out-dir outputs/] [--asr-model small] \
           [--match-threshold 0.8] [--ocr-threshold 0.65] \
           [--ocr-sample-interval 5.0] [--no-ocr] [--no-refine] [--json]
```

Default behavior: run ASR first; if it doesn't produce a confident match, automatically fall back to the OCR pipeline. `--no-ocr` disables the fallback entirely (e.g. for a known audio-only input, to save time). Output format, exit codes unchanged from the ASR-only draft, with `source` now included in both the printed and JSON output.

## 13. Independence from a future UI

Unchanged: `pipeline.locate_dialogue(url, text, config) -> PipelineResult` is the only entrypoint; `cli.py` is a thin formatter around it. Adding the OCR branch didn't require touching this boundary — it's entirely internal to the pipeline's orchestration logic.
