# Video Dialogue Locator

Given a video URL and a target line of dialogue, finds the exact frame and
timestamp where that dialogue first appears, and extracts the frame as an
image. Speech-to-text (ASR) is the primary pipeline; OCR is a secondary
fallback for dialogue that's shown as on-screen text rather than spoken.

See `DESIGN.md` for the full architecture and the reasoning behind every
design choice, `AGENTS.md` for the engineering rules this codebase follows,
and `prompt.txt` for the record of prompts/decisions that shaped it.

## Prerequisites

These must be installed and on `PATH` (not bundled in this repo, not
installed by `pip install`):

| Tool | Required? | Why |
|---|---|---|
| Python 3.10+ | Yes | Runtime |
| `git` | Yes | To clone this repo |
| `ffmpeg` / `ffprobe` | Yes | Video/audio decoding, frame extraction, shot detection. The ASR pipeline cannot run at all without it — acquisition fails immediately with `FileNotFoundError: ffprobe` |
| `tesseract-ocr` | No | Only needed for the OCR fallback to actually run. Its absence doesn't break the ASR pipeline; OCR calls simply error if triggered without it installed |

### Installing the prerequisites

**Ubuntu / Debian**

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv git ffmpeg tesseract-ocr
```

**macOS** (via [Homebrew](https://brew.sh))

```bash
brew install python git ffmpeg tesseract
```

**Windows**

Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with an
Ubuntu distro and follow the Ubuntu/Debian instructions above — this
project has not been tested on native Windows. Native alternative: install
[Python 3.10+](https://www.python.org/downloads/), [Git](https://git-scm.com/download/win),
and [ffmpeg](https://ffmpeg.org/download.html#build-windows) (add its `bin/`
folder to `PATH`); Tesseract is optional (installer at
[UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)).

**Verify before continuing:**

```bash
python3 --version   # 3.10 or higher
ffmpeg -version      # any recent version
ffprobe -version     # ships alongside ffmpeg
tesseract --version   # optional — only if you installed it
```

If `ffmpeg`/`ffprobe` print "command not found" after installing, open a
new shell (or re-source your profile) so the updated `PATH` takes effect.

## Setup

```bash
git clone https://github.com/sharonprabhu11/Video-Dialogue-Locator.git
cd Video-Dialogue-Locator

python3 -m venv .venv
source .venv/bin/activate      # Windows (native): .venv\Scripts\activate

pip install -e ".[dev]"
```

**Verify the install:**

```bash
vdl locate --help
```

You should see the full argument list. If `vdl: command not found`, make
sure the virtualenv is activated (`source .venv/bin/activate`) — the `vdl`
entry point only exists inside it.

## Usage

```bash
vdl locate --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation"
```

The first run against a given URL downloads the video, which — depending
on length/connection — can take anywhere from under a minute to tens of
minutes; transcription (ASR) then runs on CPU by default and is the other
significant chunk of wall-clock time (see **Performance** below). Re-runs
against the *same* URL are much faster: the download is skipped entirely
via the on-disk cache (see `--cache-dir` below).

Output:

```
Timestamp : HH:MM:SS.sss
Frame     : <frame number>
Text      : "My mind rebels at stagnation"
Source    : asr | ocr
Image     : outputs/frame_<n>.png
```

Exit codes: `0` = confident answer found, `1` = ran fine but no confident
answer (ambiguous or not found), `2` = error.

### Flags

All optional — see `vdl locate --help` for the authoritative list.

| Flag | Default | Purpose |
|---|---|---|
| `--out-dir DIR` | `outputs` | Where to write the extracted frame image |
| `--asr-model NAME` | `small` | faster-whisper model size (`tiny`, `base`, `small`, ...) — smaller is faster but less accurate |
| `--asr-device {cpu,cuda}` | `cpu` | Run ASR on GPU if one is available (see **Performance**) |
| `--match-threshold` | `0.80` | ASR fuzzy-match sensitivity, 0..1 |
| `--ocr-threshold` | `0.65` | OCR fuzzy-match sensitivity, 0..1 (more lenient — OCR misreads are noisier) |
| `--scene-threshold` | `0.4` | ffmpeg scene-change sensitivity for OCR frame sampling, 0..1 |
| `--video-format SELECTOR` | low-but-verified quality | yt-dlp format selector; raise for a higher-quality output frame image |
| `--no-ocr` | off | Disable the OCR fallback entirely |
| `--vad-snap` | off | Enable the optional audio-onset precision pass (`DESIGN.md` §9) |
| `--cache-dir DIR` | `.vdl_cache` | Persist downloaded videos here, keyed by URL+format, so re-running against the same URL skips the download |
| `--no-cache` | off | Disable download caching; always fetch fresh |
| `--concurrent-fragments N` | `8` | yt-dlp fragment download concurrency (`-N`); speeds up fragmented (HLS/DASH) sources |
| `-v`, `--verbose` | off | DEBUG-level logging |
| `--json` | off | Also print the full structured result as JSON |

## Performance

Two independent bottlenecks, worth telling apart when something feels slow:

- **Download (network-bound):** determined by the source's connection
  speed and, for HLS/DASH sources, fragment concurrency (`--concurrent-fragments`).
  A GPU has no effect here. The `.vdl_cache/` directory makes this a
  one-time cost per URL — see `--cache-dir` above.
- **ASR transcription (compute-bound):** runs on CPU by default using
  `int8` quantization (already the fastest correct CPU precision for this
  model). A CUDA-capable GPU (`--asr-device cuda`) is substantially faster
  here, since faster-whisper's underlying engine (CTranslate2) is
  optimized for GPU batch inference and additionally switches to
  `float16` on GPU. If your machine has no NVIDIA GPU, `notebooks/gpu_asr_benchmark.ipynb`
  runs the same transcription step on a free Colab GPU so you can see the
  CPU-vs-GPU difference without owning one.

## Troubleshooting

- **`FileNotFoundError: ffprobe`** — `ffmpeg`/`ffprobe` isn't installed or
  isn't on `PATH`. See **Prerequisites** above.
- **yt-dlp fails with `Connection reset by peer`** — a transient failure on
  the source site's end (observed intermittently against `ok.ru` during
  development), not a bug in this pipeline. Plain retries typically
  succeed within a few attempts; `vdl` doesn't currently retry
  automatically, so just re-run the command.
- **OCR fallback errors** — `tesseract-ocr` isn't installed. Either install
  it (see **Prerequisites**) or pass `--no-ocr` if you only need the ASR
  path.
- **Stuck/slow with no visible progress** — the process is almost always
  still working: yt-dlp prints per-fragment progress that gets buffered
  when not attached to a terminal (e.g. piped through `tee`/`tail`), and
  ASR transcription gives no progress output at all until it finishes. Use
  `-v` for more logging, or check CPU usage (`ps aux | grep vdl`) — high
  CPU means it's transcribing, not hung.
- **A re-run re-downloads instead of hitting the cache** — the cache key is
  `sha256(url + video_format)`; a different `--video-format` (including
  the default changing between versions) is a different cache entry by
  design, not a bug.

## Testing

```bash
python -m pytest tests/ --ignore=tests/integration   # unit tests: fast, offline, all external deps mocked
python -m pytest tests/integration/                   # integration tests: real ffmpeg/tesseract/ASR, requires prerequisites above
```

Integration tests that need a binary not installed on the machine are
skipped (not failed), with the reason printed. One ASR integration test
(`test_real_asr_transcribe_end_to_end`) is skipped unconditionally because
it downloads a real model from the network — run it manually if you want
to exercise that path.

## Project layout

```
src/vdl/                        pipeline package (see DESIGN.md section 2 for the full breakdown)
tests/                          unit tests, one file per module, all external deps mocked
tests/integration/               real-component tests against a locally-generated fixture video
notebooks/                      standalone Colab notebooks (dev tooling, not part of the shipped pipeline)
outputs/                        frame images written by CLI runs
.vdl_cache/                     downloaded source videos, cached by url+format (gitignored)
```
