# Video Dialogue Locator

Given a video URL and a target line of dialogue, finds the exact frame and
timestamp where that dialogue first appears, and extracts the frame as an
image. Speech-to-text (ASR) is the primary pipeline; OCR is a secondary
fallback for dialogue that's shown as on-screen text rather than spoken.

See `DESIGN.md` for the full architecture and the reasoning behind every
design choice, `AGENTS.md` for the engineering rules this codebase follows,
and `prompt.txt` for the record of prompts/decisions that shaped it.

## Prerequisites

These must be installed and on `PATH` (not bundled in this repo):

- Python 3.10+
- `ffmpeg` / `ffprobe` — video/audio decoding, frame extraction, shot detection
- `tesseract-ocr` — only needed for the OCR fallback pipeline to actually run
  (its absence doesn't break the ASR pipeline; OCR calls will simply error
  if triggered without it installed)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
vdl locate --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation"
```

Output:

```
Timestamp : HH:MM:SS.sss
Frame     : <frame number>
Text      : "My mind rebels at stagnation"
Source    : asr | ocr
Image     : outputs/frame_<n>.png
```

Useful flags (all optional — see `vdl locate --help` for the full list):

- `--out-dir DIR` — where to write the extracted frame image (default `outputs/`)
- `--asr-model NAME` — faster-whisper model size, e.g. `tiny`, `base`, `small` (default `small`)
- `--match-threshold`, `--ocr-threshold` — fuzzy-match sensitivity for each pipeline
- `--no-ocr` — disable the OCR fallback entirely
- `--vad-snap` — enable the optional audio-onset precision pass (see `DESIGN.md` §9)
- `--json` — also print the full structured result as JSON

Exit codes: `0` = confident answer found, `1` = ran fine but no confident
answer (ambiguous or not found), `2` = error.

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
src/vdl/            pipeline package (see DESIGN.md section 2 for the full breakdown)
tests/               unit tests, one file per module, all external deps mocked
tests/integration/   real-component tests against a locally-generated fixture video
outputs/             frame images written by CLI runs
```
