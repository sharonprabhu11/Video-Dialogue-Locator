# AGENTS.md

Instructions for any AI coding agent working in this repository. This file
organizes decisions and constraints already established by the project
owner (see `DESIGN.md` and `prompt.txt`) — it does not introduce new
architectural decisions of its own.

## 1. Project Context

- The system resolves a video URL to the exact frame/timestamp where a
  specified target dialogue first appears, and extracts the dialogue text.
- Two localization pipelines: speech-to-text (ASR) for spoken dialogue, OCR
  for visually displayed on-screen text. ASR is primary (tried first); OCR
  is secondary/fallback, used when ASR finds no confident match.
- `DESIGN.md` is the authoritative architecture reference — component
  responsibilities, interfaces, data structures, and the reasoning behind
  each choice. Read it before touching pipeline code.
- `prompt.txt` is the authoritative record of the prompts and decisions
  that shaped this design.

## 2. Architecture Principles

- Modular, single-responsibility components, per `DESIGN.md` §1. No module
  reaches into another module's internals.
- `pipeline.locate_dialogue(...)` is the only entrypoint. The CLI (and any
  future UI) is a thin wrapper around it with no business logic of its own.
- Shared logic (e.g. text normalization/fuzzy matching) is implemented once
  and reused by both the ASR and OCR paths — not duplicated per pipeline.
- No VLM, database, queue, or cloud infrastructure unless an actual,
  demonstrated requirement forces it. "Might be useful" or "more robust in
  theory" is not sufficient justification on its own.
- The agent does not change the architecture in `DESIGN.md` unilaterally.
  If implementation reveals a real problem with the design, stop and raise
  it — do not silently diverge from the doc, and do not silently rewrite
  the doc to match code that was already written.

## 3. Module Responsibilities

Condensed from `DESIGN.md` §1–4 (see there for full interfaces/data types):

- **Acquisition** — URL → local video file + measured fps/duration/frame count.
- **AudioExtraction** — video → audio track for ASR.
- **TranscriptionEngine** — audio → timestamped transcript (segment + word level).
- **ASR DialogueMatcher** / **OCRMatcher** — transcript/frames → match candidates, using shared text-matching logic.
- **OnsetRefiner** / **VisualRefiner** — approximate match → precise onset time / exact frame.
- **FrameExtractor** — refined time → frame index + saved image, fps-aware.
- **Pipeline** — orchestrates the above in order, owns error handling and result assembly.
- **CLI** — argument parsing and output formatting only.

## 4. Testing Requirements

- Every non-trivial function has a unit test.
- Unit tests mock/isolate all external ML and I/O dependencies (ASR model,
  OCR engine, network access via yt-dlp, video decoding) — they must run
  offline and deterministically.
- Integration tests exercise real component interaction (actual
  ffmpeg/yt-dlp/ASR/OCR calls) against small local fixtures, kept separate
  from unit tests (`tests/` vs. `tests/integration/`).
- The full test suite is run after every implementation change, before the
  change is considered complete.
- Never weaken a test to make it pass: no loosening thresholds, deleting or
  skipping assertions, or mocking away the thing under test. If a test
  fails, fix the code — or raise the discrepancy with the user — rather
  than adjust the test to match broken behavior.

## 5. Error Handling

- A structured exception hierarchy rooted at one base error type, with one
  subtype per failure category (acquisition, transcription, etc.) — see
  `DESIGN.md` §11.
- The pipeline is the only layer that catches these at the boundary and
  converts them into a structured result (`status="error"`); it does not
  swallow errors silently elsewhere.
- No silent fallbacks that could produce a confidently wrong answer (e.g.
  guessing an fps when it can't be measured) — fail loudly instead.

## 6. Logging

- Python's standard `logging` module; one logger per module (e.g.
  `vdl.acquisition`).
- INFO for stage start/end and key metrics (duration, fps, candidate
  counts); DEBUG for verbose internals (per-window match scores).
- No `print()` except for final CLI-formatted output.

## 7. Development Workflow

- CLI is the interface for now; the core pipeline must remain usable
  without it (see §2, §9).
- Consult `DESIGN.md` before implementing a pipeline stage. If a task
  requires deviating from it, pause and confirm with the user rather than
  reinterpreting the design mid-implementation.
- Run the test suite after every change; don't hand back a change with
  failing or skipped tests.
- Record meaningful development prompts in `prompt.txt` only when the user
  explicitly asks, using the format already established there (prompt
  number, purpose, exact verbatim prompt, resulting decision/outcome).
  Never edit a previously recorded entry.
- Record engineering decisions in `DESIGN.md`, keeping it current as the
  design evolves — this is the separate decision record `prompt.txt`
  entries point back to.

## 8. Scope-Control Rules

- No hardcoded video URL, target dialogue text, fps, timestamps, or frame
  numbers anywhere in source — these are always inputs (CLI args/config),
  never literals, since the evaluator may substitute a different video or
  target text at any time.
- No new architecture (VLM, database, queue, cloud, frontend framework,
  etc.) without a demonstrated requirement — state the concrete requirement
  it satisfies before adding it.
- Do not invent requirements that aren't in `DESIGN.md`, `prompt.txt`, or
  explicit user instruction.

## 9. AI-Assisted Development Principles

- All AI-generated code must be reviewed and understood by the user before
  it's considered accepted — the user must be able to explain and defend
  any part of it.
- The agent does not make architectural decisions unilaterally (§2, §7) —
  it implements decided architecture and surfaces tradeoffs or problems for
  the user to decide.
- Meaningful prompts go in `prompt.txt`; meaningful decisions go in
  `DESIGN.md` — both only as the design actually evolves, not speculatively
  ahead of it.

## 10. Definition of Done

A change is done when:

- It matches the current `DESIGN.md` (or `DESIGN.md` was explicitly updated
  first, with user approval, to reflect an approved change).
- Unit tests exist for new non-trivial logic, with external dependencies
  mocked.
- Integration tests are added/updated where real component interaction
  changed.
- The full test suite passes.
- No hardcoded URL/text/fps/timestamp/frame-number values were introduced.
- Logging and structured error handling are in place for new failure paths.
- The user has reviewed and understands the resulting code.
