"""ASR transcription (DESIGN.md sections 1, 5).

Wraps faster-whisper behind a narrow function so the rest of the codebase
never imports the ASR library directly — unit tests mock transcribe()
itself (or _run_model below) rather than needing a real model loaded.
"""

from __future__ import annotations

import logging

from vdl.errors import TranscriptionError
from vdl.models import AudioAsset, Transcript, TranscriptSegment, Word

logger = logging.getLogger("vdl.transcription")


def transcribe(audio: AudioAsset, model_name: str, device: str = "cpu") -> Transcript:
    """Run ASR once over the full audio track, with word-level timestamps.

    Falls back to segment-level-only timing (word_level=False) if the
    backend doesn't return word timestamps for some segment — this is
    surfaced to the caller via Transcript.word_level rather than silently
    treated as equal precision (DESIGN.md section 5).
    """
    logger.info("transcribing %s with model=%s device=%s", audio.path, model_name, device)
    try:
        raw_segments = _run_model(audio, model_name, device)
    except Exception as exc:  # noqa: BLE001 - any backend failure is a TranscriptionError
        raise TranscriptionError(f"ASR failed on '{audio.path}': {exc}") from exc

    segments: list[TranscriptSegment] = []
    word_level = True
    for seg in raw_segments:
        words = [
            Word(text=w.word.strip(), start_s=w.start, end_s=w.end, confidence=getattr(w, "probability", None))
            for w in (seg.words or [])
        ]
        if not words:
            word_level = False
        segments.append(TranscriptSegment(start_s=seg.start, end_s=seg.end, text=seg.text.strip(), words=words))

    logger.info("transcription complete: %d segments, word_level=%s", len(segments), word_level)
    return Transcript(segments=segments, language="en", model_name=model_name, word_level=word_level)


def _run_model(audio: AudioAsset, model_name: str, device: str):
    """Thin seam over the faster-whisper library, isolated so tests can
    monkeypatch this single function instead of loading a real model.
    """
    from faster_whisper import WhisperModel  # imported lazily: heavy, ML-only dependency

    model = WhisperModel(model_name, device=device)
    segments, _info = model.transcribe(str(audio.path), word_timestamps=True)
    return list(segments)
