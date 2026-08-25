"""Shared data structures passed between pipeline stages.

See DESIGN.md section 4 for the rationale behind each field. Modules never
reach into each other's internals — they only exchange these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class AcquiredVideo:
    source_url: str
    local_path: Path
    duration_s: float
    fps: float  # measured from the decoded stream, never assumed
    frame_count: int | None
    is_vfr: bool
    start_offset_s: float = 0.0


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
    confidence: float | None = None


@dataclass
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    segments: list[TranscriptSegment]
    language: str
    model_name: str
    word_level: bool  # False when the ASR backend only gave segment-level timing

    def words(self) -> list[Word]:
        """Flattened, segment-boundary-agnostic view used by the matcher."""
        return [w for seg in self.segments for w in seg.words]


@dataclass
class MatchCandidate:
    """A candidate match found by the ASR pipeline."""

    matched_text: str
    score: float  # 0..1 fuzzy similarity
    start_s: float
    end_s: float
    word_span: tuple[int, int]


@dataclass
class TimeWindow:
    start_s: float
    end_s: float


@dataclass
class OCRCandidate:
    """A candidate match found by the OCR pipeline."""

    matched_text: str
    score: float
    window: TimeWindow


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
    text: str | None = None
    timestamp: str | None = None  # HH:MM:SS.sss
    frame_index: int | None = None
    image_path: Path | None = None
    confidence: float | None = None
    asr_candidates: list[MatchCandidate] = field(default_factory=list)
    ocr_candidates: list[OCRCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
