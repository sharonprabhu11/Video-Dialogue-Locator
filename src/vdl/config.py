"""Tunable thresholds and model choices, in one place.

Nothing here is video-specific: no URL, target text, fps, timestamp, or
frame number ever lives in this file or in any default below — those are
always supplied as explicit call arguments (see pipeline.locate_dialogue),
never baked in as literals, because the evaluator may substitute a
different video or target text at any time (see AGENTS.md section 8).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchConfig:
    """ASR dialogue-matching thresholds."""

    match_threshold: float = 0.80  # fuzzy score in [0, 1] to count as a hit
    ambiguous_margin: float = 0.05  # candidates within this score of the best are "ambiguous"


@dataclass
class RefineConfig:
    """Temporal refinement behavior."""

    use_vad_snap: bool = False  # optional ASR-path precision pass (see DESIGN.md section 9)
    vad_window_s: float = 1.0  # local audio window searched around the ASR onset


@dataclass
class OCRConfig:
    """OCR fallback pipeline thresholds and sampling behavior."""

    match_threshold: float = 0.65  # more lenient than ASR: OCR misreads are noisier
    scene_change_threshold: float = 0.4  # ffmpeg scene filter sensitivity, 0..1
    ocr_lang: str = "eng"


@dataclass
class PipelineConfig:
    """Top-level configuration for a single locate_dialogue() run."""

    asr_model_name: str = "small"
    asr_device: str = "cpu"
    match: MatchConfig = None  # type: ignore[assignment]
    refine: RefineConfig = None  # type: ignore[assignment]
    ocr: OCRConfig = None  # type: ignore[assignment]
    enable_ocr_fallback: bool = True
    out_dir: str = "outputs"

    def __post_init__(self) -> None:
        if self.match is None:
            self.match = MatchConfig()
        if self.refine is None:
            self.refine = RefineConfig()
        if self.ocr is None:
            self.ocr = OCRConfig()
