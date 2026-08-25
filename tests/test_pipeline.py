from pathlib import Path
from unittest.mock import patch

from vdl.config import MatchConfig, PipelineConfig
from vdl.errors import AcquisitionError
from vdl.models import (
    AcquiredVideo,
    AudioAsset,
    FrameResult,
    MatchCandidate,
    OCRCandidate,
    RefinedTime,
    TimeWindow,
    Transcript,
)
from vdl.pipeline import locate_dialogue

_TARGET = "My mind rebels at stagnation"


def _video():
    return AcquiredVideo(
        source_url="https://example.com/v", local_path=Path("source.mp4"),
        duration_s=60.0, fps=24.0, frame_count=1440, is_vfr=False,
    )


def _audio_asset():
    return AudioAsset(path=Path("audio.wav"), sample_rate=16_000, duration_s=60.0)


def _empty_transcript():
    return Transcript(segments=[], language="en", model_name="small", word_level=True)


def _patched(**overrides):
    """Patch every pipeline stage with sensible no-op defaults, overridable per test."""
    defaults = dict(
        acquire_video=_video(),
        extract_audio=_audio_asset(),
        transcribe=_empty_transcript(),
        find_dialogue=[],
        refine_onset=RefinedTime(onset_s=1.0, method="asr_word_timestamp", confidence=0.9),
        sample_candidate_windows=[],
        find_onscreen_dialogue=[],
        refine_frame_boundary=RefinedTime(onset_s=1.0, method="ocr_window_scan", confidence=0.7),
        extract_frame=FrameResult(frame_index=24, timestamp_s=1.0, image_path=Path("frame_24.png")),
    )
    defaults.update(overrides)
    return (
        patch("vdl.pipeline.acquisition.acquire_video", return_value=defaults["acquire_video"]),
        patch("vdl.pipeline.audio.extract_audio", return_value=defaults["extract_audio"]),
        patch("vdl.pipeline.transcription.transcribe", return_value=defaults["transcribe"]),
        patch("vdl.pipeline.matching.find_dialogue", return_value=defaults["find_dialogue"]),
        patch("vdl.pipeline.refinement.refine_onset", return_value=defaults["refine_onset"]),
        patch("vdl.pipeline.ocr.sample_candidate_windows", return_value=defaults["sample_candidate_windows"]),
        patch("vdl.pipeline.ocr.find_onscreen_dialogue", return_value=defaults["find_onscreen_dialogue"]),
        patch("vdl.pipeline.visual_refinement.refine_frame_boundary", return_value=defaults["refine_frame_boundary"]),
        patch("vdl.pipeline.frames.extract_frame", return_value=defaults["extract_frame"]),
    )


def _apply(patches):
    return [p.start() for p in patches], patches


def _stop(patches):
    for p in patches:
        p.stop()


def test_asr_confident_match_short_circuits_ocr():
    asr_candidates = [MatchCandidate(matched_text=_TARGET, score=0.95, start_s=1.0, end_s=2.0, word_span=(0, 5))]
    patches = _patched(find_dialogue=asr_candidates)
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig())
    finally:
        _stop(patches)

    assert result.status == "ok"
    assert result.source == "asr"
    assert result.text == _TARGET
    assert result.frame_index == 24
    mocks[6].assert_not_called()  # ocr.find_onscreen_dialogue never invoked


def test_asr_ambiguous_candidates_reported_without_frame_extraction():
    asr_candidates = [
        MatchCandidate(matched_text=_TARGET, score=0.90, start_s=1.0, end_s=2.0, word_span=(0, 5)),
        MatchCandidate(matched_text=_TARGET, score=0.88, start_s=30.0, end_s=31.0, word_span=(100, 105)),
    ]
    patches = _patched(find_dialogue=asr_candidates)
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig(match=MatchConfig(ambiguous_margin=0.05)))
    finally:
        _stop(patches)

    assert result.status == "ambiguous"
    assert result.source == "asr"
    assert len(result.asr_candidates) == 2
    mocks[8].assert_not_called()  # frames.extract_frame never invoked


def test_no_asr_match_falls_back_to_ocr_and_succeeds():
    ocr_candidates = [OCRCandidate(matched_text=_TARGET, score=0.8, window=TimeWindow(0.0, 5.0))]
    patches = _patched(find_dialogue=[], find_onscreen_dialogue=ocr_candidates)
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig())
    finally:
        _stop(patches)

    assert result.status == "ok"
    assert result.source == "ocr"
    mocks[6].assert_called_once()  # ocr.find_onscreen_dialogue was invoked


def test_no_match_anywhere_returns_not_found():
    patches = _patched(find_dialogue=[], find_onscreen_dialogue=[])
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig())
    finally:
        _stop(patches)

    assert result.status == "not_found"
    assert result.source == "none"


def test_ocr_disabled_skips_fallback_entirely():
    patches = _patched(find_dialogue=[])
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue(
            "https://example.com/v", _TARGET, PipelineConfig(enable_ocr_fallback=False)
        )
    finally:
        _stop(patches)

    assert result.status == "not_found"
    assert result.source == "none"
    mocks[5].assert_not_called()  # ocr.sample_candidate_windows never invoked


def test_ocr_ambiguous_candidates_reported():
    ocr_candidates = [
        OCRCandidate(matched_text=_TARGET, score=0.75, window=TimeWindow(0.0, 5.0)),
        OCRCandidate(matched_text=_TARGET, score=0.74, window=TimeWindow(100.0, 105.0)),
    ]
    patches = _patched(find_dialogue=[], find_onscreen_dialogue=ocr_candidates)
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig())
    finally:
        _stop(patches)

    assert result.status == "ambiguous"
    assert result.source == "ocr"


def test_acquisition_failure_returns_error_result():
    with patch("vdl.pipeline.acquisition.acquire_video", side_effect=AcquisitionError("bad url")):
        result = locate_dialogue("https://example.com/bad", _TARGET, PipelineConfig())

    assert result.status == "error"
    assert result.source == "none"
    assert "bad url" in result.error


def test_degraded_word_timestamps_produce_warning():
    degraded_transcript = Transcript(segments=[], language="en", model_name="small", word_level=False)
    asr_candidates = [MatchCandidate(matched_text=_TARGET, score=0.95, start_s=1.0, end_s=2.0, word_span=(0, 5))]
    patches = _patched(transcribe=degraded_transcript, find_dialogue=asr_candidates)
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig())
    finally:
        _stop(patches)

    assert any("segment-level" in w for w in result.warnings)
