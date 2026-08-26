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


# --- Candidate-selection regression tests (first-occurrence fix) ---------
#
# These exercise locate_dialogue() end-to-end with only the ASR-model-level
# calls mocked (find_dialogue), so matching.dedupe_by_occurrence() and the
# pipeline's earliest-occurrence selection run for real -- not just the
# matcher in isolation.


def test_asr_selects_earliest_occurrence_over_higher_scoring_later_one():
    earlier = MatchCandidate(matched_text="earlier reading", score=0.85, start_s=5.0, end_s=6.0, word_span=(0, 5))
    later = MatchCandidate(matched_text="later reading", score=0.98, start_s=100.0, end_s=101.0, word_span=(500, 505))
    patches = _patched(find_dialogue=[earlier, later])
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig())
    finally:
        _stop(patches)

    assert result.status == "ok"
    assert result.text == "earlier reading"  # earliest valid occurrence wins despite the lower score
    selected = mocks[4].call_args[0][0]  # refinement.refine_onset(best, ...)
    assert selected.start_s == 5.0


def test_asr_overlapping_windows_collapse_to_single_occurrence():
    wide = MatchCandidate(matched_text="wide window", score=0.90, start_s=10.0, end_s=12.0, word_span=(10, 16))
    narrow = MatchCandidate(matched_text="narrow window", score=0.95, start_s=10.3, end_s=11.8, word_span=(11, 16))
    patches = _patched(find_dialogue=[wide, narrow])
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig())
    finally:
        _stop(patches)

    assert result.status == "ok"  # one occurrence, not two -- must not be reported ambiguous
    assert len(result.asr_candidates) == 1
    assert result.text == "narrow window"  # higher-scoring representative within the single cluster


def test_asr_distinct_occurrences_both_represented_in_result():
    first = MatchCandidate(matched_text="first", score=0.85, start_s=5.0, end_s=6.0, word_span=(0, 5))
    second = MatchCandidate(matched_text="second", score=0.98, start_s=100.0, end_s=101.0, word_span=(500, 505))
    patches = _patched(find_dialogue=[first, second])
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig())
    finally:
        _stop(patches)

    assert result.status == "ok"
    assert len(result.asr_candidates) == 2  # both distinct occurrences preserved, not collapsed
    assert sorted(c.start_s for c in result.asr_candidates) == [5.0, 100.0]


def test_asr_multiple_window_sizes_around_one_occurrence_collapse_and_pick_best():
    # Mirrors a real run against the target video: one true spoken
    # occurrence produced 6 overlapping-span candidates via the
    # target_len +/-1 window-size search, with start_s spread across ~4.4s
    # because the +1-extended variant pulled in a word from the end of an
    # unrelated prior sentence, across a real speech pause.
    candidates = [
        MatchCandidate(matched_text="My mind rebels at stagnation", score=1.0,
                        start_s=324.52, end_s=327.68, word_span=(290, 295)),
        MatchCandidate(matched_text="mind rebels at stagnation", score=0.943,
                        start_s=325.38, end_s=327.68, word_span=(291, 295)),
        MatchCandidate(matched_text="time My mind rebels at stagnation", score=0.918,
                        start_s=320.96, end_s=327.68, word_span=(289, 295)),
        MatchCandidate(matched_text="My mind rebels at stagnation Give", score=0.918,
                        start_s=324.52, end_s=330.16, word_span=(290, 296)),
        MatchCandidate(matched_text="mind rebels at stagnation Give", score=0.862,
                        start_s=325.38, end_s=330.16, word_span=(291, 296)),
        MatchCandidate(matched_text="mind rebels at stagnation Give me", score=0.820,
                        start_s=325.38, end_s=330.28, word_span=(291, 297)),
    ]
    patches = _patched(find_dialogue=candidates)
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig())
    finally:
        _stop(patches)

    # Raw top-2 scores (1.0, 0.943) are only 0.057 apart -- barely above the
    # default 0.05 margin. Pre-fix, this was one narrowly-averted false
    # "ambiguous" away from misreporting a single occurrence as two.
    assert result.status == "ok"
    assert len(result.asr_candidates) == 1  # all 6 collapse to one true occurrence
    assert result.text == "My mind rebels at stagnation"
    selected = mocks[4].call_args[0][0]
    assert selected.start_s == 324.52  # correct onset, not the span-extended 320.96 (mid "time.")


def test_asr_ambiguity_ignores_window_delta_duplicates_within_occurrences():
    # Two genuinely distinct occurrences, each ALSO split into 2 window-delta
    # duplicates. Ambiguity must be judged on the 2 distinct occurrences
    # (post-dedup), not on all 4 raw candidates.
    first_a = MatchCandidate(matched_text="first a", score=0.90, start_s=5.0, end_s=6.0, word_span=(0, 5))
    first_b = MatchCandidate(matched_text="first b", score=0.87, start_s=5.2, end_s=6.2, word_span=(1, 5))
    second_a = MatchCandidate(matched_text="second a", score=0.88, start_s=100.0, end_s=101.0, word_span=(500, 505))
    second_b = MatchCandidate(matched_text="second b", score=0.85, start_s=100.2, end_s=101.2, word_span=(501, 505))
    patches = _patched(find_dialogue=[first_a, first_b, second_a, second_b])
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue(
            "https://example.com/v", _TARGET, PipelineConfig(match=MatchConfig(ambiguous_margin=0.05))
        )
    finally:
        _stop(patches)

    assert result.status == "ambiguous"
    assert len(result.asr_candidates) == 2  # 4 raw candidates collapse to 2 distinct occurrences
    assert sorted((c.score for c in result.asr_candidates), reverse=True) == [0.90, 0.88]
    mocks[8].assert_not_called()  # frames.extract_frame never invoked


def test_ocr_selects_earliest_shot_over_higher_scoring_later_one():
    earlier = OCRCandidate(matched_text="earlier text", score=0.70, window=TimeWindow(5.0, 6.0))
    later = OCRCandidate(matched_text="later text", score=0.95, window=TimeWindow(100.0, 101.0))
    # find_onscreen_dialogue always returns score-descending (ocr.py sorts
    # it) -- mock must match that real contract for _classify_candidates'
    # scores[0]/scores[1] assumption to mean what it means in production.
    patches = _patched(find_dialogue=[], find_onscreen_dialogue=[later, earlier])
    mocks, patches = _apply(patches)
    try:
        result = locate_dialogue("https://example.com/v", _TARGET, PipelineConfig())
    finally:
        _stop(patches)

    assert result.status == "ok"
    assert result.text == "earlier text"  # earliest valid shot wins despite the lower score
    selected = mocks[7].call_args[0][0]  # visual_refinement.refine_frame_boundary(best, ...)
    assert selected.window.start_s == 5.0
