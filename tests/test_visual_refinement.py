from unittest.mock import patch

import pytest

from vdl.config import OCRConfig
from vdl.models import AcquiredVideo, OCRCandidate, TimeWindow
from vdl.visual_refinement import refine_frame_boundary

_TARGET = "My mind rebels at stagnation"


def _video(fps=10.0):
    return AcquiredVideo(
        source_url="https://example.com/v", local_path="source.mp4",
        duration_s=10.0, fps=fps, frame_count=100, is_vfr=False, start_offset_s=0.0,
    )


def _candidate(window: TimeWindow, score: float = 0.9) -> OCRCandidate:
    return OCRCandidate(matched_text=_TARGET, score=score, window=window)


def test_refine_finds_first_of_two_consecutive_hits():
    video = _video(fps=10.0)
    window = TimeWindow(start_s=0.0, end_s=1.0)  # 11 sampled instants at fps=10: i=0..10

    # hits at i=3 and i=4 only (t=0.3, t=0.4); everything else misses.
    ocr_outputs = ["" for _ in range(11)]
    ocr_outputs[3] = _TARGET
    ocr_outputs[4] = _TARGET

    with (
        patch("vdl.visual_refinement.extract_keyframe", return_value="dummy-frame"),
        patch("vdl.visual_refinement.pytesseract.image_to_string", side_effect=ocr_outputs),
    ):
        refined = refine_frame_boundary(_candidate(window), video, _TARGET, OCRConfig())

    assert refined.method == "ocr_window_scan"
    assert refined.onset_s == pytest.approx(0.3)
    assert refined.confidence == 0.9


def test_refine_falls_back_when_no_stable_two_frame_run():
    video = _video(fps=10.0)
    window = TimeWindow(start_s=0.0, end_s=1.0)

    # isolated single hits, never two in a row
    ocr_outputs = ["" for _ in range(11)]
    ocr_outputs[2] = _TARGET
    ocr_outputs[7] = _TARGET

    with (
        patch("vdl.visual_refinement.extract_keyframe", return_value="dummy-frame"),
        patch("vdl.visual_refinement.pytesseract.image_to_string", side_effect=ocr_outputs),
    ):
        refined = refine_frame_boundary(_candidate(window, score=0.9), video, _TARGET, OCRConfig())

    assert refined.onset_s == window.start_s
    assert refined.confidence == 0.45  # score * 0.5, signals a weaker/unconfirmed result


def test_refine_treats_undecodable_frame_as_a_miss():
    video = _video(fps=10.0)
    window = TimeWindow(start_s=0.0, end_s=1.0)

    with patch("vdl.visual_refinement.extract_keyframe", return_value=None) as mock_extract:
        refined = refine_frame_boundary(_candidate(window), video, _TARGET, OCRConfig())

    assert refined.onset_s == window.start_s
    assert mock_extract.called
