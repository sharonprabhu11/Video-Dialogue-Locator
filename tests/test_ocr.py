import subprocess
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from vdl.config import OCRConfig
from vdl.models import AcquiredVideo, TimeWindow
from vdl.ocr import extract_keyframe, find_onscreen_dialogue, has_probable_text_region, sample_candidate_windows


def _video(**overrides):
    defaults = dict(
        source_url="https://example.com/v", local_path="source.mp4",
        duration_s=10.0, fps=24.0, frame_count=240, is_vfr=False, start_offset_s=0.0,
    )
    defaults.update(overrides)
    return AcquiredVideo(**defaults)


def _blank_frame(size=64):
    return np.zeros((size, size, 3), dtype=np.uint8)


def _textured_frame(size=64):
    # solid rectangular bars on a plain background, like a block of on-screen
    # text: coarse, high-contrast edges that survive Canny's internal
    # smoothing (unlike a single-pixel-period checkerboard, which does not).
    frame = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (size // 8, size // 3), (size - size // 8, size // 3 + 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (size // 8, size // 2), (size - size // 4, size // 2 + 10), (0, 0, 0), -1)
    return frame


def test_has_probable_text_region_rejects_blank_frame():
    assert has_probable_text_region(_blank_frame()) == False  # noqa: E712 (numpy bool, avoid `is False`)


def test_has_probable_text_region_accepts_high_contrast_frame():
    assert has_probable_text_region(_textured_frame()) == True  # noqa: E712 (numpy bool, avoid `is True`)


def test_sample_candidate_windows_from_scene_boundaries():
    video = _video(duration_s=10.0)
    stderr = (
        "[Parsed_showinfo_1 @ 0x1] ... pts_time:2.500 ...\n"
        "[Parsed_showinfo_1 @ 0x1] ... pts_time:6.000 ...\n"
    )
    with patch(
        "vdl.ocr.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=stderr),
    ):
        windows = sample_candidate_windows(video, OCRConfig())

    assert windows == [
        TimeWindow(start_s=0.0, end_s=2.5),
        TimeWindow(start_s=2.5, end_s=6.0),
        TimeWindow(start_s=6.0, end_s=10.0),
    ]


def test_sample_candidate_windows_no_boundaries_returns_single_window():
    video = _video(duration_s=10.0)
    with patch(
        "vdl.ocr.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    ):
        windows = sample_candidate_windows(video, OCRConfig())

    assert windows == [TimeWindow(start_s=0.0, end_s=10.0)]


def test_extract_keyframe_returns_none_on_failure(tmp_path):
    video = _video(local_path=str(tmp_path / "source.mp4"))
    with patch(
        "vdl.ocr.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"seek failed"),
    ):
        assert extract_keyframe(video, 1.0) is None


def test_extract_keyframe_decodes_bmp_bytes(tmp_path):
    video = _video(local_path=str(tmp_path / "source.mp4"))
    frame = _textured_frame(size=32)
    ok, encoded = cv2.imencode(".bmp", frame)
    assert ok

    with patch(
        "vdl.ocr.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=encoded.tobytes(), stderr=b""),
    ):
        decoded = extract_keyframe(video, 1.0)

    assert decoded is not None
    assert decoded.shape[:2] == (32, 32)


def test_find_onscreen_dialogue_matches_target_text(tmp_path):
    video = _video(local_path=str(tmp_path / "source.mp4"))
    windows = [TimeWindow(start_s=0.0, end_s=5.0)]

    with (
        patch("vdl.ocr.extract_keyframe", return_value=_textured_frame()),
        patch("vdl.ocr.pytesseract.image_to_string", return_value="My mind rebels at stagnation"),
    ):
        candidates = find_onscreen_dialogue(video, windows, "My mind rebels at stagnation", OCRConfig())

    assert len(candidates) == 1
    assert candidates[0].score == pytest.approx(1.0)
    assert candidates[0].window == windows[0]


def test_find_onscreen_dialogue_skips_blank_frames_without_calling_ocr(tmp_path):
    video = _video(local_path=str(tmp_path / "source.mp4"))
    windows = [TimeWindow(start_s=0.0, end_s=5.0)]

    with (
        patch("vdl.ocr.extract_keyframe", return_value=_blank_frame()),
        patch("vdl.ocr.pytesseract.image_to_string") as mock_ocr,
    ):
        candidates = find_onscreen_dialogue(video, windows, "My mind rebels at stagnation", OCRConfig())

    assert candidates == []
    mock_ocr.assert_not_called()


def test_find_onscreen_dialogue_skips_low_similarity_text(tmp_path):
    video = _video(local_path=str(tmp_path / "source.mp4"))
    windows = [TimeWindow(start_s=0.0, end_s=5.0)]

    with (
        patch("vdl.ocr.extract_keyframe", return_value=_textured_frame()),
        patch("vdl.ocr.pytesseract.image_to_string", return_value="completely unrelated text here"),
    ):
        candidates = find_onscreen_dialogue(video, windows, "My mind rebels at stagnation", OCRConfig())

    assert candidates == []
