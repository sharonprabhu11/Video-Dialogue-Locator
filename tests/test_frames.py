import subprocess
from unittest.mock import patch

import pytest

from vdl.errors import FrameExtractionError
from vdl.frames import extract_frame, format_timestamp, onset_to_frame_index
from vdl.models import AcquiredVideo, RefinedTime


def _video(**overrides):
    defaults = dict(
        source_url="https://example.com/v", local_path="source.mp4",
        duration_s=60.0, fps=24.0, frame_count=1440, is_vfr=False, start_offset_s=0.0,
    )
    defaults.update(overrides)
    return AcquiredVideo(**defaults)


def test_onset_to_frame_index_basic():
    video = _video(fps=24.0, frame_count=1440)
    assert onset_to_frame_index(video, 1.0) == 24
    assert onset_to_frame_index(video, 0.0) == 0
    assert onset_to_frame_index(video, 10.5) == int(10.5 * 24)


def test_onset_to_frame_index_respects_start_offset():
    video = _video(fps=24.0, start_offset_s=2.0)
    assert onset_to_frame_index(video, 3.0) == 24  # 1s into the stream after the offset


def test_onset_to_frame_index_clamped_to_valid_range():
    video = _video(fps=24.0, frame_count=100)
    assert onset_to_frame_index(video, -5.0) == 0
    assert onset_to_frame_index(video, 1000.0) == 99


def test_onset_to_frame_index_unbounded_when_frame_count_unknown():
    video = _video(fps=24.0, frame_count=None)
    assert onset_to_frame_index(video, 5.0) == 120


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0.0, "00:00:00.000"),
        (1.5, "00:00:01.500"),
        (61.234, "00:01:01.234"),
        (3661.001, "01:01:01.001"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_extract_frame_success(tmp_path):
    video = _video(local_path=str(tmp_path / "source.mp4"), fps=24.0)
    out_dir = tmp_path / "out"
    refined = RefinedTime(onset_s=1.0, method="asr_word_timestamp", confidence=0.9)

    def fake_run(cmd, capture_output, text):
        image_path = out_dir / "frame_24.png"
        image_path.write_bytes(b"fake png")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("vdl.frames.subprocess.run", side_effect=fake_run):
        result = extract_frame(video, refined, out_dir)

    assert result.frame_index == 24
    assert result.timestamp_s == 1.0
    assert result.image_path == out_dir / "frame_24.png"


def test_extract_frame_raises_on_ffmpeg_failure(tmp_path):
    video = _video(local_path=str(tmp_path / "source.mp4"))
    out_dir = tmp_path / "out"
    refined = RefinedTime(onset_s=1.0, method="asr_word_timestamp", confidence=0.9)

    with patch(
        "vdl.frames.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="seek out of range"),
    ):
        with pytest.raises(FrameExtractionError):
            extract_frame(video, refined, out_dir)
