import subprocess
from unittest.mock import patch

import pytest

from vdl.audio import extract_audio
from vdl.errors import AudioExtractionError
from vdl.models import AcquiredVideo


def _video(tmp_path):
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake")
    return AcquiredVideo(
        source_url="https://example.com/v", local_path=src,
        duration_s=60.0, fps=24.0, frame_count=1440, is_vfr=False,
    )


def _completed(returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def test_extract_audio_success(tmp_path):
    video = _video(tmp_path)
    workdir = tmp_path / "work"

    def fake_run(cmd, capture_output, text):
        (workdir / "audio.wav").write_bytes(b"fake wav")
        return _completed(returncode=0)

    with patch("vdl.audio.subprocess.run", side_effect=fake_run):
        audio = extract_audio(video, workdir)

    assert audio.path == workdir / "audio.wav"
    assert audio.sample_rate == 16_000
    assert audio.duration_s == 60.0


def test_extract_audio_raises_on_ffmpeg_failure(tmp_path):
    video = _video(tmp_path)
    workdir = tmp_path / "work"

    with patch("vdl.audio.subprocess.run", return_value=_completed(returncode=1, stderr="no audio stream")):
        with pytest.raises(AudioExtractionError):
            extract_audio(video, workdir)


def test_extract_audio_raises_when_no_output_file(tmp_path):
    video = _video(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()

    with patch("vdl.audio.subprocess.run", return_value=_completed(returncode=0)):
        with pytest.raises(AudioExtractionError):
            extract_audio(video, workdir)
