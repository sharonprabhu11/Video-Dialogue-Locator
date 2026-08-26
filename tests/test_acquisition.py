import json
import subprocess
from unittest.mock import patch

import pytest

from vdl.acquisition import acquire_video, probe_video
from vdl.errors import AcquisitionError, VideoProbeError


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _ffprobe_json(r_frame_rate="24/1", avg_frame_rate="24/1", nb_frames="1000",
                   duration="41.666", start_time="0.0", format_duration=None):
    return json.dumps({
        "streams": [{
            "r_frame_rate": r_frame_rate,
            "avg_frame_rate": avg_frame_rate,
            "nb_frames": nb_frames,
            "duration": duration,
            "start_time": start_time,
        }],
        "format": {"duration": format_duration or duration},
    })


def test_probe_video_constant_frame_rate(tmp_path):
    fake_file = tmp_path / "source.mp4"
    fake_file.write_bytes(b"not a real video")

    with patch("vdl.acquisition.subprocess.run", return_value=_completed(stdout=_ffprobe_json())):
        meta = probe_video(fake_file)

    assert meta.fps == pytest.approx(24.0)
    assert meta.duration_s == pytest.approx(41.666)
    assert meta.frame_count == 1000
    assert meta.is_vfr is False
    assert meta.start_offset_s == pytest.approx(0.0)


def test_probe_video_detects_vfr(tmp_path):
    fake_file = tmp_path / "source.mp4"
    fake_file.write_bytes(b"x")

    vfr_json = _ffprobe_json(r_frame_rate="30/1", avg_frame_rate="23.7/1")
    with patch("vdl.acquisition.subprocess.run", return_value=_completed(stdout=vfr_json)):
        meta = probe_video(fake_file)

    assert meta.is_vfr is True


def test_probe_video_raises_on_ffprobe_failure(tmp_path):
    fake_file = tmp_path / "source.mp4"
    fake_file.write_bytes(b"x")

    with patch("vdl.acquisition.subprocess.run", return_value=_completed(returncode=1, stderr="no such file")):
        with pytest.raises(VideoProbeError):
            probe_video(fake_file)


def test_probe_video_raises_when_fps_missing(tmp_path):
    fake_file = tmp_path / "source.mp4"
    fake_file.write_bytes(b"x")

    bad_json = _ffprobe_json(r_frame_rate="0/0", avg_frame_rate="0/0")
    with patch("vdl.acquisition.subprocess.run", return_value=_completed(stdout=bad_json)):
        with pytest.raises(VideoProbeError):
            probe_video(fake_file)


def test_acquire_video_success(tmp_path):
    def fake_run(cmd, capture_output, text):
        if cmd[0] == "yt-dlp":
            (tmp_path / "source.mp4").write_bytes(b"fake video bytes")
            return _completed(returncode=0)
        elif cmd[0] == "ffprobe":
            return _completed(stdout=_ffprobe_json())
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("vdl.acquisition.subprocess.run", side_effect=fake_run):
        video = acquire_video("https://example.com/video", tmp_path)

    assert video.source_url == "https://example.com/video"
    assert video.local_path == tmp_path / "source.mp4"
    assert video.fps == pytest.approx(24.0)


def test_acquire_video_raises_on_yt_dlp_failure(tmp_path):
    with patch(
        "vdl.acquisition.subprocess.run",
        return_value=_completed(returncode=1, stderr="video unavailable"),
    ):
        with pytest.raises(AcquisitionError):
            acquire_video("https://example.com/bad-video", tmp_path)


def test_acquire_video_raises_when_no_output_file(tmp_path):
    with patch("vdl.acquisition.subprocess.run", return_value=_completed(returncode=0)):
        with pytest.raises(AcquisitionError):
            acquire_video("https://example.com/video", tmp_path)


def test_acquire_video_uses_low_cost_format_by_default(tmp_path):
    seen_commands = []

    def fake_run(cmd, capture_output, text):
        seen_commands.append(cmd)
        if cmd[0] == "yt-dlp":
            (tmp_path / "source.mp4").write_bytes(b"fake")
            return _completed(returncode=0)
        return _completed(stdout=_ffprobe_json())

    with patch("vdl.acquisition.subprocess.run", side_effect=fake_run):
        acquire_video("https://example.com/video", tmp_path)

    yt_dlp_cmd = seen_commands[0]
    assert "-f" in yt_dlp_cmd
    format_arg = yt_dlp_cmd[yt_dlp_cmd.index("-f") + 1]
    assert format_arg == "wv*[height>=240]+wa/w[height>=240]/bv*+ba/b"
    assert "bestaudio" not in format_arg  # would risk a video-less file; frame extraction always needs video
    assert format_arg.endswith("bv*+ba/b")  # falls back to BEST, not worst, when no quality floor is verifiable


def test_acquire_video_respects_custom_format_selector(tmp_path):
    seen_commands = []

    def fake_run(cmd, capture_output, text):
        seen_commands.append(cmd)
        if cmd[0] == "yt-dlp":
            (tmp_path / "source.mp4").write_bytes(b"fake")
            return _completed(returncode=0)
        return _completed(stdout=_ffprobe_json())

    with patch("vdl.acquisition.subprocess.run", side_effect=fake_run):
        acquire_video("https://example.com/video", tmp_path, format_selector="bv*+ba/b")

    yt_dlp_cmd = seen_commands[0]
    format_arg = yt_dlp_cmd[yt_dlp_cmd.index("-f") + 1]
    assert format_arg == "bv*+ba/b"
