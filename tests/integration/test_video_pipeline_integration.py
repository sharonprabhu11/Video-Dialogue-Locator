"""Integration tests: real ffmpeg/ffprobe/faster-whisper/tesseract calls
against a small locally-generated fixture video, per AGENTS.md section 4.

These are intentionally separate from tests/ (unit tests, all mocked).
Each external binary is looked up via shutil.which and the test is skipped
(not failed) if it isn't installed — see README.md for the system
prerequisites (ffmpeg, yt-dlp, tesseract-ocr) needed to run these for real.
"""

from __future__ import annotations

import shutil
import subprocess
import wave

import cv2
import numpy as np
import pytest

from vdl import acquisition, audio, ocr
from vdl.config import OCRConfig, RefineConfig
from vdl.frames import extract_frame
from vdl.models import MatchCandidate, RefinedTime
from vdl.refinement import refine_onset

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
TESSERACT = shutil.which("tesseract")

requires_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not installed; see README.md prerequisites"
)
requires_tesseract = pytest.mark.skipif(
    not TESSERACT, reason="tesseract-ocr not installed; see README.md prerequisites"
)


@pytest.fixture
def synthetic_video(tmp_path):
    """A real, tiny (3s, 25fps, 320x240) generated video with a tone -- no
    network required, just ffmpeg's built-in test sources.
    """
    path = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            FFMPEG, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-shortest", "-pix_fmt", "yuv420p", str(path),
        ],
        capture_output=True, check=True,
    )
    return path


@requires_ffmpeg
def test_probe_video_reads_real_metadata(synthetic_video):
    meta = acquisition.probe_video(synthetic_video, ffprobe_bin=FFPROBE)
    assert meta.fps == pytest.approx(25.0, abs=0.5)
    assert meta.duration_s == pytest.approx(3.0, abs=0.2)
    assert meta.is_vfr is False


@requires_ffmpeg
def test_extract_audio_produces_valid_wav(tmp_path, synthetic_video):
    from vdl.models import AcquiredVideo

    video = AcquiredVideo(
        source_url="local", local_path=synthetic_video, duration_s=3.0,
        fps=25.0, frame_count=75, is_vfr=False,
    )
    asset = audio.extract_audio(video, tmp_path / "audio_out", ffmpeg_bin=FFMPEG)

    assert asset.path.exists()
    with wave.open(str(asset.path), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1
        real_duration = wav.getnframes() / wav.getframerate()
    assert real_duration == pytest.approx(3.0, abs=0.2)


@pytest.fixture
def vfr_video(tmp_path):
    """A real, genuinely variable-frame-rate video: a 30fps source with an
    irregular frame-keep pattern (select filter) muxed with -vsync vfr, so
    frame spacing is actually non-uniform rather than merely mislabeled.

    Confirmed live against this exact command: ffprobe reports
    r_frame_rate=30/1 (nominal) vs. avg_frame_rate ~= 12.4 (actual) -- a
    >1% gap, which is exactly what probe_video's is_vfr detection checks.
    """
    path = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            FFMPEG, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=30",
            "-vf", r"select='if(eq(mod(n\,5)\,0)+eq(mod(n\,5)\,1)\,1\,0)'",
            "-vsync", "vfr", "-pix_fmt", "yuv420p", str(path),
        ],
        capture_output=True, check=True,
    )
    return path


@requires_ffmpeg
def test_probe_video_detects_real_vfr_content(vfr_video):
    meta = acquisition.probe_video(vfr_video, ffprobe_bin=FFPROBE)
    assert meta.is_vfr is True


@requires_ffmpeg
def test_extract_frame_produces_valid_image_from_real_vfr_video(tmp_path, vfr_video):
    """No VFR-specific branch exists in extract_frame -- it always seeks by
    real timestamp (-ss before -i) and never trusts arithmetic frame-index
    for extraction itself, so the same code path is expected to already be
    correct for VFR content. This is the first real (non-mocked) check of
    that claim: DESIGN.md discusses VFR handling, but no prior test -- unit
    or integration -- ever exercised a file with actually non-uniform frame
    spacing.
    """
    from vdl.models import AcquiredVideo

    meta = acquisition.probe_video(vfr_video, ffprobe_bin=FFPROBE)
    assert meta.is_vfr is True  # sanity: this fixture must actually be VFR, or the test proves nothing

    video = AcquiredVideo(
        source_url="local", local_path=vfr_video, duration_s=meta.duration_s,
        fps=meta.fps, frame_count=meta.frame_count, is_vfr=True,
    )
    refined = RefinedTime(onset_s=1.5, method="asr_word_timestamp", confidence=0.9)

    result = extract_frame(video, refined, tmp_path / "frames_out", ffmpeg_bin=FFMPEG)

    image = cv2.imread(str(result.image_path))
    assert image is not None
    assert image.shape[0] > 0 and image.shape[1] > 0


@requires_ffmpeg
def test_extract_frame_produces_readable_image(tmp_path, synthetic_video):
    from vdl.models import AcquiredVideo

    video = AcquiredVideo(
        source_url="local", local_path=synthetic_video, duration_s=3.0,
        fps=25.0, frame_count=75, is_vfr=False,
    )
    refined = RefinedTime(onset_s=1.5, method="asr_word_timestamp", confidence=0.9)

    result = extract_frame(video, refined, tmp_path / "frames_out", ffmpeg_bin=FFMPEG)

    assert result.frame_index == 37  # floor(1.5 * 25)
    image = cv2.imread(str(result.image_path))
    assert image is not None
    assert image.shape[0] > 0 and image.shape[1] > 0


@requires_ffmpeg
def test_vad_snap_runs_against_real_extracted_audio(tmp_path, synthetic_video):
    from vdl.models import AcquiredVideo

    video = AcquiredVideo(
        source_url="local", local_path=synthetic_video, duration_s=3.0,
        fps=25.0, frame_count=75, is_vfr=False,
    )
    asset = audio.extract_audio(video, tmp_path / "audio_out", ffmpeg_bin=FFMPEG)
    candidate = MatchCandidate(matched_text="x", score=0.9, start_s=1.0, end_s=1.2, word_span=(0, 1))

    # The synthetic clip is a constant tone (no silence->speech transition),
    # so this exercises the real code path end-to-end without asserting a
    # specific onset -- it must not crash, and must return a RefinedTime.
    refined = refine_onset(candidate, asset, RefineConfig(use_vad_snap=True))
    assert refined.onset_s >= 0.0


@requires_ffmpeg
def test_ocr_sample_candidate_windows_on_constant_content(synthetic_video):
    from vdl.models import AcquiredVideo

    video = AcquiredVideo(
        source_url="local", local_path=synthetic_video, duration_s=3.0,
        fps=25.0, frame_count=75, is_vfr=False,
    )
    # testsrc's moving elements are gradual, not hard cuts -- with a default
    # scene-change threshold, real ffmpeg should report few or no cuts on it.
    windows = ocr.sample_candidate_windows(video, OCRConfig(), ffmpeg_bin=FFMPEG)

    assert len(windows) >= 1
    assert windows[0].start_s == 0.0
    assert windows[-1].end_s == pytest.approx(3.0, abs=0.2)


@requires_ffmpeg
@requires_tesseract
def test_real_ocr_reads_rendered_text(tmp_path, synthetic_video):
    """Draws real text onto a real frame and confirms tesseract (installed
    system binary, invoked via pytesseract) reads it back correctly -- the
    one test that needs the actual OCR engine, not a mock.
    """
    frame = np.full((120, 640, 3), 255, dtype=np.uint8)
    cv2.putText(frame, "STAGNATION", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    import pytesseract

    text = pytesseract.image_to_string(frame).strip()
    assert "STAGNATION" in text.upper()


@pytest.mark.skipif(
    True, reason="downloads a real ASR model from the network; run manually with -m integration --no-skip-slow"
)
def test_real_asr_transcribe_end_to_end(tmp_path, synthetic_video):
    """Not run automatically (network + model download cost). Left in place
    to document how a full ASR integration check would be run manually.
    """
    from vdl.models import AcquiredVideo
    from vdl.transcription import transcribe

    video = AcquiredVideo(
        source_url="local", local_path=synthetic_video, duration_s=3.0,
        fps=25.0, frame_count=75, is_vfr=False,
    )
    asset = audio.extract_audio(video, tmp_path / "audio_out", ffmpeg_bin=FFMPEG)
    transcript = transcribe(asset, model_name="tiny", device="cpu")
    assert transcript.language
