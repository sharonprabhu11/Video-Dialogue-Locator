"""Audio extraction: pull a clean mono track from the acquired video for ASR
(DESIGN.md section 1). 16kHz mono PCM WAV is the standard input format for
Whisper-family models.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from vdl.errors import AudioExtractionError
from vdl.models import AcquiredVideo, AudioAsset

logger = logging.getLogger("vdl.audio")

_TARGET_SAMPLE_RATE = 16_000


def extract_audio(video: AcquiredVideo, workdir: Path, ffmpeg_bin: str = "ffmpeg") -> AudioAsset:
    workdir.mkdir(parents=True, exist_ok=True)
    audio_path = workdir / "audio.wav"

    logger.info("extracting audio from %s", video.local_path)
    result = subprocess.run(
        [
            ffmpeg_bin, "-y", "-i", str(video.local_path),
            "-vn", "-ac", "1", "-ar", str(_TARGET_SAMPLE_RATE),
            "-f", "wav", str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg failed to extract audio from '{video.local_path}': {result.stderr.strip()}"
        )
    if not audio_path.exists():
        raise AudioExtractionError(
            f"ffmpeg reported success but no audio file was produced for '{video.local_path}'"
        )

    logger.info("extracted audio: %s", audio_path)
    return AudioAsset(path=audio_path, sample_rate=_TARGET_SAMPLE_RATE, duration_s=video.duration_s)
