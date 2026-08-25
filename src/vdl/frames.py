"""Frame extraction (DESIGN.md sections 7, 8).

For constant frame rate video: frame_index = floor((t - start_offset) * fps),
computed from the video's own measured fps/offset, never an assumed
constant. For VFR video, arithmetic frame numbering is unreliable (frame
spacing isn't uniform), so extraction instead seeks by timestamp and reads
back whichever frame the decoder actually presents, then reports the frame
index consistent with that same timestamp/fps basis for reproducibility.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path

from vdl.errors import FrameExtractionError
from vdl.models import AcquiredVideo, FrameResult, RefinedTime

logger = logging.getLogger("vdl.frames")


def onset_to_frame_index(video: AcquiredVideo, onset_s: float) -> int:
    """Deterministic timestamp -> frame index mapping (DESIGN.md section 7)."""
    frame_index = math.floor((onset_s - video.start_offset_s) * video.fps)
    if video.frame_count is not None:
        frame_index = max(0, min(frame_index, video.frame_count - 1))
    else:
        frame_index = max(0, frame_index)
    return frame_index


def extract_frame(
    video: AcquiredVideo, refined: RefinedTime, out_dir: Path, ffmpeg_bin: str = "ffmpeg"
) -> FrameResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_index = onset_to_frame_index(video, refined.onset_s)
    image_path = out_dir / f"frame_{frame_index}.png"

    logger.info(
        "extracting frame at t=%.3fs (method=%s) -> frame_index=%d",
        refined.onset_s, refined.method, frame_index,
    )

    # -ss before -i seeks by timestamp in the demuxer; for VFR streams this
    # reliably returns whichever frame is actually displayed at that time,
    # rather than trusting linear fps arithmetic (DESIGN.md section 8).
    result = subprocess.run(
        [
            ffmpeg_bin, "-y", "-ss", f"{refined.onset_s:.6f}", "-i", str(video.local_path),
            "-frames:v", "1", str(image_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FrameExtractionError(
            f"ffmpeg failed to extract frame at t={refined.onset_s:.3f}s from "
            f"'{video.local_path}': {result.stderr.strip()}"
        )
    if not image_path.exists():
        raise FrameExtractionError(
            f"ffmpeg reported success but no image was produced for t={refined.onset_s:.3f}s"
        )

    return FrameResult(frame_index=frame_index, timestamp_s=refined.onset_s, image_path=image_path)


def format_timestamp(seconds: float) -> str:
    """HH:MM:SS.sss, matching the problem statement's required output format."""
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
