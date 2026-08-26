"""Video acquisition: resolve a URL to a local, decodable file plus its
measured stream metadata (DESIGN.md sections 1, 7, 8).

fps/duration/VFR are always measured from the actually-downloaded file via
ffprobe — never assumed or hardcoded — because the same code has to handle
whatever fps/container the evaluator's substituted video happens to use.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from vdl.errors import AcquisitionError, VideoProbeError
from vdl.models import AcquiredVideo

logger = logging.getLogger("vdl.acquisition")

# Prefer a verified-floor low-quality stream (>=240p, both video+audio
# present) over best quality, since ASR doesn't need video quality and
# frame extraction only needs one still frame. Falls back to best quality
# -- not to the unqualified "worst" -- when no format reports a height we
# can check: a real run against the target video showed plain "wv*+wa/w"
# resolving to an even-lower, unverifiable-quality tier than intended
# (ok.ru's "mobile" format, whose audio/resolution metadata reports as
# unknown) and it measurably degraded transcription accuracy ("rebels at"
# misheard as "verbels its" -- see prompt.txt). Correctness takes priority
# over download speed when the quality floor can't be confirmed.
# Deliberately not "bestaudio" alone: a source with separate audio-only
# streams would then yield a file with no video track at all, breaking the
# frame-extraction step that always has to run regardless of which
# pipeline (ASR or OCR) resolves the match.
DEFAULT_FORMAT_SELECTOR = "wv*[height>=240]+wa/w[height>=240]/bv*+ba/b"


def acquire_video(
    url: str, workdir: Path, yt_dlp_bin: str = "yt-dlp", format_selector: str = DEFAULT_FORMAT_SELECTOR
) -> AcquiredVideo:
    """Download the given URL to workdir and return its measured metadata.

    Raises AcquisitionError if the URL can't be resolved/downloaded, or
    VideoProbeError if the downloaded file's stream metadata can't be read.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    output_template = str(workdir / "source.%(ext)s")

    logger.info("acquiring video: %s (format=%s)", url, format_selector)
    result = subprocess.run(
        [yt_dlp_bin, "--no-warnings", "-o", output_template, "-f", format_selector, url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AcquisitionError(
            f"yt-dlp failed to acquire '{url}': {result.stderr.strip()}"
        )

    downloaded = _find_downloaded_file(workdir)
    if downloaded is None:
        raise AcquisitionError(f"yt-dlp reported success but no output file was found for '{url}'")

    metadata = probe_video(downloaded)
    logger.info(
        "acquired video: fps=%.3f duration=%.2fs frame_count=%s vfr=%s",
        metadata.fps, metadata.duration_s, metadata.frame_count, metadata.is_vfr,
    )
    return AcquiredVideo(
        source_url=url,
        local_path=downloaded,
        duration_s=metadata.duration_s,
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        is_vfr=metadata.is_vfr,
        start_offset_s=metadata.start_offset_s,
    )


def _find_downloaded_file(workdir: Path) -> Path | None:
    candidates = [p for p in workdir.glob("source.*") if p.is_file()]
    return candidates[0] if candidates else None


class _ProbedMetadata:
    def __init__(self, fps: float, duration_s: float, frame_count: int | None, is_vfr: bool, start_offset_s: float):
        self.fps = fps
        self.duration_s = duration_s
        self.frame_count = frame_count
        self.is_vfr = is_vfr
        self.start_offset_s = start_offset_s


def probe_video(path: Path, ffprobe_bin: str = "ffprobe") -> _ProbedMetadata:
    """Measure fps, duration, frame count, VFR-ness, and start offset via ffprobe.

    VFR is detected by comparing r_frame_rate (the container's nominal rate)
    against avg_frame_rate (the actual average over the stream) — a
    meaningful difference means frame spacing isn't uniform (DESIGN.md
    section 8), which downstream frame extraction needs to know about.
    """
    result = subprocess.run(
        [
            ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,nb_frames,duration,start_time",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise VideoProbeError(f"ffprobe failed on '{path}': {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise VideoProbeError(f"ffprobe returned unparseable output for '{path}': {exc}") from exc

    r_fps = _parse_rational(stream.get("r_frame_rate"))
    avg_fps = _parse_rational(stream.get("avg_frame_rate"))
    if r_fps is None and avg_fps is None:
        raise VideoProbeError(f"ffprobe could not determine fps for '{path}'")
    fps = avg_fps or r_fps

    duration_s = _parse_float(stream.get("duration")) or _parse_float(data.get("format", {}).get("duration"))
    if duration_s is None:
        raise VideoProbeError(f"ffprobe could not determine duration for '{path}'")

    frame_count = _parse_int(stream.get("nb_frames"))
    start_offset_s = _parse_float(stream.get("start_time")) or 0.0

    is_vfr = (
        r_fps is not None
        and avg_fps is not None
        and abs(r_fps - avg_fps) > 0.01 * max(r_fps, avg_fps)
    )

    return _ProbedMetadata(
        fps=fps, duration_s=duration_s, frame_count=frame_count,
        is_vfr=is_vfr, start_offset_s=start_offset_s,
    )


def _parse_rational(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        num, _, den = value.partition("/")
        den_f = float(den)
        if den_f == 0:
            return None
        return float(num) / den_f
    return float(value)


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
