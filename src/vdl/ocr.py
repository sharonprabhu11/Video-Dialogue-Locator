"""OCR fallback pipeline: shot-boundary sampling, cascade prefilter, OCR +
matching (DESIGN.md sections 1, 6, 9).

Triggered only when the ASR pipeline finds no confident match. Coarse
localization uses ffmpeg's scene-change detection to sample one keyframe
per shot -- a text/dialogue card is almost always its own static shot --
rather than blind fixed-interval sampling. Full OCR only runs on keyframes
that pass a cheap text-region prefilter (reject-cheap-negatives-first,
same principle as a Viola-Jones cascade).
"""

from __future__ import annotations

import logging
import re
import subprocess

import cv2
import numpy as np
import pytesseract

from vdl.config import OCRConfig
from vdl.models import AcquiredVideo, OCRCandidate, TimeWindow
from vdl.text_match import similarity

logger = logging.getLogger("vdl.ocr")

_PTS_TIME_RE = re.compile(r"pts_time:([\d.]+)")


def sample_candidate_windows(
    video: AcquiredVideo, cfg: OCRConfig, ffmpeg_bin: str = "ffmpeg"
) -> list[TimeWindow]:
    """One time window per detected shot, spanning the full video duration."""
    boundaries = _detect_scene_boundaries(video, cfg, ffmpeg_bin)
    boundaries = [0.0] + [b for b in boundaries if 0.0 < b < video.duration_s] + [video.duration_s]
    boundaries = sorted(set(boundaries))
    return [TimeWindow(start_s=a, end_s=b) for a, b in zip(boundaries, boundaries[1:])]


def _detect_scene_boundaries(video: AcquiredVideo, cfg: OCRConfig, ffmpeg_bin: str) -> list[float]:
    result = subprocess.run(
        [
            ffmpeg_bin, "-i", str(video.local_path),
            "-vf", f"select='gt(scene,{cfg.scene_change_threshold})',showinfo",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    # ffmpeg writes showinfo lines to stderr; exit code from "-f null" isn't
    # a reliable success signal here, so we just parse whatever came out —
    # the file's own validity was already established during acquisition.
    return [float(m) for m in _PTS_TIME_RE.findall(result.stderr)]


def has_probable_text_region(frame: np.ndarray, min_edge_density: float = 0.02) -> bool:
    """Cheap prefilter: does this frame plausibly contain a text region?

    Deliberately permissive (favors false positives over false negatives):
    its only job is to reject frames with clearly no chance of containing
    text before spending a full OCR call on them.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size
    return bool(edge_density >= min_edge_density)


def find_onscreen_dialogue(
    video: AcquiredVideo,
    windows: list[TimeWindow],
    target_text: str,
    cfg: OCRConfig,
    ffmpeg_bin: str = "ffmpeg",
) -> list[OCRCandidate]:
    candidates: list[OCRCandidate] = []
    for window in windows:
        midpoint_s = (window.start_s + window.end_s) / 2
        frame = extract_keyframe(video, midpoint_s, ffmpeg_bin)
        if frame is None or not has_probable_text_region(frame):
            continue

        text = pytesseract.image_to_string(frame, lang=cfg.ocr_lang).strip()
        if not text:
            continue

        score = similarity(text, target_text)
        if score >= cfg.match_threshold:
            candidates.append(OCRCandidate(matched_text=text, score=score, window=window))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def extract_keyframe(video: AcquiredVideo, timestamp_s: float, ffmpeg_bin: str = "ffmpeg") -> np.ndarray | None:
    """Decode a single frame at timestamp_s directly to memory (no temp files).

    Shared by find_onscreen_dialogue (coarse pass) and visual_refinement
    (fine pass) so both use identical decoding behavior.
    """
    result = subprocess.run(
        [
            ffmpeg_bin, "-y", "-ss", f"{timestamp_s:.6f}", "-i", str(video.local_path),
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "bmp", "-",
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        logger.warning("could not extract keyframe at t=%.3fs", timestamp_s)
        return None
    buffer = np.frombuffer(result.stdout, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
