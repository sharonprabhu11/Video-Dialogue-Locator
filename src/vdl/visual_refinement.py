"""OCR-path frame-boundary refinement (DESIGN.md section 9).

A linear scan of the shot's real frames is used here deliberately, not a
bisection: OCR is a noisy oracle, and a single misread at a bisected
midpoint would send a binary search into the wrong half with no way to
recover. The window is already small (one shot, from
ocr.sample_candidate_windows), so scanning every frame in it costs a
negligible amount more and is fully robust to a one-off misread — it
requires the match to hold on two consecutive real frames before accepting
the first of them as the answer (DESIGN.md section 6).
"""

from __future__ import annotations

import logging

import pytesseract

from vdl.config import OCRConfig
from vdl.models import AcquiredVideo, OCRCandidate, RefinedTime
from vdl.ocr import extract_keyframe
from vdl.text_match import similarity

logger = logging.getLogger("vdl.visual_refinement")


def refine_frame_boundary(
    candidate: OCRCandidate,
    video: AcquiredVideo,
    target_text: str,
    cfg: OCRConfig,
    ffmpeg_bin: str = "ffmpeg",
) -> RefinedTime:
    window = candidate.window
    n_frames = max(1, round((window.end_s - window.start_s) * video.fps))
    frame_interval_s = 1.0 / video.fps

    prev_hit = False
    prev_t = window.start_s
    for i in range(n_frames + 1):
        t = window.start_s + i * frame_interval_s
        hit = _is_text_match(video, t, target_text, cfg, ffmpeg_bin)

        if hit and prev_hit:
            return RefinedTime(onset_s=prev_t, method="ocr_window_scan", confidence=candidate.score)

        prev_hit = hit
        prev_t = t

    logger.warning(
        "could not confirm a stable text run within shot window [%.3f, %.3f]; using window start",
        window.start_s, window.end_s,
    )
    return RefinedTime(onset_s=window.start_s, method="ocr_window_scan", confidence=candidate.score * 0.5)


def _is_text_match(
    video: AcquiredVideo, t: float, target_text: str, cfg: OCRConfig, ffmpeg_bin: str
) -> bool:
    frame = extract_keyframe(video, t, ffmpeg_bin)
    if frame is None:
        return False
    text = pytesseract.image_to_string(frame, lang=cfg.ocr_lang).strip()
    return bool(text) and similarity(text, target_text) >= cfg.match_threshold
