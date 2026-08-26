"""Pipeline orchestration (DESIGN.md sections 1, 10, 13).

locate_dialogue() is the single entrypoint every caller (CLI, future UI,
tests) uses. It owns stage ordering and the error boundary: exceptions from
any stage are caught here and turned into a structured PipelineResult
rather than propagating to the caller (AGENTS.md section 5).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from vdl import acquisition, audio, frames, matching, ocr, refinement, transcription, visual_refinement
from vdl.config import PipelineConfig
from vdl.errors import VDLError
from vdl.models import MatchCandidate, OCRCandidate, PipelineResult, RefinedTime

logger = logging.getLogger("vdl.pipeline")


def locate_dialogue(url: str, target_text: str, cfg: PipelineConfig | None = None) -> PipelineResult:
    cfg = cfg or PipelineConfig()

    try:
        with tempfile.TemporaryDirectory(prefix="vdl_") as workdir_str:
            workdir = Path(workdir_str)

            cache_dir = Path(cfg.download_cache_dir) if cfg.download_cache_dir else None
            video = acquisition.acquire_video(
                url, workdir / "video", format_selector=cfg.video_format,
                cache_dir=cache_dir, concurrent_fragments=cfg.concurrent_fragments,
            )
            audio_asset = audio.extract_audio(video, workdir / "audio")

            transcript = transcription.transcribe(audio_asset, cfg.asr_model_name, cfg.asr_device)
            warnings: list[str] = []
            if not transcript.word_level:
                warnings.append("ASR backend returned segment-level timestamps only; onset precision is reduced")

            asr_candidates = matching.find_dialogue(transcript, target_text, cfg.match)
            result = _resolve_asr(asr_candidates, video, audio_asset, target_text, cfg, warnings)
            if result is not None:
                return result

            if not cfg.enable_ocr_fallback:
                return PipelineResult(
                    status="not_found", source="none", asr_candidates=asr_candidates, warnings=warnings
                )

            logger.info("ASR found no confident match; falling back to OCR pipeline")
            ocr_candidates = _run_ocr_pipeline(video, target_text, cfg)
            result = _resolve_ocr(ocr_candidates, video, target_text, cfg, asr_candidates, warnings)
            return result

    except VDLError as exc:
        logger.error("pipeline failed: %s", exc)
        return PipelineResult(status="error", source="none", error=str(exc))


def _resolve_asr(
    candidates: list[MatchCandidate], video, audio_asset, target_text, cfg: PipelineConfig, warnings: list[str]
) -> PipelineResult | None:
    if not candidates:
        return None

    # Collapse window-size-delta duplicates of the same utterance before
    # judging ambiguity or picking a winner (matching.dedupe_by_occurrence),
    # so near-duplicate candidates for one true occurrence can't masquerade
    # as two competing occurrences.
    occurrences = matching.dedupe_by_occurrence(candidates)
    scores_desc = sorted((c.score for c in occurrences), reverse=True)

    decision = _classify_candidates(scores_desc, cfg.match.ambiguous_margin)
    if decision == "ambiguous":
        return PipelineResult(
            status="ambiguous", source="asr", asr_candidates=occurrences, warnings=warnings,
            confidence=scores_desc[0],
        )

    # occurrences is sorted chronologically (dedupe_by_occurrence), so [0] is
    # the earliest valid occurrence -- required by the problem statement's
    # "first appears" -- never the highest-scoring one if that's a later,
    # distinct occurrence.
    best = occurrences[0]
    refined = refinement.refine_onset(best, audio_asset, cfg.refine)
    return _build_ok_result(video, refined, best.matched_text, "asr", occurrences, [], cfg, warnings)


def _run_ocr_pipeline(video, target_text: str, cfg: PipelineConfig) -> list[OCRCandidate]:
    windows = ocr.sample_candidate_windows(video, cfg.ocr)
    return ocr.find_onscreen_dialogue(video, windows, target_text, cfg.ocr)


def _resolve_ocr(
    candidates: list[OCRCandidate], video, target_text: str, cfg: PipelineConfig,
    asr_candidates: list[MatchCandidate], warnings: list[str],
) -> PipelineResult:
    if not candidates:
        return PipelineResult(
            status="not_found", source="none", asr_candidates=asr_candidates,
            ocr_candidates=candidates, warnings=warnings,
        )

    decision = _classify_candidates([c.score for c in candidates], cfg.match.ambiguous_margin)
    if decision == "ambiguous":
        return PipelineResult(
            status="ambiguous", source="ocr", asr_candidates=asr_candidates, ocr_candidates=candidates,
            warnings=warnings, confidence=candidates[0].score,
        )

    # OCR candidates are already one-per-shot (find_onscreen_dialogue never
    # produces window-delta duplicates the way ASR matching does), so no
    # dedup step is needed here -- just pick the earliest valid shot rather
    # than the highest-scoring one, same reasoning as the ASR path.
    best = min(candidates, key=lambda c: c.window.start_s)
    refined = visual_refinement.refine_frame_boundary(best, video, target_text, cfg.ocr)
    return _build_ok_result(video, refined, best.matched_text, "ocr", asr_candidates, candidates, cfg, warnings)


def _build_ok_result(
    video, refined: RefinedTime, text: str, source: str,
    asr_candidates: list[MatchCandidate], ocr_candidates: list[OCRCandidate],
    cfg: PipelineConfig, warnings: list[str],
) -> PipelineResult:
    frame_result = frames.extract_frame(video, refined, Path(cfg.out_dir))
    return PipelineResult(
        status="ok",
        source=source,
        text=text,
        timestamp=frames.format_timestamp(frame_result.timestamp_s),
        frame_index=frame_result.frame_index,
        image_path=frame_result.image_path,
        confidence=refined.confidence,
        asr_candidates=asr_candidates,
        ocr_candidates=ocr_candidates,
        warnings=warnings,
    )


def _classify_candidates(scores: list[float], ambiguous_margin: float) -> str:
    """'ok' if one candidate clearly dominates, else 'ambiguous' (DESIGN.md section 10)."""
    if len(scores) == 1:
        return "ok"
    best, second = scores[0], scores[1]
    return "ambiguous" if (best - second) <= ambiguous_margin else "ok"
