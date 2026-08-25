"""CLI: argument parsing and output formatting only (DESIGN.md sections 12,
13; AGENTS.md section 2). No business logic lives here — everything is
delegated to pipeline.locate_dialogue().
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

from vdl.config import MatchConfig, OCRConfig, PipelineConfig, RefineConfig
from vdl.pipeline import locate_dialogue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vdl", description="Locate the exact frame where a dialogue first appears in a video.")
    sub = parser.add_subparsers(dest="command", required=True)

    locate = sub.add_parser("locate", help="Locate a target dialogue in a video URL.")
    locate.add_argument("--url", required=True, help="Video URL to search (any yt-dlp-supported source).")
    locate.add_argument("--text", required=True, help="Target dialogue text to locate.")
    locate.add_argument("--out-dir", default="outputs", help="Directory to write the extracted frame image into.")
    locate.add_argument("--asr-model", default="small", help="faster-whisper model name.")
    locate.add_argument("--asr-device", default="cpu", choices=["cpu", "cuda"])
    locate.add_argument("--match-threshold", type=float, default=0.80, help="ASR fuzzy match threshold, 0..1.")
    locate.add_argument("--ocr-threshold", type=float, default=0.65, help="OCR fuzzy match threshold, 0..1.")
    locate.add_argument("--scene-threshold", type=float, default=0.4, help="ffmpeg scene-change sensitivity, 0..1.")
    locate.add_argument("--no-ocr", action="store_true", help="Disable the OCR fallback pipeline.")
    locate.add_argument("--vad-snap", action="store_true", help="Enable the optional ASR onset VAD refinement.")
    locate.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    locate.add_argument("--json", action="store_true", help="Also print the full result as JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = PipelineConfig(
        asr_model_name=args.asr_model,
        asr_device=args.asr_device,
        match=MatchConfig(match_threshold=args.match_threshold),
        refine=RefineConfig(use_vad_snap=args.vad_snap),
        ocr=OCRConfig(match_threshold=args.ocr_threshold, scene_change_threshold=args.scene_threshold),
        enable_ocr_fallback=not args.no_ocr,
        out_dir=args.out_dir,
    )

    result = locate_dialogue(args.url, args.text, cfg)
    _print_result(result)
    if args.json:
        print(json.dumps(_result_to_jsonable(result), indent=2))

    return {"ok": 0, "ambiguous": 1, "not_found": 1, "error": 2}[result.status]


def _print_result(result) -> None:
    if result.status == "error":
        print(f"Error: {result.error}", file=sys.stderr)
        return
    if result.status == "not_found":
        print("No confident match found (ASR and OCR both attempted).")
        return
    if result.status == "ambiguous":
        print(f"Ambiguous: {len(result.asr_candidates) + len(result.ocr_candidates)} candidates found, none dominant.")
        for c in result.asr_candidates:
            print(f"  [asr] score={c.score:.2f} text={c.matched_text!r} at {c.start_s:.3f}s")
        for c in result.ocr_candidates:
            print(f"  [ocr] score={c.score:.2f} text={c.matched_text!r} in window [{c.window.start_s:.3f}, {c.window.end_s:.3f}]s")
        return

    print(f"Timestamp : {result.timestamp}")
    print(f"Frame     : {result.frame_index}")
    print(f"Text      : \"{result.text}\"")
    print(f"Source    : {result.source}")
    print(f"Image     : {result.image_path}")
    for w in result.warnings:
        print(f"Warning   : {w}")


def _result_to_jsonable(result) -> dict:
    data = asdict(result)
    if data.get("image_path") is not None:
        data["image_path"] = str(data["image_path"])
    return data


if __name__ == "__main__":
    sys.exit(main())
