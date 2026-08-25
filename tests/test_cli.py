from pathlib import Path
from unittest.mock import patch

from vdl.cli import main
from vdl.models import MatchCandidate, PipelineResult


def test_main_ok_result_prints_expected_fields_and_returns_zero(capsys):
    result = PipelineResult(
        status="ok", source="asr", text="My mind rebels at stagnation",
        timestamp="00:12:03.500", frame_index=17364, image_path=Path("outputs/frame_17364.png"),
        confidence=0.97,
    )
    with patch("vdl.cli.locate_dialogue", return_value=result):
        code = main(["locate", "--url", "https://example.com/v", "--text", "My mind rebels at stagnation"])

    captured = capsys.readouterr()
    assert code == 0
    assert "Timestamp : 00:12:03.500" in captured.out
    assert "Frame     : 17364" in captured.out
    assert 'Text      : "My mind rebels at stagnation"' in captured.out
    assert "Source    : asr" in captured.out


def test_main_not_found_returns_nonzero(capsys):
    result = PipelineResult(status="not_found", source="none")
    with patch("vdl.cli.locate_dialogue", return_value=result):
        code = main(["locate", "--url", "https://example.com/v", "--text", "nonexistent phrase"])

    assert code == 1
    assert "No confident match" in capsys.readouterr().out


def test_main_ambiguous_lists_candidates(capsys):
    result = PipelineResult(
        status="ambiguous", source="asr",
        asr_candidates=[
            MatchCandidate(matched_text="my mind rebels", score=0.9, start_s=1.0, end_s=2.0, word_span=(0, 3)),
            MatchCandidate(matched_text="my mind rebels", score=0.88, start_s=30.0, end_s=31.0, word_span=(50, 53)),
        ],
    )
    with patch("vdl.cli.locate_dialogue", return_value=result):
        code = main(["locate", "--url", "https://example.com/v", "--text", "my mind rebels"])

    out = capsys.readouterr().out
    assert code == 1
    assert "Ambiguous: 2 candidates" in out


def test_main_error_prints_to_stderr_and_returns_two(capsys):
    result = PipelineResult(status="error", source="none", error="yt-dlp failed: video unavailable")
    with patch("vdl.cli.locate_dialogue", return_value=result):
        code = main(["locate", "--url", "https://example.com/bad", "--text", "anything"])

    assert code == 2
    assert "yt-dlp failed" in capsys.readouterr().err


def test_main_passes_cli_flags_into_pipeline_config():
    result = PipelineResult(status="not_found", source="none")
    with patch("vdl.cli.locate_dialogue", return_value=result) as mock_locate:
        main([
            "locate", "--url", "https://example.com/v", "--text", "hi",
            "--match-threshold", "0.9", "--no-ocr", "--vad-snap",
        ])

    args, kwargs = mock_locate.call_args
    cfg = args[2] if len(args) > 2 else kwargs["cfg"]
    assert cfg.match.match_threshold == 0.9
    assert cfg.enable_ocr_fallback is False
    assert cfg.refine.use_vad_snap is True
