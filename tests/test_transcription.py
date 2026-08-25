from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vdl.errors import TranscriptionError
from vdl.models import AudioAsset
from vdl.transcription import transcribe


def _audio(tmp_path):
    p = tmp_path / "audio.wav"
    p.write_bytes(b"fake")
    return AudioAsset(path=p, sample_rate=16_000, duration_s=10.0)


def _fake_word(word, start, end, probability=0.95):
    return SimpleNamespace(word=word, start=start, end=end, probability=probability)


def _fake_segment(text, start, end, words):
    return SimpleNamespace(text=text, start=start, end=end, words=words)


def test_transcribe_produces_word_level_transcript(tmp_path):
    audio = _audio(tmp_path)
    fake_segments = [
        _fake_segment(
            "my mind rebels at stagnation", 10.0, 11.6,
            [
                _fake_word(" my", 10.0, 10.2),
                _fake_word(" mind", 10.2, 10.5),
                _fake_word(" rebels", 10.5, 10.9),
                _fake_word(" at", 10.9, 11.0),
                _fake_word(" stagnation", 11.0, 11.6),
            ],
        )
    ]

    with patch("vdl.transcription._run_model", return_value=fake_segments):
        transcript = transcribe(audio, model_name="small")

    assert transcript.word_level is True
    assert len(transcript.segments) == 1
    words = transcript.words()
    assert [w.text for w in words] == ["my", "mind", "rebels", "at", "stagnation"]
    assert words[0].start_s == 10.0
    assert words[-1].end_s == 11.6


def test_transcribe_flags_missing_word_timestamps(tmp_path):
    audio = _audio(tmp_path)
    fake_segments = [_fake_segment("hello world", 0.0, 1.0, words=[])]

    with patch("vdl.transcription._run_model", return_value=fake_segments):
        transcript = transcribe(audio, model_name="small")

    assert transcript.word_level is False
    assert transcript.segments[0].text == "hello world"


def test_transcribe_wraps_backend_failure(tmp_path):
    audio = _audio(tmp_path)

    with patch("vdl.transcription._run_model", side_effect=RuntimeError("model load failed")):
        with pytest.raises(TranscriptionError):
            transcribe(audio, model_name="small")
