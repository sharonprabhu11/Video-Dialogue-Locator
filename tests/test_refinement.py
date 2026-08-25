import wave

import numpy as np
import pytest

from vdl.config import RefineConfig
from vdl.models import AudioAsset, MatchCandidate
from vdl.refinement import refine_onset

_SAMPLE_RATE = 16_000


def _write_wav(path, samples: np.ndarray, sample_rate: int = _SAMPLE_RATE):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype(np.int16).tobytes())


def _silence_then_tone(silence_s: float, tone_s: float, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    silence = np.zeros(int(silence_s * sample_rate))
    t = np.arange(int(tone_s * sample_rate)) / sample_rate
    tone = (20000 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)
    return np.concatenate([silence, tone])


def _candidate(start_s: float) -> MatchCandidate:
    return MatchCandidate(matched_text="my mind rebels", score=0.95, start_s=start_s, end_s=start_s + 0.5, word_span=(0, 3))


def test_refine_onset_returns_baseline_when_vad_disabled(tmp_path):
    audio_path = tmp_path / "audio.wav"
    _write_wav(audio_path, _silence_then_tone(1.0, 1.0))
    audio = AudioAsset(path=audio_path, sample_rate=_SAMPLE_RATE, duration_s=2.0)

    refined = refine_onset(_candidate(1.0), audio, RefineConfig(use_vad_snap=False))

    assert refined.method == "asr_word_timestamp"
    assert refined.onset_s == 1.0


def test_vad_snap_finds_speech_onset(tmp_path):
    audio_path = tmp_path / "audio.wav"
    _write_wav(audio_path, _silence_then_tone(1.0, 1.0))
    audio = AudioAsset(path=audio_path, sample_rate=_SAMPLE_RATE, duration_s=2.0)

    # ASR reported the onset slightly early; VAD should snap close to the true 1.0s onset.
    refined = refine_onset(_candidate(0.9), audio, RefineConfig(use_vad_snap=True, vad_window_s=1.0))

    assert refined.method == "vad_snap"
    assert refined.onset_s == pytest.approx(1.0, abs=0.1)


def test_vad_snap_falls_back_to_baseline_on_silence(tmp_path):
    audio_path = tmp_path / "audio.wav"
    _write_wav(audio_path, np.zeros(int(2.0 * _SAMPLE_RATE)))  # all silence, no speech anywhere
    audio = AudioAsset(path=audio_path, sample_rate=_SAMPLE_RATE, duration_s=2.0)

    refined = refine_onset(_candidate(1.0), audio, RefineConfig(use_vad_snap=True))

    assert refined.method == "asr_word_timestamp"
    assert refined.onset_s == 1.0


def test_refine_onset_falls_back_gracefully_on_corrupt_audio(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"this is not a valid wav file")
    audio = AudioAsset(path=audio_path, sample_rate=_SAMPLE_RATE, duration_s=2.0)

    refined = refine_onset(_candidate(1.0), audio, RefineConfig(use_vad_snap=True))

    assert refined.method == "asr_word_timestamp"
    assert refined.onset_s == 1.0
