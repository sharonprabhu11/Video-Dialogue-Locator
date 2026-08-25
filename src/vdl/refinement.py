"""ASR-path temporal refinement (DESIGN.md section 9).

Baseline: trust the ASR word timestamp directly. Optional precision pass:
snap to the actual speech onset within a short local window using simple
short-time energy (no ML, no new dependency — stdlib `wave` + numpy, which
is already a transitive dependency via OpenCV).

A linear scan is used over the local window rather than a bisection: the
"silence -> speech" transition is a noisy predicate (background noise,
breath sounds), and DESIGN.md section 9 already established that a linear
scan over a small window is preferable to a search algorithm that can't
recover from one bad read at the probed point.
"""

from __future__ import annotations

import logging
import wave

import numpy as np

from vdl.config import RefineConfig
from vdl.models import AudioAsset, MatchCandidate, RefinedTime

logger = logging.getLogger("vdl.refinement")

_CHUNK_S = 0.02  # 20ms analysis chunks, standard for short-time energy


def refine_onset(candidate: MatchCandidate, audio: AudioAsset, cfg: RefineConfig) -> RefinedTime:
    baseline = RefinedTime(onset_s=candidate.start_s, method="asr_word_timestamp", confidence=candidate.score)
    if not cfg.use_vad_snap:
        return baseline

    try:
        snapped_s = _vad_snap(audio, candidate.start_s, cfg.vad_window_s)
    except Exception:  # noqa: BLE001 - refinement is best-effort, never fatal
        logger.warning("VAD snap failed for onset %.3fs; falling back to ASR timestamp", candidate.start_s)
        return baseline

    if snapped_s is None:
        logger.info("VAD snap found no clear transition near %.3fs; using ASR timestamp", candidate.start_s)
        return baseline

    return RefinedTime(onset_s=snapped_s, method="vad_snap", confidence=candidate.score)


def _vad_snap(audio: AudioAsset, approx_onset_s: float, window_s: float) -> float | None:
    with wave.open(str(audio.path), "rb") as wav:
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()

        window_start_s = max(0.0, approx_onset_s - window_s)
        window_end_s = min(n_frames / sample_rate, approx_onset_s + window_s)
        start_frame = int(window_start_s * sample_rate)
        end_frame = int(window_end_s * sample_rate)

        wav.setpos(start_frame)
        raw = wav.readframes(end_frame - start_frame)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width)
    if dtype is None:
        return None
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if samples.size == 0:
        return None

    chunk_len = max(1, int(_CHUNK_S * sample_rate))
    n_chunks = samples.size // chunk_len
    if n_chunks < 2:
        return None

    energies = np.array(
        [np.sqrt(np.mean(samples[i * chunk_len:(i + 1) * chunk_len] ** 2)) for i in range(n_chunks)]
    )
    peak = energies.max()
    if peak <= 0:
        return None
    threshold = 0.3 * peak

    # Linear scan (not bisection, see module docstring): first chunk whose
    # energy crosses the threshold and stays above it for the next chunk too,
    # to avoid snapping to a single noise spike.
    for i in range(n_chunks - 1):
        if energies[i] >= threshold and energies[i + 1] >= threshold:
            return window_start_s + i * _CHUNK_S

    return None
