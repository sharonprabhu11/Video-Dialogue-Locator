"""ASR dialogue matcher (DESIGN.md sections 3 and 6).

Slides a window over the flattened, segment-boundary-agnostic word stream
and scores each window against the target phrase using the shared fuzzy
text-matching logic. Operating on the flattened word list (rather than
per-segment text) is what makes a phrase split across two ASR segments a
non-issue: the window just happens to include words from both.
"""

from __future__ import annotations

from vdl.config import MatchConfig
from vdl.models import MatchCandidate, Transcript, Word
from vdl.text_match import similarity, word_count

# Tolerate one ASR insertion/deletion by also trying window sizes +/-1
# relative to the target phrase's own word count.
_WINDOW_SIZE_DELTAS = (-1, 0, 1)


def find_dialogue(
    transcript: Transcript, target_text: str, cfg: MatchConfig
) -> list[MatchCandidate]:
    """Return every window of the transcript scoring above cfg.match_threshold.

    All candidates above threshold are returned, sorted by score descending
    — the caller (pipeline) decides whether one dominates or whether the
    result is ambiguous (DESIGN.md section 10). This function never picks a
    single winner itself.
    """
    words = transcript.words()
    target_len = word_count(target_text)
    if target_len == 0 or not words:
        return []

    candidates: list[MatchCandidate] = []
    seen_spans: set[tuple[int, int]] = set()

    for delta in _WINDOW_SIZE_DELTAS:
        size = target_len + delta
        if size <= 0:
            continue
        for start in range(0, len(words) - size + 1):
            end = start + size
            span = (start, end)
            if span in seen_spans:
                continue
            seen_spans.add(span)

            window_words = words[start:end]
            window_text = " ".join(w.text for w in window_words)
            score = similarity(window_text, target_text)
            if score >= cfg.match_threshold:
                candidates.append(_to_candidate(window_words, window_text, score, span))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _to_candidate(
    window_words: list[Word], matched_text: str, score: float, span: tuple[int, int]
) -> MatchCandidate:
    return MatchCandidate(
        matched_text=matched_text,
        score=score,
        start_s=window_words[0].start_s,
        end_s=window_words[-1].end_s,
        word_span=span,
    )
