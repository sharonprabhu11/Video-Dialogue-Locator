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


def dedupe_by_occurrence(candidates: list[MatchCandidate]) -> list[MatchCandidate]:
    """Collapse window-size-delta duplicates of the same utterance into one
    representative per distinct occurrence, sorted chronologically.

    find_dialogue() tries window sizes target_len-1/0/+1 (see
    _WINDOW_SIZE_DELTAS above), so a single true occurrence is very often
    returned as several overlapping candidates -- trimmed or extended by one
    word at either boundary. Real evidence this matters: a single spoken
    occurrence produced 6 candidates whose word_span all overlapped the
    canonical (delta=0) span, but whose start_s values ranged across 4.4
    seconds, because the +1-word-extended variant happened to pull in a word
    from the end of an unrelated prior sentence across a real speech pause.
    A seconds-based proximity threshold cannot safely separate "same
    occurrence, wide time spread due to an adjacent pause" from "genuinely
    different occurrence, narrow time gap" -- so grouping instead uses
    word_span overlap, which is exact: every delta-generated variant of one
    occurrence is guaranteed (by construction, since only one word is
    trimmed/added at a boundary) to share indices with the canonical span.
    Two truly distinct occurrences only risk overlapping if they fall within
    one target-phrase-length of each other in the word stream -- essentially
    back-to-back stutter repetition, a rare edge case not worth a special
    case without evidence it occurs.

    Within a cluster, the representative is the highest-scoring candidate:
    the tightest-fitting window is the one least contaminated by a spurious
    adjacent word, so it's also the most accurate estimate of that
    occurrence's true start_s (confirmed against the real 6-candidate
    example above: the highest-scoring candidate's start_s was the
    genuinely correct onset; the lowest-scoring, span-extended one was
    several seconds off, still mid-way through the *previous* sentence).
    """
    if not candidates:
        return []

    by_span_start = sorted(candidates, key=lambda c: c.word_span[0])
    clusters: list[list[MatchCandidate]] = [[by_span_start[0]]]
    cluster_end = by_span_start[0].word_span[1]

    for candidate in by_span_start[1:]:
        if candidate.word_span[0] < cluster_end:
            clusters[-1].append(candidate)
            cluster_end = max(cluster_end, candidate.word_span[1])
        else:
            clusters.append([candidate])
            cluster_end = candidate.word_span[1]

    representatives = [max(cluster, key=lambda c: c.score) for cluster in clusters]
    representatives.sort(key=lambda c: c.start_s)
    return representatives
