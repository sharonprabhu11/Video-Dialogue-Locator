from vdl.config import MatchConfig
from vdl.matching import dedupe_by_occurrence, find_dialogue
from vdl.models import MatchCandidate, Transcript, TranscriptSegment, Word


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start_s=start, end_s=end, confidence=0.9)


def _transcript(*segments: TranscriptSegment) -> Transcript:
    return Transcript(segments=list(segments), language="en", model_name="test", word_level=True)


def test_finds_exact_phrase_in_single_segment():
    words = [
        _word("my", 10.0, 10.2),
        _word("mind", 10.2, 10.5),
        _word("rebels", 10.5, 10.9),
        _word("at", 10.9, 11.0),
        _word("stagnation", 11.0, 11.6),
    ]
    seg = TranscriptSegment(start_s=10.0, end_s=11.6, text="my mind rebels at stagnation", words=words)
    transcript = _transcript(seg)

    candidates = find_dialogue(transcript, "My mind rebels at stagnation", MatchConfig())

    assert len(candidates) >= 1
    best = candidates[0]
    assert best.score > 0.95
    assert best.start_s == 10.0
    assert best.end_s == 11.6


def test_finds_phrase_split_across_two_segments():
    seg1 = TranscriptSegment(
        start_s=0.0, end_s=1.0, text="my mind rebels",
        words=[_word("my", 0.0, 0.2), _word("mind", 0.2, 0.5), _word("rebels", 0.5, 1.0)],
    )
    seg2 = TranscriptSegment(
        start_s=1.0, end_s=2.0, text="at stagnation",
        words=[_word("at", 1.0, 1.2), _word("stagnation", 1.2, 2.0)],
    )
    transcript = _transcript(seg1, seg2)

    candidates = find_dialogue(transcript, "My mind rebels at stagnation", MatchConfig())

    assert len(candidates) >= 1
    assert candidates[0].score > 0.95
    assert candidates[0].start_s == 0.0
    assert candidates[0].end_s == 2.0


def test_tolerates_asr_misrecognition():
    words = [
        _word("my", 0.0, 0.2),
        _word("mind", 0.2, 0.5),
        _word("rebells", 0.5, 0.9),  # misheard
        _word("at", 0.9, 1.0),
        _word("stagnation", 1.0, 1.6),
    ]
    seg = TranscriptSegment(start_s=0.0, end_s=1.6, text="", words=words)
    transcript = _transcript(seg)

    candidates = find_dialogue(transcript, "My mind rebels at stagnation", MatchConfig(match_threshold=0.80))

    assert len(candidates) >= 1
    assert candidates[0].score >= 0.80


def test_no_match_returns_empty_list():
    words = [_word(w, i, i + 0.5) for i, w in enumerate(["the", "quick", "brown", "fox"])]
    seg = TranscriptSegment(start_s=0.0, end_s=2.0, text="", words=words)
    transcript = _transcript(seg)

    candidates = find_dialogue(transcript, "My mind rebels at stagnation", MatchConfig())

    assert candidates == []


def test_multiple_occurrences_all_returned():
    phrase = ["my", "mind", "rebels", "at", "stagnation"]
    filler = ["and", "then", "watson", "said", "quietly"]
    words = []
    t = 0.0
    for w in phrase:
        words.append(_word(w, t, t + 0.4))
        t += 0.4
    for w in filler:
        words.append(_word(w, t, t + 0.4))
        t += 0.4
    for w in phrase:
        words.append(_word(w, t, t + 0.4))
        t += 0.4

    seg = TranscriptSegment(start_s=0.0, end_s=t, text="", words=words)
    transcript = _transcript(seg)

    candidates = find_dialogue(transcript, "My mind rebels at stagnation", MatchConfig())

    high_scoring = [c for c in candidates if c.score > 0.95]
    # both occurrences should be found (allow for +/-1 window-size duplicates)
    distinct_starts = {round(c.start_s, 1) for c in high_scoring}
    assert len(distinct_starts) == 2


def test_empty_transcript_returns_empty_list():
    transcript = _transcript(TranscriptSegment(start_s=0.0, end_s=0.0, text="", words=[]))
    assert find_dialogue(transcript, "My mind rebels at stagnation", MatchConfig()) == []


def test_empty_target_text_returns_empty_list():
    words = [_word("hello", 0.0, 0.5)]
    seg = TranscriptSegment(start_s=0.0, end_s=0.5, text="hello", words=words)
    transcript = _transcript(seg)
    assert find_dialogue(transcript, "", MatchConfig()) == []


def _candidate(text: str, score: float, start_s: float, span: tuple[int, int]) -> MatchCandidate:
    return MatchCandidate(matched_text=text, score=score, start_s=start_s, end_s=start_s + 1.0, word_span=span)


def test_dedupe_by_occurrence_empty_list():
    assert dedupe_by_occurrence([]) == []


def test_dedupe_by_occurrence_single_candidate_unchanged():
    c = _candidate("x", 0.9, 1.0, (0, 5))
    assert dedupe_by_occurrence([c]) == [c]


def test_dedupe_by_occurrence_merges_overlapping_spans_keeping_best_score():
    low = _candidate("low", 0.80, 1.0, (0, 6))
    high = _candidate("high", 0.95, 1.2, (1, 6))  # overlaps [0,6) at indices 1-5
    result = dedupe_by_occurrence([low, high])

    assert len(result) == 1
    assert result[0] is high  # highest-scoring candidate in the cluster wins


def test_dedupe_by_occurrence_keeps_non_overlapping_spans_separate():
    first = _candidate("first", 0.98, 100.0, (500, 505))  # deliberately out of chronological order
    second = _candidate("second", 0.80, 5.0, (0, 5))
    result = dedupe_by_occurrence([first, second])

    assert len(result) == 2
    assert [c.start_s for c in result] == [5.0, 100.0]  # sorted chronologically, not by input order or score


def test_dedupe_by_occurrence_real_six_candidate_example():
    # Real candidates from one true spoken occurrence (see prompt.txt /
    # DESIGN.md): the target_len +/-1 window search produced 6 overlapping-
    # span candidates, with start_s spread across 4.4s because the span
    # extended one word left pulled in a word from an unrelated prior
    # sentence, across a real speech pause.
    candidates = [
        _candidate("My mind rebels at stagnation", 1.0, 324.52, (290, 295)),
        _candidate("mind rebels at stagnation", 0.943, 325.38, (291, 295)),
        _candidate("time My mind rebels at stagnation", 0.918, 320.96, (289, 295)),
        _candidate("My mind rebels at stagnation Give", 0.918, 324.52, (290, 296)),
        _candidate("mind rebels at stagnation Give", 0.862, 325.38, (291, 296)),
        _candidate("mind rebels at stagnation Give me", 0.820, 325.38, (291, 297)),
    ]
    result = dedupe_by_occurrence(candidates)

    assert len(result) == 1
    assert result[0].matched_text == "My mind rebels at stagnation"
    assert result[0].start_s == 324.52  # the correct onset, not the span-extended 320.96
