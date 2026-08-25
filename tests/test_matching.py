from vdl.config import MatchConfig
from vdl.matching import find_dialogue
from vdl.models import Transcript, TranscriptSegment, Word


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
