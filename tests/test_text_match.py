from vdl.text_match import normalize, similarity, word_count


def test_normalize_lowercases():
    assert normalize("My Mind Rebels") == "my mind rebels"


def test_normalize_strips_punctuation():
    assert normalize("My mind rebels at stagnation!") == "my mind rebels at stagnation"
    assert normalize("\"Rebels,\" he said.") == "rebels he said"


def test_normalize_collapses_whitespace():
    assert normalize("my   mind\n\trebels") == "my mind rebels"


def test_normalize_empty_string():
    assert normalize("") == ""
    assert normalize("   !!! ") == ""


def test_similarity_identical_strings_is_one():
    assert similarity("My mind rebels at stagnation", "my mind rebels at stagnation") == 1.0


def test_similarity_ignores_case_punctuation_whitespace():
    a = "My mind rebels at stagnation"
    b = "my   MIND, rebels  at STAGNATION."
    assert similarity(a, b) == 1.0


def test_similarity_tolerates_small_transcription_error():
    # one substituted word ("rebells" misheard/misread) should still score high
    a = "My mind rebels at stagnation"
    b = "my mind rebells at stagnation"
    assert similarity(a, b) > 0.9


def test_similarity_unrelated_strings_is_low():
    a = "My mind rebels at stagnation"
    b = "the quick brown fox jumps"
    assert similarity(a, b) < 0.3


def test_similarity_empty_string_is_zero():
    assert similarity("", "my mind rebels") == 0.0
    assert similarity("my mind rebels", "") == 0.0
    assert similarity("!!!", "???") == 0.0  # both normalize to empty


def test_word_count():
    assert word_count("My mind rebels at stagnation") == 5
    assert word_count("") == 0
    assert word_count("   ") == 0
    assert word_count("one, two, three!") == 3
