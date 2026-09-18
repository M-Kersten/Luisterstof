from pipeline.audio.wer import edit_distance, normalise, wer, word_error_rate
from pipeline.models import Glossary, LexiconEntry


def test_normalise():
    assert normalise("Héél, mooi! (ja) 'twee'") == ["heel", "mooi", "ja", "2"]
    assert normalise("de hele—") == ["de", "hele"]


def test_wer_values():
    assert wer(["a", "b", "c"], ["a", "b", "c"]) == 0.0
    assert wer(["a", "b", "c"], ["a", "c"]) == 1 / 3
    assert wer([], ["x"]) == 1.0 and wer([], []) == 0.0
    assert edit_distance(["a"], ["b", "a"]) == 1


def test_glossary_tolerance():
    g = Glossary(book_id="b", entries=[LexiconEntry(surface="gradient descent", kind="loanword_en", spoken="greedient discent")])
    assert word_error_rate("we doen gradient descent", "we doen greedient discent", g) == 0.0
    assert word_error_rate("we doen gradient descent", "we doen greedient discent") > 0.0
