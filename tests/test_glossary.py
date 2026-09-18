from pipeline.models import Glossary, LexiconEntry
from pipeline.plan.glossary import (
    apply_lexicon,
    canonicalise,
    load_defaults,
    load_or_create_glossary,
    propose_lexicon,
    scan_candidates,
)


def _glossary(cast_dir):
    g = Glossary(book_id="b")
    g.merge(load_defaults(cast_dir))
    g.merge([LexiconEntry(surface="gradient descent", kind="loanword_en", spoken="greedient discent", lock=True),
             LexiconEntry(surface="descent", kind="loanword_en", spoken="discent"),
             LexiconEntry(surface="P(A|B)", kind="notation", spoken="P van A gegeven B")])
    return g


def test_apply_lexicon_boundaries_and_case(cast_dir):
    g = _glossary(cast_dir)
    out = apply_lexicon("Bijv. gradient descent met o.a. P(A|B) en 16,7 %. Descent alleen. Rebijv.x", g)
    assert out.startswith("Bijvoorbeeld greedient discent met onder andere P van A gegeven B en 16,7 procent.")
    assert "Discent alleen." in out
    assert "Rebijv.x" in out  # no replacement inside a word


def test_apply_lexicon_longest_first_single_pass(cast_dir):
    g = Glossary(book_id="b")
    g.merge([LexiconEntry(surface="a", kind="loanword_en", spoken="b"), LexiconEntry(surface="b", kind="loanword_en", spoken="c")])
    assert apply_lexicon("a b", g) == "b c"


def test_scan_and_canonicalise(cast_dir):
    cands = scan_candidates("De kans is ca. 2,3 % bij P(A|B), zgn. CBS-cijfers en 1/6.")
    assert "ca." in cands["abbreviations"] and "zgn." in cands["abbreviations"]
    assert "CBS" in cands["acronyms"] and "1/6" in cands["notation"] and "P(A|B)" in cands["notation"]
    g = _glossary(cast_dir)
    assert canonicalise("greedient discent", g).split() == canonicalise("gradient descent", g).split()


def test_propose_and_persist(book, fake_llm, cast_dir, tmp_path):
    path = tmp_path / "g.json"
    g = load_or_create_glossary(path, "demo", cast_dir)
    assert g.find("bijv.") is not None
    new = propose_lexicon(book.chapter("ch01"), g, fake_llm)
    assert any(e.surface == "1/6" for e in new)
    g.merge(new)
    g.save(path)
    again = load_or_create_glossary(path, "demo", cast_dir)
    assert again.find("1/6").spoken == "1 gedeeld door 6"
    assert propose_lexicon(book.chapter("ch01"), again, fake_llm) == []  # nothing new the second time
