"""The scorer of the audio simulation (#96), on small hand-made inputs."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "skills" / "score_audio_simulation.py"


def load_scorer() -> ModuleType:
    """Import the scoring script, which is not in a package."""
    spec = importlib.util.spec_from_file_location("score_audio_simulation",
                                                  SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scorer = load_scorer()


def seg(speaker: str, start: float, end: float, text: str,
        overlap: bool = False, sid: str = "") -> Dict[str, Any]:
    """Return one sentence or utterance."""
    return {"id": sid, "speaker": speaker, "start": start, "end": end,
            "text": text, "overlap": overlap}


REFERENCE: List[Dict[str, Any]] = [
    seg("ana", 0.0, 2.0, "Buenos días, Luis.", sid="r0"),
    seg("ana", 2.2, 4.0, "¿Empezamos ya?", sid="r1"),
    seg("luis", 3.5, 5.0, "Sí, empecemos.", overlap=True, sid="r2"),
    seg("luis", 5.5, 7.0, "Traigo los datos.", sid="r3"),
]


def test_words_drop_punctuation_and_case() -> None:
    """Punctuation goes, case is lowered, accents stay."""
    assert scorer.words("¿Empezamos YA? Sí, ¡vale!") == [
        "empezamos", "ya", "sí", "vale"]


def test_edit_counts_separate_the_three_kinds() -> None:
    """One substitution, one deletion and one insertion are told apart."""
    counts = scorer.edit_counts(list("abcdefg"), list("axcefgh"))
    assert counts == {"substitutions": 1, "deletions": 1, "insertions": 1}
    assert scorer.edit_counts([], ["a"]) == {
        "substitutions": 0, "deletions": 0, "insertions": 1}


def test_word_error_reads_both_sides_in_time_order() -> None:
    """Utterances out of order are sorted by start before comparing."""
    hypothesis = [seg("x", 3.5, 7.0, "sí empecemos traigo los datos"),
                  seg("y", 0.0, 3.4, "buenos días luis empezamos")]
    result = scorer.word_error(REFERENCE, hypothesis)
    assert result["reference_words"] == 10
    assert result["deletions"] == 1  # "ya"
    assert result["wer"] == pytest.approx(0.1)
    assert scorer.word_error([], [])["wer"] == 0.0


def test_best_match_takes_the_longest_overlap() -> None:
    """The utterance sharing more seconds wins; none when nothing meets."""
    a = seg("ana", 0.0, 1.0, "a")
    b = seg("luis", 1.0, 5.0, "b")
    assert scorer.best_match(seg("?", 0.5, 3.0, ""), [a, b]) is b
    assert scorer.best_match(seg("?", 9.0, 10.0, ""), [a, b]) is None


def test_score_counts_right_wrong_missed_and_overlaps() -> None:
    """Every figure on a transcript with one wrong speaker and one miss."""
    utterances = [
        seg("ana", 0.0, 3.4, "Buenos días Luis. Empezamos", sid="u0"),
        seg("ana", 3.4, 5.0, "Sí empecemos", sid="u1"),  # wrong speaker
    ]
    result = scorer.score(REFERENCE, utterances)
    assert result["attribution"] == {
        "sentences": 4, "right": 2, "wrong": 1, "missed": 1,
        "accuracy": 0.5}
    assert result["by_speaker"]["ana"]["accuracy"] == 1.0
    assert result["by_speaker"]["luis"]["missed"] == 1
    assert result["overlapping"]["wrong"] == 1
    assert result["cut_into"]["right"] == 1  # r1, the sentence cut into
    assert result["transcript_speakers"] == ["ana"]
    # ana: all 3.8 s right (u0 and u1); luis: none of 3.0 s.
    assert result["time_attribution"] == pytest.approx(3.8 / 6.8)
    assert result["sentences"][3]["utterance"] is None


def test_time_attribution_caps_at_the_sentence_length() -> None:
    """Two utterances over one sentence count it once."""
    reference = [seg("ana", 0.0, 1.0, "a")]
    twice = [seg("ana", 0.0, 1.0, "a"), seg("ana", 0.0, 1.0, "a")]
    assert scorer.time_attribution(reference, twice) == 1.0
    assert scorer.time_attribution([], twice) == 0.0


def test_main_prints_and_writes_json(tmp_path: Path,
                                     capsys: pytest.CaptureFixture[str]
                                     ) -> None:
    """The command line reads both files and writes the JSON asked for."""
    reference = tmp_path / "ref.json"
    transcript = tmp_path / "t.json"
    out = tmp_path / "out.json"
    reference.write_text(json.dumps({"utterances": REFERENCE}),
                         encoding="utf-8")
    transcript.write_text(json.dumps({"utterances": REFERENCE}),
                          encoding="utf-8")
    assert scorer.main([str(reference), str(transcript),
                        "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "WER 0.000" in printed
    assert "overlapping starts: 1/1 right speaker" in printed
    assert json.loads(out.read_text(encoding="utf-8"))[
        "attribution"]["accuracy"] == 1.0
