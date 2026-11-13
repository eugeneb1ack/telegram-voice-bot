"""Offline corpus contracts; these checks do not claim model semantic quality."""

import json
from pathlib import Path

import pytest

from voice_bot.dialogue_summary import has_summary_content

CORPUS = json.loads((Path(__file__).parent / "fixtures/summary-eval-v1.json").read_text())


def test_versioned_corpus_covers_required_semantic_cases():
    assert CORPUS["version"] == 1
    assert {case["id"] for case in CORPUS["cases"]} == {
        "past_tense",
        "negation",
        "uncertainty",
        "correction",
        "recipient",
        "injection",
        "mixed_authors",
        "filler_skip",
    }
    assert len(CORPUS["cases"]) == 8


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_corpus_has_valid_sources_and_reviewable_expectations(case):
    assert case["mode"] in {"single", "dialogue", "batch"}
    assert case["sources"]
    ids = [source["message_id"] for source in case["sources"]]
    assert ids == sorted(set(ids)) and all(mid > 0 for mid in ids)
    for source in case["sources"]:
        assert isinstance(source["text"], str) and source["text"].strip()
        for identity in (source["speaker"], source["reply_recipient"]):
            if identity is not None:
                assert isinstance(identity["id"], int)
                assert isinstance(identity["name"], str) and identity["name"].strip()
    expected = case["expected"]
    assert (
        any(has_summary_content(source["text"]) for source in case["sources"])
        is expected["should_summarize"]
    )
    for rubric in ("required_facts", "forbidden_claims"):
        assert expected[rubric]
        assert all(isinstance(rule, str) and rule.strip() for rule in expected[rubric])
    if case["mode"] == "single":
        assert len(case["sources"]) == 1
    elif case["mode"] == "dialogue":
        assert len(case["sources"]) == 3
        assert len({source["speaker"]["id"] for source in case["sources"]}) == 1
    else:
        assert len({source["speaker"]["id"] for source in case["sources"]}) == 2
