import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voice_bot.dialogue_summary import dialogue_payload


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_summaries.py"
    spec = importlib.util.spec_from_file_location("evaluate_summaries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(text, *, case_id="case", summarize=True):
    return {
        "id": case_id,
        "mode": "single",
        "sources": [{"message_id": 10, "speaker": {"id": 101, "name": "Анна"}, "text": text}],
        "expected": {
            "should_summarize": summarize,
            "required_facts": ["Факт"],
            "forbidden_claims": ["Вымысел"],
        },
    }


def test_preview_does_not_load_credentials_or_make_api_calls(runner, monkeypatch, tmp_path, capsys):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"version": 1, "cases": [sample("Нет.")]}))
    monkeypatch.setattr("sys.argv", ["evaluate_summaries", "--corpus", str(corpus)])

    def forbidden_settings():
        raise AssertionError("Dry run must not load credentials")

    monkeypatch.setattr(runner, "Settings", forbidden_settings)
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out) == {"dry_run": True, "cases": ["case"]}


async def test_evaluation_requires_human_review_and_skips_filler_without_api(
    runner, settings, monkeypatch
):
    summarize = AsyncMock(return_value="Анна: Завтра не приду.")
    monkeypatch.setattr(runner, "HTTPSummarizer", lambda _: SimpleNamespace(summarize=summarize))
    cases = [sample("Завтра не приду."), sample("Привет. Спасибо.", summarize=False)]
    report = await runner.evaluate(cases, settings)
    summarize.assert_awaited_once()
    assert [row["status"] for row in report] == ["review_required", "skipped_without_api"]
    assert report[0]["expected"] == cases[0]["expected"]
    assert "summary" not in report[1]


def test_dialogue_evaluation_keeps_recipient_identity_separate(runner):
    case = sample("Отправь, пожалуйста, черновик к пятнице.")
    case["mode"] = "dialogue"
    case["sources"][0]["reply_recipient"] = {"id": 202, "name": "Борис"}
    context = runner.source_context(case)
    primary = dialogue_payload(context.dialogue, 1000)["primary_messages"]
    assert primary[0]["speaker"] == {"id": 101, "name": "Анна"}
    assert primary[0]["reply_recipient"] == {"id": 202, "name": "Борис"}
    assert primary[0]["text"] == case["sources"][0]["text"]
