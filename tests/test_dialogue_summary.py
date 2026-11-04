import json
from dataclasses import replace

import httpx
import pytest
from test_semantic_context import response

from voice_bot.dialogue_graph import DialogueBranch, DialogueMessage, SharedDialogue
from voice_bot.dialogue_summary import dialogue_payload
from voice_bot.errors import ProviderError
from voice_bot.providers import APIClient, HTTPSummarizer, SummaryContext


def music_dialogue():
    return SharedDialogue(
        1,
        (
            DialogueMessage(1, 2, "Борис", None, "text", "Я играю джаз. А ты играешь?"),
            DialogueMessage(2, 3, "Вера", None, "text", "Я профессиональный музыкант."),
            DialogueMessage(10, 1, "Антон", None, "voice", "Люблю слушать джаз.", 1),
            DialogueMessage(
                11, 1, "Антон", None, "voice", "Играть не умею, только одну мелодию пальцем.", 1
            ),
            DialogueMessage(12, 1, "Антон", None, "voice", "Сейчас попробую её сыграть."),
        ),
        (
            DialogueBranch("b1", 2, "Борис", None, (10, 11), True, (1,)),
            DialogueBranch("b2", None, None, None, (12,)),
        ),
    )


async def test_own_speech_and_other_speaker_background_are_separate_on_wire(settings):
    graph = music_dialogue()
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        data = json.loads(payload["input"])["dialogue"]
        assert "branches" not in data
        primary = data["primary_messages"]
        assert [m["message_id"] for m in primary] == [10, 11, 12]
        assert primary[0]["reply_to"]["speaker"] == {"id": 2, "name": "Борис"}
        assert primary[1]["reply_to"]["text"] == "Я играю джаз. А ты играешь?"
        assert primary[2]["reply_to"] is None and not primary[2]["background"]
        assert "профессиональный" not in payload["input"]
        assert json.loads(payload["input"])["context"]["speaker_name"] == "Антон"
        return response(
            {
                "summary": (
                    "Антон любит слушать джаз, но играть не умеет, "
                    "кроме одной мелодии одним пальцем. "
                    "Собирается попробовать её сыграть."
                ),
            }
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await HTTPSummarizer(APIClient(client, settings)).summarize(
            "Три записи", context=SummaryContext(author_name="Антон", dialogue=graph)
        )
    assert result == (
        "Антон любит слушать джаз, но играть не умеет, кроме одной мелодии одним пальцем. "
        "Собирается попробовать её сыграть."
    )
    assert len(calls) == 1  # no extra verifier/model round trip


@pytest.mark.parametrize("ids", [(1, 10), (99,), (10, 10), (10, 12)])
def test_foreign_missing_or_duplicate_primary_cannot_reach_model(ids):
    graph = music_dialogue()
    graph = replace(
        graph, branches=(replace(graph.branches[0], message_ids=ids), graph.branches[1])
    )
    with pytest.raises(ProviderError, match="авторство"):
        dialogue_payload(graph, 10000)


def test_repeated_reply_text_counts_towards_request_limit():
    graph = music_dialogue()
    payload = dialogue_payload(graph, 10000)
    total = sum(
        len(m["text"]) + (len(m["reply_to"]["text"]) if m["reply_to"] else 0)
        for m in payload["primary_messages"]
    )
    dialogue_payload(graph, total)
    with pytest.raises(ProviderError, match="длинный"):
        dialogue_payload(graph, total - 1)


def test_unknown_background_author_never_becomes_same_speaker():
    graph = music_dialogue()
    graph = replace(
        graph, messages=(replace(graph.messages[0], author_id=None), *graph.messages[1:])
    )
    assert (
        dialogue_payload(graph, 10000)["primary_messages"][0]["reply_to"]["speaker"]["id"] is None
    )


@pytest.mark.parametrize(
    "name,username,label",
    [
        ("  Анна\nПетрова  ", "handle", "Анна Петрова"),
        (None, "handle", "@handle"),
        ("<example/>", "handle", "<example/>"),
    ],
)
async def test_speaker_display_name_reaches_model_without_duplicate_heading(
    settings, name, username, label
):
    graph = music_dialogue()
    graph = replace(graph, branches=(graph.branches[1],), messages=(graph.messages[-1],))
    body = f"{label} надеется получить понятный пересказ."

    def respond(request):
        payload = json.loads(request.content)
        assert json.loads(payload["input"])["context"]["speaker_name"] == label
        assert label not in payload["instructions"]
        return response({"summary": body})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await HTTPSummarizer(APIClient(client, settings)).summarize(
            "Надеюсь получить понятный пересказ.",
            context=SummaryContext(author_name=name, author_username=username, dialogue=graph),
        )
    assert result == body


def test_interleaved_recipients_are_chronological_and_each_reply_keeps_its_own_source():
    graph = music_dialogue()
    messages = list(graph.messages)
    messages[3] = replace(messages[3], reply_to_message_id=2)
    messages[4] = replace(messages[4], reply_to_message_id=1)
    graph = replace(
        graph,
        messages=tuple(messages),
        branches=(
            DialogueBranch("b1", 2, "Борис", None, (10, 12), True, (1,)),
            DialogueBranch("b2", 3, "Вера", None, (11,), True, (2,)),
        ),
    )
    primary = dialogue_payload(graph, 10000)["primary_messages"]
    assert [m["message_id"] for m in primary] == [10, 11, 12]
    assert [m["reply_to"]["speaker"]["id"] for m in primary] == [2, 3, 2]
    assert "профессиональный" not in json.dumps(primary[0], ensure_ascii=False)
    assert "профессиональный" in primary[1]["reply_to"]["text"]


def test_reply_to_own_message_keeps_literal_speaker_and_nested_recipient():
    graph = music_dialogue()
    graph = replace(
        graph,
        messages=tuple(
            replace(m, reply_to_message_id=10) if m.message_id == 11 else m for m in graph.messages
        ),
    )
    primary = dialogue_payload(graph, 10000)["primary_messages"]
    reply = primary[1]["reply_to"]
    assert reply["speaker"] == {"id": 1, "name": "Антон"}
    assert reply["reply_to"]["speaker"] == {"id": 2, "name": "Борис"}
    assert primary[2]["reply_to"] is None  # no inherited recipient without evidence


def test_branch_background_cannot_leak_into_unlinked_siblings():
    graph = music_dialogue()
    graph = replace(
        graph,
        branches=(DialogueBranch("b1", None, None, None, (10, 11, 12), False, (1, 2)),),
        messages=tuple(
            replace(m, reply_to_message_id=None, context_message_ids=(2,))
            if m.message_id == 11
            else m
            for m in graph.messages
        ),
    )
    primary = dialogue_payload(graph, 10000)["primary_messages"]
    assert primary[0]["reply_to"]["speaker"]["id"] == 2
    assert not primary[0]["background"]
    assert primary[1]["background"][0]["speaker"]["id"] == 3
    assert not primary[2]["background"] and primary[2]["reply_to"] is None


def test_missing_forward_or_cyclic_reply_does_not_invent_a_recipient():
    graph = music_dialogue()
    graph = replace(
        graph,
        messages=tuple(
            replace(m, reply_to_message_id=m.message_id + 1) if m.message_id >= 10 else m
            for m in graph.messages
        ),
    )
    assert all(m["reply_to"] is None for m in dialogue_payload(graph, 10000)["primary_messages"])
