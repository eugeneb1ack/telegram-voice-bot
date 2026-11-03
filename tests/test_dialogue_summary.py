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
        assert "messages" not in data
        b1, b2 = data["branches"]
        assert [m["text"] for m in b1["primary_messages"]] == [
            "Люблю слушать джаз.",
            "Играть не умею, только одну мелодию пальцем.",
        ]
        assert {m["author_id"] for m in b1["primary_messages"]} == {1}
        assert b1["background_messages"][0]["author_id"] == 2
        assert b1["background_messages"][0]["role"] == "other_speaker"
        assert "recipient_name" not in b1  # addressee isn't a speaker instruction
        assert not b2["background_messages"]
        assert [m["message_id"] for m in b2["primary_messages"]] == [12]
        assert "профессиональный" not in payload["input"]
        assert json.loads(payload["input"])["context"]["speaker_name"] == "Антон"
        assert all("author_name" not in m for m in b1["primary_messages"])
        return response(
            {
                "b1": "Любит слушать джаз. Играть не умеет, кроме одной мелодии одним пальцем.",
                "b2": "Собирается попробовать её сыграть.",
            }
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await HTTPSummarizer(APIClient(client, settings)).summarize(
            "Три записи", context=SummaryContext(author_name="Антон", dialogue=graph)
        )
    assert result == (
        "• Антон → Борис:\nЛюбит слушать джаз. Играть не умеет, кроме одной мелодии одним пальцем."
        "\n\n• Антон:\nСобирается попробовать её сыграть."
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


def test_background_shared_by_branches_counts_towards_request_limit():
    graph = music_dialogue()
    graph = replace(
        graph,
        branches=(graph.branches[0], replace(graph.branches[1], context_message_ids=(1,))),
    )
    payload = dialogue_payload(graph, 10000)
    total = sum(
        len(m["text"])
        for b in payload["branches"]
        for field in ("primary_messages", "background_messages")
        for m in b[field]
    )
    dialogue_payload(graph, total)
    with pytest.raises(ProviderError, match="длинный"):
        dialogue_payload(graph, total - 1)


def test_unknown_background_author_never_becomes_same_speaker():
    graph = music_dialogue()
    graph = replace(
        graph, messages=(replace(graph.messages[0], author_id=None), *graph.messages[1:])
    )
    assert dialogue_payload(graph, 10000)["branches"][0]["background_messages"][0]["role"] == (
        "other_speaker"
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
        return response({"b2": body})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await HTTPSummarizer(APIClient(client, settings)).summarize(
            "Надеюсь получить понятный пересказ.",
            context=SummaryContext(author_name=name, author_username=username, dialogue=graph),
        )
    assert result == body
