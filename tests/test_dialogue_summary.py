import json
from dataclasses import replace

import httpx
import pytest

from voice_bot.dialogue_graph import DialogueBranch, DialogueMessage, SharedDialogue
from voice_bot.dialogue_summary import dialogue_payload
from voice_bot.errors import ProviderError
from voice_bot.providers import APIClient, HTTPSummarizer, SummaryContext


def music_dialogue():
    return SharedDialogue(
        1,
        (
            DialogueMessage(1, 2, "Борис", None, "text", "UnrelatedOldTopic"),
            DialogueMessage(2, 3, "Вера", None, "video_note", "UnrelatedVideoSpeech"),
            DialogueMessage(10, 1, "Антон", None, "voice", "Люблю слушать джаз.", 1),
            DialogueMessage(11, 1, "Антон", None, "voice", "Играть не умею.", 2),
            DialogueMessage(12, 1, "Антон", None, "voice", "Пойду на концерт."),
        ),
        (DialogueBranch("b1", 2, "Борис", None, (10, 11, 12)),),
    )


def response(value):
    return httpx.Response(
        200,
        json={
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": json.dumps(value)}],
                }
            ],
        },
    )


async def test_only_current_voices_and_recipient_names_reach_one_model_request(settings):
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        primary = json.loads(payload["input"])["primary_messages"]
        assert [item["message_id"] for item in primary] == [10, 11, 12]
        assert primary[0]["reply_recipient"] == {"id": 2, "name": "Борис"}
        assert primary[1]["reply_recipient"] == {"id": 3, "name": "Вера"}
        assert primary[2]["reply_recipient"] is None
        assert "Unrelated" not in payload["input"]
        return response({"summary": "Антон любит джаз, не умеет играть и собирается на концерт."})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        summary = await HTTPSummarizer(APIClient(client, settings)).summarize(
            "Три записи", context=SummaryContext(author_name="Антон", dialogue=music_dialogue())
        )
    assert summary.startswith("Антон любит джаз")
    assert len(calls) == 1


@pytest.mark.parametrize("ids", [(1, 10), (99,), (10, 10)])
def test_foreign_missing_or_duplicate_primary_is_rejected(ids):
    graph = replace(music_dialogue(), branches=(DialogueBranch("b", None, None, None, ids),))
    with pytest.raises(ProviderError, match="авторство"):
        dialogue_payload(graph, 10000)


def test_request_limit_counts_current_speech_only():
    graph = music_dialogue()
    total = sum(len(m.text) for m in graph.messages if m.message_id >= 10)
    dialogue_payload(graph, total)
    with pytest.raises(ProviderError, match="длинный"):
        dialogue_payload(graph, total - 1)


@pytest.mark.parametrize(
    "value", [None, {}, {"summary": 4}, {"summary": ""}, {"summary": "x", "extra": 1}]
)
async def test_invalid_structured_summary_is_rejected(settings, value):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response(value))
    ) as client:
        with pytest.raises(ProviderError):
            await HTTPSummarizer(APIClient(client, settings)).summarize(
                "Три записи", context=SummaryContext(author_name="Антон", dialogue=music_dialogue())
            )


@pytest.mark.parametrize(
    "name,username,label",
    [
        (" Анна\nПетрова ", "handle", "Анна Петрова"),
        (None, "handle", "@handle"),
        ("<example/>", "handle", "<example/>"),
    ],
)
def test_display_names_are_literal_with_username_fallback(name, username, label):
    graph = music_dialogue()
    graph = replace(
        graph,
        messages=tuple(
            replace(m, author_name=name, author_username=username) if m.message_id >= 10 else m
            for m in graph.messages
        ),
    )
    assert dialogue_payload(graph, 10000)["primary_messages"][0]["speaker"]["name"] == label


async def test_first_long_voice_has_current_author_name_without_saved_history(settings):
    graph = SharedDialogue(
        1,
        (DialogueMessage(10, 1, None, None, "voice", "Пойду на концерт."),),
        (DialogueBranch("b", None, None, None, (10,)),),
    )

    def respond(request):
        data = json.loads(json.loads(request.content)["input"])
        assert data["primary_messages"][0]["speaker"]["name"] == "Антон"
        return response({"summary": "Антон собирается на концерт."})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assert (
            await HTTPSummarizer(APIClient(client, settings)).summarize(
                "Пойду на концерт.", context=SummaryContext(author_name="Антон", dialogue=graph)
            )
            == "Антон собирается на концерт."
        )
