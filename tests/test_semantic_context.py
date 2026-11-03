import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from test_dialogue import setup, voice
from test_shared_dialogue import memory, ref, remember

from voice_bot.conversation import Entry
from voice_bot.dialogue_graph import shared_dialogue
from voice_bot.providers import APIClient, HTTPSummarizer, SummaryContext
from voice_bot.semantic_context import apply_links, select_continuations


def continuation():
    mem = memory()
    remember(mem, 1)
    mem.sessions[(-100, 0, 1)].entries[0] = Entry(
        (1,), 1, "Выбрал синий велосипед. У него широкие колёса."
    )
    mem.remember((-100, 0, 2), 2, (2,), "Встреча в библиотеке завтра в семь.", kind="text")
    sources = [
        Entry((10,), 10, "У этого велосипеда ещё удобное седло."),
        Entry((11,), 11, "Куплю его завтра."),
        Entry((12,), 12, "Насос подберу отдельно."),
    ]
    return shared_dialogue(mem, (-100, 0, 1), sources, (), 12, 12)


def response(data):
    return httpx.Response(
        200,
        json={
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": json.dumps(data, ensure_ascii=False)}
                    ],
                }
            ],
        },
    )


async def test_only_semantically_selected_source_reaches_summary(settings):
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if payload["text"]["format"]["name"] == "dialogue_context_links":
            return response(
                {
                    "links": [
                        {
                            "message_id": 10,
                            "source_id": 1,
                            "relation": "specific_reference",
                            "confidence": "high",
                            "current_quote": "У этого велосипеда",
                            "source_quote": "Выбрал синий велосипед",
                        }
                    ]
                }
            )
        graph = json.loads(payload["input"])["dialogue"]
        assert {m["message_id"] for m in graph["messages"]} == {1, 10, 11, 12}
        assert "candidates" not in graph
        assert graph["branches"][0]["context_message_ids"] == [1]
        return response({"b1": "У выбранного велосипеда удобное седло; купит его завтра."})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        await HTTPSummarizer(APIClient(client, settings)).summarize(
            "Новые сообщения", context=SummaryContext(dialogue=continuation(), is_conversation=True)
        )
    assert len(calls) == 2


def link(**overrides):
    return {
        "message_id": 10,
        "source_id": 1,
        "relation": "specific_reference",
        "confidence": "high",
        "current_quote": "У этого велосипеда",
        "source_quote": "Выбрал синий велосипед",
        **overrides,
    }


def test_new_topic_has_no_history_when_selector_returns_no_links():
    graph = apply_links(continuation(), {"links": []})
    assert [m.message_id for m in graph.messages] == [10, 11, 12]
    assert not graph.candidates and not graph.branches[0].context_message_ids


@pytest.mark.parametrize(
    "bad",
    [
        {"source_id": 999},
        {"message_id": 99},
        {"source_id": True},
        {"confidence": "ambiguous"},
        {"current_quote": "не было в речи"},
        {"source_quote": "поддельный источник"},
        {"relation": "same_topic"},
        {"source_quote": "он"},
        {"message_id": []},
    ],
)
def test_unproven_or_invalid_link_cannot_admit_history(bad):
    graph = apply_links(continuation(), {"links": [link(**bad)]})
    assert [m.message_id for m in graph.messages] == [10, 11, 12]
    assert not graph.candidates


def test_future_candidate_is_rejected_and_explicit_reply_cannot_be_overridden():
    graph = continuation()
    future = replace(graph.candidates[0], message_id=11)
    graph = replace(graph, candidates=(future,))
    assert apply_links(graph, {"links": [link(source_id=11)]}).branches[0].context_message_ids == ()
    graph = continuation()
    graph = replace(
        graph, messages=(replace(graph.messages[0], reply_to_message_id=2), *graph.messages[1:])
    )
    assert apply_links(graph, {"links": [link()]}).branches[0].context_message_ids == ()


def test_semantic_link_does_not_invent_recipient_or_reply_edge():
    graph = apply_links(continuation(), {"links": [link()]})
    assert graph.branches[0].recipient_id is None
    assert not graph.branches[0].is_reply
    assert all(m.reply_to_message_id is None for m in graph.messages)
    assert graph.branches[0].message_ids == (10, 11, 12)


async def test_timeout_falls_back_to_current_block_and_cancellation_propagates(
    settings, monkeypatch
):
    from types import SimpleNamespace

    from voice_bot import semantic_context

    monkeypatch.setattr(semantic_context, "SELECTION_TIMEOUT", 0.01)

    async def hang(*args):
        await asyncio.Event().wait()

    api = SimpleNamespace(settings=settings, post=AsyncMock(side_effect=hang))
    graph = await select_continuations(api, continuation())
    assert not graph.candidates and [m.message_id for m in graph.messages] == [10, 11, 12]
    api.post.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await select_continuations(api, continuation())


@pytest.mark.parametrize(
    "body", [{}, {"status": "incomplete"}, {"status": "completed", "output": []}]
)
async def test_invalid_selector_response_does_not_block_summary(settings, body):
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if payload["text"]["format"]["name"] == "dialogue_context_links":
            return httpx.Response(200, json=body)
        graph = json.loads(payload["input"])["dialogue"]
        assert "candidates" not in graph
        assert [m["message_id"] for m in graph["messages"]] == [10, 11, 12]
        return response({"b1": "Пересказ только новых записей."})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await HTTPSummarizer(APIClient(client, settings)).summarize(
            "Текст", context=SummaryContext(dialogue=continuation())
        )
    assert "Пересказ" in result and len(calls) == 2


def test_candidate_pool_is_bounded_and_never_crosses_chat_topic_time_or_forward_scope():
    mem = memory(max_entries=50)
    for mid in range(1, 70):
        mem.remember(
            (-100, 0, 1 if mid % 2 else mid), mid, (mid,), f"Источник {mid}. " + "текст " * 300
        )
    mem.remember((-200, 0, 1), 1, (71,), "Другой чат")
    mem.remember((-100, 7, 1), 1, (72,), "Другая тема")
    mem.remember((-100, 0, 1, 1), 1, (73,), "Пересылка")
    mem.remember((-100, 0, 77), -1000, (74,), "Истёкший источник")
    mem.remember((-100, 0, 78), 100, (75,), "Будущее время")
    mem.remember((-100, 0, 79), 1, (99,), "Будущий ID")
    graph = shared_dialogue(mem, (-100, 0, 1), [Entry((80,), 80, "Новая речь")], (), 80, 80)
    assert len(graph.candidates) <= 12
    assert sum(len(n.text) for n in graph.candidates) <= 6000
    assert all(n.message_id < 70 for n in graph.candidates)
    assert any(n.author_id == 1 for n in graph.candidates)


def test_no_reply_can_continue_source_of_current_explicit_reply():
    mem = memory()
    reference = ref(10, 1, 2)
    remember(mem, 10, refs=(reference,))
    sources = [
        Entry((10,), 10, "Ответ собеседнику", reply_contexts=(reference,)),
        Entry((11,), 11, "Продолжение мысли"),
    ]
    graph = shared_dialogue(mem, (-100, 0, 1), sources, (reference,), 11, 11)
    assert any(n.message_id == 1 for n in graph.candidates)
    graph = apply_links(
        graph,
        {
            "links": [
                link(message_id=11, current_quote="Продолжение мысли", source_quote=reference.text)
            ]
        },
    )
    assert graph.branches[0].is_reply
    assert graph.branches[1].context_message_ids == (1,)
    assert not graph.branches[1].is_reply


async def test_thirteen_minute_gap_after_restart_starts_fresh_triplet(settings):
    service, bot, _ = setup(settings)
    for mid, seconds in [(1, 0), (2, 30)]:
        await voice(service, bot, mid, seconds=seconds)
    await service.close()
    service, bot, _ = setup(settings)
    lone = await voice(service, bot, 3, seconds=809)
    assert "Коротко" not in lone.answer.await_args.args[0]
    for mid, seconds in [(4, 820), (5, 830)]:
        last = await voice(service, bot, mid, seconds=seconds)
    assert "Коротко · 3 голосовых:" in last.answer.await_args.args[0]
    context = service.summarizer.summarize.await_args.kwargs["context"]
    assert [n.message_id for n in context.dialogue.messages] == [3, 4, 5]
    assert not context.dialogue.candidates
    await service.close()


async def test_standalone_long_voice_uses_same_semantic_gate(settings):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1, seconds=0, duration=40)
    last = await voice(service, bot, 2, seconds=30, duration=40)
    assert "Коротко:" in last.answer.await_args.args[0]
    graph = service.summarizer.summarize.await_args.kwargs["context"].dialogue
    assert [n.message_id for n in graph.messages] == [2]
    assert [n.message_id for n in graph.candidates] == [1]
    await service.close()
