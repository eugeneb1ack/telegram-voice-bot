"""Telegram reply threads must not masquerade as forum topics."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_service import identified_message, make_dependencies

from voice_bot.batching import batch_key
from voice_bot.service import conversation_position


def scoped_voice(number, *, thread=None, topic=False, forwarded=False):
    message = identified_message(number, duration=10, seconds=number * 15, forwarded=forwarded)
    message.chat = SimpleNamespace(id=-100, type="supergroup", title="Test", is_forum=topic)
    message.message_thread_id = thread
    message.is_topic_message = topic
    message.voice.file_unique_id = f"audio-{number}"
    return message


@pytest.mark.parametrize("topic_flag", [False, None])
def test_ordinary_reply_thread_shares_conversation_with_unthreaded_messages(topic_flag):
    reply = scoped_voice(1, thread=830315)
    reply.is_topic_message = topic_flag
    ordinary = scoped_voice(2)
    assert conversation_position(reply)[0] == conversation_position(ordinary)[0] == (-100, 0, 222)


async def test_reply_first_voice_is_counted_on_third_not_fourth(settings):
    settings.context_summary_every = 3
    service, bot, _ = make_dependencies(settings)
    bot.send_chat_action = AsyncMock()
    try:
        for number in range(1, 5):
            message = scoped_voice(number, thread=830315 if number == 1 else None)
            service.transcriber.transcribe.return_value = f"Мысль {number}."
            await service.handle(message, bot)
            message.answer.assert_awaited_once()
            assert ("Коротко · 3 голосовых:" in message.answer.await_args.args[0]) == (number == 3)
            assert service.summarizer.summarize.await_count == int(number >= 3)
        source = service.summarizer.summarize.await_args.args[0]
        assert all(f"Мысль {number}." in source for number in (1, 2, 3))
        assert "Мысль 4." not in source
        assert len(service.memory.sessions) == 1
    finally:
        await service.close()


async def test_genuine_forum_topics_keep_independent_triplets(settings):
    settings.context_summary_every = 3
    service, bot, _ = make_dependencies(settings)
    bot.send_chat_action = AsyncMock()
    try:
        for number, topic in [(1, 10), (2, 20), (3, 10), (4, 20), (5, 10), (6, 20)]:
            message = scoped_voice(number, thread=topic, topic=True)
            service.transcriber.transcribe.return_value = f"Тема {topic}, запись {number}."
            await service.handle(message, bot)
            assert ("Коротко · 3 голосовых:" in message.answer.await_args.args[0]) == (
                number in (5, 6)
            )
        calls = service.summarizer.summarize.await_args_list
        assert len(calls) == 2
        assert all(f"запись {number}." in calls[0].args[0] for number in (1, 3, 5))
        assert all(f"запись {number}." in calls[1].args[0] for number in (2, 4, 6))
        assert "Тема 20" not in calls[0].args[0]
        assert "Тема 10" not in calls[1].args[0]
    finally:
        await service.close()


def test_reply_threads_do_not_split_forwarded_batches():
    first = scoped_voice(1, thread=830315, forwarded=True)
    second = scoped_voice(2, forwarded=True)
    assert batch_key(first) == batch_key(second)
    first.is_topic_message = second.is_topic_message = True
    first.chat.is_forum = second.chat.is_forum = True
    first.message_thread_id, second.message_thread_id = 10, 20
    assert batch_key(first) != batch_key(second)


async def test_legacy_prepared_reply_joins_next_two_voices_after_upgrade(settings):
    settings.context_summary_every = 3
    service, bot, _ = make_dependencies(settings)
    bot.send_chat_action = AsyncMock()
    first = scoped_voice(1, thread=830315)
    prepared = {
        "summary": "Уже готовый итог.",
        "transcript": "Мысль 1.",
        "title": "Коротко:",
        "valid": [True],
        "key": [-100, 830315, 222],
        "times": [first.date.timestamp()],
        "dialogue": False,
    }
    try:
        await service._deliver_prepared([first], bot, prepared)
        first.answer.assert_awaited_once()
        assert "Уже готовый итог." in first.answer.await_args.args[0]
        assert prepared["key"] == [-100, 830315, 222]
        assert set(service.memory.sessions) == {(-100, 0, 222)}
        for number in (2, 3):
            message = scoped_voice(number)
            service.transcriber.transcribe.return_value = f"Мысль {number}."
            await service.handle(message, bot)
            assert ("Коротко · 3 голосовых:" in message.answer.await_args.args[0]) == (number == 3)
        assert service.transcriber.transcribe.await_count == 2
        service.summarizer.summarize.assert_awaited_once()
        source = service.summarizer.summarize.await_args.args[0]
        assert all(f"Мысль {number}." in source for number in (1, 2, 3))
    finally:
        await service.close()


@pytest.mark.parametrize(
    "topic,forwarded,stored_key,expected_key",
    [
        (True, False, [-100, 42, 222], (-100, 42, 222)),
        (False, True, [-100, 42, 333, 1], (-100, 0, 333, 1)),
        (False, False, None, None),
    ],
)
async def test_prepared_scope_preserves_real_topics_forwarding_and_absence(
    settings, topic, forwarded, stored_key, expected_key
):
    service, bot, _ = make_dependencies(settings)
    first = scoped_voice(1, thread=42, topic=topic, forwarded=forwarded)
    prepared = {
        "summary": None,
        "transcript": "Сохранённая расшифровка.",
        "title": "Коротко:",
        "valid": [True],
        "key": stored_key,
        "times": [first.date.timestamp()],
        "dialogue": False,
    }
    try:
        await service._deliver_prepared([first], bot, prepared)
        assert set(service.memory.sessions) == (
            {expected_key} if expected_key is not None else set()
        )
        service.transcriber.transcribe.assert_not_awaited()
        service.summarizer.summarize.assert_not_awaited()
    finally:
        await service.close()
