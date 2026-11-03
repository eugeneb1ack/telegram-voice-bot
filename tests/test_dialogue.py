import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_service import approved_group, identified_message, make_dependencies, personal_voice

from voice_bot.conversation import ConversationMemory
from voice_bot.progress import typing


def setup(settings):
    settings.context_summary_every = 3
    service, bot, paths = make_dependencies(settings)
    bot.send_message = AsyncMock()
    bot.send_chat_action = AsyncMock()
    return service, bot, paths


async def voice(service, bot, number, *, author=222, seconds=None, duration=10):
    message = identified_message(
        number,
        duration=duration,
        author=author,
        seconds=number * 15 if seconds is None else seconds,
    )
    service.transcriber.transcribe.return_value = f"  Мысль {number}.\n"
    await service.handle(message, bot)
    return message


async def text(service, number, words, *, author=222, seconds=None):
    message = identified_message(
        number, author=author, seconds=number * 15 if seconds is None else seconds
    )
    message.voice = None
    message.text = words
    await service.observe_text(message)


async def test_each_triplet_gets_one_reply_with_only_its_new_sources(settings):
    from test_rendering import ParsedReply

    service, bot, _ = setup(settings)
    for number in range(1, 7):
        message = await voice(service, bot, number)
        message.answer.assert_awaited_once()
        reply = message.answer.await_args.args[0]
        assert ParsedReply(reply).transcript == f"  Мысль {number}.\n"
        assert ("Коротко · 3 голосовых:" in reply) == (number % 3 == 0)
        assert service.summarizer.summarize.await_count == number // 3
        assert message.answer.await_args.kwargs["reply_parameters"].message_id == number
    calls = service.summarizer.summarize.await_args_list
    assert all(f"Мысль {i}." in calls[0].args[0] for i in (1, 2, 3))
    assert all(f"Мысль {i}." in calls[1].args[0] for i in (4, 5, 6))
    assert all(f"Мысль {i}." not in calls[1].args[0] for i in (1, 2, 3))
    graph = calls[1].kwargs["context"].dialogue
    assert not graph.has_previous_summary
    assert [m.text for m in graph.messages if m.kind == "summary"] == []
    assert graph.branches[0].message_ids == (4, 5, 6)
    assert all(c.kwargs["context"].is_conversation for c in calls)
    bot.send_message.assert_not_awaited()
    await service.close()


async def test_text_is_source_not_a_voice_counter_and_out_of_order_stt_keeps_event_order(settings):
    service, bot, _ = setup(settings)
    await text(service, 1, "Встреча завтра в пять.")
    entered, release = asyncio.Event(), asyncio.Event()

    async def transcribe(_, *, keywords=()):
        entered.set()
        await release.wait()
        return "Место встречи — парк."

    service.transcriber.transcribe.side_effect = transcribe
    task = asyncio.create_task(voice(service, bot, 2))
    await entered.wait()
    await text(service, 3, "Уточнение: в шесть.")
    release.set()
    await task
    service.transcriber.transcribe.side_effect = None
    await voice(service, bot, 4)
    await service.dialogue.maintain()
    bot.send_message.assert_not_awaited()
    last = await voice(service, bot, 5)
    await service.dialogue.maintain()
    source = service.summarizer.summarize.await_args.args[0]
    assert "Встреча" not in source
    assert source.index("Место") < source.index("Уточнение")
    assert "Коротко · 3 голосовых:" in last.answer.await_args.args[0]
    assert last.answer.await_args.kwargs["reply_parameters"].message_id == 5
    bot.send_message.assert_not_awaited()
    await service.close()


async def test_restart_between_second_and_third_restores_context_and_does_not_resend(settings):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1)
    await voice(service, bot, 2)
    await service.close()
    restarted, bot, _ = setup(settings)
    last = await voice(restarted, bot, 3)
    await restarted.dialogue.maintain()
    last.answer.assert_awaited_once()
    assert "Коротко · 3 голосовых:" in last.answer.await_args.args[0]
    bot.send_message.assert_not_awaited()
    assert all(f"Мысль {i}" in restarted.summarizer.summarize.await_args.args[0] for i in (1, 2, 3))
    await restarted.close()
    third, bot, _ = setup(settings)
    await third.dialogue.maintain()
    third.summarizer.summarize.assert_not_awaited()
    bot.send_message.assert_not_awaited()
    await third.close()


async def test_two_hour_gap_resets_counter_but_new_series_still_summarizes(settings):
    service, bot, _ = setup(settings)
    for number, seconds in [(1, 0), (2, 20), (3, 7200), (4, 7220), (5, 7240)]:
        await voice(service, bot, number, seconds=seconds)
        await service.dialogue.maintain()
        assert service.summarizer.summarize.await_count == int(number == 5)
    source = service.summarizer.summarize.await_args.args[0]
    assert "Мысль 1" not in source and "Мысль 2" not in source
    assert all(f"Мысль {i}" in source for i in (3, 4, 5))
    await service.close()


@pytest.mark.parametrize("separation", ["author", "topic", "chat", "forward", "private"])
async def test_conversation_does_not_mix_scopes(settings, separation):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1)
    await voice(service, bot, 2)
    third = identified_message(3, duration=10, seconds=45, forwarded=separation == "forward")
    if separation == "author":
        third.from_user = identified_message(3, author=444).from_user
    elif separation == "topic":
        third.message_thread_id = 7
        third.is_topic_message = True
    elif separation == "chat":
        third.chat.id = 555
    elif separation == "private":
        third.receiver_user = identified_message(3, author=444).from_user
    await service.handle(third, bot)
    await service.dialogue.maintain()
    bot.send_message.assert_not_awaited()
    await service.close()


async def test_later_voice_and_text_cannot_change_third_combined_reply(settings):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1)
    await voice(service, bot, 2)
    entered, release = asyncio.Event(), asyncio.Event()

    async def summarize(source, *, context):
        entered.set()
        await release.wait()
        assert "будущее" not in source
        return "Итог трёх."

    service.summarizer.summarize.side_effect = summarize
    third = asyncio.create_task(voice(service, bot, 3))
    await entered.wait()
    fourth = asyncio.create_task(voice(service, bot, 4))
    await text(service, 5, "будущее уточнение")
    release.set()
    last, continuation = await asyncio.gather(third, fourth)
    last.answer.assert_awaited_once()
    assert last.answer.await_args.kwargs["reply_parameters"].message_id == 3
    assert "Коротко · 3 голосовых:" in last.answer.await_args.args[0]
    assert "Коротко" not in continuation.answer.await_args.args[0]
    bot.send_message.assert_not_awaited()
    await service.close()


async def test_summary_failure_does_not_merge_old_block_into_next_summary(settings):
    service, bot, _ = setup(settings)
    service.summarizer.summarize.side_effect = TimeoutError("SECRET")
    for i in range(1, 4):
        message = await voice(service, bot, i)
        message.answer.assert_awaited_once()
    assert "Не удалось составить" in message.answer.await_args.args[0]
    assert "Мысль 3." in message.answer.await_args.args[0]
    service.summarizer.summarize.side_effect = None
    for i in range(4, 7):
        message = await voice(service, bot, i)
    assert service.summarizer.summarize.await_count == 2
    source = service.summarizer.summarize.await_args.args[0]
    assert all(f"Мысль {i}." not in source for i in (1, 2, 3))
    assert all(f"Мысль {i}." in source for i in (4, 5, 6))
    bot.send_message.assert_not_awaited()
    await service.close()


async def test_storage_failure_keeps_transcription_and_disables_deferred_summary(
    settings, monkeypatch
):
    service, bot, _ = setup(settings)

    def fail(snapshot):
        raise OSError("disk full")

    monkeypatch.setattr(service.memory, "_write", fail)
    for i in range(1, 4):
        message = await voice(service, bot, i)
        message.answer.assert_awaited_once()
    await service.dialogue.maintain()
    bot.send_message.assert_not_awaited()
    assert not service.memory.available
    await service.close()


async def test_corrupt_snapshot_is_preserved_without_breaking_voice_processing(settings, caplog):
    settings.context_state_file.write_text("PRIVATE invalid JSON")
    service, bot, _ = setup(settings)
    message = await voice(service, bot, 1)
    message.answer.assert_awaited_once()
    await service.close()
    assert settings.context_state_file.read_text() == "PRIVATE invalid JSON"
    assert "PRIVATE" not in caplog.text


async def test_bounded_snapshot_expiry_is_persisted_and_file_is_private(tmp_path):
    now = [10.0]
    path = tmp_path / "context.json"
    memory = ConversationMemory(15, 100, 2, clock=lambda: now[0], max_entries=4, path=path)
    for author in (1, 2, 3):
        for i in range(6):
            memory.remember((123, 0, author), i, (i,), "  мысль " * 50)
    await memory.checkpoint()
    assert path.stat().st_mode & 0o777 == 0o600
    restored = ConversationMemory(15, 100, 2, clock=lambda: now[0], max_entries=4, path=path)
    assert len(restored.sessions) == 2
    assert all(sum(len(e.text) for e in s.entries) <= 100 for s in restored.sessions.values())
    now[0] = 30
    await restored.checkpoint()
    assert json.loads(path.read_text())["sessions"] == []
    memory.clear()
    restored.clear()


async def test_blocked_group_context_is_discarded_before_model_call(settings, tmp_path):
    service, bot, _ = setup(settings)
    store = approved_group(service, tmp_path)
    for i in range(1, 4):
        message = identified_message(i, duration=10, seconds=i * 10)
        message.chat = SimpleNamespace(id=-100, type="supergroup", title="Test")
        await service.handle(message, bot)
    group = store.groups[-100]
    store.decide(-100, group.revision, "denied")
    await service.dialogue.maintain()
    bot.send_message.assert_not_awaited()
    assert not service.memory.sessions
    await service.close()


async def test_typing_repeats_and_stops_on_exit_and_errors_are_sanitized(caplog):
    bot = SimpleNamespace(send_chat_action=AsyncMock())
    async with typing(bot, 123, 5, interval=0.005):
        await asyncio.sleep(0.018)
    calls = bot.send_chat_action.await_count
    assert calls >= 2
    assert bot.send_chat_action.await_args.kwargs == dict(
        chat_id=123, message_thread_id=5, action="typing"
    )
    await asyncio.sleep(0.015)
    assert bot.send_chat_action.await_count == calls
    bot.send_chat_action.side_effect = RuntimeError("SECRET")
    async with typing(bot, 123):
        await asyncio.sleep(0.01)
    assert "SECRET" not in caplog.text


async def test_private_text_request_has_no_public_typing_or_context(settings):
    service, bot, _ = setup(settings)
    message, caller = personal_voice(111)
    await service.enqueue(message, bot, requester=caller)
    await service.close()
    bot.send_chat_action.assert_not_awaited()
    assert json.loads(settings.context_state_file.read_text())["sessions"] == []


async def test_many_interleaved_chats_topics_and_authors_never_mix_sources(settings):
    import re

    from aiogram.types import User

    service, bot, _ = setup(settings)
    keys = [
        (chat, topic, author) for chat in range(1, 6) for topic in (0, 7) for author in range(1, 5)
    ]
    messages = []
    for key in keys:
        base = key[2] * 1000 + key[1] * 100
        label = f"scope={key[0]}/{key[1]}/{key[2]}"
        for number in (1, 2):
            sent_at = identified_message(number, seconds=number * 15).date.timestamp()
            service.memory.remember(key, sent_at, (base + number,), f"{label} часть {number}")
        message = identified_message(base + 3, duration=10, author=key[2], seconds=45)
        message.chat.id = key[0]
        message.message_thread_id = key[1] or None
        message.is_topic_message = bool(key[1])
        message.from_user = User(id=key[2], first_name="Одинаковое имя", is_bot=False)
        messages.append(message)

    async def transcribe(message, _):
        scope = f"{message.chat.id}/{message.message_thread_id or 0}/{message.from_user.id}"
        return f"scope={scope} часть 3"

    service._transcribe = AsyncMock(side_effect=transcribe)
    running = peak = 0
    seen = []

    async def summarize(source, *, context):
        nonlocal running, peak
        scopes = set(re.findall(r"scope=(\d+/\d+/\d+)", source))
        assert len(scopes) == 1
        scope = scopes.pop()
        seen.append(scope)
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.002)
        running -= 1
        return scope

    service.summarizer.summarize.side_effect = summarize
    await asyncio.gather(*(service.handle(m, bot) for m in reversed(messages)))
    assert len(seen) == len(set(seen)) == 40
    assert peak <= settings.max_concurrent_jobs
    for message in messages:
        message.answer.assert_awaited_once()
        call = message.answer.await_args
        assert (
            f"{message.chat.id}/{message.message_thread_id or 0}/{message.from_user.id}"
            in call.args[0]
        )
        assert call.kwargs["reply_parameters"].message_id == message.message_id
    bot.send_message.assert_not_awaited()
    await service.close()


async def test_idle_maintenance_never_creates_summary_requests_or_messages(settings):
    service, bot, _ = setup(settings)
    for number in range(1, 10):
        service.memory.remember((123, 0, 222), number, (number,), "Речь")
    await service.dialogue.maintain()
    service.summarizer.summarize.assert_not_awaited()
    bot.send_message.assert_not_awaited()
    await service.close()


async def test_cancelled_checkpoint_finishes_before_newer_snapshot(tmp_path, monkeypatch):
    import threading

    memory = ConversationMemory(900, 800, 2, path=tmp_path / "context.json")
    entered, release = threading.Event(), threading.Event()
    original = memory._write

    def slow_write(snapshot):
        entered.set()
        release.wait(timeout=2)
        original(snapshot)

    monkeypatch.setattr(memory, "_write", slow_write)
    memory.remember((1, 0, 1), 1, (1,), "Первая")
    first = asyncio.create_task(memory.checkpoint())
    await asyncio.to_thread(entered.wait, 2)
    first.cancel()
    memory.remember((1, 0, 1), 2, (2,), "Вторая")
    second = asyncio.create_task(memory.checkpoint())
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second
    assert "Вторая" in memory.path.read_text()
    memory.clear()


async def test_router_collects_only_allowed_human_text_and_never_replies(settings):
    from aiogram import Bot, Dispatcher
    from aiogram.types import Update

    from voice_bot.handlers import build_router

    service, _, _ = setup(settings)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(service))
    bot = Bot("123456:test-token")
    try:
        for index, words, is_bot in [
            (1, "Уточнение плана", False),
            (2, "/unknown", False),
            (3, "bot text", True),
        ]:
            update = Update.model_validate(
                {
                    "update_id": index,
                    "message": {
                        "message_id": index,
                        "date": 1800000000,
                        "chat": {"id": 123, "type": "private"},
                        "from": {"id": 222, "is_bot": is_bot, "first_name": "Имя"},
                        "text": words,
                    },
                }
            )
            await dispatcher.feed_update(bot, update)
        entries = service.memory.sessions[(123, 0, 222)].entries
        assert [e.text for e in entries] == ["Уточнение плана"]
        service.transcriber.transcribe.assert_not_awaited()
        service.summarizer.summarize.assert_not_awaited()
    finally:
        await service.close()
        await bot.session.close()


async def test_disabled_summaries_still_remove_expired_text_from_disk(settings):
    service, bot, _ = setup(settings)
    settings.context_summary_every = 0
    now = [10.0]
    service.memory.clock = lambda: now[0]
    await text(service, 1, "Истекающий текст")
    assert "Истекающий текст" in settings.context_state_file.read_text()
    now[0] += settings.context_ttl_seconds + 1
    service.start(bot)
    for _ in range(100):
        if json.loads(settings.context_state_file.read_text())["sessions"] == []:
            break
        await asyncio.sleep(0.005)
    assert json.loads(settings.context_state_file.read_text())["sessions"] == []
    bot.send_message.assert_not_awaited()
    await service.close()


async def test_legacy_separate_summary_watermark_does_not_repeat_an_old_triple(settings):
    service, bot, _ = setup(settings)
    for number in (1, 2, 3):
        sent_at = identified_message(number, seconds=number * 15).date.timestamp()
        service.memory.remember((123, 0, 222), sent_at, (number,), f"Старое {number}")
    service.memory.sessions[(123, 0, 222)].claimed_through = 3
    for number in (4, 5, 6):
        message = await voice(service, bot, number)
        message.answer.assert_awaited_once()
        assert ("Коротко · 3 голосовых:" in message.answer.await_args.args[0]) == (number == 6)
    call = service.summarizer.summarize.await_args
    assert "Старое" not in call.args[0]
    assert call.kwargs["context"].recent_summaries == ()
    await service.close()
