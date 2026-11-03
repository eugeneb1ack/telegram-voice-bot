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


async def test_third_and_sixth_voice_summarize_all_sources_without_intermediate_calls(settings):
    service, bot, _ = setup(settings)
    for number in range(1, 7):
        message = await voice(service, bot, number)
        assert "Расшифровка:" in message.answer.await_args.args[0]
        await service.dialogue.process_due(bot)
        assert bot.send_message.await_count == number // 3
    calls = service.summarizer.summarize.await_args_list
    assert len(calls) == 2
    assert all(f"Мысль {i}." in calls[0].args[0] for i in (1, 2, 3))
    assert all(f"Мысль {i}." in calls[1].args[0] for i in range(1, 7))
    assert [c.kwargs["reply_parameters"].message_id for c in bot.send_message.await_args_list] == [
        3,
        6,
    ]
    assert all(c.kwargs["context"].is_conversation for c in calls)
    await service.close()


async def test_text_is_source_not_a_voice_counter_and_out_of_order_stt_keeps_event_order(settings):
    service, bot, _ = setup(settings)
    await text(service, 1, "Встреча завтра в пять.")
    entered, release = asyncio.Event(), asyncio.Event()

    async def transcribe(_):
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
    await service.dialogue.process_due(bot)
    bot.send_message.assert_not_awaited()
    await voice(service, bot, 5)
    await service.dialogue.process_due(bot)
    source = service.summarizer.summarize.await_args.args[0]
    assert source.index("Встреча") < source.index("Место") < source.index("Уточнение")
    assert "голосовых: 3, текстовых: 2" in bot.send_message.await_args.kwargs["text"]
    assert bot.send_message.await_args.kwargs["reply_parameters"].message_id == 5
    await service.close()


async def test_restart_between_second_and_third_restores_context_and_does_not_resend(settings):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1)
    await voice(service, bot, 2)
    await service.close()
    restarted, bot, _ = setup(settings)
    await voice(restarted, bot, 3)
    await restarted.dialogue.process_due(bot)
    bot.send_message.assert_awaited_once()
    assert all(f"Мысль {i}" in restarted.summarizer.summarize.await_args.args[0] for i in (1, 2, 3))
    await restarted.close()
    third, bot, _ = setup(settings)
    await third.dialogue.process_due(bot)
    third.summarizer.summarize.assert_not_awaited()
    bot.send_message.assert_not_awaited()
    await third.close()


async def test_three_minute_gap_resets_counter_but_new_series_still_summarizes(settings):
    service, bot, _ = setup(settings)
    for number, seconds in [(1, 0), (2, 20), (3, 250), (4, 270), (5, 290)]:
        await voice(service, bot, number, seconds=seconds)
        await service.dialogue.process_due(bot)
        assert bot.send_message.await_count == int(number == 5)
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
    elif separation == "chat":
        third.chat.id = 555
    elif separation == "private":
        third.receiver_user = identified_message(3, author=444).from_user
    await service.handle(third, bot)
    await service.dialogue.process_due(bot)
    bot.send_message.assert_not_awaited()
    await service.close()


async def test_fourth_and_fifth_do_not_cancel_third_summary_or_consume_next_group(settings):
    service, bot, _ = setup(settings)
    for i in range(1, 4):
        await voice(service, bot, i)
    entered, release = asyncio.Event(), asyncio.Event()

    async def summarize(source, *, context):
        entered.set()
        await release.wait()
        return "Итог трёх."

    service.summarizer.summarize.side_effect = summarize
    task = asyncio.create_task(service.dialogue.process_due(bot))
    await entered.wait()
    await voice(service, bot, 4)
    await voice(service, bot, 5)
    release.set()
    await task
    assert bot.send_message.await_args.kwargs["reply_parameters"].message_id == 3
    await service.dialogue.process_due(bot)
    bot.send_message.assert_awaited_once()
    await voice(service, bot, 6)
    await service.dialogue.process_due(bot)
    assert bot.send_message.await_args.kwargs["reply_parameters"].message_id == 6
    await service.close()


async def test_ambiguous_send_is_not_retried_on_restart_and_transcripts_survive(settings):
    service, bot, _ = setup(settings)
    for i in range(1, 4):
        message = await voice(service, bot, i)
        message.answer.assert_awaited_once()
    bot.send_message.side_effect = TimeoutError("private content must not be logged")
    await service.dialogue.process_due(bot)
    await service.close()
    restarted, bot, _ = setup(settings)
    await restarted.dialogue.process_due(bot)
    restarted.summarizer.summarize.assert_not_awaited()
    await restarted.close()


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
    await service.dialogue.process_due(bot)
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
    await service.dialogue.process_due(bot)
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

    from voice_bot.conversation import Anchor

    service, bot, _ = setup(settings)
    keys = [
        (chat, topic, author) for chat in range(1, 6) for topic in (0, 7) for author in range(1, 5)
    ]
    # Same local message IDs and visible names deliberately collide across scopes.
    for number in range(1, 4):
        for key in reversed(keys):
            label = f"scope={key[0]}/{key[1]}/{key[2]}"
            service.memory.remember(
                key,
                number * 15,
                (number,),
                f"{label} часть {number}",
                anchor=Anchor(number, "Одинаковое имя", None, "private"),
            )
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
    while len(seen) < len(keys):
        await service.dialogue.process_due(bot)
        assert len(service.dialogue.jobs) <= settings.max_concurrent_jobs
    assert len(seen) == len(set(seen)) == 40
    assert peak == settings.max_concurrent_jobs
    for call in bot.send_message.await_args_list:
        scope = re.search(r"\n\n(\d+)/(\d+)/(\d+)$", call.kwargs["text"])
        assert scope
        assert call.kwargs["chat_id"] == int(scope[1])
        assert (call.kwargs["message_thread_id"] or 0) == int(scope[2])
        assert call.kwargs["reply_parameters"].message_id == 3
    await service.close()


async def test_summary_workers_are_bounded_while_voice_capacity_is_busy(settings):
    from voice_bot.conversation import Anchor

    service, bot, _ = setup(settings)
    for chat in range(1, 21):
        for number in range(1, 4):
            service.memory.remember(
                (chat, 0, 222),
                number,
                (number,),
                "Речь",
                anchor=Anchor(number, "Имя", None, "private"),
            )
    for _ in range(settings.max_concurrent_jobs):
        await service._capacity.acquire()
    for _ in range(3):
        await service.dialogue.process_due(bot, wait=False)
    assert len(service.dialogue.jobs) == settings.max_concurrent_jobs
    service.summarizer.summarize.assert_not_awaited()
    for _ in range(settings.max_concurrent_jobs):
        service._capacity.release()
    await service.dialogue.process_due(bot)
    assert service.summarizer.summarize.await_count == settings.max_concurrent_jobs
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
