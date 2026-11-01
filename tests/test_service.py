import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError, TelegramRetryAfter
from aiogram.methods import SendMessage, SendRichMessage
from aiogram.types import (
    Chat,
    MessageOriginChannel,
    MessageOriginChat,
    MessageOriginHiddenUser,
    MessageOriginUser,
    Update,
    User,
)

from voice_bot.audio import BoundedFile
from voice_bot.errors import ProviderError, UserFacingError
from voice_bot.handlers import build_router
from voice_bot.service import VoiceService, conversation_position, summary_context


def make_message(*, duration=30, size=100, chat_id=123):
    return SimpleNamespace(
        voice=SimpleNamespace(duration=duration, file_size=size, file_id="voice-id"),
        chat=SimpleNamespace(id=chat_id),
        message_id=42,
        answer=AsyncMock(return_value=SimpleNamespace(message_id=1000)),
    )


def make_dependencies(settings):
    paths = []

    async def download(path, output, **kwargs):
        output.write(b"mock-ogg")

    async def converter(source, destination, **kwargs):
        paths.append(source.parent)
        assert source.read_bytes() == b"mock-ogg"
        destination.write_bytes(b"mock-wav")

    bot = SimpleNamespace(
        get_file=AsyncMock(return_value=SimpleNamespace(file_path="voice/file.ogg", file_size=100)),
        download_file=AsyncMock(side_effect=download),
        send_rich_message=AsyncMock(return_value=SimpleNamespace(message_id=1000)),
        copy_message=AsyncMock(return_value=SimpleNamespace(message_id=2000)),
        delete_message=AsyncMock(),
    )
    transcriber = SimpleNamespace(transcribe=AsyncMock(return_value="Полный <текст> & да."))
    summarizer = SimpleNamespace(summarize=AsyncMock(return_value="Кратко."))
    return VoiceService(settings, transcriber, summarizer, converter), bot, paths


def test_author_comes_from_telegram_metadata():
    message = make_message()
    message.from_user = SimpleNamespace(full_name="Настоящее имя", username="actual_user")
    assert summary_context(message).author_name == "Настоящее имя"
    assert summary_context(message).author_username == "actual_user"
    message.sender_chat = SimpleNamespace(title="Анонимная группа", username=None)
    assert summary_context(message).author_name == "Анонимная группа"


@pytest.mark.parametrize("kind", ["user", "hidden", "chat", "channel", "unknown"])
def test_forwarded_voice_uses_original_author_not_forwarder(kind):
    message = make_message()
    message.from_user = SimpleNamespace(full_name="Переславший", username="forwarder")
    original = User(id=333, is_bot=False, first_name="Оригинальный автор", username="original")
    channel = Chat(id=-100333, type="channel", title="Исходный канал", username="source_channel")
    origins = {
        "user": MessageOriginUser(date=0, sender_user=original),
        "hidden": MessageOriginHiddenUser(date=0, sender_user_name="Скрытый автор"),
        "chat": MessageOriginChat(date=0, sender_chat=channel),
        "channel": MessageOriginChannel(date=0, chat=channel, message_id=7),
        "unknown": SimpleNamespace(type="unknown"),
    }
    message.forward_origin = origins[kind]
    context = summary_context(message)
    assert context.is_forwarded
    assert context.author_name != "Переславший"
    assert context.author_username != "forwarder"
    if kind == "unknown":
        assert context.author_name is None and context.author_username is None
        assert context.author_source == "unknown"
    else:
        assert context.author_source == "forwarded"
        assert (
            context.author_name
            == {
                "user": "Оригинальный автор",
                "hidden": "Скрытый автор",
                "chat": "Исходный канал",
                "channel": "Исходный канал",
            }[kind]
        )


async def test_summary_gets_author_chat_and_bounded_reply_context(settings):
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    message.from_user = SimpleNamespace(id=111, full_name="Анна", username="anna")
    message.chat.type, message.chat.title = "supergroup", "Рабочая группа"
    message.reply_to_message = SimpleNamespace(text="План встречи " * 200, caption=None)
    await service.handle(message, bot)
    context = service.summarizer.summarize.await_args.kwargs["context"]
    assert context.author_name == "Анна" and context.author_username == "anna"
    assert context.author_source == "sender" and not context.is_forwarded
    assert context.chat_type == "supergroup" and context.chat_title == "Рабочая группа"
    assert context.duration_seconds == 30
    assert len(context.reply_to_text) == 1000
    assert context.reply_to_text == message.reply_to_message.text[:1000]


def test_missing_identity_and_username_only_fallback():
    message = make_message()
    assert summary_context(message).author_source == "unknown"
    message.from_user = SimpleNamespace(full_name="", username="nickname")
    context = summary_context(message)
    assert context.author_name is None and context.author_username == "nickname"
    assert context.author_source == "sender"


async def test_voice_reply_collapsible_transcript_and_cleanup(settings):
    service, bot, paths = make_dependencies(settings)
    message = make_message()
    await service.handle(message, bot)
    message.answer.assert_awaited_once()
    bot.send_rich_message.assert_not_awaited()
    call = message.answer.await_args
    assert "<blockquote expandable><b>Расшифровка:</b>\n" in call.args[0]
    assert "Полный &lt;текст&gt; &amp; да.</blockquote>" in call.args[0]
    assert call.kwargs["reply_parameters"].message_id == 42
    assert call.kwargs["parse_mode"] == "HTML"
    assert not paths[0].exists()
    assert service.active_jobs == 0
    assert call.kwargs["reply_parameters"].allow_sending_without_reply is False


@pytest.mark.parametrize(
    "duration,expected_summary", [(0, 0), (14, 0), (15, 0), (20, 0), (29, 0), (30, 1), (31, 1)]
)
async def test_short_notes_are_transcribed_and_summary_starts_at_thirty_seconds(
    settings, duration, expected_summary
):
    service, bot, _ = make_dependencies(settings)
    message = make_message(duration=duration)
    await service.handle(message, bot)
    bot.get_file.assert_awaited_once()
    service.transcriber.transcribe.assert_awaited_once()
    assert service.summarizer.summarize.await_count == expected_summary
    message.answer.assert_awaited_once()
    bot.send_rich_message.assert_not_awaited()
    response = message.answer.await_args.args[0]
    if expected_summary:
        assert "<b>Коротко:</b>" in response
    else:
        assert "Коротко:" not in response
    assert "<blockquote expandable><b>Расшифровка:</b>\n" in response
    assert "Полный &lt;текст&gt; &amp; да.</blockquote>" in response


async def test_summary_activation_threshold_is_configurable(settings):
    settings.summary_min_voice_seconds = 45
    service, bot, _ = make_dependencies(settings)
    for index, duration in enumerate((30, 44, 45)):
        message = make_message(duration=duration)
        message.message_id = index + 1
        await service.handle(message, bot)
    assert service.transcriber.transcribe.await_count == 3
    assert service.summarizer.summarize.await_count == 1


@pytest.mark.parametrize("status", ["pending", "denied", "allowed"])
async def test_only_approved_groups_reach_models(settings, tmp_path, status):
    from voice_bot.admin import GroupAdmin
    from voice_bot.groups import GroupStore

    store = GroupStore(tmp_path / "groups.json")
    group, _ = store.observe(-100, "Test")
    store.mark_notified(group)
    store.decide(-100, group.revision, status)
    service, bot, _ = make_dependencies(settings)
    service.admin = GroupAdmin(settings, store)
    message = make_message(chat_id=-100)
    message.chat.type, message.chat.title = "supergroup", "Test"
    await service.handle(message, bot)
    assert service.transcriber.transcribe.await_count == int(status == "allowed")


async def test_forwarded_private_voice_works_for_non_owner_without_group_approval(
    settings, tmp_path
):
    from voice_bot.admin import GroupAdmin
    from voice_bot.groups import GroupStore

    settings.owner_id = 987654321
    service, bot, _ = make_dependencies(settings)
    store = GroupStore(tmp_path / "groups.json")
    service.admin = GroupAdmin(settings, store)
    message = make_message(chat_id=222)
    message.chat.type = "private"
    message.from_user = SimpleNamespace(id=222, full_name="Переславший", username="forwarder")
    message.forward_origin = MessageOriginHiddenUser(date=0, sender_user_name="Оригинальный автор")
    await service.handle(message, bot)
    service.transcriber.transcribe.assert_awaited_once()
    context = service.summarizer.summarize.await_args.kwargs["context"]
    assert context.author_name == "Оригинальный автор" and context.is_forwarded
    assert context.chat_type == "private"
    assert store.groups == {}
    assert not store.path.exists()
    message.answer.assert_awaited_once()


async def test_revocation_during_transcription_stops_summary_and_delivery(settings, tmp_path):
    from voice_bot.admin import GroupAdmin
    from voice_bot.groups import GroupStore

    store = GroupStore(tmp_path / "groups.json")
    group, _ = store.observe(-100, "Test")
    store.decide(-100, group.revision, "allowed")
    service, bot, paths = make_dependencies(settings)
    service.admin = GroupAdmin(settings, store)

    async def transcribe_and_revoke(path):
        store.decide(-100, store.groups[-100].revision, "denied")
        return "Private transcript"

    service.transcriber.transcribe.side_effect = transcribe_and_revoke
    message = make_message(chat_id=-100)
    message.chat.type, message.chat.title = "supergroup", "Test"
    await service.handle(message, bot)
    service.summarizer.summarize.assert_not_awaited()
    message.answer.assert_not_awaited()
    assert service.active_jobs == 0
    assert not paths[0].exists()


@pytest.mark.parametrize("reason", ["duration", "size", "busy", "allowlist", "not-voice"])
async def test_reject_before_download(settings, reason):
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    if reason == "duration":
        message.voice.duration = settings.max_voice_seconds + 1
    elif reason == "size":
        message.voice.file_size = settings.max_download_bytes + 1
    elif reason == "busy":
        service.active_jobs = settings.max_concurrent_jobs
    elif reason == "allowlist":
        settings.allowed_chat_ids = {999}
    else:
        message.voice = None
    await service.handle(message, bot)
    bot.get_file.assert_not_awaited()
    assert message.answer.await_count == (0 if reason in {"allowlist", "not-voice"} else 1)


@pytest.mark.parametrize("failure", [ProviderError("upstream detail"), RuntimeError("SECRET")])
async def test_summary_failure_still_returns_full_transcript(settings, failure, caplog):
    service, bot, paths = make_dependencies(settings)
    service.summarizer.summarize.side_effect = failure
    message = make_message()
    await service.handle(message, bot)
    response = message.answer.await_args.args[0]
    assert "Не удалось составить краткое содержание" in response
    assert "Полный &lt;текст&gt; &amp; да.</blockquote>" in response
    assert "upstream detail" not in response
    assert "SECRET" not in response + caplog.text
    assert not paths[0].exists()


async def test_stt_failure_cleans_up_and_does_not_log_body(settings, caplog):
    service, bot, paths = make_dependencies(settings)
    service.transcriber.transcribe.side_effect = RuntimeError("SECRET audio transcript")
    message = make_message()
    await service.handle(message, bot)
    assert not paths[0].exists()
    assert service.active_jobs == 0
    assert "SECRET" not in caplog.text
    assert "SECRET" not in message.answer.await_args.args[0]


async def test_timeout_after_stt_preserves_transcript(settings):
    settings.job_timeout_seconds = 0.02
    service, bot, paths = make_dependencies(settings)

    async def slow_summary(transcript, *, context=None):
        await asyncio.Event().wait()

    service.summarizer.summarize.side_effect = slow_summary
    message = make_message()
    await service.handle(message, bot)
    assert "Полный &lt;текст&gt; &amp; да.</blockquote>" in message.answer.await_args.args[0]
    assert not paths[0].exists()
    assert service.active_jobs == 0


async def test_cancellation_cleans_up(settings):
    service, bot, paths = make_dependencies(settings)
    entered = asyncio.Event()

    async def wait_for_cancel(audio):
        entered.set()
        await asyncio.Event().wait()

    service.transcriber.transcribe.side_effect = wait_for_cancel
    message = make_message()
    task = asyncio.create_task(service.handle(message, bot))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not paths[0].exists()
    assert service.active_jobs == 0
    message.answer.assert_not_awaited()


def test_actual_download_size_is_bounded(tmp_path):
    path = Path(tmp_path) / "audio.ogg"
    with BoundedFile(path, 10) as file:
        file.write(b"123456")
        with pytest.raises(UserFacingError):
            file.write(b"123456")
    assert path.stat().st_size == 6


async def test_download_failure_removes_partial_audio(settings):
    service, bot, _ = make_dependencies(settings)
    paths = []

    async def fail_download(path, output, **kwargs):
        paths.append(Path(output.name).parent)
        output.write(b"partial")
        raise RuntimeError("secret file URL")

    bot.download_file.side_effect = fail_download
    message = make_message()
    await service.handle(message, bot)
    assert not paths[0].exists()
    assert service.active_jobs == 0
    assert "secret" not in message.answer.await_args.args[0]


async def test_telegram_floodwait_has_one_bounded_retry(settings, monkeypatch):
    service, _, _ = make_dependencies(settings)
    message = make_message()
    error = TelegramRetryAfter(
        method=SendMessage(chat_id=123, text="private"), message="wait", retry_after=1
    )
    message.answer.side_effect = [error, None]
    sleep = AsyncMock()
    monkeypatch.setattr("voice_bot.service.asyncio.sleep", sleep)
    await service.send(message, "safe")
    assert message.answer.await_count == 2
    sleep.assert_awaited_once_with(1)


async def test_aiogram_router_dispatches_only_voice_updates(settings):
    service, _, _ = make_dependencies(settings)
    handled = AsyncMock()

    async def record_handle(message, bot, **kwargs):
        await handled(message, bot)

    service.enqueue = record_handle
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(service))
    async with Bot(token="123456:TEST_TOKEN") as bot:
        for update_id, content in [
            (1, {"text": "обычный текст"}),
            (
                2,
                {
                    "voice": {
                        "file_id": "id",
                        "file_unique_id": "unique",
                        "duration": 10,
                    }
                },
            ),
        ]:
            update = Update.model_validate(
                {
                    "update_id": update_id,
                    "message": {
                        "message_id": update_id,
                        "date": 0,
                        "chat": {"id": 123, "type": "private"},
                        **content,
                    },
                }
            )
            await dispatcher.feed_update(bot, update)
    handled.assert_awaited_once()


async def test_text_command_processes_original_voice_and_replies_to_it(settings):
    service, _, _ = make_dependencies(settings)
    handled = AsyncMock()

    async def record_handle(message, bot, **kwargs):
        await handled(message, bot)

    service.enqueue = record_handle
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(service))
    async with Bot(token="123456:TEST_TOKEN") as bot:
        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 43,
                    "date": 0,
                    "chat": {"id": 123, "type": "private"},
                    "from": {"id": 111, "is_bot": False, "first_name": "Команда"},
                    "text": "/text",
                    "reply_to_message": {
                        "message_id": 42,
                        "date": 0,
                        "chat": {"id": 123, "type": "private"},
                        "from": {"id": 222, "is_bot": False, "first_name": "Другой участник"},
                        "voice": {"file_id": "id", "file_unique_id": "unique", "duration": 25},
                    },
                },
            }
        )
        await dispatcher.feed_update(bot, update)
    handled.assert_awaited_once()
    assert handled.await_args.args[0].message_id == 42
    assert handled.await_args.args[0].from_user.id == 222


async def test_text_command_accepts_external_voice_in_same_chat(settings):
    service, _, _ = make_dependencies(settings)
    handled = AsyncMock()

    async def record_handle(message, bot, **kwargs):
        await handled(message, bot)

    service.enqueue = record_handle
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(service))
    async with Bot(token="123456:TEST_TOKEN") as bot:
        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 43,
                    "date": 0,
                    "text": "/text",
                    "chat": {"id": -100, "type": "supergroup", "title": "Test"},
                    "external_reply": {
                        "origin": {
                            "type": "user",
                            "date": 0,
                            "sender_user": {
                                "id": 222,
                                "is_bot": False,
                                "first_name": "Другой участник",
                            },
                        },
                        "chat": {"id": -100, "type": "supergroup", "title": "Test"},
                        "message_id": 42,
                        "voice": {"file_id": "id", "file_unique_id": "unique", "duration": 25},
                    },
                },
            }
        )
        await dispatcher.feed_update(bot, update)
    handled.assert_awaited_once()
    original = handled.await_args.args[0]
    assert original.message_id == 42 and original.chat.id == -100
    assert summary_context(original).author_name == "Другой участник"


async def test_text_command_does_not_process_external_voice_from_other_chat(settings):
    service, _, _ = make_dependencies(settings)
    handled = AsyncMock()
    service.enqueue = handled
    service.send = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(service))
    async with Bot(token="123456:TEST_TOKEN") as bot:
        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 43,
                    "date": 0,
                    "text": "/text",
                    "chat": {"id": -100, "type": "supergroup", "title": "Test"},
                    "external_reply": {
                        "origin": {"type": "hidden_user", "date": 0, "sender_user_name": "Автор"},
                        "chat": {"id": -200, "type": "supergroup", "title": "Other"},
                        "message_id": 42,
                        "voice": {"file_id": "id", "file_unique_id": "unique", "duration": 25},
                    },
                },
            }
        )
        await dispatcher.feed_update(bot, update)
    handled.assert_not_awaited()
    assert "Перешлите саму запись" in service.send.await_args.args[1]


async def test_successful_voice_processing_logs_only_operational_metadata(settings, caplog):
    import logging

    caplog.set_level(logging.INFO)
    service, bot, _ = make_dependencies(settings)
    await service.handle(make_message(), bot)
    assert "voice_job_started" in caplog.text
    assert "voice_job_delivered" in caplog.text
    assert service.transcriber.transcribe.return_value not in caplog.text
    assert service.summarizer.summarize.return_value not in caplog.text


async def test_voice_burst_waits_in_queue_instead_of_busy_replies(settings):
    service, bot, paths = make_dependencies(settings)
    messages = [make_message(duration=7) for _ in range(12)]
    for index, message in enumerate(messages):
        message.message_id = index + 1
        await service.accept(message, bot)
    assert service.transcriber.transcribe.await_count == 0
    await service.close()
    assert service.transcriber.transcribe.await_count == len(messages)
    assert all(message.answer.await_count == 1 for message in messages)
    assert all("заполнена" not in message.answer.await_args.args[0] for message in messages)
    assert all(not path.exists() for path in paths)
    assert service.active_jobs == 0 and not service.queue.workers


async def test_separate_chats_use_all_configured_workers(settings):
    settings.max_concurrent_jobs = 4
    service, bot, paths = make_dependencies(settings)
    entered, release = asyncio.Event(), asyncio.Event()
    running = peak = 0

    async def transcribe(path):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        if running == settings.max_concurrent_jobs:
            entered.set()
        await release.wait()
        running -= 1
        return "Тестовая речь"

    service.transcriber.transcribe.side_effect = transcribe
    for chat_id in range(8):
        await service.accept(make_message(duration=7, chat_id=chat_id), bot)
    await asyncio.wait_for(entered.wait(), 2)
    assert peak == settings.max_concurrent_jobs
    release.set()
    await service.close()
    assert service.transcriber.transcribe.await_count == 8
    assert all(not path.exists() for path in paths)


def identified_message(index, *, duration=30, author=222, seconds=0, forwarded=False):
    message = make_message(duration=duration)
    message.message_id = index
    message.date = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    message.from_user = User(id=author, is_bot=False, first_name=f"Имя {author}")
    if forwarded:
        message.forward_origin = MessageOriginUser(
            date=message.date,
            sender_user=User(id=333, is_bot=False, first_name="Исходный автор"),
        )
    return message


async def test_short_voice_context_is_reused_without_extra_summary_request(settings):
    service, bot, _ = make_dependencies(settings)
    first = identified_message(1, duration=7)
    await service.handle(first, bot)
    service.summarizer.summarize.assert_not_awaited()
    second = identified_message(2, seconds=30)
    await service.handle(second, bot)
    context = service.summarizer.summarize.await_args.kwargs["context"]
    assert context.recent_summaries == ("Полный <текст> & да.",)
    await service.handle(second, bot)
    assert service.summarizer.summarize.await_count == 1
    context = service.summarizer.summarize.await_args.kwargs["context"]
    assert context.recent_summaries == ("Полный <текст> & да.",)
    await service.close()


@pytest.mark.parametrize("separation", ["author", "chat", "topic", "day", "old"])
async def test_context_does_not_cross_identity_scope_or_time(settings, separation):
    service, bot, _ = make_dependencies(settings)
    await service.handle(identified_message(1), bot)
    next_message = identified_message(2, seconds=30)
    if separation == "author":
        next_message.from_user = User(id=444, is_bot=False, first_name="Другой")
    elif separation == "chat":
        next_message.chat.id = 999
    elif separation == "topic":
        next_message.message_thread_id = 55
    elif separation == "day":
        next_message.date += timedelta(days=1)
    else:
        next_message.date -= timedelta(days=1)
    await service.handle(next_message, bot)
    assert service.summarizer.summarize.await_args.kwargs["context"].recent_summaries == ()
    await service.close()


def test_memory_uses_original_identity_and_does_not_guess_hidden_authors():
    message = identified_message(1, forwarded=True)
    key, sent_at = conversation_position(message)
    assert key == (123, 0, 333)
    assert sent_at == message.forward_origin.date.timestamp()
    message.forward_origin = MessageOriginHiddenUser(date=0, sender_user_name="Имя")
    assert conversation_position(message) is None
    message.forward_origin = None
    message.sender_chat = Chat(id=-100, type="supergroup", title="Анонимно")
    assert conversation_position(message) is None


async def test_concurrent_author_messages_receive_preceding_summary_in_order(settings):
    service, bot, _ = make_dependencies(settings)
    entered, release = asyncio.Event(), asyncio.Event()

    async def transcribe(path):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return "Речь"

    service.transcriber.transcribe.side_effect = transcribe
    first = asyncio.create_task(service.handle(identified_message(1), bot))
    await entered.wait()
    second = asyncio.create_task(service.handle(identified_message(2, seconds=10), bot))
    await asyncio.sleep(0)
    assert service.transcriber.transcribe.await_count == 1
    release.set()
    await asyncio.gather(first, second)
    calls = service.summarizer.summarize.await_args_list
    assert calls[0].kwargs["context"].recent_summaries == ()
    assert calls[1].kwargs["context"].recent_summaries == ("Кратко.",)
    await service.close()


async def test_forwarded_burst_gets_one_summary_one_folded_reply_to_last_voice(settings):
    service, bot, paths = make_dependencies(settings)
    bot.send_rich_message = AsyncMock()
    service.transcriber.transcribe.side_effect = ["  Первый\n", "Второй & <текст>", "Третий"]
    messages = [identified_message(i, duration=5, forwarded=True) for i in (1, 2, 3)]
    for message in messages:
        await service.accept(message, bot)
    assert not paths  # collection downloads nothing
    await service.close()
    assert service.transcriber.transcribe.await_count == 3
    service.summarizer.summarize.assert_awaited_once()
    call = service.summarizer.summarize.await_args
    assert call.args[0] == (
        "Голосовое 1/3:\n  Первый\n\n\nГолосовое 2/3:\nВторой & <текст>\n\nГолосовое 3/3:\nТретий"
    )
    assert len(call.kwargs["context"].voices) == 3
    bot.send_rich_message.assert_awaited_once()
    result = bot.send_rich_message.await_args.kwargs
    assert result["reply_parameters"].message_id == 3
    details = result["rich_message"].blocks[-1]
    assert details.summary == "Расшифровка" and details.is_open is False
    assert details.blocks[0].blocks[0].text == call.args[0]
    assert all(not path.exists() for path in paths)
    assert all(message.answer.await_count == 0 for message in messages)
    assert service.active_jobs == 0 and not service.memory.sessions


async def test_mixed_forwarded_authors_are_explicit_and_do_not_share_memory(settings):
    service, bot, _ = make_dependencies(settings)
    bot.send_rich_message = AsyncMock()
    messages = [identified_message(i, forwarded=True) for i in (1, 2)]
    messages[1].forward_origin = MessageOriginUser(
        date=messages[1].date, sender_user=User(id=444, is_bot=False, first_name="Другой автор")
    )
    await service._handle_messages(messages, bot)
    context = service.summarizer.summarize.await_args.kwargs["context"]
    assert context.author_name is None and context.author_source == "multiple"
    assert [voice.author_name for voice in context.voices] == ["Исходный автор", "Другой автор"]
    assert not service.memory.sessions


async def test_partial_batch_preserves_successes_and_marks_failed_record(settings, caplog):
    service, bot, paths = make_dependencies(settings)
    bot.send_rich_message = AsyncMock()
    service.transcriber.transcribe.side_effect = ["Первая", RuntimeError("SECRET"), "Третья"]
    messages = [identified_message(i, forwarded=True) for i in (1, 2, 3)]
    await service._handle_messages(messages, bot)
    rich = bot.send_rich_message.await_args.kwargs["rich_message"]
    transcript = rich.blocks[-1].blocks[0].blocks[0].text
    assert "Первая" in transcript and "Третья" in transcript
    assert "Голосовое 2/3:\n[Не удалось распознать запись.]" in transcript
    assert rich.blocks[2].text.startswith("Часть записей не обработана.")
    assert "SECRET" not in rich.model_dump_json() + caplog.text
    assert all(not path.exists() for path in paths)
    assert not service.memory.sessions


async def test_batch_timeout_delivers_completed_transcripts_and_cleans_all_files(settings):
    settings.job_timeout_seconds = 0.02
    service, bot, paths = make_dependencies(settings)
    bot.send_rich_message = AsyncMock()

    async def transcribe(path):
        if len(paths) == 1:
            return "Готовая часть"
        await asyncio.Event().wait()

    service.transcriber.transcribe.side_effect = transcribe
    await service._handle_messages([identified_message(i, forwarded=True) for i in (1, 2)], bot)
    rich = bot.send_rich_message.await_args.kwargs["rich_message"]
    assert "Готовая часть" in rich.blocks[-1].blocks[0].blocks[0].text
    assert "превышено время" in rich.blocks[-1].blocks[0].blocks[0].text
    assert all(not path.exists() for path in paths)
    assert service.active_jobs == 0


async def test_definitive_rich_rejection_falls_back_without_losing_transcript(settings):
    from test_rendering import ParsedReply

    service, bot, _ = make_dependencies(settings)
    bot.send_rich_message = AsyncMock(
        side_effect=TelegramBadRequest(
            method=SendRichMessage(chat_id=123, rich_message={"html": "test"}),
            message="SECRET response body",
        )
    )
    text = "  😀<&\n" * 2000
    service.transcriber.transcribe.return_value = text
    messages = [identified_message(i, forwarded=True) for i in (1, 2)]
    await service._handle_messages(messages, bot)
    calls = messages[-1].answer.await_args_list
    assert all(call.kwargs["reply_parameters"].message_id == 2 for call in calls)
    transcript = "".join(ParsedReply(call.args[0]).transcript for call in calls)
    assert transcript == f"Голосовое 1/2:\n{text}\n\nГолосовое 2/2:\n{text}"


async def test_repeated_text_commands_share_one_queued_job(settings, monkeypatch):
    service, fake_bot, _ = make_dependencies(settings)
    service.send = AsyncMock(return_value=1000)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(service))
    original = {
        "message_id": 42,
        "date": 0,
        "chat": {"id": 123, "type": "private"},
        "voice": {"file_id": "id", "file_unique_id": "unique", "duration": 35},
    }
    async with Bot(token="123456:TEST_TOKEN") as bot:
        monkeypatch.setattr(bot, "get_file", fake_bot.get_file)
        monkeypatch.setattr(bot, "download_file", fake_bot.download_file)
        monkeypatch.setattr(bot, "copy_message", fake_bot.copy_message)
        await dispatcher.feed_update(
            bot, Update.model_validate({"update_id": 1, "message": original})
        )
        for index in range(20):
            command = {
                "message_id": index + 100,
                "date": 0,
                "chat": original["chat"],
                "text": "/text",
                "reply_to_message": original,
            }
            await dispatcher.feed_update(
                bot, Update.model_validate({"update_id": index + 2, "message": command})
            )
        await service.close()
    fake_bot.get_file.assert_awaited_once()
    fake_bot.download_file.assert_awaited_once()
    service.transcriber.transcribe.assert_awaited_once()
    service.summarizer.summarize.assert_awaited_once()
    service.send.assert_awaited_once()


async def test_completed_voice_stays_protected_after_service_restart(settings):
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    await service.handle(message, bot)
    await service.close()
    restarted, new_bot, _ = make_dependencies(settings)
    await restarted.handle(message, new_bot)
    await restarted.enqueue(message, new_bot)
    await restarted.close()
    new_bot.get_file.assert_not_awaited()
    restarted.transcriber.transcribe.assert_not_awaited()
    restarted.summarizer.summarize.assert_not_awaited()
    new_bot.copy_message.assert_awaited_once()
    message.answer.assert_awaited_once()


async def test_repeated_forwarded_audio_is_skipped_before_batch_collection(settings):
    service, bot, _ = make_dependencies(settings)
    first = identified_message(1, duration=7, forwarded=True)
    first.voice.file_unique_id = "same-recording"
    duplicate = identified_message(2, duration=7, forwarded=True)
    duplicate.voice.file_unique_id = "same-recording"
    await service.accept(first, bot)
    for _ in range(20):
        await service.accept(duplicate, bot)
        await service.enqueue(first, bot)
    await service.close()
    service.transcriber.transcribe.assert_awaited_once()
    service.summarizer.summarize.assert_not_awaited()
    first.answer.assert_awaited_once()
    duplicate.answer.assert_not_awaited()
    assert not service.processed.pending


async def test_partial_batch_retries_only_failed_voice_after_cooldown(settings):
    service, bot, _ = make_dependencies(settings)
    now = [0.0]
    service.processed.clock = lambda: now[0]
    service.transcriber.transcribe.side_effect = [
        "Первая",
        ProviderError("Ошибка"),
        "Третья",
        "Вторая",
    ]
    messages = [identified_message(i, forwarded=True) for i in (1, 2, 3)]
    await service._handle_messages(messages, bot)
    await service._handle_messages(messages, bot)
    assert service.transcriber.transcribe.await_count == 3
    now[0] = 61
    await service._handle_messages(messages, bot)
    assert service.transcriber.transcribe.await_count == 4
    assert bot.get_file.await_count == 4
    assert messages[1].answer.await_count == 0
    assert bot.send_rich_message.await_count == 2
    bot.copy_message.assert_awaited_once()
    assert all(service.processed.reply_ids(message) == (1000,) for message in messages)


async def test_failed_attempt_cannot_be_retriggered_by_command_spam(settings):
    service, bot, _ = make_dependencies(settings)
    now = [0.0]
    service.processed.clock = lambda: now[0]
    service.transcriber.transcribe.side_effect = [ProviderError("Не отвечает"), "Готово"]
    message = make_message(duration=7)
    await service.handle(message, bot)
    for _ in range(100):
        await service.enqueue(message, bot)
    assert service.transcriber.transcribe.await_count == 1
    assert message.answer.await_count == 1
    assert not service.processed.completed
    now[0] = 61
    await service.handle(message, bot)
    assert service.transcriber.transcribe.await_count == 2
    assert message.answer.await_count == 2
    assert len(service.processed.records) == 1


async def test_delivery_retry_reuses_transcript_and_summary_without_download_or_api(settings):
    service, bot, _ = make_dependencies(settings)
    now = [0.0]
    service.processed.clock = lambda: now[0]
    message = make_message()
    message.answer.side_effect = [
        TelegramNetworkError(method=SendMessage(chat_id=123, text="test"), message="SECRET"),
        None,
    ]
    await service.handle(message, bot)
    assert not service.processed.completed
    now[0] = 61
    await service.handle(message, bot)
    bot.get_file.assert_awaited_once()
    bot.download_file.assert_awaited_once()
    service.transcriber.transcribe.assert_awaited_once()
    service.summarizer.summarize.assert_awaited_once()
    assert message.answer.await_count == 2
    assert len(service.processed.records) == 1


async def test_receipt_write_failure_does_not_resend_result_or_allow_more_api(
    settings, monkeypatch
):
    service, bot, _ = make_dependencies(settings)

    def fail_write(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(service.processed, "_write", fail_write)
    message = make_message()
    await service.handle(message, bot)
    await service.handle(identified_message(99), bot)
    service.transcriber.transcribe.assert_awaited_once()
    message.answer.assert_awaited_once()
    assert not service.processed.available


async def test_queue_rejection_releases_reservation(settings, monkeypatch):
    service, bot, _ = make_dependencies(settings)
    now = [0.0]
    service.processed.clock = lambda: now[0]
    monkeypatch.setattr(service.queue, "add", lambda messages, bot: False)
    message = make_message()
    await service.enqueue(message, bot)
    assert not service.processed.pending
    assert not service.processed.claim(message)
    now[0] = 61
    assert service.processed.claim(message)
    service.processed.release(message)


async def test_retry_of_split_reply_does_not_resend_confirmed_parts(settings):
    service, bot, _ = make_dependencies(settings)
    now = [0.0]
    service.processed.clock = lambda: now[0]
    message = make_message(duration=7)
    service.transcriber.transcribe.return_value = "Длинный текст. " * 700
    message.answer.side_effect = [
        None,
        TelegramNetworkError(method=SendMessage(chat_id=123, text="test"), message="SECRET"),
    ]
    await service.handle(message, bot)
    confirmed_first_part = message.answer.await_args_list[0].args[0]
    message.answer.side_effect = None
    now[0] = 61
    await service.handle(message, bot)
    assert sum(call.args[0] == confirmed_first_part for call in message.answer.await_args_list) == 1
    service.transcriber.transcribe.assert_awaited_once()
    bot.get_file.assert_awaited_once()
    assert len(service.processed.records) == 1


async def test_concurrent_duplicate_requests_do_not_start_multiple_jobs(settings):
    service, bot, _ = make_dependencies(settings)
    entered, release = asyncio.Event(), asyncio.Event()

    async def transcribe(path):
        entered.set()
        await release.wait()
        return "Одна расшифровка"

    service.transcriber.transcribe.side_effect = transcribe
    message = make_message(duration=7)
    first = asyncio.create_task(service.handle(message, bot))
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.gather(*(service.handle(message, bot) for _ in range(100)))
    bot.get_file.assert_awaited_once()
    release.set()
    await first
    message.answer.assert_awaited_once()


async def test_private_quota_blocks_before_download_and_notifies_only_once(settings):
    settings.private_daily_voice_limit = 2
    service, bot, _ = make_dependencies(settings)
    messages = [identified_message(i, duration=7, forwarded=True) for i in range(1, 7)]
    for message in messages:
        message.chat.type = "private"
        await service._handle_messages([message], bot)
    assert bot.get_file.await_count == 2
    assert service.transcriber.transcribe.await_count == 2
    assert sum(message.answer.await_count for message in messages) == 3
    assert "2 голосовых" in messages[2].answer.await_args.args[0]
    await service.handle(messages[0], bot)
    assert list(service.private_quota.users.values())[0].used == 2


async def test_each_private_batch_voice_counts_toward_daily_limit(settings):
    settings.private_daily_voice_limit = 2
    service, bot, _ = make_dependencies(settings)
    messages = [identified_message(i, duration=7, forwarded=True) for i in range(1, 5)]
    for message in messages:
        message.chat.type = "private"
        await service.accept(message, bot)
    await service.close()
    assert service.transcriber.transcribe.await_count == 2
    bot.send_rich_message.assert_awaited_once()
    assert list(service.private_quota.users.values())[0].used == 2


async def test_private_allowance_is_refunded_when_queue_rejects_job(settings, monkeypatch):
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    message.chat.type = "private"
    monkeypatch.setattr(service.queue, "add", lambda messages, bot: False)
    await service.enqueue(message, bot)
    assert list(service.private_quota.users.values())[0].used == 0
    assert not service.processed.pending


async def test_one_group_sender_cannot_fill_queue_using_text_on_others_voices(settings):
    settings.max_pending_voices_per_user = 2
    settings.private_daily_voice_limit = 1
    service, bot, _ = make_dependencies(settings)
    caller = identified_message(900, author=777)
    caller.chat.type = "supergroup"
    messages = [identified_message(i, duration=7, author=i) for i in range(1, 5)]
    for message in messages:
        message.chat.type = "supergroup"
        await service.enqueue(message, bot, requester=caller)
    another_sender = identified_message(5, duration=7, author=555)
    another_sender.chat.type = "supergroup"
    await service.enqueue(another_sender, bot)
    assert service.queue.size == 3
    caller.answer.assert_awaited_once()
    await service.close()
    assert service.transcriber.transcribe.await_count == 3
    assert not service.private_quota.users
    assert not service._pending_by_sender and not service._owners


async def test_repeated_invalid_commands_do_not_spam_group(settings):
    service, _, _ = make_dependencies(settings)
    messages = [identified_message(i, author=i) for i in range(100)]
    for message in messages:
        message.chat.type = "supergroup"
        await service.notify(message, "Ответьте на голосовое сообщение.")
    assert sum(message.answer.await_count for message in messages) == 1


async def test_group_daily_use_is_not_limited_by_private_quota(settings):
    settings.private_daily_voice_limit = 1
    service, bot, _ = make_dependencies(settings)
    for index in range(35):
        message = identified_message(index, duration=7)
        message.chat.type = "supergroup"
        await service.handle(message, bot)
        message.answer.assert_awaited_once()
    assert service.transcriber.transcribe.await_count == 35
    assert not service.private_quota.users


async def test_full_forwarded_group_batch_fits_sender_allowance(settings):
    service, bot, _ = make_dependencies(settings)
    messages = [identified_message(i, duration=7, forwarded=True) for i in range(10)]
    for message in messages:
        message.chat.type = "supergroup"
        await service.accept(message, bot)
    await service.close()
    assert service.transcriber.transcribe.await_count == 10
    bot.send_rich_message.assert_awaited_once()
    assert not service._owners and not service._pending_by_sender


async def test_private_quota_write_failure_never_reaches_download(settings, monkeypatch):
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    message.chat.type = "private"

    def fail_write(users):
        raise OSError("disk unavailable")

    monkeypatch.setattr(service.private_quota, "_write", fail_write)
    await service.handle(message, bot)
    await service.handle(message, bot)
    bot.get_file.assert_not_awaited()
    service.transcriber.transcribe.assert_not_awaited()
    assert not service.processed.pending


async def test_repeated_forward_copies_complete_telegram_reply_without_api(settings):
    service, bot, _ = make_dependencies(settings)
    first = identified_message(1)
    first.voice.file_unique_id = "recording"
    await service.handle(first, bot)
    duplicate = identified_message(2, forwarded=True)
    duplicate.voice.file_unique_id = "recording"
    await service.handle(duplicate, bot)
    bot.get_file.assert_awaited_once()
    service.transcriber.transcribe.assert_awaited_once()
    service.summarizer.summarize.assert_awaited_once()
    bot.copy_message.assert_awaited_once()
    call = bot.copy_message.await_args.kwargs
    assert call["from_chat_id"] == call["chat_id"] == 123
    assert call["message_id"] == 1000
    assert call["reply_parameters"].message_id == 2
    duplicate.answer.assert_not_awaited()
    assert service.processed.reply_ids(duplicate) == (2000,)


async def test_deleted_cached_reply_falls_back_to_processing(settings):
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    await service.processed.complete([message], (999,))
    bot.copy_message.side_effect = TelegramBadRequest(
        method=SendMessage(chat_id=123, text="test"), message="message not found"
    )
    await service.handle(message, bot)
    bot.copy_message.assert_awaited_once()
    bot.get_file.assert_awaited_once()
    service.transcriber.transcribe.assert_awaited_once()
    assert service.processed.reply_ids(message) == (1000,)


async def test_legacy_receipt_without_reply_id_never_silently_drops_voice(settings):
    service, bot, _ = make_dependencies(settings)
    message = make_message(duration=25)
    await service.processed.complete([message])
    await service.handle(message, bot)
    service.transcriber.transcribe.assert_awaited_once()
    service.summarizer.summarize.assert_not_awaited()
    message.answer.assert_awaited_once()
    assert "Расшифровка:" in message.answer.await_args.args[0]


async def test_cached_private_reply_is_free_even_when_daily_allowance_is_used(settings):
    settings.private_daily_voice_limit = 1
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    message.chat.type = "private"
    await service.handle(message, bot)
    assert list(service.private_quota.users.values())[0].used == 1
    await service.handle(message, bot)
    bot.copy_message.assert_awaited_once()
    assert list(service.private_quota.users.values())[0].used == 1
    service.transcriber.transcribe.assert_awaited_once()


async def test_cached_batch_is_copied_once_with_all_transcripts(settings):
    service, bot, _ = make_dependencies(settings)
    messages = [identified_message(i, forwarded=True) for i in (1, 2, 3)]
    await service.processed.complete(messages, (999,))
    await service._handle_messages(messages, bot)
    bot.copy_message.assert_awaited_once()
    assert bot.copy_message.await_args.kwargs["reply_parameters"].message_id == 3
    bot.get_file.assert_not_awaited()
    service.transcriber.transcribe.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()


async def test_copy_network_timeout_never_starts_paid_fallback(settings):
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    await service.processed.complete([message], (999,))
    bot.copy_message.side_effect = TelegramNetworkError(
        method=SendMessage(chat_id=123, text="test"), message="SECRET timeout"
    )
    await service.handle(message, bot)
    bot.get_file.assert_not_awaited()
    service.transcriber.transcribe.assert_not_awaited()


async def test_repeated_cached_requests_cannot_flood_telegram(settings):
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    await service.processed.complete([message], (999,))
    for _ in range(20):
        await service.handle(message, bot)
    bot.copy_message.assert_awaited_once()
    bot.get_file.assert_not_awaited()


async def test_deleted_private_reply_does_not_bypass_daily_allowance(settings):
    settings.private_daily_voice_limit = 1
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    message.chat.type = "private"
    await service.private_quota.admit(message)
    await service.processed.complete([message], (999,))
    bot.copy_message.side_effect = TelegramBadRequest(
        method=SendMessage(chat_id=123, text="test"), message="missing"
    )
    await service.handle(message, bot)
    bot.get_file.assert_not_awaited()
    service.transcriber.transcribe.assert_not_awaited()
    assert list(service.private_quota.users.values())[0].used == 1


async def test_missing_split_reply_cleans_partial_copy_before_regenerating(settings):
    service, bot, _ = make_dependencies(settings)
    message = make_message()
    await service.processed.complete([message], (999, 1001))
    bot.copy_message.side_effect = [
        SimpleNamespace(message_id=2001),
        TelegramBadRequest(method=SendMessage(chat_id=123, text="test"), message="missing"),
    ]
    await service.handle(message, bot)
    bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=2001)
    service.transcriber.transcribe.assert_awaited_once()
