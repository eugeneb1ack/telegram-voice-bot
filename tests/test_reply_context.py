"""Explicit reply sources enrich summaries without becoming the author's speech."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Message, User
from test_conversation_scope import scoped_voice
from test_service import make_dependencies

from voice_bot.journal import pack
from voice_bot.reply_context import as_voice, reply_target

FOREIGN_TEXT = "Ирина предлагает провести встречу в библиотеке в семь часов."


def foreign_message(*, chat=-100, kind="text", duration=15, size=100):
    media = {
        "file_id": "foreign-file",
        "file_unique_id": "foreign-unique",
        "duration": duration,
        "file_size": size,
    }
    if kind == "video":
        media.update(width=320, height=240)
    elif kind == "video_note":
        media.update(length=240)
    data = {
        "message_id": 1,
        "date": datetime(2026, 1, 1, tzinfo=UTC),
        "chat": {"id": chat, "type": "supergroup"},
        "from_user": {"id": 444, "is_bot": False, "first_name": "Ирина", "username": "irina"},
    }
    if kind == "text":
        data["text"] = FOREIGN_TEXT
    else:
        data[kind] = media
    return Message.model_validate(data)


def reply_voice(number=10, target=None):
    message = scoped_voice(number, thread=1 if target is not None else None)
    message.reply_to_message = target
    return message


def setup(settings):
    settings.context_summary_every = 3
    service, bot, paths = make_dependencies(settings)
    bot.send_chat_action = AsyncMock()
    return service, bot, paths


async def process_voice(service, bot, number, target=None):
    message = reply_voice(number, target)
    service.transcriber.transcribe.return_value = f"Собственная мысль {number}."
    await service.handle(message, bot)
    return message


async def test_first_voice_reply_reaches_third_summary_without_becoming_own_speech(settings):
    service, bot, _ = setup(settings)
    try:
        for number in (10, 11, 12):
            message = await process_voice(
                service, bot, number, foreign_message() if number == 10 else None
            )
            message.answer.assert_awaited_once()
            assert ("Коротко · 3 голосовых:" in message.answer.await_args.args[0]) == (number == 12)
        service.summarizer.summarize.assert_awaited_once()
        call = service.summarizer.summarize.await_args
        assert all(f"Собственная мысль {number}." in call.args[0] for number in (10, 11, 12))
        assert FOREIGN_TEXT not in call.args[0]
        contexts = call.kwargs["context"].reply_contexts
        assert len(contexts) == 1
        context = contexts[0]
        assert context.message_id == 10 and context.reply_to_message_id == 1
        assert context.author_name == "Ирина" and context.author_username == "irina"
        assert context.text == FOREIGN_TEXT
        entries = service.memory.sessions[(-100, 0, 222)].entries
        assert [entry.text for entry in entries] == [
            f"Собственная мысль {number}." for number in (10, 11, 12)
        ]
    finally:
        await service.close()


async def test_reply_on_first_voice_survives_restart_before_third(settings):
    service, bot, _ = setup(settings)
    await process_voice(service, bot, 10, foreign_message())
    await process_voice(service, bot, 11)
    await service.close()
    restarted, bot, _ = setup(settings)
    try:
        message = await process_voice(restarted, bot, 12)
        assert "Коротко · 3 голосовых:" in message.answer.await_args.args[0]
        contexts = restarted.summarizer.summarize.await_args.kwargs["context"].reply_contexts
        assert [(item.message_id, item.reply_to_message_id, item.text) for item in contexts] == [
            (10, 1, FOREIGN_TEXT)
        ]
        restarted.transcriber.transcribe.assert_awaited_once()
    finally:
        await restarted.close()


async def test_next_triple_does_not_inherit_foreign_reply_sources(settings):
    service, bot, _ = setup(settings)
    try:
        for number in range(10, 16):
            await process_voice(service, bot, number, foreign_message() if number == 10 else None)
        calls = service.summarizer.summarize.await_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["context"].reply_contexts
        assert calls[1].kwargs["context"].reply_contexts == ()
        assert FOREIGN_TEXT not in calls[1].args[0]
        assert all(FOREIGN_TEXT not in text for text in calls[1].kwargs["context"].recent_summaries)
        assert all(f"Собственная мысль {number}." in calls[1].args[0] for number in (13, 14, 15))
    finally:
        await service.close()


@pytest.mark.parametrize("private_kind", ["other_chat", "receiver", "ephemeral"])
async def test_private_and_cross_chat_targets_never_reach_stt(settings, private_kind):
    service, bot, _ = setup(settings)
    target = foreign_message(chat=-200 if private_kind == "other_chat" else -100, kind="voice")
    if private_kind == "receiver":
        target = target.model_copy(
            update={"receiver_user": User(id=222, first_name="Reader", is_bot=False)}
        )
    elif private_kind == "ephemeral":
        target = target.model_copy(update={"ephemeral_message_id": 123})
    try:
        message = reply_voice(target=target)
        assert reply_target(message) is None
        assert await service._reply_context(message, bot) is None
        bot.get_file.assert_not_awaited()
        service.transcriber.transcribe.assert_not_awaited()
    finally:
        await service.close()


async def test_exact_saved_source_is_reused_without_another_transcription(settings):
    service, bot, _ = setup(settings)
    target = foreign_message(kind="voice")
    # A similar message from another chat must never win the lookup.
    service.memory.remember((-200, 0, 444), target.date.timestamp(), (1,), "Чужой чат.")
    service.memory.remember((-100, 0, 444), target.date.timestamp(), (1,), FOREIGN_TEXT)
    try:
        context = await service._reply_context(reply_voice(target=target), bot)
        assert context is not None and context.text == FOREIGN_TEXT
        assert context.reply_to_message_id == 1
        bot.get_file.assert_not_awaited()
        service.transcriber.transcribe.assert_not_awaited()
    finally:
        await service.close()


@pytest.mark.parametrize("kind", ["video", "video_note"])
async def test_media_reply_uses_bounded_audio_path_and_cleans_files(settings, kind):
    service, bot, paths = setup(settings)
    target = foreign_message(kind=kind)
    adapted = as_voice(target)
    assert adapted is not None
    assert adapted.chat.id == target.chat.id and adapted.message_id == target.message_id
    assert (
        adapted.voice.file_id == "foreign-file" and adapted.voice.file_unique_id == "foreign-unique"
    )
    assert adapted.voice.duration == 15 and adapted.voice.file_size == 100
    service.transcriber.transcribe.return_value = FOREIGN_TEXT
    try:
        context = await service._reply_context(reply_voice(target=target), bot)
        assert context is not None and context.kind == kind and context.text == FOREIGN_TEXT
        service.transcriber.transcribe.assert_awaited_once()
        assert len(paths) == 1 and not paths[0].exists()
        assert not service.memory.sessions
    finally:
        await service.close()


@pytest.mark.parametrize("kind", ["video", "video_note"])
@pytest.mark.parametrize("limit", ["duration", "size"])
async def test_media_reply_limits_skip_download_without_failing_main_voice(settings, kind, limit):
    service, bot, _ = setup(settings)
    target = foreign_message(
        kind=kind,
        duration=settings.max_voice_seconds + 1 if limit == "duration" else 15,
        size=settings.max_download_bytes + 1 if limit == "size" else 100,
    )
    try:
        context = await service._reply_context(reply_voice(target=target), bot)
        assert context is None or context.unavailable_reason
        bot.get_file.assert_not_awaited()
        service.transcriber.transcribe.assert_not_awaited()
        # The source's size/duration failure must not prevent this author's own transcript.
        message = await process_voice(service, bot, 10, target)
        message.answer.assert_awaited_once()
        assert "Собственная мысль 10." in message.answer.await_args.args[0]
    finally:
        await service.close()


@pytest.mark.parametrize("kind", ["text", "voice", "video", "video_note"])
def test_journal_packs_only_one_reply_level_and_preserves_target_metadata(kind):
    target = foreign_message(kind=kind)
    nested = foreign_message(chat=-200).model_copy(update={"message_id": 999, "text": "Не хранить"})
    target = target.model_copy(update={"reply_to_message": nested})
    message = Message.model_validate(
        {
            "message_id": 10,
            "date": target.date,
            "chat": target.chat,
            "from_user": {"id": 222, "is_bot": False, "first_name": "Автор"},
            "voice": {"file_id": "own", "file_unique_id": "own-unique", "duration": 10},
            "reply_to_message": target,
            "message_thread_id": 1,
            "is_topic_message": False,
        }
    )
    restored = Message.model_validate(pack(message))
    reply = restored.reply_to_message
    assert reply is not None and reply.message_id == 1 and reply.chat.id == -100
    assert reply.from_user is not None and reply.from_user.id == 444
    assert reply.reply_to_message is None
    if kind == "text":
        assert reply.text == FOREIGN_TEXT
    else:
        assert getattr(reply, kind).file_id == "foreign-file"
    assert "Не хранить" not in str(pack(message))


@pytest.mark.parametrize("private_kind", ["other_chat", "receiver", "ephemeral"])
def test_journal_omits_rejected_reply_targets(private_kind):
    target = foreign_message(chat=-200 if private_kind == "other_chat" else -100)
    if private_kind == "receiver":
        target = target.model_copy(
            update={"receiver_user": User(id=222, first_name="Reader", is_bot=False)}
        )
    elif private_kind == "ephemeral":
        target = target.model_copy(update={"ephemeral_message_id": 123})
    message = Message.model_validate(
        {
            "message_id": 10,
            "date": target.date,
            "chat": {"id": -100, "type": "supergroup"},
            "from_user": {"id": 222, "is_bot": False, "first_name": "Автор"},
            "voice": {"file_id": "own", "file_unique_id": "own-unique", "duration": 10},
            "reply_to_message": target,
        }
    )
    packed = pack(message)
    assert "reply_to_message" not in packed
    assert FOREIGN_TEXT not in str(packed)
    assert reply_target(Message.model_validate(packed)) is None


async def test_context_budget_includes_foreign_text_without_merging_it_into_transcript(tmp_path):
    from voice_bot.conversation import ConversationMemory
    from voice_bot.reply_context import ReplyContext

    path = tmp_path / "context.json"
    memory = ConversationMemory(900, 120, 10, clock=lambda: 10, max_entries=4, path=path)
    key = (-100, 0, 222)
    try:
        for number in range(1, 5):
            reference = ReplyContext(number, 100 + number, "Ирина", None, "text", "F" * 1000)
            memory.remember(key, number, (number,), "O" * 1000, reply_contexts=(reference,))
        await memory.checkpoint()
        restored = ConversationMemory(900, 120, 10, clock=lambda: 10, max_entries=4, path=path)
        try:
            for saved in (memory, restored):
                entries = saved.sessions[key].entries
                assert (
                    sum(
                        len(entry.text) + sum(len(ref.text) for ref in entry.reply_contexts)
                        for entry in entries
                    )
                    <= 120
                )
                assert all("F" not in entry.text for entry in entries)
                assert all(entry.kind == "voice" for entry in entries)
                assert all(
                    ref.message_id in entry.message_ids
                    for entry in entries
                    for ref in entry.reply_contexts
                )
        finally:
            restored.clear()
    finally:
        memory.clear()


async def test_frozen_prepared_reply_context_survives_restart_without_stt(settings, tmp_path):
    from dataclasses import asdict

    from test_journal import idle, runtime, stop, voice

    from voice_bot.journal import QueueJournal
    from voice_bot.reply_context import ReplyContext

    path = tmp_path / "queue.db"
    journal = QueueJournal(path)
    target = foreign_message(kind="voice")
    original = voice(10, author=222).model_copy(update={"reply_to_message": target})
    job_id = journal.admit(original, None, collecting=False, charged=False)
    reference = ReplyContext(10, 1, "Ирина", "irina", "voice", FOREIGN_TEXT)
    prepared = {
        "summary": "Уже подготовленный итог.",
        "transcript": "Собственная расшифровка.",
        "title": "Коротко:",
        "valid": [True],
        "key": [-100, 0, 222],
        "times": [original.date.timestamp()],
        "dialogue": False,
        "reply_contexts": [asdict(reference)],
    }
    journal.set(job_id, "result", prepared)
    journal.close()
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    service._reply_context = AsyncMock(side_effect=AssertionError("frozen result was recomputed"))
    try:
        await service.recover(bot)
        await idle(service)
        assert len(sent) == 1 and "Уже подготовленный итог." in sent[0].text
        service._reply_context.assert_not_awaited()
        service.transcriber.transcribe.assert_not_awaited()
        service.summarizer.summarize.assert_not_awaited()
        bot.get_file.assert_not_awaited()
        entries = service.memory.sessions[(-100, 0, 222)].entries
        assert entries[0].reply_contexts == (reference,)
        assert entries[0].text == "Собственная расшифровка."
    finally:
        await stop(service, bot, journal)


def test_legacy_reply_without_author_or_privacy_metadata_is_not_reused():
    target = foreign_message().model_copy(update={"from_user": None})
    message = reply_voice(target=target)
    assert reply_target(message) is None
