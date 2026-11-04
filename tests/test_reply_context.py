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
    if kind in {"video", "animation", "photo"}:
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
        data[kind] = [media] if kind == "photo" else media
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


@pytest.mark.parametrize(
    "kind", ["text", "voice", "video_note", "video", "document", "audio", "photo"]
)
async def test_reply_preserves_identity_without_text_media_or_extra_stt(settings, kind):
    service, bot, _ = setup(settings)
    target = foreign_message(kind=kind).model_copy(update={"caption": "ForeignCaption"})
    message = await process_voice(service, bot, 10, target)
    context = await service._reply_context(message, bot)
    assert context.author_name == "Ирина"
    assert context.author_id == 444
    assert context.text == "" and context.pending_media is None
    service.transcriber.transcribe.assert_awaited_once()
    assert bot.get_file.await_count == 1
    stored = pack(
        foreign_message(kind="voice").model_copy(
            update={"message_id": 10, "reply_to_message": target}
        )
    )["reply_to_message"]
    assert stored["from_user"]["id"] == 444
    assert not set(stored) & {"text", "caption", "voice", "video_note", "reply_to_message"}
    await service.close()


@pytest.mark.parametrize("restriction", ["chat", "receiver_user", "ephemeral_message_id"])
async def test_private_or_cross_chat_reply_is_excluded(settings, restriction):
    service, bot, _ = setup(settings)
    target = foreign_message()
    if restriction == "chat":
        target = foreign_message(chat=-200)
    elif restriction == "receiver_user":
        target = target.model_copy(
            update={"receiver_user": User(id=444, is_bot=False, first_name="Ирина")}
        )
    else:
        target = target.model_copy(update={"ephemeral_message_id": "private"})
    message = reply_voice(target=target)
    assert reply_target(message) is None
    assert await service._reply_context(message, bot) is None
    assert "reply_to_message" not in pack(
        foreign_message(kind="voice").model_copy(
            update={"message_id": 10, "reply_to_message": target}
        )
    )
    await service.close()


async def test_recipient_identity_survives_restart_without_source_body(settings):
    service, bot, _ = setup(settings)
    await process_voice(service, bot, 10, foreign_message())
    await process_voice(service, bot, 11)
    await service.close()
    service, bot, _ = setup(settings)
    await process_voice(service, bot, 12)
    context = service.summarizer.summarize.await_args.kwargs["context"]
    assert context.reply_contexts[0].author_name == "Ирина"
    assert context.reply_contexts[0].text == ""
    assert FOREIGN_TEXT not in service.summarizer.summarize.await_args.args[0]
    service.transcriber.transcribe.assert_awaited_once()
    await service.close()


def test_video_note_is_never_converted_to_a_voice():
    note = foreign_message(kind="video_note")
    assert as_voice(note) is None
    assert as_voice(note.model_copy(update={"voice": foreign_message(kind="voice").voice})) is None
