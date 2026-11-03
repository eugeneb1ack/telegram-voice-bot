import json
import re
from types import SimpleNamespace

import httpx
import pytest
from aiogram.types import (
    Chat,
    MessageOriginChannel,
    MessageOriginChat,
    MessageOriginHiddenUser,
    MessageOriginUser,
    User,
)

from voice_bot.providers import APIClient, HTTPSummarizer, SummaryContext, VoicePartContext
from voice_bot.reply_context import ReplyContext
from voice_bot.service import summary_context


def author_message(user=None, *, origin=None, sender_chat=None):
    return SimpleNamespace(
        from_user=user,
        sender_chat=sender_chat,
        forward_origin=origin,
        chat=Chat(id=-100, type="supergroup", title="Обсуждение"),
    )


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"full_name": "  Анна\u00a0  Петрова \n", "first_name": "Анна"}, "Анна Петрова"),
        ({"full_name": None, "first_name": "  Ирина\t"}, "Ирина"),
        ({"full_name": " \t\n", "first_name": " Ирина \n"}, "Ирина"),
        ({"first_name": "Олег"}, "Олег"),
        ({"full_name": "", "first_name": ""}, None),
        ({"full_name": " \t", "first_name": " \n"}, None),
        ({}, None),
    ],
)
def test_author_name_normalizes_whitespace_and_falls_back_to_first_name(fields, expected):
    context = summary_context(author_message(SimpleNamespace(username="nickname", **fields)))

    assert context.author_name == expected
    assert context.author_username == "nickname"
    assert context.author_source == "sender"


def test_full_name_preserves_last_name_and_username_metadata():
    user = User(
        id=777,
        is_bot=False,
        first_name="  Анна  ",
        last_name="  Петрова  ",
        username="anna_nickname",
    )

    context = summary_context(author_message(user))

    assert context.author_name == "Анна Петрова"
    assert context.author_username == "anna_nickname"


@pytest.mark.parametrize(
    "title,expected", [("  Команда\n  проекта  ", "Команда проекта"), (" \t", None)]
)
def test_sender_chat_uses_its_title_before_username(title, expected):
    context = summary_context(
        author_message(
            User(id=777, is_bot=False, first_name="Скрытый отправитель", username="sender"),
            sender_chat=Chat(id=-200, type="channel", title=title, username="team_channel"),
        )
    )

    assert context.author_name == expected
    assert context.author_username == "team_channel"


@pytest.mark.parametrize("kind", ["user", "hidden", "chat", "channel"])
def test_forwarded_author_name_is_normalized_without_using_forwarder(kind):
    original = User(
        id=333,
        is_bot=False,
        first_name=" Исходный ",
        last_name=" Автор ",
        username="original",
    )
    channel = Chat(id=-333, type="channel", title=" Исходный\n Автор ", username="original")
    origins = {
        "user": MessageOriginUser(date=0, sender_user=original),
        "hidden": MessageOriginHiddenUser(date=0, sender_user_name=" Исходный\t Автор "),
        "chat": MessageOriginChat(date=0, sender_chat=channel),
        "channel": MessageOriginChannel(date=0, chat=channel, message_id=12),
    }

    context = summary_context(
        author_message(
            User(id=777, is_bot=False, first_name="Переславший", username="forwarder"),
            origin=origins[kind],
        )
    )

    assert context.author_name == "Исходный Автор"
    assert context.author_username == (None if kind == "hidden" else "original")
    assert context.author_source == "forwarded"
    assert context.is_forwarded


@pytest.mark.parametrize("name", ["Анна Петрова", None])
async def test_summary_payload_prioritizes_names_with_username_as_fallback(settings, name):
    context = SummaryContext(
        author_name=name,
        author_username="anna_nickname",
        author_source="sender",
        voices=(VoicePartContext(1, "Олег", "oleg_nickname", 15),),
        reply_contexts=(ReplyContext(10, 5, "Ирина", "irina_nickname", "text", "Встреча в семь."),),
    )
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "Итог."}]}
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        await HTTPSummarizer(APIClient(client, settings)).summarize(
            "Согласна на встречу в семь.", context=context
        )

    assert len(requests) == 1
    payload = requests[0]
    metadata = json.loads(payload["input"])["context"]
    assert metadata["author_name"] == name
    assert metadata["author_username"] == "anna_nickname"
    assert metadata["voices"][0]["author_name"] == "Олег"
    assert metadata["reply_contexts"][0]["author_name"] == "Ирина"
    # Check the actual provider contract, not a fabricated model response.
    assert re.search(
        r"context\.author_name[^.]*только если имя отсутствует или пустое[^.]*"
        r"context\.author_username",
        payload["instructions"],
    )
    assert (
        "в voices и reply_contexts: при наличии имени никнейм не добавляй"
        in payload["instructions"]
    )
    assert "anna_nickname" not in payload["instructions"]
    assert payload["max_output_tokens"] == settings.summary_max_output_tokens
    assert payload["model"] == settings.summary_model
