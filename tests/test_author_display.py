import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
from test_rendering import ParsedReply

from voice_bot.providers import APIClient, HTTPSummarizer, SummaryContext, VoicePartContext
from voice_bot.rendering import render_replies
from voice_bot.reply_context import ReplyContext
from voice_bot.service import summary_context


def author_message(user=None, *, origin=None, sender_chat=None):
    return SimpleNamespace(
        from_user=user,
        sender_chat=sender_chat,
        forward_origin=origin,
        chat=Chat(id=-100, type="supergroup", title="Обсуждение"),
    )


async def summarize_body(settings, context, body="Встреча в семь."):
    api = SimpleNamespace(
        settings=settings,
        post=AsyncMock(
            return_value={
                "output": [{"type": "message", "content": [{"type": "output_text", "text": body}]}]
            }
        ),
    )
    return await HTTPSummarizer(api).summarize("Согласна на встречу в семь.", context=context)


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
async def test_sender_chat_uses_its_title_before_username(settings, title, expected):
    context = summary_context(
        author_message(
            User(id=777, is_bot=False, first_name="Скрытый отправитель", username="sender"),
            sender_chat=Chat(id=-200, type="channel", title=title, username="team_channel"),
        )
    )

    assert context.author_name == expected
    assert context.author_username == "team_channel"
    assert await summarize_body(settings, context) == (
        f"{expected or '@team_channel'}: Встреча в семь."
    )


@pytest.mark.parametrize("kind", ["user", "hidden", "chat", "channel"])
async def test_forwarded_author_name_is_normalized_without_using_forwarder(settings, kind):
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
    assert await summarize_body(settings, context) == "Исходный Автор: Встреча в семь."


@pytest.mark.parametrize(
    "name,username,label",
    [
        ("Анна Петрова", "anna_nickname", "Анна Петрова"),
        ("  Анна\u00a0 Петрова \n", "anna_nickname", "Анна Петрова"),
        (None, "anna_nickname", "@anna_nickname"),
        (" \t\n", " nickname ", "@nickname"),
        (None, None, "Автор"),
        (" \t", " \n", "Автор"),
        ("<disroot/>", "public_handle", "<disroot/>"),
    ],
)
async def test_summary_result_prioritizes_name_even_when_model_omits_author(
    settings, name, username, label
):
    context = SummaryContext(
        author_name=name,
        author_username=username,
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
        summary = await HTTPSummarizer(APIClient(client, settings)).summarize(
            "Согласна на встречу в семь.", context=context
        )

    assert len(requests) == 1
    assert summary == f"{label}: Итог."
    payload = requests[0]
    metadata = json.loads(payload["input"])["context"]
    assert metadata["author_name"] == name
    assert metadata["author_username"] == username
    assert metadata["voices"][0]["author_name"] == "Олег"
    assert metadata["reply_recipients"][0]["author_name"] == "Ирина"
    assert "anna_nickname" not in payload["instructions"]
    assert payload["max_output_tokens"] == settings.summary_max_output_tokens
    assert payload["model"] == settings.summary_model


async def test_literal_author_name_is_html_escaped_only_during_rendering(settings):
    context = summary_context(
        author_message(
            User(id=777, is_bot=False, first_name="<disroot/>", username="public_handle")
        )
    )
    summary = await summarize_body(settings, context)
    transcript = "  Встреча <&> в семь.\n"

    replies = render_replies(summary, transcript)

    assert len(replies) == 1
    assert replies[0].startswith("<b>Коротко:</b>\n&lt;disroot/&gt;: Встреча в семь.")
    parsed = ParsedReply(replies[0])
    assert "<disroot/>: Встреча в семь." in parsed.plain
    assert "public_handle" not in parsed.plain
    assert parsed.transcript == transcript


async def test_multiple_authors_do_not_get_a_single_author_prefix(settings):
    context = SummaryContext(
        author_name="Переславший",
        author_username="forwarder",
        author_source="multiple",
        voices=(VoicePartContext(1, "Ирина", "irina", 15), VoicePartContext(2, "Олег", "oleg", 15)),
    )
    body = "Ирина предлагает семь часов; Олег согласен."

    assert await summarize_body(settings, context, body) == body


async def test_missing_context_keeps_provider_body_without_inventing_author(settings):
    assert await summarize_body(settings, None, "  Встреча в семь.  ") == "Встреча в семь."


async def test_unknown_forward_does_not_attribute_summary_to_forwarder(settings):
    context = summary_context(
        author_message(
            User(id=777, is_bot=False, first_name="Переславший", username="forwarder"),
            origin=SimpleNamespace(type="unknown"),
        )
    )

    assert context.author_source == "unknown"
    assert await summarize_body(settings, context) == "Автор: Встреча в семь."
