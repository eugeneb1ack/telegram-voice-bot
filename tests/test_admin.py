import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.methods import EditMessageText, GetMe, SendMessage
from aiogram.types import Message, Update, User

from voice_bot.admin import (
    GREETING,
    AdminAction,
    GroupAdmin,
    build_admin_router,
    group_card,
    group_greeting,
)
from voice_bot.groups import GroupStore


class RecordingSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        pass

    async def stream_content(self, *args, **kwargs):
        yield b""

    async def make_request(self, bot, method, timeout=None):  # noqa: ASYNC109
        self.calls.append(method)
        if isinstance(method, GetMe):
            return User(id=123456, is_bot=True, first_name="Voice", username="voice_test_bot")
        return True


@pytest.fixture
def admin(settings, tmp_path):
    settings.owner_id = 987654321
    return GroupAdmin(settings, GroupStore(tmp_path / "groups.json"))


def query(admin, *, user_id=None, chat_id=None):
    message = MagicMock(spec=Message)
    message.chat = MagicMock()
    message.chat.type = "private"
    message.chat.id = admin.settings.owner_id if chat_id is None else chat_id
    message.edit_text = AsyncMock()
    value = MagicMock()
    value.from_user.id = admin.settings.owner_id if user_id is None else user_id
    value.message = message
    value.answer = AsyncMock()
    return value


async def test_owner_notification_retries_after_start_and_does_not_leak_error(admin, caplog):
    group, _ = admin.store.observe(-100, "Test")
    bot = MagicMock()
    bot.send_message = AsyncMock(
        side_effect=TelegramForbiddenError(
            method=SendMessage(chat_id=admin.settings.owner_id, text="private"),
            message="SECRET token",
        )
    )
    await admin.notify(group, bot)
    assert not admin.store.groups[-100].notified
    assert "SECRET" not in caplog.text
    bot.send_message.side_effect = None
    admin.notification_attempts.clear()
    await admin.notify_pending(bot)
    assert admin.store.groups[-100].notified
    await admin.notify_pending(bot)
    assert bot.send_message.await_count == 2


async def test_concurrent_start_and_join_do_not_duplicate_notification(admin):
    group, _ = admin.store.observe(-100, "Test")
    entered, finish = asyncio.Event(), asyncio.Event()

    async def delayed_send(*args, **kwargs):
        entered.set()
        await finish.wait()

    bot = MagicMock(send_message=AsyncMock(side_effect=delayed_send))
    first = asyncio.create_task(admin.notify(group, bot))
    await entered.wait()
    admin.notification_attempts.clear()
    second = asyncio.create_task(admin.notify(group, bot))
    finish.set()
    await asyncio.gather(first, second)
    bot.send_message.assert_awaited_once()


@pytest.mark.parametrize("bad_user,bad_chat", [(123, None), (None, 123)])
async def test_callbacks_require_owner_and_private_owner_chat(admin, bad_user, bad_chat):
    group, _ = admin.store.observe(-100, "Test")
    value = query(admin, user_id=bad_user, chat_id=bad_chat)
    await admin.callback(
        value, AdminAction(action="allow", chat_id=-100, revision=group.revision), MagicMock()
    )
    assert not admin.store.allowed(-100)
    assert value.answer.await_args.kwargs["show_alert"]
    value.message.edit_text.assert_not_awaited()


async def test_allow_then_deny_and_stale_callback(admin):
    group, _ = admin.store.observe(-100, "Test")
    bot = MagicMock(send_message=AsyncMock())
    value = query(admin)
    action = AdminAction(action="allow", chat_id=-100, revision=group.revision)
    await admin.callback(value, action, bot)
    assert admin.store.allowed(-100)
    await admin.callback(value, action, bot)
    assert value.answer.await_args.args[0] == "Уже разрешено"
    await admin.callback(
        value,
        AdminAction(action="deny", chat_id=-100, revision=admin.store.groups[-100].revision),
        bot,
    )
    assert not admin.store.allowed(-100)
    assert not GroupStore(admin.store.path).allowed(-100)
    assert bot.send_message.await_count == 2
    await admin.callback(value, action, bot)
    assert "устарела" in value.answer.await_args.args[0]
    assert not admin.store.allowed(-100)


@pytest.mark.parametrize("action", ["allow", "deny"])
async def test_repeated_and_concurrent_decisions_notify_group_once(admin, action):
    group, _ = admin.store.observe(-100, "Test")
    bot = MagicMock(send_message=AsyncMock())
    value = query(admin)
    data = AdminAction(action=action, chat_id=-100, revision=group.revision)
    await asyncio.gather(*(admin.callback(value, data, bot) for _ in range(4)))
    current = admin.store.groups[-100]
    await admin.callback(
        value, AdminAction(action=action, chat_id=-100, revision=current.revision), bot
    )
    bot.send_message.assert_awaited_once()
    value.message.edit_text.assert_awaited_once()
    assert admin.store.groups[-100].revision == current.revision


async def test_telegram_not_modified_does_not_turn_refresh_into_an_error(admin):
    value = query(admin)
    value.message.edit_text.side_effect = TelegramBadRequest(
        method=EditMessageText(text="safe"), message="Bad Request: message is not modified"
    )
    await admin.callback(value, AdminAction(action="list"), MagicMock())
    value.answer.assert_awaited_once()


def test_group_titles_are_escaped_and_panel_is_paginated(admin):
    group, _ = admin.store.observe(-100, "<b>Injected</b>&")
    text, markup = group_card(group)
    assert "&lt;b&gt;Injected&lt;/b&gt;&amp;" in text
    assert all(len(b.callback_data.encode()) <= 64 for row in markup.inline_keyboard for b in row)
    for number in range(20):
        admin.store.observe(-1000 - number, f"Group {number}")
    text, markup = admin.panel(1)
    assert "21" in text
    assert "сек" not in text
    assert len(markup.inline_keyboard) == 8
    assert any(b.text == "2/4" for row in markup.inline_keyboard for b in row)


async def test_real_aiogram_routes_group_join_and_promotion(admin):
    dispatcher = Dispatcher()
    dispatcher.include_router(build_admin_router(admin))
    assert set(dispatcher.resolve_used_update_types()) == {
        "message",
        "my_chat_member",
        "callback_query",
    }
    session = RecordingSession()
    async with Bot(token="123456:TEST_TOKEN", session=session) as bot:
        for number, old, new in [(1, "left", "member"), (2, "member", "administrator")]:
            member = {"id": 123456, "is_bot": True, "first_name": "Voice"}
            new_member = {"user": member, "status": new}
            if new == "administrator":
                new_member.update(
                    can_be_edited=False,
                    is_anonymous=False,
                    can_manage_chat=False,
                    can_delete_messages=False,
                    can_manage_video_chats=False,
                    can_restrict_members=False,
                    can_promote_members=False,
                    can_change_info=False,
                    can_invite_users=False,
                    can_post_stories=False,
                    can_edit_stories=False,
                    can_delete_stories=False,
                    can_send_welcome_messages=False,
                )
            update = Update.model_validate(
                {
                    "update_id": number,
                    "my_chat_member": {
                        "chat": {"id": -100, "type": "supergroup", "title": "Test"},
                        "from": {"id": 777, "is_bot": False, "first_name": "Member"},
                        "date": 0,
                        "old_chat_member": {"user": member, "status": old},
                        "new_chat_member": new_member,
                    },
                }
            )
            await dispatcher.feed_update(bot, update)
        assert admin.store.groups[-100].status == "pending"
        sends = [call for call in session.calls if isinstance(call, SendMessage)]
        assert len(sends) == 2
        assert sends[0].chat_id == admin.settings.owner_id
        assert sends[0].reply_markup.inline_keyboard[0][0].text == "✅ Разрешить"
        assert sends[1].text == group_greeting(admin.store.groups[-100], admin.settings)
        assert "Ожидаю разрешения владельца" in sends[1].text


@pytest.mark.parametrize("owner", [True, False])
async def test_admin_command_only_responds_to_owner(admin, owner):
    dispatcher = Dispatcher()
    dispatcher.include_router(build_admin_router(admin))
    session = RecordingSession()
    user_id = admin.settings.owner_id if owner else 777
    async with Bot(token="123456:TEST_TOKEN", session=session) as bot:
        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": 0,
                    "from": {"id": user_id, "is_bot": False, "first_name": "User"},
                    "chat": {"id": user_id, "type": "private"},
                    "text": "/admin",
                },
            }
        )
        await dispatcher.feed_update(bot, update)
    assert len(session.calls) == int(owner)


def help_update(*, text="/start", chat_id=777, chat_type="private", voice=False):
    data = {
        "message_id": 1,
        "date": 0,
        "from": {"id": 777, "is_bot": False, "first_name": "User"},
        "chat": {"id": chat_id, "type": chat_type},
        "text": text,
    }
    if voice:
        data.pop("text")
        data["voice"] = {"file_id": "voice", "file_unique_id": "unique", "duration": 10}
    return Update.model_validate({"update_id": 1, "message": data})


@pytest.mark.parametrize("command", ["/start", "/help", "/start group"])
async def test_private_help_explains_first_step_and_group_approval(admin, command):
    dispatcher = Dispatcher()
    dispatcher.include_router(build_admin_router(admin))
    session = RecordingSession()
    async with Bot(token="123456:TEST_TOKEN", session=session) as bot:
        await dispatcher.feed_update(bot, help_update(text=command))
    sends = [call for call in session.calls if isinstance(call, SendMessage)]
    assert len(sends) == 1
    reply = sends[0]
    assert reply.text == GREETING
    assert reply.parse_mode == "HTML"
    assert "Пришлите или перешлите голосовое" in reply.text
    assert "Дождитесь разрешения владельца" in reply.text
    assert len(reply.text) < 650
    assert reply.reply_markup.inline_keyboard[0][0].url == (
        "https://t.me/voice_test_bot?startgroup=true"
    )
    assert len(reply.reply_markup.inline_keyboard) == 1


@pytest.mark.parametrize(
    "status,mode,expected",
    [
        ("pending", "auto", "Ожидаю разрешения владельца"),
        ("denied", "auto", "Обработка отключена владельцем"),
        ("allowed", "auto", "Обычный режим: автоматически"),
        ("allowed", "quiet", "Тихий режим: обрабатываю голосовые только по /text"),
    ],
)
async def test_group_help_reflects_actual_access_and_mode(admin, status, mode, expected):
    group, _ = admin.store.observe(-100, "Test")
    group = admin.store.decide(-100, group.revision, status)
    admin.store.set_mode(-100, group.revision, mode)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_admin_router(admin))
    session = RecordingSession()
    async with Bot(token="123456:TEST_TOKEN", session=session) as bot:
        await dispatcher.feed_update(
            bot, help_update(text="/help", chat_id=-100, chat_type="supergroup")
        )
    replies = [c for c in session.calls if isinstance(c, SendMessage) and c.chat_id == -100]
    assert len(replies) == 1
    assert expected in replies[0].text
    assert "Нажмите «Добавить в группу»" not in replies[0].text
    assert replies[0].reply_markup.inline_keyboard[0][0].callback_data.startswith("mode:")


def test_group_help_does_not_promise_processing_outside_allowed_chat_list(admin):
    group, _ = admin.store.observe(-100, "Test")
    group = admin.store.decide(-100, group.revision, "allowed")
    admin.settings.allowed_chat_ids = {123}
    assert "группа не входит в список разрешённых" in group_greeting(group, admin.settings)


@pytest.mark.parametrize(
    "text,chat_type,voice,expected",
    [
        ("привет", "private", False, True),
        ("привет", "supergroup", False, False),
        ("/settings", "private", False, False),
        (None, "private", True, False),
    ],
)
async def test_private_text_hint_does_not_swallow_commands_or_voices(
    admin, text, chat_type, voice, expected
):
    dispatcher = Dispatcher()
    dispatcher.include_router(build_admin_router(admin))
    session = RecordingSession()
    async with Bot(token="123456:TEST_TOKEN", session=session) as bot:
        await dispatcher.feed_update(
            bot,
            help_update(
                text=text,
                chat_id=777 if chat_type == "private" else -100,
                chat_type=chat_type,
                voice=voice,
            ),
        )
    sends = [call for call in session.calls if isinstance(call, SendMessage)]
    assert len(sends) == int(expected)
    if expected:
        assert "голосовое сообщение" in sends[0].text
        assert "/help" in sends[0].text
