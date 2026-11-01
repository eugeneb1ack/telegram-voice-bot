from types import SimpleNamespace

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditEphemeralMessageText, EditMessageText, GetChatMember, SendMessage
from aiogram.types import Update
from test_admin import RecordingSession

from voice_bot.admin import GroupAdmin, build_admin_router
from voice_bot.groups import GroupStore
from voice_bot.modes import ModeAction, build_modes_router


class ModeSession(RecordingSession):
    def __init__(self, *, user_status="administrator", bot_status="administrator"):
        super().__init__()
        self.user_status = user_status
        self.bot_status = bot_status
        self.edit_error = None

    async def make_request(self, bot, method, timeout=None):  # noqa: ASYNC109
        if isinstance(method, (EditEphemeralMessageText, EditMessageText)) and self.edit_error:
            self.calls.append(method)
            raise TelegramBadRequest(method=method, message=self.edit_error)
        if isinstance(method, GetChatMember):
            self.calls.append(method)
            return SimpleNamespace(
                status=self.bot_status if method.user_id == bot.id else self.user_status
            )
        return await super().make_request(bot, method, timeout)


@pytest.fixture
def setup_modes(settings, tmp_path):
    store = GroupStore(tmp_path / "groups.json")
    group, _ = store.observe(-100, "Test")
    store.mark_notified(group)
    store.decide(-100, group.revision, "allowed")
    admin = GroupAdmin(settings, store)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_admin_router(admin))
    dispatcher.include_router(build_modes_router(admin))
    return dispatcher, store


def command_update():
    return Update.model_validate(
        {
            "update_id": 1,
            "message": {
                "message_id": 0,
                "ephemeral_message_id": 999,
                "date": 0,
                "chat": {"id": -100, "type": "supergroup", "title": "Test"},
                "from": {"id": 111, "is_bot": False, "first_name": "Admin"},
                "text": "/settings",
            },
        }
    )


def callback_update(action, revision, *, ephemeral=True):
    return Update.model_validate(
        {
            "update_id": 2,
            "callback_query": {
                "id": "callback-id",
                "chat_instance": "instance",
                "from": {"id": 111, "is_bot": False, "first_name": "Admin"},
                "message": {
                    "message_id": 0 if ephemeral else 213,
                    "ephemeral_message_id": 1000 if ephemeral else None,
                    "date": 0,
                    "chat": {"id": -100, "type": "supergroup", "title": "Test"},
                },
                "data": ModeAction(action=action, revision=revision).pack(),
            },
        }
    )


async def test_settings_are_private_and_show_both_buttons(setup_modes):
    dispatcher, _ = setup_modes
    session = ModeSession()
    async with Bot(token="123456:TEST_TOKEN", session=session) as bot:
        await dispatcher.feed_update(bot, command_update())
    sends = [c for c in session.calls if isinstance(c, SendMessage)]
    assert len(sends) == 1
    assert sends[0].ephemeral_message_parameters.receiver_user_id == 111
    assert sends[0].reply_parameters.ephemeral_message_id == 999
    buttons = sends[0].reply_markup.inline_keyboard[0]
    assert [b.text for b in buttons] == ["✓ Обычный", "Тихий"]


@pytest.mark.parametrize("status", ["member", "restricted", "left"])
async def test_members_cannot_change_mode(setup_modes, status):
    dispatcher, store = setup_modes
    session = ModeSession(user_status=status)
    async with Bot(token="123456:TEST_TOKEN", session=session) as bot:
        await dispatcher.feed_update(bot, command_update())
        await dispatcher.feed_update(bot, callback_update("quiet", store.groups[-100].revision))
    assert store.automatic(-100)
    sends = [c for c in session.calls if isinstance(c, SendMessage)]
    assert len(sends) == 1 and sends[0].reply_markup is None
    assert sends[0].ephemeral_message_parameters.receiver_user_id == 111


async def test_bot_admin_required_and_old_buttons_cannot_overwrite(setup_modes):
    dispatcher, store = setup_modes
    session = ModeSession(bot_status="member")
    revision = store.groups[-100].revision
    async with Bot(token="123456:TEST_TOKEN", session=session) as bot:
        await dispatcher.feed_update(bot, callback_update("quiet", revision))
        assert store.automatic(-100)
        session.bot_status = "administrator"
        await dispatcher.feed_update(bot, callback_update("quiet", revision))
        assert not store.automatic(-100)
        await dispatcher.feed_update(bot, callback_update("auto", revision))
        assert not store.automatic(-100)
    assert not GroupStore(store.path).automatic(-100)
    edits = [c for c in session.calls if isinstance(c, EditEphemeralMessageText)]
    assert len(edits) == 2
    assert all(c.receiver_user_id == 111 and c.ephemeral_message_id == 1000 for c in edits)
    assert edits[-1].reply_markup.inline_keyboard[0][1].text == "✓ Тихий"
    assert not [c for c in session.calls if isinstance(c, SendMessage)]


@pytest.mark.parametrize("ephemeral", [True, False])
async def test_checkmark_and_revision_change_in_same_menu_both_directions(setup_modes, ephemeral):
    dispatcher, store = setup_modes
    session = ModeSession()
    async with Bot(token="123456:TEST_TOKEN", session=session) as bot:
        for action, selected in (("quiet", 1), ("auto", 0), ("quiet", 1)):
            await dispatcher.feed_update(
                bot, callback_update(action, store.groups[-100].revision, ephemeral=ephemeral)
            )
            method = EditEphemeralMessageText if ephemeral else EditMessageText
            edit = [c for c in session.calls if isinstance(c, method)][-1]
            buttons = edit.reply_markup.inline_keyboard[0]
            assert [b.text.startswith("✓") for b in buttons] == [selected == 0, selected == 1]
            assert all(
                ModeAction.unpack(b.callback_data).revision == store.groups[-100].revision
                for b in buttons
            )
            assert edit.ephemeral_message_id == 1000 if ephemeral else edit.message_id == 213


@pytest.mark.parametrize(
    "error", ["Bad Request: message is not modified", "Bad Request: message not found"]
)
async def test_unchanged_or_expired_private_menu_is_handled_without_public_reply(
    setup_modes, error
):
    dispatcher, store = setup_modes
    session = ModeSession()
    session.edit_error = error
    async with Bot(token="123456:TEST_TOKEN", session=session) as bot:
        await dispatcher.feed_update(bot, callback_update("quiet", store.groups[-100].revision))
    assert not store.automatic(-100)
    sends = [c for c in session.calls if isinstance(c, SendMessage)]
    if "not modified" in error:
        assert not sends
    else:
        assert len(sends) == 1
        assert sends[0].ephemeral_message_parameters.receiver_user_id == 111
        assert sends[0].reply_markup.inline_keyboard[0][1].text == "✓ Тихий"
