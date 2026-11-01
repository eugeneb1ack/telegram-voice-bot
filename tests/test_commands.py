import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.methods import DeleteMessage
from aiogram.types import Message, Update

from voice_bot.commands import CommandCleanup
from voice_bot.delivery import personal, reply_to


def message(*, chat_type="supergroup", ephemeral=False):
    return Message.model_validate(
        {
            "message_id": 0 if ephemeral else 43,
            "ephemeral_message_id": 999 if ephemeral else None,
            "date": datetime.now(UTC),
            "chat": {"id": -100 if chat_type == "supergroup" else 111, "type": chat_type},
            "from": {"id": 111, "is_bot": False, "first_name": "Reader"},
            "text": "/text",
        }
    )


def context(bot, flag=True):
    return {"bot": bot, "handler": SimpleNamespace(flags={"cleanup_command": flag})}


async def test_deletes_only_public_group_command_after_delay(monkeypatch):
    cleanup = CommandCleanup()
    bot = SimpleNamespace(delete_message=AsyncMock())
    handler = AsyncMock(return_value="handled")
    sleep = AsyncMock()
    monkeypatch.setattr("voice_bot.commands.asyncio.sleep", sleep)
    msg = message()
    assert await cleanup(handler, msg, context(bot)) == "handled"
    await asyncio.gather(*cleanup.tasks)
    delay = sleep.await_args.args[0]
    assert 3 <= delay <= 5
    bot.delete_message.assert_awaited_once_with(-100, 43)


@pytest.mark.parametrize(
    "private,ephemeral,flag", [(True, False, True), (False, True, True), (False, False, False)]
)
async def test_does_not_delete_private_ephemeral_or_unmarked_messages(private, ephemeral, flag):
    cleanup = CommandCleanup(delay=0)
    bot = SimpleNamespace(delete_message=AsyncMock())
    msg = message(chat_type="private" if private else "supergroup", ephemeral=ephemeral)
    await cleanup(AsyncMock(), msg, context(bot, flag))
    assert not cleanup.tasks
    bot.delete_message.assert_not_awaited()


async def test_missing_delete_rights_do_not_leak_or_affect_handler(caplog):
    cleanup = CommandCleanup(delay=0)
    bot = SimpleNamespace(
        delete_message=AsyncMock(
            side_effect=TelegramBadRequest(
                method=DeleteMessage(chat_id=-100, message_id=43), message="SECRET request"
            )
        )
    )
    handler = AsyncMock()
    await cleanup(handler, message(), context(bot))
    await asyncio.gather(*cleanup.tasks)
    handler.assert_awaited_once()
    assert "SECRET" not in caplog.text


async def test_pending_deletions_are_bounded_and_cancelled_on_shutdown():
    cleanup = CommandCleanup(delay=60, max_pending=1)
    bot = SimpleNamespace(delete_message=AsyncMock())
    for _ in range(3):
        await cleanup(AsyncMock(), message(), context(bot))
    assert len(cleanup.tasks) == 1
    await cleanup.close()
    assert not cleanup.tasks
    bot.delete_message.assert_not_awaited()


async def test_real_aiogram_handler_flags_schedule_cleanup_only_for_commands(monkeypatch):
    cleanup = CommandCleanup(delay=0)
    dispatcher = Dispatcher()
    dispatcher.message.middleware(cleanup)
    router = Router()

    async def handle(message):
        pass

    router.message.register(handle, Command("text"), flags={"cleanup_command": True})
    dispatcher.include_router(router)
    delete = AsyncMock()
    monkeypatch.setattr(Bot, "delete_message", delete)
    async with Bot(token="123456:TEST_TOKEN") as bot:
        await dispatcher.feed_update(bot, Update(update_id=1, message=message()))
        await asyncio.gather(*cleanup.tasks)
    delete.assert_awaited_once_with(-100, 43)


def test_private_notifications_do_not_reply_to_deleted_public_commands():
    msg = personal(message())
    assert reply_to(msg) is None
    native = personal(message(ephemeral=True))
    assert reply_to(native).ephemeral_message_id == 999
