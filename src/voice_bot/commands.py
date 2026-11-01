"""Best-effort cleanup of public command messages from older Telegram clients."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from aiogram import BaseMiddleware, Bot
from aiogram.dispatcher.flags import get_flag
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message, TelegramObject

logger = logging.getLogger(__name__)


class CommandCleanup(BaseMiddleware):
    def __init__(self, *, delay: float = 5, max_pending: int = 100) -> None:
        self.delay = delay
        self.max_pending = max_pending
        self.tasks: set[asyncio.Task[None]] = set()

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if (
            isinstance(event, Message)
            and get_flag(data, "cleanup_command")
            and event.chat.type in {"group", "supergroup"}
            and not event.ephemeral_message_id
            and event.message_id > 0
            and len(self.tasks) < self.max_pending
        ):
            task = asyncio.create_task(self._delete(event, data["bot"]), name="command-cleanup")
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        return await handler(event, data)

    async def _delete(self, message: Message, bot: Bot) -> None:
        age = (datetime.now(UTC) - message.date).total_seconds()
        await asyncio.sleep(max(0, self.delay - age))
        try:
            await bot.delete_message(message.chat.id, message.message_id)
        except TelegramAPIError:
            # Missing delete rights or an already-deleted command must not affect the result.
            logger.info("command_cleanup_unavailable chat_id=%s", message.chat.id)
        except Exception as exc:
            logger.warning("command_cleanup_failed error_type=%s", type(exc).__name__)

    async def close(self) -> None:
        tasks = set(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
