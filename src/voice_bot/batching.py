"""Bounded debounce for a burst of forwarded voice messages."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from aiogram import Bot
from aiogram.types import Message

from voice_bot.delivery import forum_topic_id

logger = logging.getLogger(__name__)
BatchKey = tuple[int, int, str, int]
BatchHandler = Callable[[list[Message], Bot], Awaitable[None]]


def batch_key(message: Message) -> BatchKey | None:
    if getattr(message, "forward_origin", None) is None:
        return None
    sender_chat = getattr(message, "sender_chat", None)
    user = getattr(message, "from_user", None)
    if sender_chat:
        sender, sender_id = "chat", sender_chat.id
    elif user:
        sender, sender_id = "user", user.id
    else:
        return None
    return (message.chat.id, forum_topic_id(message) or 0, sender, sender_id)


@dataclass
class PendingBatch:
    bot: Bot
    started_at: float
    last_at: float
    messages: list[Message] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    ready: bool = False


class VoiceBatcher:
    def __init__(
        self,
        handler: BatchHandler,
        pause_seconds: float,
        max_messages: int,
        max_pending: int,
    ) -> None:
        self.handler = handler
        self.pause = pause_seconds
        self.max_messages = max_messages
        self.max_pending = max_pending
        self.pending: dict[BatchKey, PendingBatch] = {}
        self.tasks: set[asyncio.Task[None]] = set()
        self.closed = False

    def add(self, message: Message, bot: Bot) -> bool:
        key = batch_key(message)
        if self.closed or key is None:
            return False
        now = time.monotonic()
        batch = self.pending.get(key)
        # Old queued Telegram updates cannot join a new burst just because they arrived together.
        if batch and abs((message.date - batch.messages[0].date).total_seconds()) > 10:
            self._release(key, batch)
            batch = None
        if batch is None:
            if len(self.tasks) >= self.max_pending:
                return False
            batch = PendingBatch(bot=bot, started_at=now, last_at=now)
            self.pending[key] = batch
            task = asyncio.create_task(self._run(key, batch), name="forwarded-voices")
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        # Telegram retries must not create duplicate parts or extend the timer.
        if any(item.message_id == message.message_id for item in batch.messages):
            return True
        batch.messages.append(message)
        batch.last_at = now
        if len(batch.messages) >= self.max_messages:
            self._release(key, batch)
        batch.changed.set()
        return True

    def _release(self, key: BatchKey, batch: PendingBatch) -> None:
        if self.pending.get(key) is batch:
            del self.pending[key]
        batch.ready = True
        batch.changed.set()

    async def _run(self, key: BatchKey, batch: PendingBatch) -> None:
        try:
            while not batch.ready:
                batch.changed.clear()
                deadline = min(batch.last_at + self.pause, batch.started_at + 10)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    async with asyncio.timeout(remaining):
                        await batch.changed.wait()
                except TimeoutError:
                    break
            self._release(key, batch)
            await self.handler(sorted(batch.messages, key=lambda item: item.message_id), batch.bot)
        except Exception as exc:
            # Background tasks must never leak request data through asyncio's exception handler.
            logger.warning("voice_batch_failed error_type=%s", type(exc).__name__)
        finally:
            if self.pending.get(key) is batch:
                del self.pending[key]

    def finish(self) -> None:
        self.closed = True
        for key, batch in list(self.pending.items()):
            self._release(key, batch)
