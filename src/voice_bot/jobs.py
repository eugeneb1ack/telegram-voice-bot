"""Bounded in-memory voice queue with round-robin scheduling between chats."""

import asyncio
import logging
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from aiogram import Bot
from aiogram.types import Message

logger = logging.getLogger(__name__)
JobHandler = Callable[[list[Message], Bot], Awaitable[None]]


@dataclass(frozen=True)
class VoiceJob:
    messages: tuple[Message, ...]
    bot: Bot
    queued_at: float = field(default_factory=time.monotonic)


class VoiceJobQueue:
    def __init__(
        self, handler: JobHandler, concurrency: int, max_pending: int, max_per_chat: int
    ) -> None:
        self.handler = handler
        self.concurrency = concurrency
        self.max_pending = max_pending
        self.max_per_chat = max_per_chat
        self.pending: OrderedDict[int, deque[VoiceJob]] = OrderedDict()
        self.active_chats: set[int] = set()
        self.workers: set[asyncio.Task[None]] = set()
        self.size = 0
        self.closed = False
        self._changed = asyncio.Event()

    def add(self, messages: list[Message], bot: Bot) -> bool:
        if self.closed or not messages or self.size >= self.max_pending:
            return False
        chat_id = messages[-1].chat.id
        waiting = self.pending.get(chat_id)
        if waiting is not None and len(waiting) >= self.max_per_chat:
            return False
        if waiting is None:
            waiting = deque()
            self.pending[chat_id] = waiting
        waiting.append(VoiceJob(tuple(messages), bot))
        self.size += 1
        if not self.workers:
            for index in range(self.concurrency):
                task = asyncio.create_task(self._run(), name=f"voice-worker-{index + 1}")
                self.workers.add(task)
                task.add_done_callback(self.workers.discard)
        self._changed.set()
        return True

    async def _next(self) -> VoiceJob | None:
        while True:
            for chat_id in list(self.pending):
                if chat_id in self.active_chats:
                    continue
                waiting = self.pending.pop(chat_id)
                job = waiting.popleft()
                if waiting:
                    self.pending[chat_id] = waiting
                self.size -= 1
                self.active_chats.add(chat_id)
                return job
            if self.closed and not self.size:
                return None
            self._changed.clear()
            await self._changed.wait()

    async def _run(self) -> None:
        while (job := await self._next()) is not None:
            chat_id = job.messages[-1].chat.id
            try:
                logger.info(
                    "voice_queue_wait message_id=%s elapsed_ms=%s",
                    job.messages[-1].message_id,
                    round((time.monotonic() - job.queued_at) * 1000),
                )
                await self.handler(list(job.messages), job.bot)
            except Exception as exc:
                logger.warning("voice_queue_job_failed error_type=%s", type(exc).__name__)
            finally:
                self.active_chats.remove(chat_id)
                self._changed.set()

    async def close(self, grace_seconds: float) -> None:
        self.closed = True
        self._changed.set()
        workers = set(self.workers)
        if workers:
            try:
                _, unfinished = await asyncio.wait(workers, timeout=grace_seconds)
                if unfinished:
                    logger.warning(
                        "voice_queue_shutdown_incomplete pending=%s active=%s",
                        self.size,
                        len(self.active_chats),
                    )
            finally:
                for task in workers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                self.workers.difference_update(workers)
        self.pending.clear()
        self.size = 0
