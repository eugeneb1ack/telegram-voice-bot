"""Build the next three-voice summary inside the current voice job."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from voice_bot.config import Settings
from voice_bot.conversation import ConversationKey, ConversationMemory, Entry, Session
from voice_bot.reply_context import ReplyContext, bounded_contexts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DialogueInput:
    text: str
    background: tuple[str, ...]
    voice_count: int
    duration_seconds: int
    reply_contexts: tuple[ReplyContext, ...] = ()


class DialogueContext:
    def __init__(
        self,
        memory: ConversationMemory,
        settings: Settings,
        allowed: Callable[[ConversationKey, Session], bool],
    ) -> None:
        self.memory = memory
        self.settings = settings
        self.allowed = allowed
        self.task: asyncio.Task[None] | None = None

    def prepare(
        self,
        key: ConversationKey | None,
        sent_at: float,
        message_id: int,
        transcript: str,
        duration_seconds: int,
        reply_contexts: tuple[ReplyContext, ...] = (),
    ) -> DialogueInput | None:
        self.memory.purge()
        every = self.settings.context_summary_every
        if key is None or len(key) != 3 or not every or not self.memory.available:
            return None
        session = self.memory.sessions.get(key)
        if session is None or not self.allowed(key, session):
            return None
        previous = [
            e
            for e in session.entries
            if 0 <= sent_at - e.sent_at <= self.memory.ttl
            and (e.sent_at, max(e.message_ids)) < (sent_at, message_id)
        ]
        entries = previous + [
            Entry((message_id,), sent_at, transcript, "voice", duration_seconds, reply_contexts)
        ]
        start = 0
        for index in range(1, len(entries)):
            if (
                entries[index].sent_at - entries[index - 1].sent_at
                > self.settings.context_series_gap_seconds
            ):
                start = index
        # Old separate-message snapshots retain their watermark during the upgrade.
        through = max(session.summarized_through, session.claimed_through)
        unseen = [e for e in entries[start:] if e.kind == "voice" and max(e.message_ids) > through]
        if len(unseen) < every:
            return None
        voices = unseen[-every:]
        first_id = min(voices[0].message_ids)
        sources = [
            e
            for e in entries[start:]
            if max(e.message_ids) > through and (e.kind == "text" or min(e.message_ids) >= first_id)
        ]
        background = tuple(e.text for e in previous if e not in sources)
        references = bounded_contexts(
            tuple(context for entry in sources for context in entry.reply_contexts),
            min(6000, self.settings.context_max_chars // 2),
        )
        return DialogueInput(
            text="\n\n".join(
                f"{i}. {'Голосовое' if e.kind == 'voice' else 'Текст'}"
                + (f" (message_id={max(e.message_ids)})" if references else "")
                + f":\n{e.text}"
                for i, e in enumerate(sources, 1)
            ),
            background=background,
            reply_contexts=references,
            voice_count=len(voices),
            duration_seconds=sum(e.duration_seconds for e in voices),
        )

    def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self._run(), name="conversation-maintenance")

    async def _run(self) -> None:
        while True:
            try:
                await self.maintain()
            except Exception as exc:
                logger.warning("conversation_worker_failed error_type=%s", type(exc).__name__)
            await asyncio.sleep(1)

    async def maintain(self) -> None:
        self.memory.purge()
        for key, session in list(self.memory.sessions.items()):
            if not self.allowed(key, session):
                self.memory.sessions.pop(key, None)
        await self.memory.checkpoint()

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        await self.memory.checkpoint()
