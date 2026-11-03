"""A cumulative summary on every third delivered voice in an author's series."""

import asyncio
import logging
import time
from collections.abc import Callable
from html import escape

from aiogram import Bot
from aiogram.exceptions import TelegramRetryAfter
from aiogram.types import LinkPreviewOptions, ReplyParameters

from voice_bot.config import Settings
from voice_bot.conversation import ConversationKey, ConversationMemory, Entry, Session
from voice_bot.progress import typing
from voice_bot.providers import Summarizer, SummaryContext
from voice_bot.rendering import split_text

logger = logging.getLogger(__name__)


class DialogueSummaries:
    def __init__(
        self,
        memory: ConversationMemory,
        settings: Settings,
        summarizer: Summarizer,
        capacity: asyncio.Semaphore,
        busy: Callable[[ConversationKey, int], bool],
        allowed: Callable[[ConversationKey, Session], bool],
    ) -> None:
        self.memory = memory
        self.settings = settings
        self.summarizer = summarizer
        self.capacity = capacity
        self.busy = busy
        self.allowed = allowed
        self.task: asyncio.Task[None] | None = None
        self.jobs: dict[ConversationKey, asyncio.Task[None]] = {}

    def start(self, bot: Bot) -> None:
        if self.task is None and self.memory.ttl and self.settings.context_summary_every:
            self.task = asyncio.create_task(self._run(bot), name="conversation-summaries")

    async def _run(self, bot: Bot) -> None:
        while True:
            try:
                await self.process_due(bot, wait=False)
            except Exception as exc:
                logger.warning("conversation_worker_failed error_type=%s", type(exc).__name__)
            await asyncio.sleep(0.5)

    def source(self, key: ConversationKey, session: Session) -> tuple[Entry, ...]:
        every = self.settings.context_summary_every
        if not (
            every
            and self.memory.available
            and len(key) == 3
            and session.anchor
            and session.expires_at > self.memory.clock()
            and self.allowed(key, session)
        ):
            return ()
        # Completed voices can be interleaved with already received text. Event order wins.
        start = 0
        for index in range(1, len(session.entries)):
            if (
                session.entries[index].sent_at - session.entries[index - 1].sent_at
                > self.settings.context_series_gap_seconds
            ):
                start = index
        entries = session.entries[start:]
        through = max(session.summarized_through, session.claimed_through)
        unseen = 0
        for index, entry in enumerate(entries):
            if entry.kind == "voice" and max(entry.message_ids) > through:
                unseen += 1
                if unseen == every:
                    return tuple(entries[: index + 1])
        return ()

    async def process_due(self, bot: Bot, *, wait: bool = True) -> None:
        self.memory.purge()
        for key, session in list(self.memory.sessions.items()):
            if not self.allowed(key, session):
                self.memory.sessions.pop(key, None)
        if not await self.memory.checkpoint():
            return
        for key, session in list(self.memory.sessions.items()):
            if len(self.jobs) >= self.settings.max_concurrent_jobs:
                break
            if key in self.jobs:
                continue
            # Wait for earlier accepted voices before choosing the third source message.
            entries = self.source(key, session)
            if not entries or self.busy(key, max(entries[-1].message_ids)):
                continue
            task = asyncio.create_task(
                self._process_one(key, session, bot), name="dialogue-summary"
            )
            self.jobs[key] = task
            self.memory.sessions.move_to_end(key)
        if wait and self.jobs:
            await asyncio.gather(*list(self.jobs.values()))

    async def _process_one(self, key: ConversationKey, session: Session, bot: Bot) -> None:
        try:
            async with self.capacity:
                if self.memory.sessions.get(key) is not session:
                    return
                entries = self.source(key, session)
                if entries and not self.busy(key, max(entries[-1].message_ids)):
                    await self._summarize(key, session, entries, bot)
        except Exception as exc:
            logger.warning("conversation_job_failed error_type=%s", type(exc).__name__)
        finally:
            self.jobs.pop(key, None)

    async def _summarize(
        self, key: ConversationKey, session: Session, entries: tuple[Entry, ...], bot: Bot
    ) -> None:
        anchor = session.anchor
        assert anchor is not None
        target = max(entries[-1].message_ids)
        voices = sum(e.kind == "voice" for e in entries)
        texts = len(entries) - voices
        source = "\n\n".join(
            f"{i}. {'Голосовое' if e.kind == 'voice' else 'Текст'}:\n{e.text}"
            for i, e in enumerate(entries, 1)
        )
        context = SummaryContext(
            author_name=anchor.author_name,
            author_username=anchor.author_username,
            author_source="sender",
            chat_type=anchor.chat_type,
            duration_seconds=sum(e.duration_seconds for e in entries),
            is_conversation=True,
        )

        def current() -> bool:
            # Later messages do not cancel the summary on the third voice; source changes do.
            return (
                self.memory.sessions.get(key) is session
                and session.expires_at > self.memory.clock()
                and self.allowed(key, session)
                and tuple(
                    e
                    for e in session.entries
                    if (entries[0].sent_at, max(entries[0].message_ids))
                    <= (e.sent_at, max(e.message_ids))
                    <= (entries[-1].sent_at, target)
                )
                == entries
            )

        started = time.monotonic()
        async with typing(bot, key[0], key[1] or None, enabled=current):
            try:
                async with asyncio.timeout(self.settings.job_timeout_seconds):
                    summary = await self.summarizer.summarize(source, context=context)
                    if not current():
                        logger.info("conversation_summary_superseded")
                        return
                    session.claimed_through = target
                    if not await self.memory.checkpoint() or not current():
                        return
                    header = f"Кратко по серии · голосовых: {voices}, текстовых: {texts}"
                    for part in split_text(summary):
                        if not current():
                            return
                        for attempt in range(2):
                            try:
                                await bot.send_message(
                                    chat_id=key[0],
                                    message_thread_id=key[1] or None,
                                    text=f"<b>{header}</b>\n\n{escape(part)}",
                                    parse_mode="HTML",
                                    reply_parameters=ReplyParameters(
                                        message_id=target,
                                        allow_sending_without_reply=False,
                                    ),
                                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                                )
                                break
                            except TelegramRetryAfter as exc:
                                if (
                                    attempt
                                    or exc.retry_after
                                    > self.settings.telegram_max_retry_after_seconds
                                ):
                                    raise
                                await asyncio.sleep(exc.retry_after)
                                if not current():
                                    return
                    session.summarized_through = target
                    logger.info(
                        "conversation_summary_delivered voices=%s texts=%s elapsed_ms=%s",
                        voices,
                        texts,
                        round((time.monotonic() - started) * 1000),
                    )
            except Exception as exc:
                # Failed groups are retried with the next three voices; never loop paid calls.
                session.summarized_through = target
                logger.warning("conversation_summary_failed error_type=%s", type(exc).__name__)
            finally:
                await self.memory.checkpoint()

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        jobs = list(self.jobs.values())
        for task in jobs:
            task.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        await self.memory.checkpoint()
