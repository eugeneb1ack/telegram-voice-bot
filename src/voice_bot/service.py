import asyncio
import hashlib
import json
import logging
import tempfile
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest, TelegramRetryAfter
from aiogram.types import (
    LinkPreviewOptions,
    Message,
    MessageOriginChannel,
    MessageOriginChat,
    MessageOriginHiddenUser,
    MessageOriginUser,
    ReplyParameters,
    User,
)

from voice_bot.admin import GroupAdmin
from voice_bot.audio import BoundedFile, convert_audio
from voice_bot.batching import VoiceBatcher, batch_key
from voice_bot.config import Settings
from voice_bot.conversation import Anchor, ConversationKey, ConversationMemory, Session
from voice_bot.deduplication import ProcessedVoices, RecentResults, voice_keys
from voice_bot.delivery import ephemeral, forum_topic_id, recipient, reply_to
from voice_bot.dialogue import DialogueContext, DialogueInput
from voice_bot.errors import ProviderError, UserFacingError
from voice_bot.jobs import VoiceJobQueue
from voice_bot.journal import DeliveryUncertain, QueueJournal, request_key
from voice_bot.limits import NoticeLimiter, SenderKey, sender_key
from voice_bot.progress import typing
from voice_bot.providers import Summarizer, SummaryContext, Transcriber, VoicePartContext
from voice_bot.quotas import PrivateVoiceQuota
from voice_bot.rendering import render_batch, render_replies
from voice_bot.reply_context import (
    ReplyContext,
    as_voice,
    bounded_contexts,
    media_kind,
    reply_target,
)
from voice_bot.stt_context import vocabulary

logger = logging.getLogger(__name__)
T = TypeVar("T")
Converter = Callable[..., Awaitable[None]]


def summary_context(message: Message) -> SummaryContext:
    origin = getattr(message, "forward_origin", None)
    user = sender_chat = None
    name = username = None
    if isinstance(origin, MessageOriginUser):
        user = origin.sender_user
    elif isinstance(origin, MessageOriginHiddenUser):
        name = origin.sender_user_name
    elif isinstance(origin, MessageOriginChat):
        sender_chat = origin.sender_chat
    elif isinstance(origin, MessageOriginChannel):
        sender_chat = origin.chat
    elif origin is None:
        sender_chat = getattr(message, "sender_chat", None)
        if sender_chat is None:
            user = getattr(message, "from_user", None)
    if sender_chat is not None:
        name, username = sender_chat.title or None, sender_chat.username
    elif user is not None:
        name = getattr(user, "full_name", None)
        if not name or not name.strip():
            name = getattr(user, "first_name", None)
        username = user.username
    if name is not None:
        name = " ".join(name.split()) or None
    reply = getattr(message, "reply_to_message", None)
    reply_text = (
        (getattr(reply, "text", None) or getattr(reply, "caption", None)) if reply else None
    )
    voice = getattr(message, "voice", None)
    return SummaryContext(
        author_name=name,
        author_username=username,
        author_source=("forwarded" if origin is not None else "sender")
        if name or username
        else "unknown",
        is_forwarded=origin is not None,
        chat_type=getattr(message.chat, "type", None),
        chat_title=getattr(message.chat, "title", None),
        duration_seconds=voice.duration if voice else None,
        reply_to_text=reply_text[:1000] if reply_text else None,
    )


def conversation_position(message: Message) -> tuple[ConversationKey, float] | None:
    # Private /text results must never feed the public conversation.
    if recipient(message):
        return None
    origin = getattr(message, "forward_origin", None)
    user: User | None
    sent_at: datetime | None
    if isinstance(origin, MessageOriginUser):
        user, sent_at = origin.sender_user, origin.date
    elif origin is None and getattr(message, "sender_chat", None) is None:
        user = getattr(message, "from_user", None)
        sent_at = getattr(message, "date", None)
    else:
        return None
    if user is None or getattr(user, "is_bot", False) or sent_at is None:
        return None
    author_id = getattr(user, "id", None)
    if author_id is None:
        return None
    key = (message.chat.id, forum_topic_id(message) or 0, author_id)
    return key + ((1,) if origin is not None else ()), sent_at.timestamp()


class VoiceService:
    def __init__(
        self,
        settings: Settings,
        transcriber: Transcriber,
        summarizer: Summarizer,
        converter: Converter = convert_audio,
        admin: GroupAdmin | None = None,
        journal: QueueJournal | None = None,
    ) -> None:
        self.journal = journal
        self._durable_job: ContextVar[str | None] = ContextVar("durable_job", default=None)
        self._delivery_part: ContextVar[str | None] = ContextVar("delivery_part", default=None)
        self._journal_keys: dict[str, str] = {}
        self.settings = settings
        self.transcriber = transcriber
        self.summarizer = summarizer
        self.converter = converter
        self.active_jobs = 0
        self.admin = admin
        self.processed = ProcessedVoices(
            settings.processed_state_file, settings.processed_max_messages
        )
        self.results = RecentResults()
        self.private_quota = PrivateVoiceQuota(
            settings.private_usage_file, settings.private_daily_voice_limit
        )
        self.notices = NoticeLimiter()
        self.replays = NoticeLimiter(sender_interval=3, chat_interval=1)
        self._quota_charged: set[str] = set()
        self._sent_ids: dict[str, tuple[int, ...]] = {}
        self._owners: dict[str, SenderKey] = {}
        self._pending_by_sender: Counter[SenderKey] = Counter()
        self._requesters: dict[str, Message] = {}
        self._capacity = asyncio.Semaphore(settings.max_concurrent_jobs)
        self.memory = ConversationMemory(
            settings.context_ttl_seconds,
            settings.context_max_chars,
            settings.context_max_sessions,
            max_entries=settings.context_max_messages,
            path=settings.context_state_file,
        )
        self.dialogue = DialogueContext(self.memory, settings, self._context_allowed)
        self.batcher = VoiceBatcher(
            self._enqueue_batch,
            settings.voice_batch_pause_seconds,
            settings.voice_batch_max_messages,
            settings.max_pending_jobs,
        )
        self.queue = VoiceJobQueue(
            self._handle_reserved_messages,
            settings.max_concurrent_jobs,
            settings.max_pending_jobs,
            settings.max_pending_jobs_per_chat,
        )
        self._jobs: set[asyncio.Task[object]] = set()

    def _context_allowed(self, key: ConversationKey, session: Session) -> bool:
        if self.settings.allowed_chat_ids and key[0] not in self.settings.allowed_chat_ids:
            return False
        if self.admin is not None and key[0] < 0:
            return self.admin.store.allowed(key[0]) and self.admin.store.automatic(key[0])
        return True

    def start(self, bot: Bot) -> None:
        self.dialogue.start()

    async def recover(self, bot: Bot) -> None:
        if self.journal is None:
            return
        restored = 0
        for job_id, state, items in self.journal.recover(bot):
            messages = [item[0] for item in items]
            if not all(self.work_allowed(message) for message in messages):
                self.journal.finish(job_id)
                continue
            for message, requester, charged in items:
                key = voice_keys(message)[0]
                self._journal_keys[key] = job_id
                self.processed.pending.update(voice_keys(message))
                if requester is not None:
                    self._requesters[key] = requester
                if charged:
                    self._quota_charged.add(key)
                actor = (
                    sender_key(requester or message)
                    if message.chat.type in {"group", "supergroup"}
                    else None
                )
                if actor is not None:
                    self._owners[key] = actor
                    self._pending_by_sender[actor] += 1
            if state == "collecting" and len(messages) == 1:
                if not self.batcher.add(messages[0], bot):
                    raise RuntimeError("Cannot restore forwarded voice buffer")
            elif not self.queue.add(messages, bot, restoring=True):
                raise RuntimeError("Cannot restore voice queue")
            restored += len(messages)
        logger.info("voice_queue_recovered voices=%s", restored)

    def _result_get(self, key: str) -> str | None:
        job_id = self._durable_job.get()
        if self.journal is not None and job_id is not None:
            value = self.journal.get(job_id)["cache"].get(key)
            if isinstance(value, str):
                return value
        return self.results.get(key)

    def _result_put(self, key: str, value: str) -> None:
        job_id = self._durable_job.get()
        if self.journal is not None and job_id is not None:
            cache = self.journal.get(job_id)["cache"]
            cache[key] = value
            self.journal.set(job_id, "cache", cache)
        self.results.put(key, value)

    async def _deliver(
        self, part: str, action: Callable[[], Awaitable[T]], identify: Callable[[T], int | None]
    ) -> T | int:
        journal, job_id = self.journal, self._durable_job.get()
        if journal is None or job_id is None:
            return await action()
        known = journal.begin_send(job_id, part)
        if known is not None:
            return known
        try:
            result = await action()
        except (TelegramBadRequest, TelegramRetryAfter):
            journal.rejected(job_id, part)
            raise
        except BaseException:
            journal.set(job_id, "state", "uncertain")
            logger.error("voice_delivery_uncertain job_id=%s", job_id)
            raise
        journal.sent(job_id, part, identify(result))
        return result

    def _admit_durable(
        self, message: Message, requester: Message | None, *, collecting: bool
    ) -> bool:
        if self.journal is None:
            return True
        job_id = self.journal.admit(
            message,
            requester,
            collecting=collecting,
            charged=voice_keys(message)[0] in self._quota_charged,
        )
        if job_id is None:
            return False
        self._journal_keys[voice_keys(message)[0]] = job_id
        return True

    async def observe_text(self, message: Message) -> None:
        text = getattr(message, "text", None)
        if (
            self.batcher.closed
            or not text
            or text.lstrip().startswith("/")
            or getattr(message, "forward_origin", None) is not None
            or not self.work_allowed(message)
        ):
            return
        position = conversation_position(message)
        if position is None:
            return
        key, sent_at = position
        self.memory.remember(key, sent_at, (message.message_id,), text, kind="text")
        await self.memory.checkpoint()

    def chat_allowed(self, message: Message) -> bool:
        if self.admin is not None:
            return self.admin.allowed(message)
        return (
            not self.settings.allowed_chat_ids or message.chat.id in self.settings.allowed_chat_ids
        )

    def work_allowed(self, message: Message) -> bool:
        return self.chat_allowed(message) and (
            recipient(message) is not None
            or self.admin is None
            or self.admin.store.automatic(message.chat.id)
        )

    async def send(self, message: Message, text: str) -> int | None:
        for attempt in range(2):
            try:

                async def transmit() -> Message:
                    return await message.answer(
                        text,
                        parse_mode="HTML",
                        reply_parameters=reply_to(message),
                        ephemeral_message_parameters=ephemeral(message),
                        link_preview_options=LinkPreviewOptions(is_disabled=True),
                    )

                sent = await self._deliver(
                    self._delivery_part.get()
                    or "text:" + hashlib.sha256(text.encode()).hexdigest(),
                    transmit,
                    lambda value: value.message_id if not recipient(message) else 0,
                )
                if isinstance(sent, int):
                    return sent or None
                if isinstance(sent, Message):
                    logger.debug(
                        "telegram_reply_sent chat_id=%s message_id=%s "
                        "reply_to=%s collapsed_blocks=%s",
                        sent.chat.id,
                        sent.message_id,
                        sent.reply_to_message.message_id if sent.reply_to_message else None,
                        sum(
                            entity.type == "expandable_blockquote" for entity in sent.entities or []
                        ),
                    )
                sent_id = getattr(sent, "message_id", None)
                return (
                    sent_id
                    if not recipient(message) and isinstance(sent_id, int) and sent_id > 0
                    else None
                )
            except TelegramRetryAfter as exc:
                if attempt or exc.retry_after > self.settings.telegram_max_retry_after_seconds:
                    raise
                await asyncio.sleep(exc.retry_after)
        return None

    async def handle(self, message: Message, bot: Bot) -> None:
        await self._handle_messages([message], bot)

    def _release(self, message: Message, *, failed: bool = False) -> None:
        key = voice_keys(message)[0]
        actor = self._owners.pop(key, None)
        if actor is not None:
            self._pending_by_sender[actor] -= 1
            if not self._pending_by_sender[actor]:
                del self._pending_by_sender[actor]
        self._requesters.pop(key, None)
        self._quota_charged.discard(key)
        self.processed.release(message, failed=failed)

    async def _charge_private(self, message: Message) -> bool:
        if getattr(message.chat, "type", None) != "private":
            return True
        key = voice_keys(message)[0]
        if key in self._quota_charged:
            return True
        decision = await self.private_quota.admit(message)
        if decision == "allowed":
            self._quota_charged.add(key)
            return True
        if decision == "notify":
            await self.notify(
                message,
                f"В личке доступно {self.settings.private_daily_voice_limit} голосовых в день. "
                "Лимит обновится в 00:00 UTC.",
                force=True,
            )
        return False

    async def _refund_private(self, message: Message) -> None:
        if voice_keys(message)[0] in self._quota_charged:
            await self.private_quota.refund(message)

    async def _claim(self, message: Message, requester: Message | None = None) -> bool:
        if self.journal is not None and self.journal.seen(request_key(message, requester)):
            return False
        if not self.processed.claim(message, completed_ok=True):
            return False
        target = requester or message
        actor = (
            sender_key(target)
            if getattr(message.chat, "type", None) in {"group", "supergroup"}
            else None
        )
        if actor is not None:
            if self._pending_by_sender[actor] >= self.settings.max_pending_voices_per_user:
                self.processed.release(message, failed=True)
                await self.notify(target, "Ваши голосовые уже в очереди. Дождитесь обработки.")
                return False
            self._owners[voice_keys(message)[0]] = actor
            self._pending_by_sender[actor] += 1
        if requester is not None:
            self._requesters[voice_keys(message)[0]] = requester
        accepted = False
        try:
            accepted = bool(self.processed.reply_ids(message)) or await self._charge_private(
                message
            )
            return accepted
        except OSError:
            logger.error("private_quota_storage_unavailable")
            return False
        finally:
            if not accepted:
                self._release(message, failed=True)

    async def enqueue(
        self, message: Message, bot: Bot, *, requester: Message | None = None
    ) -> None:
        if (
            self.work_allowed(message)
            and not self.batcher.closed
            and await self._claim(message, requester)
        ):
            if self._admit_durable(message, requester, collecting=False):
                await self._enqueue_batch([message], bot)
            else:
                await self._refund_private(message)
                self._release(message, failed=True)
                await self._send_error(message, "Очередь голосовых заполнена. Попробуйте позже.")

    async def _enqueue_batch(self, messages: list[Message], bot: Bot) -> None:
        message = messages[-1]
        queued = False
        rejected = False
        try:
            if self.admin is not None and message.chat.type in {"group", "supergroup"}:
                await self.admin.observe(message.chat, bot)
            if not self.work_allowed(message):
                return
            if self.journal is not None:
                keys = list(
                    dict.fromkeys(self._journal_keys[voice_keys(item)[0]] for item in messages)
                )
                job_id = self.journal.merge(keys)
                for item in messages:
                    self._journal_keys[voice_keys(item)[0]] = job_id
            queued = self.queue.add(messages, bot)
            if not queued:
                rejected = True
                await self._send_error(message, "Очередь голосовых заполнена. Попробуйте позже.")
            else:
                logger.info(
                    "voice_job_queued message_id=%s pending=%s", message.message_id, self.queue.size
                )
        finally:
            if not queued:
                if self.journal is not None and (rejected or not self.work_allowed(message)):
                    for job_id in {
                        self._journal_keys.pop(voice_keys(item)[0], "") for item in messages
                    }:
                        if job_id:
                            self.journal.finish(job_id)
                for item in messages:
                    try:
                        await self._refund_private(item)
                    except OSError:
                        logger.error("private_quota_storage_unavailable")
                    finally:
                        self._release(item, failed=rejected)

    async def accept(self, message: Message, bot: Bot) -> None:
        if self.batcher.closed:
            return
        if self.admin is not None and message.chat.type in {"group", "supergroup"}:
            await self.admin.observe(message.chat, bot)
        if not self.work_allowed(message):
            return
        if not self.settings.voice_batch_pause_seconds or batch_key(message) is None:
            await self.enqueue(message, bot)
            return
        if not await self._claim(message):
            return
        accepted = False
        rejected = False
        try:
            if self.admin is not None and message.chat.type in {"group", "supergroup"}:
                await self.admin.observe(message.chat, bot)
            if not self.chat_allowed(message):
                return
            accepted = self._admit_durable(message, None, collecting=True) and self.batcher.add(
                message, bot
            )
            if not accepted:
                rejected = True
                await self._send_error(message, "Очередь голосовых заполнена. Попробуйте позже.")
        finally:
            if not accepted:
                job_id = self._journal_keys.pop(voice_keys(message)[0], None)
                if self.journal is not None and job_id is not None:
                    self.journal.finish(job_id)
                try:
                    await self._refund_private(message)
                except OSError:
                    logger.error("private_quota_storage_unavailable")
                finally:
                    self._release(message, failed=rejected)

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.job_timeout_seconds
        self.batcher.finish()
        await self.dialogue.close()
        tasks = set(self.batcher.tasks)
        if tasks:
            try:
                await asyncio.wait(tasks, timeout=max(0, deadline - loop.time()))
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        await self.queue.close(grace_seconds=max(0, deadline - loop.time()))
        direct_tasks = set(self._jobs)
        if direct_tasks:
            try:
                await asyncio.wait(direct_tasks, timeout=max(0, deadline - loop.time()))
            finally:
                for direct_task in direct_tasks:
                    if not direct_task.done():
                        direct_task.cancel()
                await asyncio.gather(*direct_tasks, return_exceptions=True)
        await self.memory.checkpoint()
        self.memory.clear()
        self.processed.pending.clear()
        self._owners.clear()
        self._pending_by_sender.clear()
        self._requesters.clear()
        self._quota_charged.clear()
        self._sent_ids.clear()
        self.results.clear()

    async def _handle_messages(self, messages: list[Message], bot: Bot) -> None:
        claimed = [message for message in messages if await self._claim(message)]
        if claimed:
            await self._handle_reserved_messages(claimed, bot)

    async def _handle_reserved_messages(self, messages: list[Message], bot: Bot) -> None:
        message = messages[-1]
        async with (
            self._capacity,
            typing(
                bot,
                message.chat.id,
                forum_topic_id(message),
                enabled=lambda: not recipient(message) and self.work_allowed(message),
            ),
        ):
            job_id = self._journal_keys.get(voice_keys(message)[0])
            token = self._durable_job.set(job_id)
            try:
                finished = await self._run_reserved_messages(messages, bot)
                if finished and self.journal is not None and job_id is not None:
                    self.journal.finish(job_id)
                    for item in messages:
                        self._journal_keys.pop(voice_keys(item)[0], None)
            finally:
                self._durable_job.reset(token)

    async def _run_reserved_messages(self, messages: list[Message], bot: Bot) -> bool:
        finished = False
        message = messages[-1]
        started = time.monotonic()
        admitted = False
        completed: list[Message] = []
        task = asyncio.current_task()
        try:
            if self.admin is not None and message.chat.type in {"group", "supergroup"}:
                await self.admin.observe(message.chat, bot)
            if not self.work_allowed(message):
                logger.debug("voice_ignored reason=chat_not_allowed chat_id=%s", message.chat.id)
                return True
            if not self.processed.available:
                raise UserFacingError("Сервис временно недоступен. Попробуйте позже.")
            if self.active_jobs >= self.settings.max_concurrent_jobs:
                raise UserFacingError("Бот занят другими голосовыми. Отправьте сообщение позже.")
            # No await between checking and incrementing: admission is atomic on this event loop.
            self.active_jobs += 1
            admitted = True
            if task is not None:
                self._jobs.add(task)
            logger.info(
                "voice_job_started message_id=%s parts=%s", message.message_id, len(messages)
            )
            positions = [conversation_position(item) for item in messages]
            key = positions[0][0] if positions[0] is not None else None
            if not all(position is not None and position[0] == key for position in positions):
                key = None
            times = [position[1] for position in positions if position is not None]
            if times and max(times) - min(times) > self.settings.context_ttl_seconds:
                key = None
            deadline = asyncio.get_running_loop().time() + self.settings.job_timeout_seconds
            async with self.memory.ordered(key, deadline):
                job_id = self._durable_job.get()
                prepared = (
                    self.journal.get(job_id)["result"]
                    if self.journal is not None and job_id is not None
                    else None
                )
                saved = (
                    ()
                    if recipient(message) or prepared is not None
                    else self.processed.reply_ids(messages[0])
                )
                copy_plan = prepared is not None and "copy_ids" in prepared
                if copy_plan:
                    assert prepared is not None
                    saved = tuple(prepared["copy_ids"])
                copied = None
                if saved and (
                    copy_plan or all(self.processed.reply_ids(item) == saved for item in messages)
                ):
                    if self.journal is not None and job_id is not None and not copy_plan:
                        self.journal.set(job_id, "result", {"copy_ids": saved})
                    async with asyncio.timeout_at(deadline):
                        copied = await self._copy_result(message, bot, saved)
                if copied is not None:
                    completed = messages
                    for item in messages:
                        await self._refund_private(item)
                    if not copied:
                        return True
                    self._sent_ids[voice_keys(message)[0]] = copied
                    logger.info("voice_job_replayed message_id=%s", message.message_id)
                else:
                    if self.journal is not None and job_id is not None and (copy_plan or saved):
                        self.journal.set(job_id, "result", None)
                    completed = await self._process(messages, bot, key, times, deadline)
                try:
                    await self.processed.complete(
                        completed, self._sent_ids.pop(voice_keys(message)[0], ())
                    )
                except OSError:
                    # The result was delivered. Stop admitting paid work if receipts fail.
                    logger.error("processing_receipts_unavailable")
                else:
                    finished = True
        except TimeoutError:
            await self._send_error(
                message, "Обработка заняла слишком много времени. Попробуйте позже."
            )
        except UserFacingError as exc:
            await self._send_error(message, str(exc))
            finished = True
        except DeliveryUncertain:
            logger.error("voice_delivery_uncertain")
        except TelegramAPIError:
            # Never log Telegram exceptions: they may contain tokens or request bodies.
            logger.warning("telegram_delivery_failed")
        except Exception as exc:
            logger.warning("voice_processing_failed error_type=%s", type(exc).__name__)
            await self._send_error(message, "Не удалось обработать голосовое. Попробуйте позже.")
        finally:
            self._sent_ids.pop(voice_keys(message)[0], None)
            completed_ids = {item.message_id for item in completed}
            for item in messages:
                self._release(item, failed=admitted and item.message_id not in completed_ids)
            if admitted:
                self.active_jobs -= 1
            if task is not None:
                self._jobs.discard(task)
            logger.info(
                "voice_job_finished message_id=%s elapsed_ms=%s",
                message.message_id,
                round((time.monotonic() - started) * 1000),
            )

        return finished

    def _transcription_keywords(self, message: Message) -> tuple[str, ...]:
        if (
            self.settings.stt_model != "gpt-transcribe"
            or not self.memory.available
            or not self.memory.ttl
        ):
            return ()
        self.memory.purge()
        typed_sources: list[str] = []
        recognized_sources: list[str] = []
        if caption := getattr(message, "caption", None):
            typed_sources.append(caption)
        target = reply_target(message)
        if target is not None:
            if text := getattr(target, "text", None) or getattr(target, "caption", None):
                typed_sources.append(text)
            if media_kind(target) in {"voice", "video_note"}:
                known = self.memory.source(message.chat.id, target.message_id)
                reference = self.memory.reply_source(message.chat.id, target.message_id)
                if known:
                    recognized_sources.append(known)
                elif reference and reference.kind == media_kind(target):
                    recognized_sources.append(reference.text)
        position = conversation_position(message)
        if position is not None:
            key, sent_at = position
            if session := self.memory.sessions.get(key):
                entries = [
                    entry
                    for entry in session.entries
                    if 0 <= sent_at - entry.sent_at <= self.settings.context_series_gap_seconds
                    and (entry.sent_at, max(entry.message_ids)) < (sent_at, message.message_id)
                ][-3:]
                # Typed spelling takes precedence over earlier recognition candidates.
                typed_sources.extend(e.text for e in reversed(entries) if e.kind == "text")
                recognized_sources.extend(e.text for e in reversed(entries) if e.kind == "voice")
        return vocabulary((*typed_sources, *recognized_sources))

    async def _transcribe(self, message: Message, bot: Bot) -> str:
        settings, voice = self.settings, message.voice
        assert voice is not None
        if voice.duration > settings.max_voice_seconds:
            raise UserFacingError(
                f"Голосовое длиннее лимита {settings.max_voice_seconds} секунд. "
                "Отправьте запись короче."
            )
        if voice.file_size and voice.file_size > settings.max_download_bytes:
            raise UserFacingError("Голосовое слишком большое для обработки.")
        cache_key = "transcript:" + voice_keys(message, scoped=False)[-1]
        cached = self._result_get(cache_key)
        if cached is not None:
            return cached
        hint_key = "stt-keywords:" + voice_keys(message)[0]
        saved_hints = self._result_get(hint_key)
        if saved_hints is None:
            hints = self._transcription_keywords(message)
            # Freeze the spelling hints with the job before any STT request/retry.
            self._result_put(hint_key, json.dumps(hints, ensure_ascii=False))
        else:
            hints = tuple(json.loads(saved_hints))
        with tempfile.TemporaryDirectory(prefix="voice-bot-") as directory:
            source, wav = Path(directory) / "source.ogg", Path(directory) / "voice.wav"
            async with asyncio.timeout(settings.download_timeout_seconds):
                file = await bot.get_file(voice.file_id)
                if file.file_size and file.file_size > settings.max_download_bytes:
                    raise UserFacingError("Голосовое слишком большое для обработки.")
                if not file.file_path:
                    raise UserFacingError("Telegram не предоставил файл голосового.")
                with BoundedFile(source, settings.max_download_bytes) as output:
                    await bot.download_file(
                        file.file_path,
                        output,
                        timeout=settings.download_timeout_seconds,
                        seek=False,
                    )
            logger.info("audio_downloaded message_id=%s", message.message_id)
            await self.converter(
                source,
                wav,
                binary=settings.ffmpeg_binary,
                max_bytes=settings.max_upload_bytes,
                timeout_seconds=settings.conversion_timeout_seconds,
            )
            logger.info("audio_converted message_id=%s", message.message_id)
            if not self.work_allowed(message):
                raise UserFacingError("Обработка в этом чате отключена.")
            transcript = (
                await self.transcriber.transcribe(wav, keywords=hints)
                if hints
                else await self.transcriber.transcribe(wav)
            )
            self._result_put(cache_key, transcript)
            logger.info("transcription_completed message_id=%s", message.message_id)
            return transcript

    async def _reply_context(
        self, message: Message, bot: Bot, *, deadline: float | None = None
    ) -> ReplyContext | None:
        target = reply_target(message)
        if target is None:
            return None
        kind = media_kind(target)
        if kind is None:
            return None
        cache_key = "reply-context:" + voice_keys(message)[0]
        cached = self._result_get(cache_key)
        if cached is not None:
            saved = ReplyContext(**json.loads(cached))
            if saved.kind == kind and saved.reply_to_message_id == target.message_id:
                return saved
        previous = self.memory.reply_source(message.chat.id, target.message_id)
        if kind != "text" and previous is not None and previous.kind == kind:
            author = summary_context(target)
            context = replace(
                previous,
                message_id=message.message_id,
                author_name=author.author_name,
                author_username=author.author_username,
            )
            self._result_put(cache_key, json.dumps(asdict(context), ensure_ascii=False))
            return context
        caption = getattr(target, "text", None) or getattr(target, "caption", None) or ""
        source = caption if kind == "text" else ""
        unavailable = None
        if kind != "text":
            # Exact message lookup: no unrelated history of the other author is read.
            source = self.memory.source(message.chat.id, target.message_id) or ""
            media = as_voice(target)
            if not source and media is not None:
                try:
                    if recipient(message):
                        media = media.model_copy(update={"receiver_user": message.receiver_user})
                    remaining = 45.0
                    if deadline is not None:
                        remaining = max(
                            0.0, min(remaining, deadline - asyncio.get_running_loop().time() - 15)
                        )
                    async with asyncio.timeout(remaining):
                        source = await self._transcribe(media, bot)
                except (ProviderError, UserFacingError, TimeoutError, TelegramAPIError):
                    unavailable = "Не удалось получить речь из исходного сообщения."
                    logger.info(
                        "reply_context_unavailable message_id=%s target_id=%s",
                        message.message_id,
                        target.message_id,
                    )
                except Exception as exc:
                    logger.warning("reply_context_failed error_type=%s", type(exc).__name__)
                    unavailable = "Контекст исходного сообщения недоступен."
            if caption:
                source = f"Подпись: {caption}\nРечь: {source}" if source else f"Подпись: {caption}"
        if not source and not unavailable:
            return None
        author = summary_context(target)
        context = bounded_contexts(
            (
                ReplyContext(
                    message_id=message.message_id,
                    reply_to_message_id=target.message_id,
                    author_name=author.author_name,
                    author_username=author.author_username,
                    kind=kind,
                    text=source,
                    unavailable_reason=unavailable,
                ),
            ),
            min(3000, self.settings.context_max_chars // 4),
        )[0]
        self._result_put(cache_key, json.dumps(asdict(context), ensure_ascii=False))
        return context

    async def _process(
        self,
        messages: list[Message],
        bot: Bot,
        key: ConversationKey | None,
        times: list[float],
        deadline: float,
    ) -> list[Message]:
        job_id = self._durable_job.get()
        if self.journal is not None and job_id is not None:
            prepared = self.journal.get(job_id)["result"]
            if prepared is not None:
                return await self._deliver_prepared(messages, bot, prepared)
        message = messages[-1]
        is_batch = len(messages) > 1
        assert message.voice is not None
        needs_summary = (
            is_batch or message.voice.duration >= self.settings.summary_min_voice_seconds
        )
        ids = tuple(item.message_id for item in messages)
        session = self.memory.sessions.get(key) if key is not None else None
        if (
            not is_batch
            and key is not None
            and len(key) == 3
            and self.settings.context_summary_every
            and self.memory.available
            and session is not None
            and times
            and any(
                entry.kind == "voice"
                and 0 <= min(times) - entry.sent_at <= self.settings.context_series_gap_seconds
                and max(entry.message_ids) < min(ids)
                for entry in session.entries
            )
        ):
            needs_summary = False
        summary = (
            "Не удалось составить краткое содержание. Полная расшифровка ниже."
            if needs_summary
            else None
        )
        dialogue: DialogueInput | None = None
        reply_contexts: tuple[ReplyContext, ...] = ()
        transcripts: list[str | None] = [None] * len(messages)
        failures = ["Не обработано: превышено время ожидания."] * len(messages)
        try:
            async with asyncio.timeout_at(deadline):
                for index, item in enumerate(messages):
                    if not self.work_allowed(item):
                        return []
                    if not await self._charge_private(item):
                        if not is_batch:
                            return []
                        failures[index] = "Достигнут дневной лимит."
                        continue
                    try:
                        transcripts[index] = await self._transcribe(item, bot)
                    except (UserFacingError, ProviderError) as exc:
                        if not is_batch:
                            raise
                        failures[index] = str(exc)
                    except Exception as exc:
                        if not is_batch:
                            raise
                        logger.warning("batch_part_failed error_type=%s", type(exc).__name__)
                        failures[index] = "Не удалось распознать запись."
                if not self.work_allowed(message):
                    return []
                references: list[ReplyContext] = []
                for item, recognized in zip(messages, transcripts, strict=True):
                    if recognized is not None:
                        reference = await self._reply_context(item, bot, deadline=deadline)
                        if reference is not None:
                            references.append(reference)
                reply_contexts = bounded_contexts(
                    tuple(references), min(6000, self.settings.context_max_chars // 2)
                )
                transcript = self._joined(transcripts, failures)
                summary_input = transcript
                if not is_batch and transcripts[0] is not None:
                    dialogue = self.dialogue.prepare(
                        key,
                        min(times) if times else 0,
                        message.message_id,
                        transcript,
                        message.voice.duration,
                        reply_contexts=reply_contexts,
                    )
                    if dialogue is not None:
                        needs_summary = True
                        summary_input = dialogue.text
                        summary = (
                            "Не удалось составить краткое содержание. Полная расшифровка ниже."
                        )
                if needs_summary and any(part is not None for part in transcripts):
                    context = summary_context(messages[0])
                    context = replace(
                        context,
                        recent_summaries=self.memory.recent(key, min(times) if times else 0, ids),
                        reply_to_text=None,
                        reply_contexts=reply_contexts,
                    )
                    if dialogue is not None:
                        context = replace(
                            context,
                            recent_summaries=dialogue.background,
                            duration_seconds=dialogue.duration_seconds,
                            is_conversation=True,
                            reply_contexts=dialogue.reply_contexts,
                        )
                    if is_batch:
                        if reply_contexts:
                            summary_input = "\n\n".join(
                                f"Сообщение {item.message_id}:\n"
                                f"{part if part is not None else failures[index]}"
                                for index, (item, part) in enumerate(
                                    zip(messages, transcripts, strict=True)
                                )
                            )
                        authors = [summary_context(item) for item in messages]
                        same_author = all(
                            (item.author_name, item.author_username)
                            == (context.author_name, context.author_username)
                            for item in authors
                        )
                        context = replace(
                            context,
                            author_name=context.author_name if same_author else None,
                            author_username=context.author_username if same_author else None,
                            author_source=context.author_source if same_author else "multiple",
                            duration_seconds=sum(item.duration_seconds or 0 for item in authors),
                            voices=tuple(
                                VoicePartContext(
                                    i, item.author_name, item.author_username, item.duration_seconds
                                )
                                for i, item in enumerate(authors, 1)
                            ),
                        )
                    try:
                        cache_key = (
                            "summary:"
                            + ":".join(voice_keys(item, scoped=False)[0] for item in messages)
                            + ":"
                            + hashlib.sha256(summary_input.encode()).hexdigest()
                            + ":"
                            + hashlib.sha256(
                                json.dumps(asdict(context), sort_keys=True).encode()
                            ).hexdigest()
                        )
                        cached_summary = self._result_get(cache_key)
                        if cached_summary is None:
                            cached_summary = await self.summarizer.summarize(
                                summary_input, context=context
                            )
                            self._result_put(cache_key, cached_summary)
                        summary = cached_summary
                        logger.info("summary_completed message_id=%s", message.message_id)
                    except Exception as exc:
                        logger.warning("summary_unavailable error_type=%s", type(exc).__name__)
        except TimeoutError:
            if not any(part is not None for part in transcripts):
                raise
            logger.warning("voice_job_timeout")
        if not self.work_allowed(message):
            return []
        transcript = self._joined(transcripts, failures)
        if is_batch and all(part is None for part in transcripts):
            summary = "Не удалось распознать записи. Причины указаны в общей расшифровке."
        elif is_batch and any(part is None for part in transcripts):
            summary = "Часть записей не обработана.\n\n" + (summary or "")
        title = f"Коротко · {dialogue.voice_count} голосовых:" if dialogue else "Коротко:"
        prepared = {
            "summary": summary,
            "transcript": transcript,
            "title": title,
            "valid": [part is not None for part in transcripts],
            "key": key,
            "times": times,
            "dialogue": dialogue is not None,
            "reply_contexts": [asdict(context) for context in reply_contexts],
        }
        if self.journal is not None and job_id is not None:
            self.journal.set(job_id, "result", prepared)
        return await self._deliver_prepared(messages, bot, prepared)

    async def _deliver_prepared(
        self, messages: list[Message], bot: Bot, prepared: dict[str, Any]
    ) -> list[Message]:
        message = messages[-1]
        if not self.work_allowed(message):
            return []
        sent_ids = await self._send_result(
            message,
            bot,
            prepared["summary"],
            prepared["transcript"],
            len(messages),
            summary_title=prepared["title"],
        )
        if sent_ids is None:
            return []
        self._sent_ids[voice_keys(message)[0]] = sent_ids
        logger.info("voice_job_delivered message_id=%s", message.message_id)
        key = tuple(prepared["key"]) if prepared["key"] is not None else None
        if key is not None:
            # Old prepared jobs may have mistaken an ordinary reply thread for a topic.
            # Normalize only routing; keep author/forward scope and frozen output intact.
            key = (key[0], forum_topic_id(message) or 0, *key[2:])
        times = prepared["times"]
        if all(prepared["valid"]) and self.chat_allowed(message):
            self.memory.remember(
                key,
                max(times) if times else 0,
                tuple(item.message_id for item in messages),
                prepared["transcript"],
                reply_contexts=tuple(
                    ReplyContext(**context) for context in prepared.get("reply_contexts", [])
                ),
                duration_seconds=sum(item.voice.duration for item in messages if item.voice),
                anchor=Anchor(
                    message.message_id,
                    summary_context(message).author_name,
                    summary_context(message).author_username,
                    getattr(message.chat, "type", None),
                )
                if len(messages) == 1
                else None,
            )
            if prepared["dialogue"] and key is not None and key in self.memory.sessions:
                self.memory.sessions[key].summarized_through = message.message_id
            await self.memory.checkpoint()
        return [item for item, valid in zip(messages, prepared["valid"], strict=True) if valid]

    @staticmethod
    def _joined(transcripts: list[str | None], failures: list[str]) -> str:
        if len(transcripts) == 1:
            return transcripts[0] if transcripts[0] is not None else failures[0]
        return "\n\n".join(
            f"Голосовое {i + 1}/{len(transcripts)}:\n"
            + (part if part is not None else f"[{failures[i]}]")
            for i, part in enumerate(transcripts)
        )

    async def _send_result(
        self,
        message: Message,
        bot: Bot,
        summary: str | None,
        transcript: str,
        count: int,
        *,
        summary_title: str = "Коротко:",
    ) -> tuple[int, ...] | None:
        content_hash = hashlib.sha256(((summary or "") + "\0" + transcript).encode()).hexdigest()
        delivery_key = "delivery:" + voice_keys(message)[0] + ":" + content_hash
        if recipient(message):
            requester = self._requesters.get(voice_keys(message)[0], message)
            # Telegram may reuse an ephemeral ID after expiry. A later command is a new delivery.
            delivery_key += (
                f":{requester.ephemeral_message_id or requester.message_id}"
                f":{requester.date.timestamp()}"
            )
        if count == 1:
            return await self._send_text_result(
                message,
                summary,
                transcript,
                delivery_key,
                summary_title=summary_title,
            )
        if self._result_get("fallback:" + delivery_key) is None:
            try:
                for attempt in range(2):
                    try:

                        async def transmit_rich() -> Message:
                            return await bot.send_rich_message(
                                chat_id=message.chat.id,
                                message_thread_id=forum_topic_id(message),
                                rich_message=render_batch(summary, transcript, count),
                                ephemeral_message_parameters=ephemeral(message),
                                reply_parameters=ReplyParameters(
                                    message_id=message.message_id, allow_sending_without_reply=False
                                ),
                            )

                        sent = await self._deliver(
                            "rich",
                            transmit_rich,
                            lambda value: value.message_id if not recipient(message) else 0,
                        )
                        if isinstance(sent, int):
                            return (sent,) if sent else ()
                        sent_id = getattr(sent, "message_id", None)
                        return (
                            (sent_id,)
                            if not recipient(message) and isinstance(sent_id, int)
                            else ()
                        )
                    except TelegramRetryAfter as exc:
                        if (
                            attempt
                            or exc.retry_after > self.settings.telegram_max_retry_after_seconds
                        ):
                            raise
                        await asyncio.sleep(exc.retry_after)
            except TelegramBadRequest:
                # A definitive rejection is safe to fall back from, unlike a network timeout.
                logger.warning("rich_message_rejected")
                self._result_put("fallback:" + delivery_key, "1")
        summary = f"Получено голосовых: {count}\n\n{summary or ''}"
        return await self._send_text_result(message, summary, transcript, delivery_key)

    async def _send_text_result(
        self,
        message: Message,
        summary: str | None,
        transcript: str,
        delivery_key: str,
        *,
        summary_title: str = "Коротко:",
    ) -> tuple[int, ...] | None:
        sent_parts = int(self._result_get(delivery_key) or "0")
        ids_key = "ids:" + delivery_key
        sent_ids: list[int] = json.loads(self._result_get(ids_key) or "[]")
        for index, reply in enumerate(
            render_replies(summary, transcript, summary_title=summary_title)
        ):
            if index < sent_parts:
                continue
            if not self.work_allowed(message):
                return None
            token = self._delivery_part.set(f"text:{delivery_key}:{index}")
            try:
                sent_id = await self.send(message, reply)
            finally:
                self._delivery_part.reset(token)
            if isinstance(sent_id, int):
                sent_ids.append(sent_id)
                self._result_put(ids_key, json.dumps(sent_ids))
            self._result_put(delivery_key, str(index + 1))
        return tuple(sent_ids)

    async def _copy_result(
        self, message: Message, bot: Bot, saved: tuple[int, ...]
    ) -> tuple[int, ...] | None:
        requester = self._requesters.get(voice_keys(message)[0], message)
        if not self.replays.allow(requester):
            return ()
        signature = hashlib.sha256(json.dumps(saved).encode()).hexdigest()
        progress_key = "copy:" + voice_keys(message)[0] + ":" + signature
        copied: list[int] = json.loads(self._result_get(progress_key) or "[]")
        try:
            for source_id in saved[len(copied) :]:
                if not self.chat_allowed(message):
                    return ()
                for attempt in range(2):
                    try:

                        async def transmit_copy(source_id: int = source_id) -> Any:
                            return await bot.copy_message(
                                chat_id=message.chat.id,
                                from_chat_id=message.chat.id,
                                message_id=source_id,
                                message_thread_id=forum_topic_id(message),
                                reply_parameters=ReplyParameters(
                                    message_id=message.message_id, allow_sending_without_reply=False
                                ),
                            )

                        result = await self._deliver(
                            f"copy:{source_id}", transmit_copy, lambda value: value.message_id
                        )
                        copied.append(result if isinstance(result, int) else result.message_id)
                        self._result_put(progress_key, json.dumps(copied))
                        break
                    except TelegramRetryAfter as exc:
                        if (
                            attempt
                            or exc.retry_after > self.settings.telegram_max_retry_after_seconds
                        ):
                            raise
                        await asyncio.sleep(exc.retry_after)
        except TelegramBadRequest:
            # A deleted source can be regenerated; an uncertain network result cannot.
            self._result_put(progress_key, "[]")
            for message_id in copied:
                await bot.delete_message(chat_id=message.chat.id, message_id=message_id)
            logger.info("cached_reply_unavailable message_id=%s", message.message_id)
            return None
        return tuple(copied)

    async def _send_error(self, message: Message, text: str) -> None:
        job_id = self._durable_job.get()
        if (
            self.journal is not None
            and job_id is not None
            and self.journal.get(job_id)["state"] == "uncertain"
        ):
            return
        await self.notify(message, text)

    async def notify(self, message: Message, text: str, *, force: bool = False) -> None:
        if not self.chat_allowed(message):
            return
        target = self._requesters.get(voice_keys(message)[0], message)
        if not force and not self.notices.allow(target):
            return
        try:
            await self.send(target, text)
        except TelegramAPIError:
            logger.warning("telegram_error_delivery_failed")
