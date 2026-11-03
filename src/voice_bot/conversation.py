"""Bounded, expiring source context and atomic restart snapshots."""

import asyncio
import json
import logging
import os
import tempfile
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from voice_bot.reply_context import ReplyContext, bounded_contexts

logger = logging.getLogger(__name__)
ConversationKey = tuple[int, ...]  # chat, topic, author; optional forwarding scope
MAX_SERIES = 64


@dataclass(frozen=True)
class Entry:
    message_ids: tuple[int, ...]
    sent_at: float
    text: str
    kind: Literal["voice", "text"] = "voice"
    duration_seconds: int = 0
    reply_contexts: tuple[ReplyContext, ...] = ()
    summary: str | None = None


@dataclass
class Anchor:
    message_id: int
    author_name: str | None
    author_username: str | None
    chat_type: str | None


@dataclass
class Series:
    started_at: float
    last_at: float
    voice_count: int = 0
    counted_through: int = 0
    summarized_count: int = 0
    summary_offset: int = 0


@dataclass
class Session:
    expires_at: float
    entries: list[Entry] = field(default_factory=list)
    summarized_through: int = 0
    anchor: Anchor | None = None
    # Legacy watermark from separate-message summaries, retained for snapshot compatibility.
    claimed_through: int = 0
    series: list[Series] = field(default_factory=list)


class ConversationMemory:
    def __init__(
        self,
        ttl_seconds: float,
        max_chars: int,
        max_sessions: int,
        clock: Callable[[], float] = time.time,
        *,
        max_entries: int = 2,
        path: Path | None = None,
        series_gap: float = 900,
    ) -> None:
        self.ttl = ttl_seconds
        self.max_chars = max_chars
        self.max_sessions = max_sessions
        self.max_entries = max_entries
        self.clock = clock
        self.path = path
        self.series_gap = series_gap
        self.available = True
        self.sessions: OrderedDict[ConversationKey, Session] = OrderedDict()
        self._locks: dict[ConversationKey, tuple[asyncio.Lock, int]] = {}
        self._expiry: asyncio.TimerHandle | None = None
        self._save_lock = asyncio.Lock()
        self._saved = ""
        if path is not None and path.exists():
            try:
                if path.stat().st_size > max(
                    100_000,
                    max_sessions * (max_chars * 6 + max_entries * 500 + MAX_SERIES * 256 + 2000),
                ):
                    raise ValueError
                data = json.loads(path.read_text())
                if data["version"] != 1:
                    raise ValueError
                adapter = TypeAdapter(list[tuple[ConversationKey, Session]])
                for key, session in adapter.validate_python(data["sessions"]):
                    if len(key) not in (3, 4) or any(not e.message_ids for e in session.entries):
                        raise ValueError
                    if len(session.series) > MAX_SERIES or any(
                        s.last_at < s.started_at
                        or s.counted_through < 0
                        or not 0 <= s.summarized_count <= s.voice_count
                        or not 0 <= s.summary_offset <= s.summarized_count
                        for s in session.series
                    ):
                        raise ValueError
                    self._trim(session)
                    if not session.series:
                        for entry in session.entries:
                            if entry.kind == "voice":
                                series = self._activity(session, entry.sent_at)
                                series.voice_count += len(entry.message_ids)
                                series.counted_through = max(entry.message_ids)
                                if series.counted_through <= max(
                                    session.summarized_through, session.claimed_through
                                ):
                                    series.summarized_count = series.voice_count
                    session.expires_at = min(session.expires_at, self.clock() + self.ttl)
                    if (
                        (session.entries or session.series)
                        and session.expires_at > self.clock()
                        and self.ttl
                    ):
                        self.sessions[key] = session
                while len(self.sessions) > self.max_sessions:
                    self.sessions.popitem(last=False)
            except (OSError, ValueError, KeyError, TypeError, ValidationError):
                # Do not overwrite an unreadable snapshot or interrupt ordinary transcription.
                logger.error("conversation_storage_unavailable")
                self.available = False
                self.sessions.clear()

    @asynccontextmanager
    async def ordered(
        self, key: ConversationKey | None, deadline: float | None = None
    ) -> AsyncIterator[None]:
        if key is None or not self.ttl:
            yield
            return
        lock, users = self._locks.get(key, (asyncio.Lock(), 0))
        self._locks[key] = (lock, users + 1)
        acquired = False
        try:
            async with asyncio.timeout_at(deadline):
                await lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                lock.release()
            _, users = self._locks[key]
            if users == 1:
                del self._locks[key]
            else:
                self._locks[key] = (lock, users - 1)

    def recent(
        self, key: ConversationKey | None, sent_at: float, message_ids: tuple[int, ...]
    ) -> tuple[str, ...]:
        self.purge()
        if key is None or not self.ttl or key not in self.sessions:
            return ()
        self.sessions.move_to_end(key)
        return tuple(
            entry.text
            for entry in self.sessions[key].entries
            if 0 <= sent_at - entry.sent_at <= self.ttl
            and (entry.sent_at, max(entry.message_ids)) < (sent_at, min(message_ids))
            and not set(entry.message_ids).intersection(message_ids)
        )

    def source(self, chat_id: int, message_id: int) -> str | None:
        """Read only the exact public message explicitly referenced in this chat."""
        self.purge()
        for key, session in self.sessions.items():
            if key[0] == chat_id:
                for entry in session.entries:
                    if entry.message_ids == (message_id,):
                        return entry.text
        return None

    def reply_source(self, chat_id: int, message_id: int) -> ReplyContext | None:
        self.purge()
        for key, session in reversed(self.sessions.items()):
            if key[0] == chat_id:
                for entry in reversed(session.entries):
                    for context in entry.reply_contexts:
                        if (
                            context.reply_to_message_id == message_id
                            and context.text
                            and not context.unavailable_reason
                        ):
                            return context
        return None

    def _activity(self, session: Session, sent_at: float) -> Series:
        for series in reversed(session.series):
            if (
                series.started_at
                <= sent_at
                < min(series.last_at + self.series_gap, series.started_at + self.ttl)
            ):
                series.last_at = max(series.last_at, sent_at)
                return series
        series = Series(sent_at, sent_at)
        session.series.append(series)
        session.series.sort(key=lambda item: item.started_at)
        newest = max(item.last_at for item in session.series)
        session.series = [s for s in session.series if newest - s.last_at <= self.ttl][-MAX_SERIES:]
        return series

    def activity(self, key: ConversationKey | None, sent_at: float) -> Series | None:
        """Record event timing at admission, independently of bounded source text and STT."""
        self.purge()
        if key is None or len(key) != 3 or not self.ttl or not self.available:
            return None
        session = self.sessions.get(key)
        if session is None:
            session = Session(expires_at=self.clock() + self.ttl)
            self.sessions[key] = session
        series = self._activity(session, sent_at)
        session.expires_at = self.clock() + self.ttl
        self.sessions.move_to_end(key)
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)
        self._schedule_expiry()
        return series

    def series_at(self, key: ConversationKey, sent_at: float) -> Series | None:
        session = self.sessions.get(key)
        if session is not None:
            for series in reversed(session.series):
                if (
                    series.started_at
                    <= sent_at
                    < min(series.last_at + self.series_gap, series.started_at + self.ttl)
                ):
                    return series
        return None

    def remember(
        self,
        key: ConversationKey | None,
        sent_at: float,
        message_ids: tuple[int, ...],
        text: str,
        *,
        kind: Literal["voice", "text"] = "voice",
        duration_seconds: int = 0,
        anchor: Anchor | None = None,
        reply_contexts: tuple[ReplyContext, ...] = (),
        summary: str | None = None,
    ) -> None:
        self.purge()
        if key is None or not self.ttl or not text.strip() or not self.available:
            return
        session = self.sessions.get(key)
        if session and session.entries:
            if any(set(e.message_ids).intersection(message_ids) for e in session.entries):
                return
            latest = session.entries[-1].sent_at
            if latest - sent_at > self.ttl:
                return
        if session is None:
            session = Session(expires_at=self.clock() + self.ttl)
            self.sessions[key] = session
        if len(key) == 3 and kind == "voice":
            series = self._activity(session, sent_at)
            if max(message_ids) > series.counted_through:
                series.voice_count += len(message_ids)
                series.counted_through = max(message_ids)
        session.entries.append(
            Entry(message_ids, sent_at, text, kind, duration_seconds, reply_contexts, summary)
        )
        session.entries.sort(key=lambda e: (e.sent_at, max(e.message_ids)))
        self._trim(session)
        session.expires_at = self.clock() + self.ttl
        if anchor and (session.anchor is None or anchor.message_id >= session.anchor.message_id):
            session.anchor = anchor
        self.sessions.move_to_end(key)
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)
        self._schedule_expiry()

    def _trim(self, session: Session) -> None:
        if not session.entries:
            return
        newest = session.entries[-1].sent_at
        entries = [e for e in session.entries if newest - e.sent_at <= self.ttl]
        result: list[Entry] = []
        budget = self.max_chars
        for entry in reversed(entries[-self.max_entries :]):
            limit = min(budget, self.max_chars // 2)
            if limit < 1:
                break
            reply_contexts = bounded_contexts(entry.reply_contexts, limit // 3)
            reply_chars = sum(len(context.text) for context in reply_contexts)
            summary = entry.summary[: min(2000, limit // 3)] if entry.summary else None
            summary_chars = len(summary or "")
            text_limit = limit - reply_chars - summary_chars
            text = entry.text
            if len(text) > text_limit:
                text = text[: max(0, text_limit - 1)] + "…"
            result.append(
                Entry(
                    entry.message_ids,
                    entry.sent_at,
                    text,
                    entry.kind,
                    entry.duration_seconds,
                    reply_contexts,
                    summary,
                )
            )
            budget -= len(text) + reply_chars + summary_chars
        session.entries = list(reversed(result))

    def purge(self) -> None:
        now = self.clock()
        for key in list(self.sessions):
            if self.sessions[key].expires_at <= now:
                del self.sessions[key]
        self._schedule_expiry()

    def _schedule_expiry(self) -> None:
        if self._expiry is not None:
            self._expiry.cancel()
            self._expiry = None
        if self.sessions:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            deadline = min(session.expires_at for session in self.sessions.values())
            self._expiry = loop.call_later(max(0, deadline - self.clock()), self.purge)

    async def checkpoint(self) -> bool:
        if not self.available:
            return False
        if self.path is None:
            return True
        async with self._save_lock:
            self.purge()
            snapshot = json.dumps(
                {"version": 1, "sessions": [(key, asdict(s)) for key, s in self.sessions.items()]},
                ensure_ascii=False,
            )
            if snapshot == self._saved:
                return True
            try:
                writer = asyncio.create_task(asyncio.to_thread(self._write, snapshot))
                try:
                    await asyncio.shield(writer)
                except asyncio.CancelledError:
                    # Keep the lock until the thread finishes; a newer snapshot must win.
                    await writer
                    raise
                self._saved = snapshot
                return True
            except OSError:
                logger.error("conversation_storage_unavailable")
                self.available = False
                return False

    def _write(self, snapshot: str) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as file:
                temporary = Path(file.name)
                json.dump(json.loads(snapshot), file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
            # Also persist the rename across a host crash.
            descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def clear(self) -> None:
        self.sessions.clear()
        if self._expiry is not None:
            self._expiry.cancel()
            self._expiry = None
