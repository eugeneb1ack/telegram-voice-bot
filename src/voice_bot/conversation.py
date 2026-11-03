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


@dataclass(frozen=True)
class Entry:
    message_ids: tuple[int, ...]
    sent_at: float
    text: str
    kind: Literal["voice", "text"] = "voice"
    duration_seconds: int = 0
    reply_contexts: tuple[ReplyContext, ...] = ()


@dataclass
class Anchor:
    message_id: int
    author_name: str | None
    author_username: str | None
    chat_type: str | None


@dataclass
class Session:
    expires_at: float
    entries: list[Entry] = field(default_factory=list)
    summarized_through: int = 0
    anchor: Anchor | None = None
    # Legacy watermark from separate-message summaries, retained for snapshot compatibility.
    claimed_through: int = 0


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
    ) -> None:
        self.ttl = ttl_seconds
        self.max_chars = max_chars
        self.max_sessions = max_sessions
        self.max_entries = max_entries
        self.clock = clock
        self.path = path
        self.available = True
        self.sessions: OrderedDict[ConversationKey, Session] = OrderedDict()
        self._locks: dict[ConversationKey, tuple[asyncio.Lock, int]] = {}
        self._expiry: asyncio.TimerHandle | None = None
        self._save_lock = asyncio.Lock()
        self._saved = ""
        if path is not None and path.exists():
            try:
                if path.stat().st_size > max(
                    100_000, max_sessions * (max_chars * 6 + max_entries * 500 + 2000)
                ):
                    raise ValueError
                data = json.loads(path.read_text())
                if data["version"] != 1:
                    raise ValueError
                adapter = TypeAdapter(list[tuple[ConversationKey, Session]])
                for key, session in adapter.validate_python(data["sessions"]):
                    if len(key) not in (3, 4) or any(not e.message_ids for e in session.entries):
                        raise ValueError
                    self._trim(session)
                    session.expires_at = min(session.expires_at, self.clock() + self.ttl)
                    if session.entries and session.expires_at > self.clock() and self.ttl:
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
            if sent_at - latest > self.ttl:
                session = None
        if session is None:
            session = Session(expires_at=self.clock() + self.ttl)
            self.sessions[key] = session
        session.entries.append(
            Entry(message_ids, sent_at, text, kind, duration_seconds, reply_contexts)
        )
        session.entries.sort(key=lambda e: (e.sent_at, max(e.message_ids)))
        self._trim(session)
        session.expires_at = self.clock() + self.ttl
        if anchor and any(
            anchor.message_id in e.message_ids for e in session.entries if e.kind == "voice"
        ):
            last_voice = next(e for e in reversed(session.entries) if e.kind == "voice")
            if anchor.message_id in last_voice.message_ids:
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
            text_limit = limit - reply_chars
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
                )
            )
            budget -= len(text) + reply_chars
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
