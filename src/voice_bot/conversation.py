"""Expiring conversation context stored in memory."""

import asyncio
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

ConversationKey = tuple[int, int, int]  # chat, topic, original author


@dataclass(frozen=True)
class Entry:
    message_ids: tuple[int, ...]
    sent_at: float
    text: str


@dataclass
class Session:
    expires_at: float
    entries: list[Entry] = field(default_factory=list)


class ConversationMemory:
    def __init__(
        self,
        ttl_seconds: float,
        max_chars: int,
        max_sessions: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl = ttl_seconds
        self.max_chars = max_chars
        self.max_sessions = max_sessions
        self.clock = clock
        self.sessions: OrderedDict[ConversationKey, Session] = OrderedDict()
        self._locks: dict[ConversationKey, tuple[asyncio.Lock, int]] = {}
        self._expiry: asyncio.TimerHandle | None = None

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
        session = self.sessions[key]
        self.sessions.move_to_end(key)
        return tuple(
            entry.text
            for entry in session.entries
            if 0 <= sent_at - entry.sent_at <= self.ttl
            and (entry.sent_at, max(entry.message_ids)) < (sent_at, min(message_ids))
            and not set(entry.message_ids).intersection(message_ids)
        )

    def remember(
        self,
        key: ConversationKey | None,
        sent_at: float,
        message_ids: tuple[int, ...],
        text: str,
    ) -> None:
        self.purge()
        if key is None or not self.ttl:
            return
        # Each of the last two hints occupies at most half the total budget.
        text = " ".join(text.split())
        limit = self.max_chars // 2
        if len(text) > limit:
            text = text[: limit - 1].rsplit(" ", 1)[0] + "…"
        if not text:
            return
        session = self.sessions.get(key)
        if session and session.entries:
            last = session.entries[-1]
            # Do not rewind context when an older message arrives.
            if (sent_at, max(message_ids)) <= (last.sent_at, max(last.message_ids)):
                return
            if sent_at - last.sent_at > self.ttl:
                session.entries.clear()
        if session is None:
            session = Session(expires_at=self.clock() + self.ttl)
            self.sessions[key] = session
        session.entries.append(Entry(message_ids, sent_at, text))
        session.entries = session.entries[-2:]
        session.expires_at = self.clock() + self.ttl
        self.sessions.move_to_end(key)
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)
        self._schedule_expiry()

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

    def clear(self) -> None:
        self.sessions.clear()
        if self._expiry is not None:
            self._expiry.cancel()
            self._expiry = None
