"""Processing receipts stored as identifier hashes."""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from aiogram.types import Message

from voice_bot.delivery import recipient


def voice_keys(message: Message, *, scoped: bool = True) -> tuple[str, ...]:
    def digest(parts: tuple[object, ...]) -> str:
        if scoped and (user_id := recipient(message)) is not None:
            parts += ("recipient", user_id)
        return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()

    keys = [digest(("message", message.chat.id, message.message_id))]
    unique_id = getattr(message.voice, "file_unique_id", None)
    if unique_id:
        keys.append(
            digest(
                (
                    "voice",
                    message.chat.id,
                    getattr(message, "message_thread_id", None) or 0,
                    unique_id,
                )
            )
        )
    return tuple(keys)


@dataclass(frozen=True)
class Receipt:
    keys: tuple[str, ...]
    replies: tuple[int, ...] = ()


class ProcessedVoices:
    def __init__(
        self,
        path: Path,
        max_entries: int = 50000,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = path
        self.max_entries = max_entries
        self.clock = clock
        self.records: deque[Receipt] = deque()
        self.receipts: dict[str, Receipt] = {}
        self.completed: Counter[str] = Counter()
        self.pending: set[str] = set()
        self.cooldowns: OrderedDict[str, float] = OrderedDict()
        self.rows = 0
        self.available = True
        self._lock = asyncio.Lock()
        if path.exists():
            if path.stat().st_size > 512_000_000:
                raise ValueError("Processing receipts exceed the storage limit")
            with path.open(encoding="utf-8") as file:
                for line in file:
                    raw = json.loads(line)
                    if isinstance(raw, dict) and set(raw) != {"keys", "replies"}:
                        raise ValueError("Invalid processing receipt fields")
                    keys = raw.get("keys") if isinstance(raw, dict) else raw
                    replies = raw.get("replies", []) if isinstance(raw, dict) else []
                    if (
                        not isinstance(keys, list)
                        or not 1 <= len(keys) <= 2
                        or any(
                            not isinstance(key, str)
                            or len(key) != 64
                            or any(char not in "0123456789abcdef" for char in key)
                            for key in keys
                        )
                    ):
                        raise ValueError("Invalid processing receipt")
                    if (
                        not isinstance(replies, list)
                        or len(replies) > 1024
                        or any(type(item) is not int or item <= 0 for item in replies)
                    ):
                        raise ValueError("Invalid Telegram reply identifiers")
                    self._remember(Receipt(tuple(keys), tuple(replies)))
                    self.rows += 1
        # Refuse to start processing if receipts cannot be persisted.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab"):
            path.chmod(0o600)

    def _remember(self, receipt: Receipt) -> None:
        if len(self.records) >= self.max_entries:
            old = self.records.popleft()
            for key in old.keys:
                self.completed[key] -= 1
                if not self.completed[key]:
                    del self.completed[key]
                if self.receipts.get(key) is old:
                    del self.receipts[key]
        self.records.append(receipt)
        self.completed.update(receipt.keys)
        for key in receipt.keys:
            self.receipts[key] = receipt

    def reply_ids(self, message: Message) -> tuple[int, ...]:
        # A file can be reposted by another author or in another voice block.
        # Only the exact source message owns its complete Telegram reply.
        receipt = self.receipts.get(voice_keys(message)[0])
        return receipt.replies if receipt is not None else ()

    def claim(self, message: Message, *, completed_ok: bool = False) -> bool:
        if not self.available or not message.voice:
            return False
        now = self.clock()
        while self.cooldowns and next(iter(self.cooldowns.values())) <= now:
            self.cooldowns.popitem(last=False)
        keys = voice_keys(message)
        if any(
            (key in self.completed and not completed_ok)
            or key in self.pending
            or key in self.cooldowns
            for key in keys
        ):
            return False
        self.pending.update(keys)
        return True

    def release(self, message: Message, *, failed: bool = False) -> None:
        keys = voice_keys(message)
        self.pending.difference_update(keys)
        if failed:
            for key in keys:
                self.cooldowns[key] = self.clock() + 60
                self.cooldowns.move_to_end(key)
            while len(self.cooldowns) > 2048:
                self.cooldowns.popitem(last=False)

    def _write(self, records: list[Receipt], *, compact: bool) -> None:
        temporary: Path | None = None
        file: TextIO
        try:
            if compact:
                descriptor, name = tempfile.mkstemp(dir=self.path.parent)
                temporary = Path(name)
                file = os.fdopen(descriptor, "w", encoding="utf-8")
            else:
                file = self.path.open("a", encoding="utf-8")
            with file:
                for receipt in records:
                    data = (
                        {"keys": receipt.keys, "replies": receipt.replies}
                        if receipt.replies
                        else receipt.keys
                    )
                    file.write(json.dumps(data, separators=(",", ":")) + "\n")
                file.flush()
                os.fsync(file.fileno())
            if temporary is not None:
                os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def complete(self, messages: Iterable[Message], replies: tuple[int, ...] = ()) -> None:
        async with self._lock:
            replies = replies if len(replies) <= 1024 else ()
            fresh = [
                Receipt(voice_keys(message), () if recipient(message) else replies)
                for message in messages
            ]
            fresh = [item for item in fresh if self.receipts.get(item.keys[0]) != item]
            if not fresh:
                return
            compact = self.rows + len(fresh) > self.max_entries * 2
            records = (list(self.records) + fresh)[-self.max_entries :] if compact else fresh
            write = asyncio.create_task(asyncio.to_thread(self._write, records, compact=compact))
            try:
                try:
                    await asyncio.shield(write)
                except asyncio.CancelledError:
                    # Do not abandon a disk write or let the next writer race with its thread.
                    await write
                    raise
            except OSError:
                self.available = False
                raise
            finally:
                if write.done() and not write.cancelled() and write.exception() is None:
                    for receipt in fresh:
                        self._remember(receipt)
                    self.rows = len(records) if compact else self.rows + len(fresh)


class RecentResults:
    """Short-lived RAM cache for retrying failed delivery without new API requests."""

    def __init__(
        self,
        *,
        ttl: float = 900,
        max_chars: int = 1_000_000,
        max_entries: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl, self.max_chars, self.max_entries, self.clock = ttl, max_chars, max_entries, clock
        self.entries: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.characters = 0
        self._expiry: asyncio.TimerHandle | None = None

    def _prune(self) -> None:
        now = self.clock()
        while self.entries and next(iter(self.entries.values()))[0] <= now:
            _, text = self.entries.popitem(last=False)[1]
            self.characters -= len(text)
        self._schedule_expiry()

    def _schedule_expiry(self) -> None:
        if self._expiry is not None:
            self._expiry.cancel()
            self._expiry = None
        if self.entries:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            deadline = next(iter(self.entries.values()))[0]
            self._expiry = loop.call_later(max(0, deadline - self.clock()), self._prune)

    def get(self, key: str) -> str | None:
        self._prune()
        entry = self.entries.get(key)
        return entry[1] if entry is not None else None

    def put(self, key: str, text: str) -> None:
        self._prune()
        old = self.entries.pop(key, None)
        if old is not None:
            self.characters -= len(old[1])
        if len(text) > self.max_chars:
            return
        self.entries[key] = (self.clock() + self.ttl, text)
        self.characters += len(text)
        while self.characters > self.max_chars or len(self.entries) > self.max_entries:
            _, removed = self.entries.popitem(last=False)[1]
            self.characters -= len(removed)
        self._schedule_expiry()

    def clear(self) -> None:
        self.entries.clear()
        self.characters = 0
        if self._expiry is not None:
            self._expiry.cancel()
            self._expiry = None
