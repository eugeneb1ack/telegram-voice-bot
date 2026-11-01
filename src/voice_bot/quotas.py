"""Daily voice limits for private chats."""

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from aiogram.types import Message
from pydantic import BaseModel, Field, TypeAdapter

QuotaDecision = Literal["allowed", "notify", "limited"]


def utc_day() -> date:
    return datetime.now(UTC).date()


class DailyUsage(BaseModel, frozen=True):
    day: date
    used: int = Field(ge=0)
    notified: bool = False


class PrivateVoiceQuota:
    def __init__(
        self,
        path: Path,
        limit: int,
        *,
        max_users: int = 10000,
        clock: Callable[[], date] = utc_day,
    ) -> None:
        self.path, self.limit, self.max_users, self.clock = path, limit, max_users, clock
        self.users: dict[str, DailyUsage] = {}
        self.available = True
        self._lock = asyncio.Lock()
        if path.exists():
            if path.stat().st_size > 2_000_000:
                raise ValueError("Private quota storage exceeds its limit")
            self.users = TypeAdapter(dict[str, DailyUsage]).validate_json(path.read_bytes())
            if any(
                len(key) != 64 or any(char not in "0123456789abcdef" for char in key)
                for key in self.users
            ):
                raise ValueError("Invalid private quota identifier")
            self.users = {key: item for key, item in self.users.items() if item.day == clock()}
            path.chmod(0o600)
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(message: Message) -> str | None:
        if getattr(message.chat, "type", None) != "private":
            return None
        # For private chats chat.id identifies the recipient, including forwarded voices.
        return hashlib.sha256(f"private:{message.chat.id}".encode()).hexdigest()

    def _write(self, users: dict[str, DailyUsage]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as file:
                temporary = Path(file.name)
                json.dump(
                    {key: value.model_dump(mode="json") for key, value in users.items()}, file
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def _commit(self, users: dict[str, DailyUsage]) -> None:
        write = asyncio.create_task(asyncio.to_thread(self._write, users))
        try:
            try:
                await asyncio.shield(write)
            except asyncio.CancelledError:
                await write
                raise
        except OSError:
            self.available = False
            raise
        finally:
            if write.done() and not write.cancelled() and write.exception() is None:
                self.users = users

    async def admit(self, message: Message) -> QuotaDecision:
        key = self._key(message)
        if key is None:
            return "allowed"
        async with self._lock:
            if not self.available:
                return "limited"
            today = self.clock()
            users = {key: item for key, item in self.users.items() if item.day == today}
            usage = users.get(key, DailyUsage(day=today, used=0))
            if key not in users and len(users) >= self.max_users:
                return "limited"  # Never evict an active counter to make room for new users.
            if usage.used >= self.limit:
                if usage.notified:
                    return "limited"
                users[key] = usage.model_copy(update={"notified": True})
                await self._commit(users)
                return "notify"
            users[key] = usage.model_copy(update={"used": usage.used + 1})
            await self._commit(users)
            return "allowed"

    async def refund(self, message: Message) -> None:
        key = self._key(message)
        if key is None:
            return
        async with self._lock:
            if not self.available:
                return
            usage = self.users.get(key)
            if usage is None or usage.day != self.clock() or not usage.used:
                return
            users = self.users | {key: usage.model_copy(update={"used": usage.used - 1})}
            await self._commit(users)
