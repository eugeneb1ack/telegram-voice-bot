"""Single-owner durable queue and outbox. SQLite commits precede external effects."""

import asyncio
import fcntl
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

from aiogram import Bot
from aiogram.types import Message

from voice_bot.deduplication import voice_keys
from voice_bot.reply_context import reply_target


class JournalUnavailable(Exception):
    """Fatal: never acknowledge an update when durable admission has failed."""


class DeliveryUncertain(Exception):
    """A previous send may have reached Telegram; automatic replay is unsafe."""


def durable[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    @wraps(method)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        owner = args[0]
        assert isinstance(owner, QueueJournal)
        if owner.failed.is_set():
            raise JournalUnavailable("Queue journal unavailable")
        try:
            return method(*args, **kwargs)
        except (sqlite3.Error, OSError, JournalUnavailable):
            owner.failed.set()
            raise JournalUnavailable("Queue journal unavailable") from None

    return guarded


def request_key(message: Message, requester: Message | None = None) -> str:
    key = voice_keys(message)[0]
    if requester is not None:
        command = (
            requester.ephemeral_message_id or requester.message_id,
            requester.date.timestamp(),
        )
        key = hashlib.sha256((key + json.dumps(command)).encode()).hexdigest()
    return key


def pack(message: Message, *, include_reply: bool = True) -> dict[str, Any]:
    # Keep only fields needed for processing/reply routing, never unrelated nested history.
    fields = {
        "message_id",
        "date",
        "chat",
        "from_user",
        "sender_chat",
        "voice",
        "video",
        "video_note",
        "message_thread_id",
        "is_topic_message",
        "forward_origin",
        "receiver_user",
        "ephemeral_message_id",
    }
    data = message.model_dump(mode="json", include=fields, exclude_none=True)
    if not include_reply:
        if message.text:
            data["text"] = message.text[:8000]
        if message.caption:
            data["caption"] = message.caption[:8000]
    else:
        target = reply_target(message)
        if target is not None:
            data["reply_to_message"] = pack(target, include_reply=False)
    return data


class QueueJournal:
    def __init__(self, path: Path, *, max_jobs: int = 1100, max_receipts: int = 50000) -> None:
        self.failed = asyncio.Event()
        self.max_jobs, self.max_receipts = max_jobs, max_receipts
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = path.with_suffix(".lock").open("a+b")
        os.chmod(self._lock.name, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(descriptor)
            path.chmod(0o600)
            self.db = sqlite3.connect(path)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA secure_delete=ON")
            if self.db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
                raise ValueError("Unsupported queue journal version")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL,
                    state TEXT NOT NULL, result TEXT, cache TEXT NOT NULL DEFAULT '{}',
                    deliveries TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, job_id TEXT, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS requests_job ON requests(job_id);
                PRAGMA user_version=1;
            """)
            # An interrupted send is deliberately not retried, even if the process died
            # immediately before the socket write. Bot API has no idempotency key.
            with self.transaction():
                for row in self.db.execute("SELECT id, deliveries FROM jobs").fetchall():
                    parts = json.loads(row["deliveries"])
                    if any(part["state"] == "sending" for part in parts.values()):
                        self.db.execute(
                            "UPDATE jobs SET state='uncertain' WHERE id=?", (row["id"],)
                        )
        except BaseException:
            self._lock.close()
            raise

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            with self.db:
                yield
        except sqlite3.Error:
            raise JournalUnavailable("Queue journal unavailable") from None

    @durable
    def seen(self, key: str) -> bool:
        return self.db.execute("SELECT 1 FROM requests WHERE id=?", (key,)).fetchone() is not None

    @durable
    def admit(
        self, message: Message, requester: Message | None, *, collecting: bool, charged: bool
    ) -> str | None:
        key = request_key(message, requester)
        if self.seen(key):
            return None
        if self.db.execute("SELECT count(*) FROM jobs").fetchone()[0] >= self.max_jobs:
            return None
        payload = [
            {
                "key": key,
                "message": pack(message),
                "requester": pack(requester) if requester is not None else None,
                "charged": charged,
            }
        ]
        with self.transaction():
            self.db.execute(
                "INSERT INTO jobs(id,payload,created,state) VALUES(?,?,?,?)",
                (key, json.dumps(payload), time.time(), "collecting" if collecting else "queued"),
            )
            self.db.execute("INSERT INTO requests VALUES(?,?,?)", (key, key, time.time()))
        return key

    @durable
    def merge(self, keys: list[str]) -> str:
        # Atomic transfer from the debounce buffer to one queue job.
        rows = [self.db.execute("SELECT * FROM jobs WHERE id=?", (key,)).fetchone() for key in keys]
        if any(row is None for row in rows):
            raise JournalUnavailable("Missing durable queue member")
        payload = [item for row in rows for item in json.loads(row["payload"])]
        job_id = keys[-1]
        with self.transaction():
            self.db.execute(
                "UPDATE jobs SET payload=?, state='queued', created=? WHERE id=?",
                (json.dumps(payload), min(row["created"] for row in rows), job_id),
            )
            for key in keys:
                self.db.execute("UPDATE requests SET job_id=? WHERE job_id=?", (job_id, key))
                if key != job_id:
                    self.db.execute("DELETE FROM jobs WHERE id=?", (key,))
        return job_id

    @durable
    def get(self, job_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise JournalUnavailable("Missing durable queue job")
        value = dict(row)
        for key in ("payload", "result", "cache", "deliveries"):
            value[key] = json.loads(value[key]) if value[key] is not None else None
        return value

    @durable
    def recover(
        self, bot: Bot
    ) -> list[tuple[str, str, list[tuple[Message, Message | None, bool]]]]:
        recovered = []
        for row in self.db.execute("SELECT id, state FROM jobs ORDER BY created, rowid").fetchall():
            if row["state"] == "uncertain":
                continue
            items = self.get(row["id"])["payload"]
            recovered.append(
                (
                    row["id"],
                    row["state"],
                    [
                        (
                            Message.model_validate(item["message"]).as_(bot),
                            Message.model_validate(item["requester"]).as_(bot)
                            if item["requester"]
                            else None,
                            item["charged"],
                        )
                        for item in items
                    ],
                )
            )
        return recovered

    @durable
    def set(self, job_id: str, field: str, value: Any) -> None:
        if field not in {"result", "cache", "deliveries", "state"}:
            raise ValueError("Invalid journal field")
        with self.transaction():
            self.db.execute(
                f"UPDATE jobs SET {field}=? WHERE id=?",
                (value if field == "state" else json.dumps(value), job_id),
            )

    @durable
    def begin_send(self, job_id: str, part: str) -> int | None:
        deliveries = self.get(job_id)["deliveries"]
        previous = deliveries.get(part)
        if previous is not None:
            if previous["state"] == "sent":
                return int(previous["id"])
            raise DeliveryUncertain
        deliveries[part] = {"state": "sending"}
        self.set(job_id, "deliveries", deliveries)
        return None

    @durable
    def sent(self, job_id: str, part: str, message_id: int | None) -> None:
        deliveries = self.get(job_id)["deliveries"]
        deliveries[part] = {"state": "sent", "id": message_id or 0}
        self.set(job_id, "deliveries", deliveries)

    @durable
    def rejected(self, job_id: str, part: str) -> None:
        deliveries = self.get(job_id)["deliveries"]
        deliveries.pop(part, None)
        self.set(job_id, "deliveries", deliveries)

    @durable
    def finish(self, job_id: str) -> None:
        with self.transaction():
            self.db.execute("UPDATE requests SET job_id=NULL WHERE job_id=?", (job_id,))
            self.db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            self.db.execute(
                """DELETE FROM requests WHERE id IN (
                SELECT id FROM requests WHERE job_id IS NULL ORDER BY created DESC, rowid DESC
                LIMIT -1 OFFSET ?)""",
                (self.max_receipts,),
            )

    def close(self) -> None:
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()
        self._lock.close()
