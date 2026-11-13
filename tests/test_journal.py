import asyncio
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramForbiddenError, TelegramNetworkError, TelegramRetryAfter
from aiogram.methods import SendMessage
from aiogram.types import Message

from voice_bot.journal import DeliveryUncertain, QueueJournal, request_key
from voice_bot.service import VoiceService


def voice(number=1, *, chat=-100, author=10, topic=1, forwarded=False, receiver=None):
    data = dict(
        message_id=number,
        date=datetime.now(UTC),
        chat={"id": chat, "type": "supergroup"},
        from_user={"id": author, "is_bot": False, "first_name": "Same name"},
        message_thread_id=topic,
        voice={"file_id": f"file-{number}", "file_unique_id": f"unique-{number}", "duration": 40},
    )
    if forwarded:
        data["forward_origin"] = {
            "type": "user",
            "date": datetime.now(UTC),
            "sender_user": data["from_user"],
        }
    if receiver:
        data["receiver_user"] = {"id": receiver, "is_bot": False, "first_name": "Reader"}
    return Message.model_validate(data)


def runtime(settings, path, sent):
    journal = QueueJournal(path)
    bot = Bot("123456:test-token")

    async def request(bot, method, **kwargs):
        if isinstance(method, SendMessage):
            sent.append(method)
            return Message(
                message_id=1000 + len(sent),
                date=datetime.now(UTC),
                chat={"id": method.chat_id, "type": "supergroup"},
            )
        return True

    bot.session.make_request = AsyncMock(side_effect=request)
    bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="voice.ogg", file_size=100))

    async def download(path, output, **kwargs):
        output.write(b"audio")

    bot.download_file = AsyncMock(side_effect=download)

    async def convert(source, destination, **kwargs):
        destination.write_bytes(b"wav")

    transcriber = SimpleNamespace(
        transcribe=AsyncMock(return_value="  Исходный текст.\n\nПродолжение.  ")
    )
    summarizer = SimpleNamespace(summarize=AsyncMock(return_value="Итог."))
    service = VoiceService(settings, transcriber, summarizer, converter=convert, journal=journal)
    return service, bot, journal


async def stop(service, bot, journal):
    await service.close()
    journal.close()
    await bot.session.close()


async def idle(service):
    async with asyncio.timeout(3):
        while service.queue.size or service.queue.active_chats or service.batcher.tasks:  # noqa: ASYNC110
            await asyncio.sleep(0.005)


async def deferred(service):
    async with asyncio.timeout(3):
        while service.queue.active_chats or not any(  # noqa: ASYNC110
            waiting[0].not_before > time.monotonic() for waiting in service.queue.pending.values()
        ):
            await asyncio.sleep(0.005)


async def test_pending_and_active_jobs_survive_shutdown(settings, tmp_path):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    entered = asyncio.Event()

    async def blocked(_):
        entered.set()
        await asyncio.Event().wait()

    service.transcriber.transcribe.side_effect = blocked
    for number in range(1, 4):
        await service.accept(voice(number).as_(bot), bot)
    await asyncio.wait_for(entered.wait(), 2)
    await service.queue.close(0)
    await stop(service, bot, journal)

    restarted, bot, journal = runtime(settings, path, sent)
    await restarted.recover(bot)
    await idle(restarted)
    assert [item.reply_parameters.message_id for item in sent] == [1, 2, 3]
    assert journal.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    # Redelivered polling update must not copy the previous answer.
    await restarted.accept(voice(1).as_(bot), bot)
    assert len(sent) == 3
    await stop(restarted, bot, journal)


async def test_recovered_next_triplet_has_an_independent_window(settings, tmp_path):
    from datetime import timedelta

    settings.context_summary_every = 3
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    base = datetime.now(UTC) - timedelta(seconds=650)

    def message(mid, seconds):
        item = voice(mid)
        return item.model_copy(
            update={
                "date": base + timedelta(seconds=seconds),
                "voice": item.voice.model_copy(update={"duration": 10}),
            }
        ).as_(bot)

    for mid, seconds in [(1, 0), (2, 20), (3, 40)]:
        await service.accept(message(mid, seconds), bot)
    await idle(service)
    assert service.summarizer.summarize.await_count == 1
    for mid, seconds in [(4, 540), (5, 570), (6, 650)]:
        journal.admit(message(mid, seconds), None, collecting=False, charged=False)
    await stop(service, bot, journal)
    restarted, bot, journal = runtime(settings, path, sent)
    await restarted.recover(bot)
    await idle(restarted)
    assert [item.reply_parameters.message_id for item in sent] == [1, 2, 3, 4, 5, 6]
    assert restarted.summarizer.summarize.await_count == 1
    context = restarted.summarizer.summarize.await_args.kwargs["context"]
    assert context.dialogue.branches[0].message_ids == (4, 5, 6)
    assert "Коротко · 3 голосовых:" in sent[-1].text
    assert journal.recover(bot) == []
    await stop(restarted, bot, journal)


async def test_transcript_checkpoint_avoids_second_stt_after_restart(settings, tmp_path):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    service.summarizer.summarize.side_effect = blocked
    await service.accept(voice().as_(bot), bot)
    await asyncio.wait_for(entered.wait(), 2)
    await service.queue.close(0)
    await stop(service, bot, journal)
    restarted, bot, journal = runtime(settings, path, sent)
    await restarted.recover(bot)
    await idle(restarted)
    restarted.transcriber.transcribe.assert_not_awaited()
    restarted.summarizer.summarize.assert_awaited_once()
    assert len(sent) == 1
    await stop(restarted, bot, journal)


@pytest.mark.parametrize("after_receipt", [False, True])
async def test_acknowledged_reply_is_not_copied_after_restart(settings, tmp_path, after_receipt):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    entered = asyncio.Event()
    if after_receipt:
        original = service.processed.complete

        async def blocked(*args, **kwargs):
            await original(*args, **kwargs)
            entered.set()
            await asyncio.Event().wait()

        service.processed.complete = blocked
    else:

        async def blocked(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        service.memory.checkpoint = blocked
    await service.accept(voice().as_(bot), bot)
    await asyncio.wait_for(entered.wait(), 2)
    await service.queue.close(0)
    # The injected checkpoint is only for the job, not shutdown.
    if not after_receipt:
        service.memory.checkpoint = AsyncMock()
    await stop(service, bot, journal)
    restarted, bot, journal = runtime(settings, path, sent)
    await restarted.recover(bot)
    await idle(restarted)
    assert len(sent) == 1
    restarted.transcriber.transcribe.assert_not_awaited()
    restarted.summarizer.summarize.assert_not_awaited()
    assert journal.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    await stop(restarted, bot, journal)


async def test_long_identical_parts_resume_without_duplicates(settings, tmp_path):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    service.transcriber.transcribe.return_value = "same " * 3000
    entered = asyncio.Event()
    original = service.send
    count = 0

    async def pause(message, text):
        nonlocal count
        count += 1
        if count == 2:
            entered.set()
            await asyncio.Event().wait()
        return await original(message, text)

    service.send = pause
    await service.accept(voice().as_(bot), bot)
    await asyncio.wait_for(entered.wait(), 2)
    assert len(sent) == 1
    await service.queue.close(0)
    await stop(service, bot, journal)
    restarted, bot, journal = runtime(settings, path, sent)
    await restarted.recover(bot)
    await idle(restarted)
    from voice_bot.rendering import render_replies

    assert [item.text for item in sent] == render_replies("Итог.", "same " * 3000)
    restarted.transcriber.transcribe.assert_not_awaited()
    restarted.summarizer.summarize.assert_not_awaited()
    await stop(restarted, bot, journal)


async def test_uncertain_send_is_retained_without_automatic_retry(settings, tmp_path):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)

    async def lost_ack(bot, method, **kwargs):
        if isinstance(method, SendMessage):
            sent.append(method)
            raise TelegramNetworkError(method=method, message="SECRET")
        return True

    bot.session.make_request.side_effect = lost_ack
    await service.accept(voice().as_(bot), bot)
    await idle(service)
    assert len(sent) == 1
    await stop(service, bot, journal)
    restarted, bot, journal = runtime(settings, path, sent)
    await restarted.recover(bot)
    await restarted.accept(voice().as_(bot), bot)
    await idle(restarted)
    assert len(sent) == 1
    assert journal.db.execute("SELECT state FROM jobs").fetchone()[0] == "uncertain"
    restarted.transcriber.transcribe.assert_not_awaited()
    await stop(restarted, bot, journal)


def test_atomic_batch_transfer_and_private_scope(tmp_path):
    path = tmp_path / "queue.db"
    journal = QueueJournal(path)
    messages = [voice(i) for i in range(1, 4)]
    keys = [journal.admit(m, None, collecting=True, charged=False) for m in messages]
    job = journal.merge(keys)
    assert len(journal.get(job)["payload"]) == 3
    assert journal.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
    assert all(journal.seen(k) for k in keys)
    assert request_key(voice(receiver=10)) != request_key(voice(receiver=20))
    assert request_key(voice(chat=-200)) != request_key(voice(chat=-100))
    journal.close()


def test_wal_survives_abrupt_process_exit_and_locks_owner(tmp_path):
    path = tmp_path / "queue.db"
    code = """
import os,sys
from voice_bot.journal import QueueJournal
from aiogram.types import Message
j=QueueJournal(__import__('pathlib').Path(sys.argv[1]))
m=Message.model_validate({"message_id":1,"date":0,"chat":{"id":1,"type":"private"},"voice":{"file_id":"f","file_unique_id":"u","duration":1}})
k=j.admit(m,None,collecting=False,charged=True)
j.set(k,"result",{"transcript":"saved"})
j.begin_send(k,"text:0")
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        env=os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    journal = QueueJournal(path)
    row = journal.db.execute("SELECT id,state FROM jobs").fetchone()
    assert row["state"] == "uncertain"
    assert journal.get(row["id"])["result"]["transcript"] == "saved"
    with pytest.raises(DeliveryUncertain):
        journal.begin_send(row["id"], "text:0")
    with pytest.raises(BlockingIOError):
        QueueJournal(path)
    assert path.stat().st_mode & 0o777 == 0o600
    journal.close()


async def test_forwarded_buffer_survives_restart_and_merges_atomically(settings, tmp_path):
    settings.voice_batch_pause_seconds = 0.01
    path = tmp_path / "queue.db"
    journal = QueueJournal(path)
    for i in range(1, 4):
        journal.admit(voice(i, forwarded=True), None, collecting=True, charged=False)
    journal.close()
    service, bot, journal = runtime(settings, path, [])
    bot.send_rich_message = AsyncMock(
        return_value=Message(
            message_id=2000, date=datetime.now(UTC), chat={"id": -100, "type": "supergroup"}
        )
    )
    await service.recover(bot)
    await idle(service)
    bot.send_rich_message.assert_awaited_once()
    assert service.transcriber.transcribe.await_count == 3
    assert journal.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    assert journal.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 3
    await stop(service, bot, journal)


async def test_private_routing_and_quota_are_restored(settings, tmp_path):
    path = tmp_path / "queue.db"
    sent = []
    journal = QueueJournal(path)
    for user in (10, 20):
        message = voice(user, receiver=user)
        command = message.model_copy(
            update={"message_id": user + 100, "voice": None, "ephemeral_message_id": user + 200}
        )
        journal.admit(message, command, collecting=False, charged=True)
    journal.close()
    service, bot, journal = runtime(settings, path, sent)
    service.private_quota.admit = AsyncMock(side_effect=AssertionError("must not charge twice"))
    await service.recover(bot)
    await idle(service)
    assert sorted(m.ephemeral_message_parameters.receiver_user_id for m in sent) == [10, 20]
    assert not service.memory.sessions
    await stop(service, bot, journal)


async def test_revoked_chat_is_not_processed_on_recovery(settings, tmp_path):
    path = tmp_path / "queue.db"
    journal = QueueJournal(path)
    journal.admit(voice(), None, collecting=False, charged=False)
    journal.close()
    settings.allowed_chat_ids = {-200}
    service, bot, journal = runtime(settings, path, [])
    await service.recover(bot)
    service.transcriber.transcribe.assert_not_awaited()
    assert journal.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    await stop(service, bot, journal)


async def test_copy_receipt_restart_never_copies_the_copy(settings, tmp_path):
    path = tmp_path / "queue.db"
    service, bot, journal = runtime(settings, path, [])
    message = voice().as_(bot)
    await service.processed.complete([message], (777,))
    bot.copy_message = AsyncMock(return_value=SimpleNamespace(message_id=888))
    entered = asyncio.Event()
    original = service.processed.complete

    async def blocked(*args, **kwargs):
        await original(*args, **kwargs)
        entered.set()
        await asyncio.Event().wait()

    service.processed.complete = blocked
    await service.accept(message, bot)
    await asyncio.wait_for(entered.wait(), 2)
    await service.queue.close(0)
    await stop(service, bot, journal)
    service, bot, journal = runtime(settings, path, [])
    bot.copy_message = AsyncMock(side_effect=AssertionError("duplicate copy"))
    await service.recover(bot)
    await idle(service)
    bot.copy_message.assert_not_awaited()
    service.transcriber.transcribe.assert_not_awaited()
    await stop(service, bot, journal)


def test_disk_failure_is_fatal_and_records_are_retained(tmp_path):
    from voice_bot.journal import JournalUnavailable

    journal = QueueJournal(tmp_path / "queue.db")
    key = journal.admit(voice(), None, collecting=False, charged=False)
    journal.db.execute("PRAGMA query_only=ON")
    with pytest.raises(JournalUnavailable):
        journal.begin_send(key, "part")
    assert journal.failed.is_set()
    assert journal.db.execute("SELECT deliveries FROM jobs").fetchone()[0] == "{}"
    journal.db.execute("PRAGMA query_only=OFF")
    journal.close()


def test_corruption_refuses_start_and_receipts_are_bounded(tmp_path):
    path = tmp_path / "bad.db"
    path.write_bytes(b"broken database")
    import sqlite3

    with pytest.raises(sqlite3.DatabaseError):
        QueueJournal(path)
    assert path.read_bytes() == b"broken database"
    journal = QueueJournal(tmp_path / "queue.db", max_jobs=1, max_receipts=2)
    first = journal.admit(voice(1), None, collecting=False, charged=False)
    assert journal.admit(voice(2), None, collecting=False, charged=False) is None
    journal.finish(first)
    for i in (2, 3):
        key = journal.admit(voice(i), None, collecting=False, charged=False)
        journal.finish(key)
    assert journal.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 2
    journal.close()


@pytest.mark.parametrize("delivered", [True, False])
async def test_manual_reconciliation_continues_without_paid_calls(settings, tmp_path, delivered):
    from voice_bot.queue_admin import resolve, status

    path = tmp_path / "queue.db"
    journal = QueueJournal(path)
    m = voice()
    key = journal.admit(m, None, collecting=False, charged=False)
    prepared = {
        "summary": "Итог.",
        "transcript": "text",
        "title": "Коротко:",
        "valid": [True],
        "key": None,
        "times": [],
        "dialogue": False,
    }
    journal.set(key, "result", prepared)
    # Use the same delivery part identifier the real service will produce.
    import hashlib

    from voice_bot.deduplication import voice_keys

    content = hashlib.sha256("Итог.\0text".encode()).hexdigest()
    part = "text:delivery:" + voice_keys(m)[0] + ":" + content + ":0"
    journal.begin_send(key, part)
    journal.close()
    journal = QueueJournal(path)
    assert status(path)[0]["uncertain_parts"] == 1
    with pytest.raises(BlockingIOError):
        resolve(path, key, 999 if delivered else None)
    journal.close()
    resolve(path, key, 999 if delivered else None)
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    await service.recover(bot)
    await idle(service)
    assert len(sent) == int(not delivered)
    service.transcriber.transcribe.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()
    assert status(path) == []
    await stop(service, bot, journal)


async def test_definite_flood_rejection_retries_frozen_reply_without_blocking_other_chats(
    settings, tmp_path
):
    settings.max_concurrent_jobs = 1
    sent = []
    service, bot, journal = runtime(settings, tmp_path / "queue.db", sent)
    original = bot.session.make_request.side_effect
    rejected = False

    async def request(bot, method, **kwargs):
        nonlocal rejected
        if isinstance(method, SendMessage) and method.chat_id == -100 and not rejected:
            rejected = True
            raise TelegramRetryAfter(method=method, message="SECRET", retry_after=0.02)
        return await original(bot, method, **kwargs)

    bot.session.make_request.side_effect = request
    for item in (voice(1), voice(2), voice(3, chat=-200)):
        await service.accept(item.as_(bot), bot)
    await idle(service)
    assert [(item.chat_id, item.reply_parameters.message_id) for item in sent] == [
        (-200, 3),
        (-100, 1),
        (-100, 2),
    ]
    assert service.transcriber.transcribe.await_count == 3
    assert service.summarizer.summarize.await_count == 3
    assert not service._owners and not service.processed.pending
    assert journal.recover(bot) == []
    await stop(service, bot, journal)


async def test_long_flood_wait_is_durable_and_keeps_admission_reserved(settings, tmp_path):
    service, bot, journal = runtime(settings, tmp_path / "queue.db", [])
    rejected = asyncio.Event()
    original = bot.session.make_request.side_effect

    async def request(bot, method, **kwargs):
        if isinstance(method, SendMessage):
            rejected.set()
            raise TelegramRetryAfter(method=method, message="SECRET", retry_after=31)
        return await original(bot, method, **kwargs)

    bot.session.make_request.side_effect = request
    await service.accept(voice().as_(bot), bot)
    await asyncio.wait_for(rejected.wait(), 1)
    await deferred(service)
    row = journal.get(journal.recover(bot)[0][0])
    retry = row["cache"]["delivery_retry"]
    assert row["state"] == "retry_wait" and retry["count"] == 1
    assert retry["not_before"] < retry["deadline"]
    assert service.queue.size == 1 and not service.queue.active_chats
    assert service.processed.pending and service._owners
    assert not row["deliveries"]
    await service.queue.close(0)
    await stop(service, bot, journal)


@pytest.mark.parametrize("expired", [False, True])
async def test_deferred_retry_survives_restart_with_original_budget(
    settings, tmp_path, expired, monkeypatch
):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    original = bot.session.make_request.side_effect
    rejected = asyncio.Event()

    async def request(bot, method, **kwargs):
        if isinstance(method, SendMessage):
            rejected.set()
            raise TelegramRetryAfter(method=method, message="SECRET", retry_after=31)
        return await original(bot, method, **kwargs)

    bot.session.make_request.side_effect = request
    await service.accept(voice().as_(bot), bot)
    await asyncio.wait_for(rejected.wait(), 1)
    await deferred(service)
    job_id = journal.recover(bot)[0][0]
    retry = journal.get(job_id)["cache"]["delivery_retry"]
    await service.queue.close(0)
    await stop(service, bot, journal)
    # Advance wall time only: restart must retain count/deadline from the original job.
    monkeypatch.setattr(
        "voice_bot.journal.time.time",
        lambda: retry["deadline"] + 1 if expired else retry["not_before"] + 1,
    )
    service, bot, journal = runtime(settings, path, sent)
    assert journal.get(job_id)["cache"]["delivery_retry"] == retry
    await service.recover(bot)
    await idle(service)
    assert len(sent) == int(not expired)
    service.transcriber.transcribe.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()
    assert journal.recover(bot) == []
    await stop(service, bot, journal)


async def test_repeated_flood_rejections_exhaust_budget_and_delete_content(settings, tmp_path):
    sent = []
    service, bot, journal = runtime(settings, tmp_path / "queue.db", sent)
    attempts = 0
    original = bot.session.make_request.side_effect

    async def request(bot, method, **kwargs):
        nonlocal attempts
        if isinstance(method, SendMessage):
            attempts += 1
            raise TelegramRetryAfter(method=method, message="SECRET", retry_after=0)
        return await original(bot, method, **kwargs)

    bot.session.make_request.side_effect = request
    item = voice().as_(bot)
    await service.accept(item, bot)
    await idle(service)
    assert attempts == 4 and sent == []
    assert journal.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    assert journal.seen(request_key(item))
    service.transcriber.transcribe.assert_awaited_once()
    service.summarizer.summarize.assert_awaited_once()
    assert not service.processed.pending and not service._owners
    await stop(service, bot, journal)


async def test_forbidden_is_terminal_without_uncertain_job_or_transcript(settings, tmp_path):
    service, bot, journal = runtime(settings, tmp_path / "queue.db", [])
    original = bot.session.make_request.side_effect
    attempts = 0

    async def request(bot, method, **kwargs):
        nonlocal attempts
        if isinstance(method, SendMessage):
            attempts += 1
            raise TelegramForbiddenError(method=method, message="SECRET blocked")
        return await original(bot, method, **kwargs)

    bot.session.make_request.side_effect = request
    item = voice().as_(bot)
    await service.accept(item, bot)
    await idle(service)
    await service.accept(item, bot)
    await idle(service)
    assert attempts == 1
    assert journal.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    assert journal.seen(request_key(item))
    assert not service.processed.pending and not service._owners
    await stop(service, bot, journal)


def test_deferred_retry_count_and_deadline_cannot_reset_after_reopen(tmp_path, monkeypatch):
    path = tmp_path / "queue.db"
    now = [1000.0]
    monkeypatch.setattr("voice_bot.journal.time.time", lambda: now[0])
    journal = QueueJournal(path)
    key = journal.admit(voice(), None, collecting=False, charged=False)
    journal.set(key, "result", {"copy_ids": [777]})
    for count in range(1, 4):
        assert journal.defer_delivery(key, 31, window=300) == 31
        retry = journal.get(key)["cache"]["delivery_retry"]
        assert retry["count"] == count and retry["deadline"] == 1300
        journal.close()
        now[0] += 31
        journal = QueueJournal(path)
    assert journal.defer_delivery(key, 31, window=300) is None
    journal.close()


async def test_multipart_flood_retry_after_restart_keeps_receipts_and_private_quota(
    settings, tmp_path, monkeypatch
):
    from aiogram.types import Chat

    from voice_bot.rendering import render_replies

    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    transcript = "Речь без потерь. " * 500
    service.transcriber.transcribe.return_value = transcript
    original = bot.session.make_request.side_effect
    attempted = 0

    async def request(bot, method, **kwargs):
        nonlocal attempted
        if isinstance(method, SendMessage):
            attempted += 1
            if attempted == 2:
                raise TelegramRetryAfter(method=method, message="SECRET", retry_after=31)
        return await original(bot, method, **kwargs)

    bot.session.make_request.side_effect = request
    item = voice().model_copy(update={"chat": Chat(id=10, type="private")})
    await service.accept(item.as_(bot), bot)
    await deferred(service)
    assert len(sent) == 1
    assert next(iter(service.private_quota.users.values())).used == 1
    retry = journal.get(journal.recover(bot)[0][0])["cache"]["delivery_retry"]
    service.transcriber.transcribe.assert_awaited_once()
    service.summarizer.summarize.assert_awaited_once()
    await service.queue.close(0)
    await stop(service, bot, journal)
    monkeypatch.setattr("voice_bot.journal.time.time", lambda: retry["not_before"] + 1)
    service, bot, journal = runtime(settings, path, sent)
    await service.recover(bot)
    await idle(service)
    assert [part.text for part in sent] == render_replies("Итог.", transcript)
    assert service.processed.reply_ids(item) == tuple(range(1001, 1001 + len(sent)))
    assert next(iter(service.private_quota.users.values())).used == 1
    service.transcriber.transcribe.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()
    assert not journal.recover(bot)
    await stop(service, bot, journal)


async def test_recovered_deferred_reply_respects_quiet_mode(settings, tmp_path):
    from test_service import approved_group

    path = tmp_path / "queue.db"
    journal = QueueJournal(path)
    key = journal.admit(voice(), None, collecting=False, charged=False)
    journal.set(key, "result", {"copy_ids": [777]})
    journal.defer_delivery(key, 31, window=300)
    journal.close()
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    approved_group(service, tmp_path, "quiet")
    await service.recover(bot)
    await idle(service)
    assert not sent and not journal.recover(bot)
    assert journal.seen(request_key(voice()))
    service.transcriber.transcribe.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()
    await stop(service, bot, journal)


async def test_recovery_waits_for_the_persisted_retry_time(settings, tmp_path):
    path = tmp_path / "queue.db"
    journal = QueueJournal(path)
    key = journal.admit(voice(), None, collecting=False, charged=False)
    journal.set(key, "result", {"copy_ids": [777]})
    journal.defer_delivery(key, 31, window=300)
    retry = journal.get(key)["cache"]["delivery_retry"]
    journal.close()

    sent = []
    service, bot, journal = runtime(settings, path, sent)
    before = time.monotonic()
    await service.recover(bot)
    await asyncio.sleep(0)
    assert service.queue.size == 1 and not service.queue.active_chats
    pending = next(iter(service.queue.pending.values()))[0]
    assert pending.not_before > before + 25
    assert journal.get(key)["cache"]["delivery_retry"] == retry
    assert not sent
    service.transcriber.transcribe.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()
    await service.queue.close(0)
    await stop(service, bot, journal)


async def test_multipart_retry_starts_no_new_send_after_original_deadline(
    settings, tmp_path, monkeypatch
):
    now = [1000.0]
    monkeypatch.setattr("voice_bot.service.time.time", lambda: now[0])
    sent = []
    service, bot, journal = runtime(settings, tmp_path / "queue.db", sent)
    item = voice()
    key = journal.admit(item, None, collecting=False, charged=False)
    journal.set(
        key,
        "result",
        {
            "summary": "Итог.",
            "transcript": "Длинная расшифровка. " * 500,
            "title": "Коротко:",
            "key": None,
            "times": [],
            "valid": [True],
            "dialogue": False,
        },
    )
    journal.set(
        key,
        "cache",
        {"delivery_retry": {"count": 1, "not_before": 999, "deadline": 1010}},
    )
    journal.set(key, "state", "retry_wait")
    original = bot.session.make_request.side_effect

    async def request(bot, method, **kwargs):
        result = await original(bot, method, **kwargs)
        if isinstance(method, SendMessage):
            now[0] = 1020
        return result

    bot.session.make_request.side_effect = request
    await service.recover(bot)
    await idle(service)
    assert len(sent) == 1
    assert journal.recover(bot) == [] and journal.seen(request_key(item))
    service.transcriber.transcribe.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()
    await stop(service, bot, journal)


@pytest.mark.parametrize("restart", [False, True])
async def test_copy_flood_retry_keeps_replay_admission_and_delivered_parts(
    settings, tmp_path, monkeypatch, restart
):
    from aiogram.methods import CopyMessage

    path = tmp_path / "copy-retry.db"
    service, bot, journal = runtime(settings, path, [])
    item = voice().as_(bot)
    service.replays.clock = lambda: 1000.0
    await service.processed.complete([item], (777, 778))
    bot.copy_message = AsyncMock(
        side_effect=[
            SimpleNamespace(message_id=888),
            TelegramRetryAfter(
                method=CopyMessage(chat_id=-100, from_chat_id=-100, message_id=778),
                message="SECRET",
                retry_after=31 if restart else 0,
            ),
            SimpleNamespace(message_id=889),
        ]
    )
    await service.accept(item, bot)
    if restart:
        await deferred(service)
        assert [call.kwargs["message_id"] for call in bot.copy_message.await_args_list] == [
            777,
            778,
        ]
        key = journal.recover(bot)[0][0]
        retry = journal.get(key)["cache"]["delivery_retry"]
        await service.queue.close(0)
        await stop(service, bot, journal)
        monkeypatch.setattr("voice_bot.journal.time.time", lambda: retry["not_before"] + 1)
        service, bot, journal = runtime(settings, path, [])
        service.replays.clock = lambda: 1000.0
        # An unrelated replay used this sender's allowance after the restart.
        assert service.replays.allow(item)
        bot.copy_message = AsyncMock(return_value=SimpleNamespace(message_id=889))
        await service.recover(bot)
    await idle(service)
    assert [call.kwargs["message_id"] for call in bot.copy_message.await_args_list] == (
        [778] if restart else [777, 778, 778]
    )
    assert service.processed.reply_ids(item) == (888, 889)
    assert not journal.recover(bot)
    service.transcriber.transcribe.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()
    # A new user replay still needs its own allowance.
    other = voice(2).as_(bot)
    await service.processed.complete([other], (900,))
    count = bot.copy_message.await_count
    await service.accept(other, bot)
    await idle(service)
    assert bot.copy_message.await_count == count
    assert not journal.recover(bot)
    await stop(service, bot, journal)
