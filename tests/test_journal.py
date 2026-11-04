import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramNetworkError
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
