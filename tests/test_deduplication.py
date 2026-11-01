import asyncio
import json
import stat
import threading
from types import SimpleNamespace

import pytest

from voice_bot.deduplication import ProcessedVoices, RecentResults, voice_keys


def voice(index=1, *, chat=123, topic=None, unique="same-audio"):
    return SimpleNamespace(
        message_id=index,
        chat=SimpleNamespace(id=chat),
        message_thread_id=topic,
        voice=SimpleNamespace(file_unique_id=unique),
    )


async def test_receipts_survive_restart_without_storing_identifiers_or_content(tmp_path):
    path = tmp_path / "processed.jsonl"
    store = ProcessedVoices(path)
    message = voice(unique="private-file-identifier")
    assert store.claim(message)
    await store.complete([message])
    store.release(message)
    assert not store.claim(message)
    saved = json.loads(path.read_text())
    assert saved == list(voice_keys(message))
    assert all(len(key) == 64 for key in saved)
    assert "private-file-identifier" not in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    reloaded = ProcessedVoices(path)
    assert not reloaded.claim(message)
    assert not reloaded.claim(voice(index=2, unique="private-file-identifier"))
    # The message ID still protects an external reply lacking topic/file metadata.
    assert not reloaded.claim(voice(unique=None))


def test_pending_claim_is_atomic_and_scoped_to_chat_and_topic(tmp_path):
    store = ProcessedVoices(tmp_path / "processed.jsonl")
    message = voice()
    assert store.claim(message)
    assert not store.claim(message)
    assert not store.claim(voice(index=2))
    assert store.claim(voice(chat=456))
    assert store.claim(voice(index=3, topic=5))
    assert voice_keys(voice()) == voice_keys(voice(topic=0))
    store.release(message)
    assert store.claim(message)


def test_failed_attempt_has_bounded_cooldown_instead_of_permanent_receipt(tmp_path):
    now = [0.0]
    store = ProcessedVoices(tmp_path / "processed.jsonl", clock=lambda: now[0])
    message = voice()
    assert store.claim(message)
    store.release(message, failed=True)
    for _ in range(100):
        assert not store.claim(message)
    now[0] = 60
    assert store.claim(message)
    assert not store.completed


async def test_receipt_storage_compacts_and_retains_newest_entries(tmp_path):
    path = tmp_path / "processed.jsonl"
    store = ProcessedVoices(path, max_entries=2)
    for index in range(10):
        message = voice(index, unique=f"file-{index}")
        await store.complete([message])
    assert len(store.records) == 2
    assert len(store.completed) == 4
    assert len(path.read_text().splitlines()) <= 4
    reloaded = ProcessedVoices(path, max_entries=2)
    assert not reloaded.claim(voice(9, unique="file-9"))
    assert not reloaded.claim(voice(8, unique="file-8"))
    assert reloaded.claim(voice(0, unique="file-0"))


async def test_completing_receipt_twice_does_not_grow_file(tmp_path):
    path = tmp_path / "processed.jsonl"
    store = ProcessedVoices(path)
    await store.complete([voice()])
    before = path.read_bytes()
    await store.complete([voice()])
    assert path.read_bytes() == before


@pytest.mark.parametrize("content", ["broken", '["raw-message-id"]\n', "{}\n", "[]\n"])
def test_corrupt_receipts_fail_closed_instead_of_resetting_history(tmp_path, content):
    path = tmp_path / "processed.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError):
        ProcessedVoices(path)
    assert path.read_text() == content


async def test_storage_failure_stops_new_paid_work(tmp_path, monkeypatch):
    store = ProcessedVoices(tmp_path / "processed.jsonl")

    def fail_write(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(store, "_write", fail_write)
    with pytest.raises(OSError):
        await store.complete([voice()])
    assert not store.available and not store.completed
    assert not store.claim(voice(index=2, unique="new-audio"))


async def test_cancellation_waits_for_disk_write_and_keeps_completed_receipt(tmp_path, monkeypatch):
    path = tmp_path / "processed.jsonl"
    store = ProcessedVoices(path)
    entered, release = threading.Event(), threading.Event()
    original = store._write

    def slow_write(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        original(*args, **kwargs)

    monkeypatch.setattr(store, "_write", slow_write)
    task = asyncio.create_task(store.complete([voice()]))
    await asyncio.wait_for(asyncio.to_thread(entered.wait), 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not store.claim(voice())
    assert not ProcessedVoices(path).claim(voice())


def test_retry_results_preserve_whitespace_and_obey_memory_budget():
    now = [0.0]
    cache = RecentResults(max_chars=8, max_entries=2, ttl=10, clock=lambda: now[0])
    cache.put("first", "  а\n")
    assert cache.get("first") == "  а\n"
    cache.put("second", "12345")
    assert cache.get("first") is None
    assert cache.characters == 5
    cache.put("second", "x")
    assert cache.characters == 1
    now[0] = 10
    assert cache.get("second") is None
    assert cache.characters == 0
    cache.put("too-big", "x" * 9)
    assert not cache.entries
    cache.clear()


async def test_retry_results_expire_even_without_incoming_messages():
    cache = RecentResults(ttl=0.01)
    cache.put("voice", "private transcript")
    await asyncio.sleep(0.03)
    assert not cache.entries and cache.characters == 0
    assert cache._expiry is None
