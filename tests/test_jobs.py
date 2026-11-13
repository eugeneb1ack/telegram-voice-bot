import asyncio
from collections import defaultdict
from types import SimpleNamespace

from voice_bot.jobs import DeferredJob, VoiceJobQueue


def message(chat, number):
    return SimpleNamespace(chat=SimpleNamespace(id=chat), message_id=number)


async def test_round_robin_preserves_order_without_starving_other_chats():
    completed = []

    async def handle(messages, bot):
        completed.append((messages[0].chat.id, messages[0].message_id))

    queue = VoiceJobQueue(handle, concurrency=1, max_pending=20, max_per_chat=10)
    for chat, number in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)]:
        assert queue.add([message(chat, number)], None)
    await queue.close(grace_seconds=2)
    assert completed == [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3)]
    assert not queue.pending and not queue.workers and not queue.active_chats


async def test_four_chats_run_in_parallel_with_bounded_concurrency():
    entered, release = asyncio.Event(), asyncio.Event()
    running = peak = 0
    completed = defaultdict(list)
    active_chats = set()

    async def handle(messages, bot):
        nonlocal running, peak
        item = messages[0]
        assert item.chat.id not in active_chats
        active_chats.add(item.chat.id)
        running += 1
        peak = max(peak, running)
        if running == 4:
            entered.set()
        await release.wait()
        completed[item.chat.id].append(item.message_id)
        running -= 1
        active_chats.remove(item.chat.id)

    queue = VoiceJobQueue(handle, concurrency=4, max_pending=100, max_per_chat=20)
    for chat in range(4):
        for number in range(10):
            assert queue.add([message(chat, number)], None)
    await asyncio.wait_for(entered.wait(), 2)
    assert running == 4
    release.set()
    await queue.close(grace_seconds=2)
    assert peak == 4
    assert dict(completed) == {chat: list(range(10)) for chat in range(4)}


async def test_global_and_per_chat_capacity_leave_room_for_other_chats():
    async def handle(messages, bot):
        pass

    queue = VoiceJobQueue(handle, concurrency=1, max_pending=3, max_per_chat=2)
    assert queue.add([message(1, 1)], None)
    assert queue.add([message(1, 2)], None)
    assert not queue.add([message(1, 3)], None)
    assert queue.add([message(2, 1)], None)
    assert not queue.add([message(3, 1)], None)
    await queue.close(grace_seconds=2)
    assert queue.size == 0
    assert not queue.add([message(1, 4)], None)


async def test_failed_job_does_not_kill_worker_or_log_content(caplog):
    completed = []

    async def handle(messages, bot):
        if messages[0].message_id == 1:
            raise RuntimeError("SECRET transcript")
        completed.append(messages[0].message_id)

    queue = VoiceJobQueue(handle, concurrency=1, max_pending=5, max_per_chat=5)
    assert queue.add([message(1, 1)], None)
    assert queue.add([message(1, 2)], None)
    await queue.close(grace_seconds=2)
    assert completed == [2]
    assert "RuntimeError" in caplog.text
    assert "SECRET" not in caplog.text


async def test_shutdown_deadline_cancels_worker_and_releases_pending_messages():
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def handle(messages, bot):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    queue = VoiceJobQueue(handle, concurrency=1, max_pending=5, max_per_chat=5)
    assert queue.add([message(1, 1)], None)
    assert queue.add([message(2, 1)], None)
    await asyncio.wait_for(entered.wait(), 2)
    await queue.close(grace_seconds=0.01)
    assert cancelled.is_set()
    assert not queue.workers and not queue.pending and not queue.active_chats
    assert queue.size == 0
    await queue.close(grace_seconds=0)


async def test_deferred_head_releases_worker_but_preserves_chat_order_and_capacity():
    events = []
    attempts = 0

    async def handle(messages, bot):
        nonlocal attempts
        item = messages[0]
        if (item.chat.id, item.message_id) == (1, 1):
            attempts += 1
            if attempts <= 3:
                events.append("deferred")
                raise DeferredJob(0.01)
        events.append((item.chat.id, item.message_id))
        assert queue.size + len(queue.active_chats) <= queue.max_pending + queue.concurrency

    queue = VoiceJobQueue(handle, concurrency=1, max_pending=3, max_per_chat=2)
    assert queue.add([message(1, 1)], None)
    assert queue.add([message(1, 2)], None)
    assert queue.add([message(2, 1)], None)
    await queue.close(grace_seconds=2)
    assert events == ["deferred", (2, 1), "deferred", "deferred", (1, 1), (1, 2)]
    assert queue.size == 0 and not queue.pending and not queue.active_chats


async def test_new_chat_wakes_worker_waiting_for_a_deferred_deadline():
    deferred, completed = asyncio.Event(), asyncio.Event()

    async def handle(messages, bot):
        if messages[0].chat.id == 1:
            deferred.set()
            raise DeferredJob(60)
        completed.set()

    queue = VoiceJobQueue(handle, concurrency=1, max_pending=2, max_per_chat=2)
    assert queue.add([message(1, 1)], None)
    await asyncio.wait_for(deferred.wait(), 1)
    assert queue.add([message(2, 1)], None)
    await asyncio.wait_for(completed.wait(), 1)
    assert queue.size == 1
    await queue.close(grace_seconds=0)
