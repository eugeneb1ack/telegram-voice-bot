import asyncio

import pytest

from voice_bot.conversation import ConversationMemory


def test_context_is_bounded_and_scoped_to_chat_topic_and_author():
    memory = ConversationMemory(900, 800, 200)
    key = (1, 2, 3)
    for i in range(4):
        memory.remember(key, 100 + i, (i,), "важная мысль " * 100)
    recent = memory.recent(key, 105, (5,))
    assert len(recent) == 2
    assert sum(map(len, recent)) <= 800
    for other in [(2, 2, 3), (1, 3, 3), (1, 2, 4)]:
        assert memory.recent(other, 105, (5,)) == ()


def test_expiration_and_lru_capacity_do_not_retain_unbounded_history():
    now = [0.0]
    memory = ConversationMemory(10, 100, 2, clock=lambda: now[0])
    for author in (1, 2):
        memory.remember((1, 0, author), 100, (1,), "мысль")
    memory.recent((1, 0, 1), 101, (2,))
    memory.remember((1, 0, 3), 101, (2,), "новое")
    assert (1, 0, 2) not in memory.sessions
    now[0] = 11
    memory.purge()
    assert not memory.sessions


def test_event_time_breaks_old_dialogues_and_retries_cannot_rewind_memory():
    memory = ConversationMemory(900, 800, 200)
    key = (1, 0, 2)
    memory.remember(key, 100, (10,), "Первое")
    memory.remember(key, 110, (11, 12), "Второе")
    assert memory.recent(key, 110, (11, 12)) == ("Первое",)
    assert memory.recent(key, 90, (9,)) == ()
    memory.remember(key, 90, (9,), "Старое")
    memory.remember(key, 110, (11, 12), "Дубликат")
    assert memory.recent(key, 120, (13,)) == ("Первое", "Второе")
    assert memory.recent(key, 86400, (14,)) == ()
    memory.remember(key, 86400, (14,), "На следующий день")
    assert memory.recent(key, 86401, (15,)) == ("На следующий день",)


async def test_author_jobs_are_ordered_and_cancelled_waiters_release_locks():
    memory = ConversationMemory(900, 800, 200)
    key = (1, 0, 2)
    entered = asyncio.Event()
    release = asyncio.Event()
    seen = []

    async def first():
        async with memory.ordered(key):
            entered.set()
            await release.wait()
            memory.remember(key, 100, (1,), "Первая мысль")

    async def second():
        async with memory.ordered(key):
            seen.extend(memory.recent(key, 101, (2,)))

    task = asyncio.create_task(first())
    await entered.wait()
    waiter = asyncio.create_task(second())
    cancelled = asyncio.create_task(second())
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert not seen
    release.set()
    await asyncio.gather(task, waiter)
    assert seen == ["Первая мысль"]
    assert not memory._locks
    memory.clear()


async def test_idle_memory_expires_without_another_message():
    memory = ConversationMemory(0.01, 100, 2)
    memory.remember((1, 0, 2), 100, (1,), "мысль")
    await asyncio.sleep(0.03)
    assert not memory.sessions
    assert memory._expiry is None


def test_disabled_memory_retains_nothing():
    memory = ConversationMemory(0, 800, 200)
    memory.remember((1, 0, 2), 100, (1,), "мысль")
    assert not memory.sessions
    assert memory.recent((1, 0, 2), 101, (2,)) == ()
