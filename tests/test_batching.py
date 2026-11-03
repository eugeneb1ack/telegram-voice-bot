import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from voice_bot.batching import VoiceBatcher


def forwarded(index, sender=1, chat=1, topic=0, seconds=0):
    return SimpleNamespace(
        message_id=index,
        date=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds),
        from_user=SimpleNamespace(id=sender),
        forward_origin=object(),
        chat=SimpleNamespace(id=chat),
        message_thread_id=topic,
        is_topic_message=bool(topic),
    )


async def finish(batcher):
    batcher.finish()
    await asyncio.gather(*batcher.tasks)


async def test_burst_debounces_to_one_sorted_batch_without_duplicate_parts():
    handler = AsyncMock()
    batcher = VoiceBatcher(handler, 0.02, 10, 8)
    bot = object()
    for index in (3, 1, 1, 2):
        assert batcher.add(forwarded(index), bot)
    await asyncio.sleep(0.04)
    handler.assert_awaited_once()
    assert [m.message_id for m in handler.await_args.args[0]] == [1, 2, 3]
    assert handler.await_args.args[1] is bot
    assert not batcher.pending
    assert not batcher.tasks


async def test_different_forwarders_chats_and_topics_never_merge():
    handler = AsyncMock()
    batcher = VoiceBatcher(handler, 3, 10, 8)
    for kwargs in ({}, {"sender": 2}, {"chat": 2}, {"topic": 2}):
        assert batcher.add(forwarded(1, **kwargs), object())
    await finish(batcher)
    assert handler.await_count == 4
    assert all(len(call.args[0]) == 1 for call in handler.await_args_list)


async def test_message_count_time_gap_and_queue_capacity_bound_batches():
    handler = AsyncMock()
    batcher = VoiceBatcher(handler, 3, 2, 3)
    assert batcher.add(forwarded(1), object())
    assert batcher.add(forwarded(2), object())
    assert batcher.add(forwarded(3), object())
    assert batcher.add(forwarded(4, seconds=60), object())
    assert not batcher.add(forwarded(5, sender=2), object())
    await finish(batcher)
    assert sorted(len(call.args[0]) for call in handler.await_args_list) == [1, 1, 2]
    assert not batcher.add(forwarded(6), object())


async def test_background_failure_does_not_log_exception_body(caplog):
    batcher = VoiceBatcher(AsyncMock(side_effect=RuntimeError("SECRET")), 3, 10, 8)
    batcher.add(forwarded(1), object())
    await finish(batcher)
    assert "SECRET" not in caplog.text
    assert "RuntimeError" in caplog.text
