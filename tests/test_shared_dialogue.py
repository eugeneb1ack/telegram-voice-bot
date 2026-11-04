import asyncio

import pytest
from test_dialogue import setup, text, voice

from voice_bot.conversation import Anchor, ConversationMemory
from voice_bot.dialogue import DialogueContext
from voice_bot.dialogue_graph import shared_dialogue
from voice_bot.errors import ProviderError
from voice_bot.reply_context import ReplyContext


def memory(*, max_entries=50, max_chars=12000, path=None):
    return ConversationMemory(900, max_chars, 200, lambda: 1000, max_entries=max_entries, path=path)


def remember(mem, mid, author=1, *, time=None, kind="voice", refs=(), scope=(-100, 0)):
    mem.remember(
        (*scope, author),
        mid if time is None else time,
        (mid,),
        f"Реплика {mid}",
        kind=kind,
        anchor=Anchor(mid, "Одинаковое имя", f"user{author}", "supergroup"),
        reply_contexts=refs,
    )


def ref(mid, target, author, name="Анна", kind="text"):
    return ReplyContext(
        mid, target, name, f"user{author}", kind, "Текст источника", author_id=author
    )


def branch_fixture():
    mem = memory()
    refs = (ref(10, 1, 2), ref(11, 2, 3, name="Анна"), ref(13, 3, 2))
    for mid in (10, 11, 12, 13):
        remember(mem, mid, refs=tuple(r for r in refs if r.message_id == mid))
    return shared_dialogue(mem, (-100, 0, 1), mem.sessions[(-100, 0, 1)].entries, refs, 14, 14)


def test_recipients_use_ids_and_unaddressed_message_is_separate():
    graph = branch_fixture()
    assert [(b.recipient_id, b.message_ids) for b in graph.branches] == [
        (2, (10, 13)),
        (3, (11,)),
        (None, (12,)),
    ]
    assert [b.is_reply for b in graph.branches] == [True, True, False]
    assert all(n.author_id == 1 for n in graph.messages if n.message_id >= 10)


def test_twenty_unlinked_participants_keep_separate_counts_and_sources(settings):
    settings.context_summary_every = 3
    mem = memory()
    context = DialogueContext(mem, settings, lambda *_: True)
    for round_number in range(2):
        for author in range(1, 21):
            mid = round_number * 20 + author
            assert context.prepare((-100, 0, author), mid, mid, f"Реплика {mid}", 10) is None
            remember(mem, mid, author)
    # Foreign chat/topic/forward sources and future updates cannot enter ambient context.
    remember(mem, 3, 50, scope=(-200, 0))
    remember(mem, 4, 51, scope=(-100, 8))
    mem.remember((-100, 0, 52, 999), 5, (5,), "Пересылка")
    remember(mem, 100, 53)
    graphs = []
    for author in (1, 2):
        result = context.prepare((-100, 0, author), 41, 41, "Третье", 10)
        assert result is not None and result.voice_count == 3
        graphs.append(result.shared)
    assert [g.focus_author_id for g in graphs] == [1, 2]
    for graph in graphs:
        assert {n.author_id for n in graph.messages} == {graph.focus_author_id}
        assert len(graph.messages) == 3
        assert all(n.message_id <= 41 for n in graph.messages)
        assert len(graph.branches) == 1
    assert graphs[0].branches[0].message_ids == (1, 21, 41)
    assert graphs[1].branches[0].message_ids == (2, 22, 41)


async def test_hard_window_restarts_even_with_continuous_activity(settings):
    service, bot, _ = setup(settings)
    for mid, seconds in [(1, 0), (2, 200), (3, 400), (4, 600), (5, 800), (6, 1000)]:
        last = await voice(service, bot, mid, seconds=seconds)
        assert ("Коротко · 3 голосовых:" in last.answer.await_args.args[0]) == (mid in (3, 6))
    call = service.summarizer.summarize.await_args
    assert all(f"Мысль {mid}." in call.args[0] for mid in (4, 5, 6))
    assert all(f"Мысль {mid}." not in call.args[0] for mid in (1, 2, 3))
    await service.close()


async def test_standalone_long_summary_starts_new_triplet_without_automatic_recap(settings):
    service, bot, _ = setup(settings)
    service.summarizer.summarize.return_value = "Готовый пересказ длинной записи."
    first = await voice(service, bot, 1, seconds=0, duration=40)
    assert "Коротко:" in first.answer.await_args.args[0]
    await voice(service, bot, 2, seconds=10)
    await service.close()
    service, bot, _ = setup(settings)
    third = await voice(service, bot, 3, seconds=20)
    assert "Коротко" not in third.answer.await_args.args[0]
    fourth = await voice(service, bot, 4, seconds=30)
    assert "Коротко · 3 голосовых:" in fourth.answer.await_args.args[0]
    call = service.summarizer.summarize.await_args
    assert "Мысль 1." not in call.args[0]
    assert all(f"Мысль {n}." in call.args[0] for n in (2, 3, 4))
    graph = call.kwargs["context"].dialogue
    assert [m.message_id for m in graph.messages] == [2, 3, 4]
    assert not graph.has_previous_summary
    assert all(1 not in b.message_ids for b in graph.branches)
    for n in (5, 6, 7):
        seventh = await voice(service, bot, n, seconds=n * 10)
    assert "Коротко · 3 голосовых:" in seventh.answer.await_args.args[0]
    assert service.summarizer.summarize.await_count == 2
    await service.close()


async def test_failed_standalone_summary_does_not_mark_voice_as_summarized(settings):
    service, bot, _ = setup(settings)
    service.summarizer.summarize.side_effect = ProviderError("Недоступно")
    await voice(service, bot, 1, seconds=0, duration=40)
    service.summarizer.summarize.side_effect = None
    await voice(service, bot, 2, seconds=10)
    third = await voice(service, bot, 3, seconds=20)
    assert "Коротко · 3 голосовых:" in third.answer.await_args.args[0]
    assert "Мысль 1." in service.summarizer.summarize.await_args.args[0]
    await service.close()


@pytest.mark.parametrize("preceding", [0, 1, 2, 3])
@pytest.mark.parametrize("duration", [14, 15, 50])
async def test_long_voice_always_summarizes_at_any_position(settings, preceding, duration):
    service, bot, _ = setup(settings)
    for mid in range(1, preceding + 1):
        await voice(service, bot, mid, seconds=mid * 10)
    before = service.summarizer.summarize.await_count
    current = await voice(service, bot, preceding + 1, seconds=50, duration=duration)
    expected = duration >= 15 or preceding == 2
    assert ("Коротко" in current.answer.await_args.args[0]) == expected
    assert service.summarizer.summarize.await_count - before == int(expected)
    assert f"Мысль {preceding + 1}." in current.answer.await_args.args[0]
    if duration >= 15 and preceding == 1:
        # A long voice closes the pending short context too, so it cannot disappear.
        assert all(
            f"Мысль {mid}." in service.summarizer.summarize.await_args.args[0] for mid in (1, 2)
        )
    await service.close()


async def test_long_voice_after_restart_does_not_wait_for_triplet(settings):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1, seconds=0)
    await service.close()
    service, bot, _ = setup(settings)
    long = await voice(service, bot, 2, seconds=10, duration=50)
    assert "Коротко" in long.answer.await_args.args[0]
    service.summarizer.summarize.assert_awaited_once()
    for mid in (3, 4):
        short = await voice(service, bot, mid, seconds=mid * 10)
        assert "Коротко" not in short.answer.await_args.args[0]
    fifth = await voice(service, bot, 5, seconds=50)
    assert "Коротко · 3 голосовых:" in fifth.answer.await_args.args[0]
    assert service.summarizer.summarize.await_count == 2
    await service.close()


async def test_ignored_text_and_restart_preserve_complete_third_voice(settings):
    settings.context_max_messages = 2
    service, bot, _ = setup(settings)
    await voice(service, bot, 1, seconds=0)
    await voice(service, bot, 2, seconds=20)
    for mid in range(3, 10):
        await text(service, mid, f"Текст {mid}", seconds=mid * 10)
    await service.close()
    service, bot, _ = setup(settings)
    last = await voice(service, bot, 10, seconds=100)
    assert "Коротко · 3 голосовых:" in last.answer.await_args.args[0]
    assert not service.summarizer.summarize.await_args.kwargs["context"].dialogue.partial
    for mid in (11, 12, 13):
        last = await voice(service, bot, mid, seconds=mid * 10)
    assert "Коротко · 3 голосовых:" in last.answer.await_args.args[0]
    assert not service.summarizer.summarize.await_args.kwargs["context"].dialogue.partial
    assert service.summarizer.summarize.await_count == 2
    await service.close()


async def test_future_text_while_stt_pending_does_not_move_series(settings):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1, seconds=0)
    await voice(service, bot, 2, seconds=10)
    entered, release = asyncio.Event(), asyncio.Event()

    async def transcribe(*args, **kwargs):
        entered.set()
        await release.wait()
        return "Третья мысль"

    service.transcriber.transcribe.side_effect = transcribe
    task = asyncio.create_task(voice(service, bot, 3, seconds=20))
    await entered.wait()
    await text(service, 4, "Будущий текст", seconds=950)
    release.set()
    third = await task
    assert "Коротко · 3 голосовых" in third.answer.await_args.args[0]
    assert "Будущий текст" not in service.summarizer.summarize.await_args.args[0]
    service.transcriber.transcribe.side_effect = None
    later = await voice(service, bot, 5, seconds=960)
    assert "Коротко" not in later.answer.await_args.args[0]
    await service.close()
