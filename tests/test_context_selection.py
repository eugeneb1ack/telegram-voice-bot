"""A new voice block must not inherit an unrelated recent conversation."""

from dataclasses import replace

from test_dialogue import setup, text, voice
from test_shared_dialogue import memory, ref, remember

from voice_bot.conversation import Entry
from voice_bot.dialogue_graph import shared_dialogue


async def test_new_block_excludes_old_text_recap_and_other_participants(settings):
    service, bot, _ = setup(settings)
    await text(service, 1, "Старая тема про велосипеды.", seconds=0)
    await voice(service, bot, 2, seconds=10, duration=40)
    service.summarizer.summarize.return_value = "Новый итог."
    await text(service, 3, "Старый разговор о ремонте.", seconds=20)
    await text(service, 4, "Чужой разговор о рыбалке.", author=333, seconds=30)
    for mid in (5, 6, 7):
        last = await voice(service, bot, mid, seconds=mid * 10)
    call = service.summarizer.summarize.await_args
    graph = call.kwargs["context"].dialogue
    assert [n.message_id for n in graph.messages] == [5, 6, 7]
    assert not graph.has_previous_summary
    assert call.kwargs["context"].recent_summaries == ()
    assert "Коротко · 3 голосовых:" in last.answer.await_args.args[0]
    await service.close()


def test_only_reply_ancestors_are_eligible_not_siblings_or_same_recipient_history():
    mem = memory()
    remember(mem, 1, 2)
    remember(mem, 2, 3, refs=(ref(2, 1, 2),))
    remember(mem, 3, 4, refs=(ref(3, 1, 2),))  # Sibling, not a source for this answer.
    remember(mem, 4, 2)  # Same recipient, different topic.
    remember(mem, 10, refs=(ref(10, 2, 3),))
    remember(mem, 11)
    remember(mem, 12)
    sources = mem.sessions[(-100, 0, 1)].entries
    graph = shared_dialogue(mem, (-100, 0, 1), sources, (ref(10, 2, 3),), 12, 12)
    assert {n.message_id for n in graph.messages} == {1, 2, 10, 11, 12}
    assert graph.branches[0].recipient_id == 3
    assert graph.branches[0].context_message_ids == (1, 2)
    assert graph.branches[1].context_message_ids == ()


def test_reply_to_summarized_voice_uses_exact_transcript_not_multitopic_recap():
    mem = memory()
    remember(mem, 1)
    session = mem.sessions[(-100, 0, 1)]
    session.entries[0] = replace(session.entries[0], summary="Чужая тема из общего итога.")
    session.summarized_through = 1
    reference = replace(ref(10, 1, 1), text="Точный текст записи 1")
    graph = shared_dialogue(
        mem, (-100, 0, 1), [Entry((10,), 10, "Уточнение")], (reference,), 10, 10
    )
    assert next(n.text for n in graph.messages if n.message_id == 1) == reference.text
    assert not graph.has_previous_summary
    assert all(n.kind != "summary" for n in graph.messages)


def test_reply_chain_has_bounded_depth_and_excludes_other_scope_and_future():
    mem = memory()
    for mid in range(1, 11):
        remember(mem, mid, 2, refs=(ref(mid, mid - 1, 2),) if mid > 1 else ())
    remember(mem, 11)
    reference = ref(11, 10, 2)
    graph = shared_dialogue(mem, (-100, 0, 1), [Entry((11,), 11, "Ответ")], (reference,), 11, 11)
    assert {n.message_id for n in graph.messages} == {8, 9, 10, 11}


def test_explicit_other_topic_is_a_leaf_and_never_opens_other_history():
    mem = memory()
    remember(mem, 1, 2, scope=(-100, 7))
    remember(mem, 2, 2, scope=(-100, 7), refs=(ref(2, 1, 2),))
    remember(mem, 3, 2, scope=(-200, 0))
    remember(mem, 30, 2)  # Future arrival cannot become background.
    remember(mem, 10)
    graph = shared_dialogue(
        mem, (-100, 0, 1), [Entry((10,), 10, "Ответ")], (ref(10, 2, 2),), 10, 10
    )
    assert {n.message_id for n in graph.messages} == {2, 10}
    assert next(n for n in graph.messages if n.message_id == 2).reply_to_message_id is None


async def test_standalone_summary_does_not_read_unlinked_recent_messages(settings):
    service, bot, _ = setup(settings)
    await text(service, 1, "Совсем другая тема", seconds=0)
    await voice(service, bot, 2, seconds=10, duration=40)
    context = service.summarizer.summarize.await_args.kwargs["context"]
    assert context.recent_summaries == ()
    await service.close()
